"""
SD1.5 Feature Extraction + Symbolic Caption Synthesis (Streaming)
==================================================================
Streaming parquet writes - only holds 1 batch in memory at a time.

File: geovocab2/data/teacher/sd15_caption_integration.py
"""

import torch
from typing import Optional, Dict, List, Iterator
import numpy as np
from concurrent.futures import ProcessPoolExecutor, Future
import time
import tempfile
import shutil
from pathlib import Path

from geovocab2.data.prompt.symbolic_tree import SynthesisSystem
from geovocab2.data.teacher.extract_sd15 import SD15FeatureExtractor, SD15ExtractionConfig


class BatchedCaptionGenerator:
    """Generates caption batches for extraction pipeline."""

    def __init__(
        self,
        synthesizer: SynthesisSystem,
        batch_size: int = 32,
        num_batches: Optional[int] = None,
        complexity_distribution: Optional[Dict[int, float]] = None,
        seed: Optional[int] = None
    ):
        self.synthesizer = synthesizer
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.complexity_dist = complexity_distribution
        self.rng = np.random.RandomState(seed) if seed else np.random.RandomState()
        self.batch_count = 0

    def __iter__(self) -> Iterator[List[str]]:
        while True:
            if self.num_batches and self.batch_count >= self.num_batches:
                break

            batch = []
            for _ in range(self.batch_size):
                if self.complexity_dist:
                    complexities = list(self.complexity_dist.keys())
                    probs = list(self.complexity_dist.values())
                    complexity = self.rng.choice(complexities, p=probs)
                else:
                    complexity = 3

                result = self.synthesizer.synthesize(complexity=complexity)
                caption = result["text"]
                batch.append(caption)

            self.batch_count += 1
            yield batch

    def __len__(self):
        return self.num_batches if self.num_batches else float('inf')


class SimpleDataLoader:
    """DataLoader wrapper for BatchedCaptionGenerator."""
    def __init__(self, generator):
        self.generator = generator
    def __iter__(self):
        return iter(self.generator)
    def __len__(self):
        return len(self.generator)


def upload_parquet_file(parquet_path: str, hf_repo_id: str, private: bool, commit_msg: str):
    """Upload a parquet file to HuggingFace (runs in worker process)."""
    from datasets import Dataset as HFDataset

    dataset = HFDataset.from_parquet(parquet_path)
    dataset.push_to_hub(hf_repo_id, private=private, commit_message=commit_msg, max_shard_size="2GB")
    Path(parquet_path).unlink()


class StreamingExtractor:
    """Extracts features and streams them directly to Parquet files."""

    def __init__(self, extractor: SD15FeatureExtractor, max_workers: int = 2):
        self.extractor = extractor
        self.executor = ProcessPoolExecutor(max_workers=max_workers)
        self.pending_upload: Optional[Future] = None
        self.temp_dir = Path(tempfile.mkdtemp(prefix="sd15_extract_"))
        self.samples_since_upload = 0

    def extract_and_stream(self, dataloader, max_samples: Optional[int] = None):
        """Extract features with streaming writes."""
        from tqdm import tqdm
        import pyarrow as pa
        import pyarrow.parquet as pq

        max_samples = max_samples or self.extractor.config.max_samples

        print(f"\n{'='*80}")
        print(f"EXTRACTION PIPELINE (streaming parquet)")
        print(f"{'='*80}")
        print(f"Target samples: {max_samples or 'unlimited'}")
        print(f"Upload interval: {self.extractor.config.upload_interval:,} samples (~2GB)")
        print(f"Active blocks: 9 (full SD1.5 UNet)")
        print(f"Memory: Streaming (only current upload buffer in RAM)")
        print(f"{'='*80}\n")

        self.extractor.register_hooks(self.extractor.config.active_blocks)

        pbar = tqdm(desc="Extracting", unit="batch")
        batch_times = []

        current_parquet_path = self.temp_dir / "current.parquet"
        parquet_writer = None
        parquet_schema = None

        for batch_idx, prompts in enumerate(dataloader):
            if max_samples and self.extractor.state.total_extracted >= max_samples:
                print(f"\n✓ Reached target of {max_samples:,} samples")
                break

            batch_start = time.time()

            B = len(prompts)
            timesteps = torch.randint(0, 1000, (B,), device=self.extractor.device)

            features, clip_emb, clip_pooled = self.extractor.extract_batch(
                prompts, timesteps, accumulate=False
            )

            batch_data = {
                'prompt': pa.array(prompts),
                'timestep': pa.array(timesteps.cpu().numpy()),
                'sample_id': pa.array(np.arange(
                    self.extractor.state.total_extracted,
                    self.extractor.state.total_extracted + B
                )),
            }

            for layer_name in self.extractor.config.active_blocks:
                simple_name = self.extractor.config.block_name_mapping[layer_name]
                feat = features[simple_name].cpu().numpy()
                batch_data[f'features_{simple_name}'] = pa.array(feat.reshape(B, -1).tolist())

            if clip_emb is not None:
                clip_flat = clip_emb.numpy().reshape(B, -1)
                batch_data['clip_embeddings'] = pa.array(clip_flat.tolist())
            if clip_pooled is not None:
                batch_data['clip_pooled'] = pa.array(clip_pooled.numpy().tolist())

            table = pa.Table.from_pydict(batch_data)

            if parquet_writer is None:
                parquet_schema = table.schema
                parquet_writer = pq.ParquetWriter(current_parquet_path, parquet_schema)

            parquet_writer.write_table(table)

            self.samples_since_upload += B
            self.extractor.state.total_extracted += B

            del features, clip_emb, clip_pooled, batch_data, table
            torch.cuda.empty_cache()

            batch_time = time.time() - batch_start
            batch_times.append(batch_time)

            pbar.update(1)

            if (batch_idx + 1) % self.extractor.config.log_interval == 0:
                avg_time = np.mean(batch_times[-100:])
                sps = B / avg_time
                pbar.set_postfix({
                    'extracted': f"{self.extractor.state.total_extracted:,}",
                    'uploaded': f"{self.extractor.state.total_uploaded:,}",
                    'sps': f"{sps:.1f}",
                })

            if self._should_upload():
                parquet_writer.close()
                parquet_writer = None
                self._async_upload(current_parquet_path)
                current_parquet_path = self.temp_dir / f"chunk_{self.extractor.state.total_uploaded}.parquet"

            if (self.extractor.config.checkpoint_interval > 0 and
                self.extractor.state.total_extracted % self.extractor.config.checkpoint_interval == 0):
                self.extractor.save_checkpoint()

        pbar.close()

        if parquet_writer is not None:
            parquet_writer.close()

        if self.samples_since_upload > 0:
            self._async_upload(current_parquet_path, force=True)

        self._wait_for_upload()
        self.extractor.save_checkpoint()

        shutil.rmtree(self.temp_dir)
        self.executor.shutdown(wait=True)

        print(f"\n✓ Complete!")
        print(f"Total extracted: {self.extractor.state.total_extracted:,}")
        print(f"Total uploaded: {self.extractor.state.total_uploaded:,}")

    def _should_upload(self) -> bool:
        if self.extractor.config.upload_interval == 0:
            return False
        return self.samples_since_upload >= self.extractor.config.upload_interval

    def _async_upload(self, parquet_path: Path, force: bool = False):
        if not force and not self._should_upload():
            return
        if self.samples_since_upload == 0:
            return
        if self.extractor.config.hf_repo_id is None:
            return

        self._wait_for_upload()

        num_samples = self.samples_since_upload
        total_after = self.extractor.state.total_uploaded + num_samples

        print(f"\n{'='*70}")
        print(f"📤 UPLOADING PARQUET (async)")
        print(f"{'='*70}")
        print(f"Samples: {num_samples:,}")
        print(f"Total: {self.extractor.state.total_uploaded:,} → {total_after:,}")

        upload_path = self.temp_dir / f"upload_{total_after}.parquet"
        shutil.copy(str(parquet_path), str(upload_path))

        commit_msg = f"Add {num_samples:,} samples (total: {total_after:,})"

        self.pending_upload = self.executor.submit(
            upload_parquet_file,
            str(upload_path),
            self.extractor.config.hf_repo_id,
            self.extractor.config.private_repo,
            commit_msg
        )

        self.extractor.state.total_uploaded += num_samples
        self.extractor.state.num_uploads += 1
        self.samples_since_upload = 0

        print(f"✓ Upload queued (worker processing)")

    def _wait_for_upload(self):
        if self.pending_upload is not None:
            print("⏳ Waiting for upload...")
            try:
                self.pending_upload.result()
                print("✓ Upload complete")
            except Exception as e:
                print(f"❌ Upload failed: {e}")
            finally:
                self.pending_upload = None


def create_working_config(
    active_blocks: Optional[List[str]] = None,
    extract_clip: bool = True,
    clip_pooled: bool = True,
    hf_repo_id: Optional[str] = None,
    upload_interval: int = 4_000,
    checkpoint_interval: int = 50_000,
    num_samples: Optional[int] = None
) -> SD15ExtractionConfig:
    config = SD15ExtractionConfig()

    config.__dict__['layer_hook_configs'] = {}
    config.__dict__['default_hook_config'] = None

    class SimpleCacheSchema:
        def __init__(self):
            self.layer_configs = {}

    config.__dict__['cache_schema'] = SimpleCacheSchema()

    if active_blocks is None:
        active_blocks = [
            'down_blocks.0.resnets.1',
            'down_blocks.1.resnets.1',
            'down_blocks.2.resnets.1',
            'down_blocks.3.resnets.1',
            'mid_block.resnets.1',
            'up_blocks.0.resnets.2',
            'up_blocks.1.resnets.2',
            'up_blocks.2.resnets.2',
            'up_blocks.3.resnets.2',
        ]

    config.active_blocks = active_blocks
    config.extract_clip_embeddings = extract_clip
    config.clip_pooled = clip_pooled
    config.hf_repo_id = hf_repo_id
    config.upload_interval = upload_interval
    config.checkpoint_interval = checkpoint_interval
    config.max_samples = num_samples

    return config


def extract_sd15_with_symbolic_captions(
    num_samples: int = 10_000,
    batch_size: int = 64,
    hf_repo_id: Optional[str] = None,
    upload_interval: int = 4_000,
    checkpoint_interval: int = 50_000,
    device: str = "cuda",
    bias_weights_path: Optional[str] = None,
    complexity_distribution: Optional[Dict[int, float]] = None,
    seed: Optional[int] = None,
    extract_clip: bool = True,
    clip_pooled: bool = True,
    active_blocks: Optional[List[str]] = None,
    max_upload_workers: int = 2
):
    if bias_weights_path:
        import json
        with open(bias_weights_path, 'r') as f:
            bias_weights = json.load(f)
        print(f"✓ Loaded {len(bias_weights)} bias weights")
    else:
        bias_weights = None

    print("Initializing synthesizer...")
    synthesizer = SynthesisSystem()
    if bias_weights and hasattr(synthesizer, 'apply_bias_weights'):
        synthesizer.apply_bias_weights(bias_weights)
    print("✓ Synthesizer ready")

    caption_generator = BatchedCaptionGenerator(
        synthesizer=synthesizer,
        batch_size=batch_size,
        num_batches=num_samples // batch_size,
        complexity_distribution=complexity_distribution,
        seed=seed
    )
    dataloader = SimpleDataLoader(caption_generator)

    extraction_config = create_working_config(
        active_blocks=active_blocks,
        extract_clip=extract_clip,
        clip_pooled=clip_pooled,
        hf_repo_id=hf_repo_id,
        upload_interval=upload_interval,
        checkpoint_interval=checkpoint_interval,
        num_samples=num_samples
    )

    print(f"Initializing SD15 on {device}...")
    extractor = SD15FeatureExtractor(config=extraction_config, device=device)
    print("✓ Extractor ready\n")

    print(f"Config: {num_samples:,} samples, batch_size={batch_size}")
    print(f"Upload: every {upload_interval:,} samples (~2GB with 9 blocks)")
    if hf_repo_id:
        print(f"Repo: {hf_repo_id}")
    print()

    streaming_extractor = StreamingExtractor(extractor, max_workers=max_upload_workers)
    streaming_extractor.extract_and_stream(dataloader, max_samples=num_samples)

    print("\n✓ Complete!")
    if hf_repo_id:
        print(f"https://huggingface.co/datasets/{hf_repo_id}")

    return extractor


if __name__ == "__main__":
    print("SD1.5 + Symbolic Captions Integration (Streaming)")