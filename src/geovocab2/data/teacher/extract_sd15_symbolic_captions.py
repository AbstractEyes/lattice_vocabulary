"""
SD1.5 Feature Extraction + Symbolic Caption Synthesis (Optimized)
==================================================================
With async packaging workers and optimized uploads.

File: geovocab2/data/teacher/sd15_caption_integration.py
"""

import torch
from typing import Optional, Dict, List, Iterator
import numpy as np
from concurrent.futures import ProcessPoolExecutor, Future
from dataclasses import dataclass, field
import time

from geovocab2.data.prompt.synthesis_tree import SynthesisSystem
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


def package_and_upload_batch(data_dict: Dict, hf_repo_id: str, private: bool, commit_msg: str) -> int:
    """
    Package data and upload to HuggingFace (runs in worker process).

    Returns:
        Number of samples uploaded
    """
    from datasets import Dataset as HFDataset
    import numpy as np

    # Stack arrays for faster Arrow conversion
    stacked_data = {}
    num_samples = len(data_dict['prompt'])

    for key, values in data_dict.items():
        if key in ['prompt']:
            stacked_data[key] = values
        elif key in ['timestep', 'sample_id']:
            stacked_data[key] = np.array(values)
        elif key.startswith('features_') or key.startswith('clip_'):
            # Stack feature arrays: [N, ...] instead of list of N arrays
            stacked_data[key] = np.stack(values)

    # Create dataset from stacked arrays (much faster)
    dataset = HFDataset.from_dict(stacked_data)

    # Upload with controlled shard size (~1GB per shard)
    dataset.push_to_hub(
        hf_repo_id,
        private=private,
        commit_message=commit_msg,
        max_shard_size="1GB"  # Control shard size
    )

    return num_samples


def create_working_config(
    active_blocks: Optional[List[str]] = None,
    extract_clip: bool = True,
    clip_pooled: bool = True,
    hf_repo_id: Optional[str] = None,
    upload_interval: int = 4_000,  # Changed to 4k (~1GB)
    checkpoint_interval: int = 50_000,
    num_samples: Optional[int] = None
) -> SD15ExtractionConfig:
    """Create a properly initialized SD15ExtractionConfig."""
    config = SD15ExtractionConfig()

    # Bypass Field descriptors
    config.__dict__['layer_hook_configs'] = {}
    config.__dict__['default_hook_config'] = None

    # Create cache_schema with layer_configs dict
    class SimpleCacheSchema:
        def __init__(self):
            self.layer_configs = {}

    config.__dict__['cache_schema'] = SimpleCacheSchema()

    # Set custom values - ALL 9 BLOCKS by default
    if active_blocks is None:
        active_blocks = [
            'down_blocks.0.resnets.1',  # down_0: (320, 64, 64)
            'down_blocks.1.resnets.1',  # down_1: (640, 32, 32)
            'down_blocks.2.resnets.1',  # down_2: (1280, 16, 16)
            'down_blocks.3.resnets.1',  # down_3: (1280, 8, 8)
            'mid_block.resnets.1',       # mid: (1280, 8, 8)
            'up_blocks.0.resnets.2',     # up_0: (1280, 16, 16)
            'up_blocks.1.resnets.2',     # up_1: (1280, 32, 32)
            'up_blocks.2.resnets.2',     # up_2: (640, 64, 64)
            'up_blocks.3.resnets.2',     # up_3: (320, 64, 64)
        ]

    config.active_blocks = active_blocks
    config.extract_clip_embeddings = extract_clip
    config.clip_pooled = clip_pooled
    config.hf_repo_id = hf_repo_id
    config.upload_interval = upload_interval
    config.checkpoint_interval = checkpoint_interval
    config.max_samples = num_samples

    return config


class AsyncUploadExtractor:
    """Wrapper around SD15FeatureExtractor with async upload workers."""

    def __init__(self, extractor: SD15FeatureExtractor, max_workers: int = 2):
        self.extractor = extractor
        self.executor = ProcessPoolExecutor(max_workers=max_workers)
        self.pending_upload: Optional[Future] = None

    def extract_and_queue(self, dataloader, max_samples: Optional[int] = None):
        """Extract features with async uploads."""
        from tqdm import tqdm

        max_samples = max_samples or self.extractor.config.max_samples

        print(f"\n{'='*80}")
        print(f"EXTRACTION PIPELINE (with async uploads)")
        print(f"{'='*80}")
        print(f"Target samples: {max_samples or 'unlimited'}")
        print(f"Upload interval: {self.extractor.config.upload_interval:,} samples (~2GB)")
        print(f"Active blocks: 9 (full SD1.5 UNet)")
        print(f"Max upload workers: {self.executor._max_workers}")
        print(f"{'='*80}\n")

        # Register hooks once
        self.extractor.register_hooks(self.extractor.config.active_blocks)

        pbar = tqdm(desc="Extracting", unit="batch")
        batch_times = []

        for batch_idx, prompts in enumerate(dataloader):
            if max_samples and self.extractor.state.total_extracted >= max_samples:
                print(f"\n✓ Reached target of {max_samples:,} samples")
                break

            batch_start = time.time()

            # Extract features (GPU)
            B = len(prompts)
            timesteps = torch.randint(0, 1000, (B,), device=self.extractor.device)
            self.extractor.extract_batch(prompts, timesteps, accumulate=True)

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

            # Check if we need to upload
            if self._should_upload():
                self._async_upload()

            # Checkpoint if needed
            if (self.extractor.config.checkpoint_interval > 0 and
                self.extractor.state.total_extracted % self.extractor.config.checkpoint_interval == 0):
                self.extractor.save_checkpoint()

        pbar.close()

        # Final upload
        self._async_upload(force=True)
        self._wait_for_upload()

        self.extractor.save_checkpoint()
        self.extractor._print_stats()

        # Cleanup
        self.executor.shutdown(wait=True)

    def _should_upload(self) -> bool:
        """Check if upload needed."""
        if self.extractor.config.upload_interval == 0:
            return False

        samples_since_upload = (
            self.extractor.state.total_extracted -
            self.extractor.state.total_uploaded
        )

        return (samples_since_upload >= self.extractor.config.upload_interval or
                self.extractor.num_accumulated >= self.extractor.config.max_samples_in_memory)

    def _async_upload(self, force: bool = False):
        """Start async upload of accumulated data."""
        if not force and not self._should_upload():
            return

        if self.extractor.num_accumulated == 0:
            return

        if self.extractor.config.hf_repo_id is None:
            return

        # Wait for previous upload to finish
        self._wait_for_upload()

        # Prepare data for upload
        num_samples = self.extractor.num_accumulated
        total_after = self.extractor.state.total_uploaded + num_samples

        print(f"\n{'='*70}")
        print(f"📤 QUEUEING UPLOAD (async)")
        print(f"{'='*70}")
        print(f"Samples: {num_samples:,}")
        print(f"Total: {self.extractor.state.total_uploaded:,} → {total_after:,}")

        # Copy data for worker (avoid mutation during upload)
        data_to_upload = {
            key: list(values) for key, values in self.extractor.accumulated_data.items()
        }

        commit_msg = f"Add {num_samples:,} samples (total: {total_after:,})"

        # Submit to worker
        self.pending_upload = self.executor.submit(
            package_and_upload_batch,
            data_to_upload,
            self.extractor.config.hf_repo_id,
            self.extractor.config.private_repo,
            commit_msg
        )

        # Update state and clear buffer
        self.extractor.state.total_uploaded += num_samples
        self.extractor.state.num_uploads += 1
        self.extractor._reset_accumulation()

        print(f"✓ Upload queued (worker processing)")

    def _wait_for_upload(self):
        """Wait for pending upload to complete."""
        if self.pending_upload is not None:
            print("⏳ Waiting for upload to complete...")
            try:
                self.pending_upload.result()
                print("✓ Upload complete")
            except Exception as e:
                print(f"❌ Upload failed: {e}")
            finally:
                self.pending_upload = None


def extract_sd15_with_symbolic_captions(
    num_samples: int = 10_000,
    batch_size: int = 64,
    hf_repo_id: Optional[str] = None,
    upload_interval: int = 4_000,  # ~1GB uploads
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
    """
    Extract SD1.5 features with symbolic captions (optimized with async uploads).

    Args:
        num_samples: Total samples
        batch_size: Batch size
        hf_repo_id: HuggingFace repo
        upload_interval: Upload every N samples (~2GB at 4k with 9 blocks)
        checkpoint_interval: Checkpoint every N samples
        device: cuda/cpu
        bias_weights_path: Path to bias_weights.json
        complexity_distribution: {1: 0.1, 2: 0.3, 3: 0.4, 4: 0.2}
        seed: Random seed
        extract_clip: Extract CLIP embeddings
        clip_pooled: Extract CLIP pooled
        active_blocks: UNet blocks (None = all 9 blocks)
        max_upload_workers: Max worker processes for uploads

    Returns:
        extractor: SD15FeatureExtractor with results
    """

    # Load bias weights
    if bias_weights_path:
        import json
        with open(bias_weights_path, 'r') as f:
            bias_weights = json.load(f)
        print(f"✓ Loaded {len(bias_weights)} bias weights")
    else:
        bias_weights = None

    # Create synthesizer
    print("Initializing synthesizer...")
    synthesizer = SynthesisSystem()
    if bias_weights and hasattr(synthesizer, 'apply_bias_weights'):
        synthesizer.apply_bias_weights(bias_weights)
    print("✓ Synthesizer ready")

    # Create generator
    caption_generator = BatchedCaptionGenerator(
        synthesizer=synthesizer,
        batch_size=batch_size,
        num_batches=num_samples // batch_size,
        complexity_distribution=complexity_distribution,
        seed=seed
    )
    dataloader = SimpleDataLoader(caption_generator)

    # Create extraction config
    extraction_config = create_working_config(
        active_blocks=active_blocks,
        extract_clip=extract_clip,
        clip_pooled=clip_pooled,
        hf_repo_id=hf_repo_id,
        upload_interval=upload_interval,
        checkpoint_interval=checkpoint_interval,
        num_samples=num_samples
    )

    # Create extractor
    print(f"Initializing SD15 on {device}...")
    extractor = SD15FeatureExtractor(config=extraction_config, device=device)
    print("✓ Extractor ready\n")

    print(f"Config: {num_samples:,} samples, batch_size={batch_size}")
    print(f"Upload: every {upload_interval:,} samples (~2GB with 9 blocks)")
    if hf_repo_id:
        print(f"Repo: {hf_repo_id}")
    print()

    # Wrap with async uploader
    async_extractor = AsyncUploadExtractor(extractor, max_workers=max_upload_workers)

    # Extract with async uploads
    async_extractor.extract_and_queue(dataloader, max_samples=num_samples)

    print("\n✓ Complete!")
    if hf_repo_id:
        print(f"https://huggingface.co/datasets/{hf_repo_id}")

    return extractor


if __name__ == "__main__":
    print("SD1.5 + Symbolic Captions Integration (Optimized)")
    print("Usage: extract_sd15_with_symbolic_captions(num_samples=10_000, hf_repo_id='user/dataset')")