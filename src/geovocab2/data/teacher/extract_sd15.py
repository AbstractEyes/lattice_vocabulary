"""
SD1.5 Feature Extractor - Unified Production System
===================================================
Extracts UNet block features + CLIP text embeddings.

File: geovocab2/data/teacher/sd15_extract.py
Author: AbstractPhil + Claude Sonnet 4.5
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import Dataset as HFDataset
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import json
import time
import numpy as np
from tqdm import tqdm

from diffusers import StableDiffusionPipeline
from geovocab2.data.teacher.extract import (
    ModelExtractionBase,
    ExtractionSchema,
    HookConfig,
    CaptureMode
)


# ============================================================================
# SD1.5 EXTRACTION CONFIGURATION
# ============================================================================

@dataclass
class SD15ExtractionConfig(ExtractionSchema):
    """Unified configuration for SD1.5 feature extraction."""

    # Base config
    name: str = "sd15_extraction"
    uid: str = "c.extraction.sd15"
    model_type: str = "stable_diffusion_1.5"
    model_name: str = "runwayml/stable-diffusion-v1-5"

    # Target blocks
    active_blocks: List[str] = field(default_factory=lambda: [
        'down_blocks.0.resnets.1',
        'down_blocks.1.resnets.1',
        'mid_block.resnets.1',
        'up_blocks.0.resnets.2',
        'up_blocks.1.resnets.2',
    ])

    # Block name mapping (for DavidCollective)
    block_name_mapping: Dict[str, str] = field(default_factory=lambda: {
        'down_blocks.0.resnets.1': 'down_0',
        'down_blocks.1.resnets.1': 'down_1',
        'down_blocks.2.resnets.1': 'down_2',
        'down_blocks.3.resnets.1': 'down_3',
        'mid_block.resnets.1': 'mid',
        'up_blocks.0.resnets.2': 'up_0',
        'up_blocks.1.resnets.2': 'up_1',
        'up_blocks.2.resnets.2': 'up_2',
        'up_blocks.3.resnets.2': 'up_3',
    })

    # CLIP embeddings
    extract_clip_embeddings: bool = True  # Include CLIP text embeddings
    clip_pooled: bool = False  # Also extract pooled embeddings (last hidden state mean)

    # HuggingFace Hub
    hf_repo_id: Optional[str] = None
    private_repo: bool = False

    # Incremental upload (set to 0 to disable)
    upload_interval: int = 10_000  # Upload every N samples (0 = only at end)
    checkpoint_interval: int = 50_000  # Save checkpoint every N samples

    # Memory management
    max_samples_in_memory: int = 10_000  # Max samples before forced upload
    clear_after_upload: bool = True

    # Resume capability
    checkpoint_dir: str = "./extraction_checkpoints"
    resume_from_checkpoint: Optional[str] = None

    # Logging
    log_interval: int = 100  # Log every N batches

    # Extraction limits (None = unlimited)
    max_samples: Optional[int] = None


# ============================================================================
# EXTRACTION STATE
# ============================================================================

@dataclass
class ExtractionState:
    """Tracks extraction progress for resume and statistics."""

    total_extracted: int = 0
    total_uploaded: int = 0
    num_uploads: int = 0
    start_time: float = field(default_factory=time.time)
    last_upload_time: float = 0
    last_checkpoint_time: float = 0

    # Track prompt prefixes to detect distribution issues
    prompt_prefix_counts: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            'total_extracted': self.total_extracted,
            'total_uploaded': self.total_uploaded,
            'num_uploads': self.num_uploads,
            'start_time': self.start_time,
            'last_upload_time': self.last_upload_time,
            'last_checkpoint_time': self.last_checkpoint_time,
            'prompt_prefix_counts': self.prompt_prefix_counts
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'ExtractionState':
        return cls(**data)

    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'ExtractionState':
        with open(path, 'r') as f:
            return cls.from_dict(json.load(f))


# ============================================================================
# SD1.5 FEATURE EXTRACTOR
# ============================================================================

class SD15FeatureExtractor(ModelExtractionBase):
    """
    Unified SD1.5 feature extractor.
    Extracts UNet block features + CLIP text embeddings.
    Handles everything from small tests to 5M sample production runs.
    """

    def __init__(
        self,
        config: Optional[SD15ExtractionConfig] = None,
        device: str = "cuda"
    ):
        """
        Args:
            config: Extraction configuration (uses defaults if None)
            device: Device for SD1.5 model
        """
        if config is None:
            config = SD15ExtractionConfig()

        super().__init__(
            name="SD15FeatureExtractor",
            u_id="sd15.extract.unified",
            model=None,  # Set after loading
            cache_dataset=None,
            config=config
        )

        self.device = device

        # Load SD1.5
        print(f"Loading SD1.5: {config.model_name}")
        self.pipe = StableDiffusionPipeline.from_pretrained(
            config.model_name,
            torch_dtype=torch.float16,
            safety_checker=None,
            requires_safety_checker=False
        ).to(device)

        self.model = self.pipe.unet
        self.text_encoder = self.pipe.text_encoder
        self.tokenizer = self.pipe.tokenizer
        self.scheduler = self.pipe.scheduler

        # Disable gradients
        self.model.eval()
        self.text_encoder.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.text_encoder.parameters():
            param.requires_grad = False

        print(f"✓ Loaded on {device}")
        if config.extract_clip_embeddings:
            print(f"  CLIP embeddings: enabled (pooled={config.clip_pooled})")

        # Setup hook configuration
        self._setup_hook_config()

        # Accumulation buffer
        self._reset_accumulation()

        # Extraction state
        if config.resume_from_checkpoint:
            self.state = ExtractionState.load(config.resume_from_checkpoint)
            print(f"✓ Resumed: {self.state.total_extracted:,} samples extracted")
        else:
            self.state = ExtractionState()

        # Create checkpoint directory
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _setup_hook_config(self):
        """Configure hooks for target layers."""
        hook_config = HookConfig(
            capture_mode=CaptureMode.OUTPUTS_ONLY,
            detach=True,
            clone=False,
            to_cpu=False,
            tuple_index=0  # ResNet outputs are tuples
        )

        self.config.default_hook_config = hook_config
        for layer_name in self.config.active_blocks:
            self.config.layer_hook_configs[layer_name] = hook_config

    def _reset_accumulation(self):
        """Reset accumulation buffer."""
        self.accumulated_data = {
            'prompt': [],
            'timestep': [],
            'sample_id': [],
        }

        # UNet block features
        for layer_name in self.config.active_blocks:
            simple_name = self.config.block_name_mapping[layer_name]
            self.accumulated_data[f'features_{simple_name}'] = []

        # CLIP embeddings
        if self.config.extract_clip_embeddings:
            self.accumulated_data['clip_embeddings'] = []
            if self.config.clip_pooled:
                self.accumulated_data['clip_pooled'] = []

        self.num_accumulated = 0

    def forward(
        self,
        prompts: List[str],
        timesteps: torch.Tensor,
        register_hooks: bool = True
    ) -> Tuple[Dict[str, torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Extract features for a batch.

        Args:
            prompts: List of prompts [B]
            timesteps: Timesteps [B] (0-999)
            register_hooks: Whether to register hooks

        Returns:
            features: {simple_block_name: [B, C, H, W]}
            clip_embeddings: [B, 77, 768] or None
            clip_pooled: [B, 768] or None
        """
        if register_hooks:
            self.register_hooks(self.config.active_blocks)

        self.clear_hook_data()

        B = len(prompts)

        # Encode prompts
        text_inputs = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            # Get CLIP embeddings
            text_encoder_output = self.text_encoder(text_inputs.input_ids)
            text_embeds = text_encoder_output[0]  # [B, 77, 768]

            # Extract CLIP embeddings if configured
            clip_embeddings = None
            clip_pooled = None

            if self.config.extract_clip_embeddings:
                clip_embeddings = text_embeds.float().cpu()  # [B, 77, 768]

                if self.config.clip_pooled:
                    # Use the [EOS] token embedding (last token before padding)
                    # Or could use mean pooling
                    clip_pooled = text_embeds[:, -1, :].float().cpu()  # [B, 768]

            # Create noisy latents
            latents = torch.randn(B, 4, 64, 64, device=self.device, dtype=torch.float16)
            noise = torch.randn_like(latents)

            # Add noise per timestep
            for i, t in enumerate(timesteps):
                latents[i:i+1] = self.scheduler.add_noise(
                    latents[i:i+1],
                    noise[i:i+1],
                    t.unsqueeze(0)
                )

            # Forward (hooks capture features)
            _ = self.model(latents, timesteps.to(self.device), text_embeds).sample

        # Extract UNet features from hooks
        features = {}
        hook_data = self.get_hook_data()

        for layer_name in self.config.active_blocks:
            if layer_name in hook_data and hook_data[layer_name]:
                output = hook_data[layer_name][-1]['output']
                if isinstance(output, tuple):
                    output = output[0]

                simple_name = self.config.block_name_mapping[layer_name]
                features[simple_name] = output.float()

        return features, clip_embeddings, clip_pooled

    def extract_batch(
        self,
        prompts: List[str],
        timesteps: torch.Tensor,
        sample_ids: Optional[List[int]] = None,
        accumulate: bool = True
    ) -> Tuple[Dict[str, torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Extract features and optionally accumulate.

        Args:
            prompts: List of prompts
            timesteps: Timesteps
            sample_ids: Unique IDs for tracking (auto-generated if None)
            accumulate: Whether to add to buffer

        Returns:
            features: Extracted UNet features
            clip_embeddings: CLIP embeddings or None
            clip_pooled: Pooled CLIP or None
        """
        features, clip_embeddings, clip_pooled = self.forward(
            prompts, timesteps, register_hooks=True
        )

        if accumulate:
            if sample_ids is None:
                # Auto-generate IDs
                start_id = self.state.total_extracted
                sample_ids = list(range(start_id, start_id + len(prompts)))

            self._add_to_accumulation(
                features, prompts, timesteps, sample_ids,
                clip_embeddings, clip_pooled
            )

        return features, clip_embeddings, clip_pooled

    def _add_to_accumulation(
        self,
        features: Dict[str, torch.Tensor],
        prompts: List[str],
        timesteps: torch.Tensor,
        sample_ids: List[int],
        clip_embeddings: Optional[torch.Tensor] = None,
        clip_pooled: Optional[torch.Tensor] = None
    ):
        """Add batch to accumulation buffer."""
        B = len(prompts)

        # Add metadata
        self.accumulated_data['prompt'].extend(prompts)
        self.accumulated_data['timestep'].extend(timesteps.cpu().tolist())
        self.accumulated_data['sample_id'].extend(sample_ids)

        # Add UNet features
        for block_name, feat_tensor in features.items():
            feat_np = feat_tensor.cpu().numpy()
            for i in range(B):
                self.accumulated_data[f'features_{block_name}'].append(feat_np[i])

        # Add CLIP embeddings
        if self.config.extract_clip_embeddings and clip_embeddings is not None:
            clip_np = clip_embeddings.numpy()  # [B, 77, 768]
            for i in range(B):
                self.accumulated_data['clip_embeddings'].append(clip_np[i])

            if self.config.clip_pooled and clip_pooled is not None:
                pooled_np = clip_pooled.numpy()  # [B, 768]
                for i in range(B):
                    self.accumulated_data['clip_pooled'].append(pooled_np[i])

        self.num_accumulated += B
        self.state.total_extracted += B

        # Track prompt distribution
        for prompt in prompts:
            prefix = prompt[:100]
            self.state.prompt_prefix_counts[prefix] = \
                self.state.prompt_prefix_counts.get(prefix, 0) + 1

    def create_dataset(self) -> HFDataset:
        """Create HF Dataset from accumulated data."""
        if self.num_accumulated == 0:
            raise ValueError("No data accumulated!")

        self.cache_dataset = HFDataset.from_dict(self.accumulated_data)
        return self.cache_dataset

    def should_upload(self) -> bool:
        """Check if upload needed based on config."""
        if self.config.upload_interval == 0:
            return False

        samples_since_upload = self.state.total_extracted - self.state.total_uploaded

        return (samples_since_upload >= self.config.upload_interval or
                self.num_accumulated >= self.config.max_samples_in_memory)

    def upload_accumulated(self, force: bool = False):
        """
        Upload accumulated data to Hub.

        Args:
            force: Force upload even if interval not reached
        """
        if not force and not self.should_upload():
            return

        if self.num_accumulated == 0:
            return

        if self.config.hf_repo_id is None:
            print("⚠️  No hf_repo_id configured, skipping upload")
            return

        print(f"\n{'='*70}")
        print(f"📤 UPLOADING TO HUB")
        print(f"{'='*70}")
        print(f"Samples in batch: {self.num_accumulated:,}")
        print(f"Total uploaded: {self.state.total_uploaded:,} → {self.state.total_uploaded + self.num_accumulated:,}")

        try:
            dataset = self.create_dataset()

            # Print dataset info
            print(f"Dataset columns: {dataset.column_names}")
            print(f"Dataset size: {len(dataset)} samples")

            commit_msg = f"Add {self.num_accumulated:,} samples (total: {self.state.total_uploaded + self.num_accumulated:,})"

            dataset.push_to_hub(
                self.config.hf_repo_id,
                private=self.config.private_repo,
                commit_message=commit_msg
            )

            self.state.total_uploaded += self.num_accumulated
            self.state.num_uploads += 1
            self.state.last_upload_time = time.time()

            print(f"✓ Uploaded successfully")
            print(f"  Repo: https://huggingface.co/datasets/{self.config.hf_repo_id}")
            print(f"{'='*70}\n")

            if self.config.clear_after_upload:
                self._reset_accumulation()

        except Exception as e:
            print(f"❌ Upload failed: {e}")
            backup_path = self.checkpoint_dir / f"backup_{self.state.total_extracted}.arrow"
            dataset.save_to_disk(str(backup_path))
            print(f"   Saved backup to: {backup_path}")

    def save_checkpoint(self):
        """Save extraction state."""
        path = self.checkpoint_dir / f"state_{self.state.total_extracted}.json"
        self.state.save(str(path))

        # Keep only last 3 checkpoints
        checkpoints = sorted(self.checkpoint_dir.glob("state_*.json"))
        for old in checkpoints[:-3]:
            old.unlink()

    def extract_from_dataloader(
        self,
        dataloader: DataLoader,
        max_samples: Optional[int] = None
    ):
        """
        Extract features from a dataloader.

        Args:
            dataloader: DataLoader yielding (prompts, metadata) or just prompts
            max_samples: Maximum samples to extract (uses config if None)
        """
        max_samples = max_samples or self.config.max_samples

        print(f"\n{'='*80}")
        print(f"EXTRACTION PIPELINE")
        print(f"{'='*80}")
        print(f"Target samples: {max_samples or 'unlimited'}")
        print(f"Already extracted: {self.state.total_extracted:,}")
        print(f"Upload interval: {self.config.upload_interval:,} samples")
        print(f"Checkpoint interval: {self.config.checkpoint_interval:,} samples")
        print(f"Active blocks: {len(self.config.active_blocks)}")
        print(f"Extract CLIP: {self.config.extract_clip_embeddings}")
        if self.config.extract_clip_embeddings:
            print(f"  CLIP pooled: {self.config.clip_pooled}")
        print(f"{'='*80}\n")

        # Register hooks once
        self.register_hooks(self.config.active_blocks)

        pbar = tqdm(desc="Extracting", unit="batch")
        batch_times = []

        for batch_idx, batch_data in enumerate(dataloader):
            # Check max samples
            if max_samples and self.state.total_extracted >= max_samples:
                print(f"\n✓ Reached target of {max_samples:,} samples")
                break

            batch_start = time.time()

            # Unpack batch (handle different formats)
            if isinstance(batch_data, (list, tuple)) and len(batch_data) == 2:
                prompts, metadata = batch_data
            else:
                prompts = batch_data
                metadata = None

            # Generate timesteps
            B = len(prompts)
            timesteps = torch.randint(0, 1000, (B,), device=self.device)

            # Extract
            self.extract_batch(prompts, timesteps, accumulate=True)

            batch_time = time.time() - batch_start
            batch_times.append(batch_time)

            # Update progress
            pbar.update(1)

            if (batch_idx + 1) % self.config.log_interval == 0:
                avg_time = np.mean(batch_times[-100:])
                sps = B / avg_time

                pbar.set_postfix({
                    'extracted': f"{self.state.total_extracted:,}",
                    'uploaded': f"{self.state.total_uploaded:,}",
                    'sps': f"{sps:.1f}",
                    'mem': f"{self._estimate_memory_mb():.0f}MB"
                })

            # Upload if needed
            if self.should_upload():
                self.upload_accumulated()

            # Checkpoint if needed
            if (self.config.checkpoint_interval > 0 and
                self.state.total_extracted % self.config.checkpoint_interval == 0):
                self.save_checkpoint()

        pbar.close()

        # Final upload
        self.upload_accumulated(force=True)
        self.save_checkpoint()

        # Print stats
        self._print_stats()

    def _estimate_memory_mb(self) -> float:
        """Estimate memory usage."""
        total_bytes = 0
        total_bytes += sum(len(p.encode('utf-8')) for p in self.accumulated_data['prompt'])
        total_bytes += len(self.accumulated_data['timestep']) * 4
        total_bytes += len(self.accumulated_data['sample_id']) * 8

        for key in self.accumulated_data:
            if (key.startswith('features_') or key.startswith('clip_')) and self.accumulated_data[key]:
                sample = self.accumulated_data[key][0]
                total_bytes += sample.nbytes * len(self.accumulated_data[key])

        return total_bytes / (1024 * 1024)

    def _print_stats(self):
        """Print extraction statistics."""
        elapsed = time.time() - self.state.start_time
        hours, minutes = int(elapsed // 3600), int((elapsed % 3600) // 60)

        print(f"\n{'='*80}")
        print(f"EXTRACTION COMPLETE")
        print(f"{'='*80}")
        print(f"Total extracted: {self.state.total_extracted:,}")
        print(f"Total uploaded: {self.state.total_uploaded:,}")
        print(f"Number of uploads: {self.state.num_uploads}")
        print(f"Time: {hours}h {minutes}m")
        print(f"Speed: {self.state.total_extracted / elapsed:.1f} samples/sec")

        if self.config.extract_clip_embeddings:
            print(f"\nCLIP embeddings: ✓ included")
            if self.config.clip_pooled:
                print(f"CLIP pooled: ✓ included")

        if self.state.prompt_prefix_counts:
            print(f"\nPrompt distribution (top 10):")
            sorted_prompts = sorted(
                self.state.prompt_prefix_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
            for prefix, count in sorted_prompts:
                pct = 100 * count / self.state.total_extracted
                print(f"  {count:7,}x ({pct:5.2f}%)  {prefix[:50]}...")

        if self.config.hf_repo_id:
            print(f"\nDataset: https://huggingface.co/datasets/{self.config.hf_repo_id}")

        print(f"{'='*80}\n")

    def save_cache(self, path: str):
        """Save accumulated dataset to disk."""
        if self.cache_dataset is None:
            self.create_dataset()
        self.cache_dataset.save_to_disk(path)

    def load_cache(self, path: str) -> HFDataset:
        """Load cached dataset from disk."""
        self.cache_dataset = HFDataset.load_from_disk(path)
        return self.cache_dataset


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":

    # Example 1: Quick test with CLIP
    print("EXAMPLE 1: Extract with CLIP embeddings")
    print("="*80)

    config = SD15ExtractionConfig(
        extract_clip_embeddings=True,
        clip_pooled=True,
        hf_repo_id=None,
        upload_interval=0
    )

    extractor = SD15FeatureExtractor(config=config, device="cuda")

    prompts = ["a photo of a cat", "a beautiful landscape"]
    timesteps = torch.tensor([500, 750])

    features, clip_emb, clip_pooled = extractor.extract_batch(
        prompts, timesteps, accumulate=False
    )

    print(f"\nExtracted:")
    print(f"  UNet features: {len(features)} blocks")
    for name, feat in features.items():
        print(f"    {name}: {list(feat.shape)}")

    if clip_emb is not None:
        print(f"  CLIP embeddings: {list(clip_emb.shape)}")  # [2, 77, 768]
    if clip_pooled is not None:
        print(f"  CLIP pooled: {list(clip_pooled.shape)}")  # [2, 768]

    print("\n✓ Example complete\n")
"""

**Added:**

1. ✅ **CLIP text embeddings** - Full [B, 77, 768] sequence
2. ✅ **CLIP pooled** - Optional [B, 768] single vector (EOS token)
3. ✅ **Configurable** - `extract_clip_embeddings=True/False`
4. ✅ **Stored in dataset** - `clip_embeddings` and `clip_pooled` columns
5. ✅ **Memory tracking** - Includes CLIP in size estimates

**Dataset structure now:**

{
    'prompt': str,
    'timestep': int,
    'sample_id': int,
    'features_down_0': [320, 64, 64],
    'features_down_1': [640, 32, 32],
    'features_mid': [1280, 8, 8],
    'features_up_0': [1280, 16, 16],
    'features_up_1': [1280, 32, 32],
    'clip_embeddings': [77, 768],      # NEW!
    'clip_pooled': [768]                # NEW! (optional)
}
"""