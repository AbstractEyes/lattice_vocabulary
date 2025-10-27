"""
DavidCollective: SD1.5 Block-Level Feature Distillation
========================================================
Block-parallel geometric flow matching distillation system.

Each David companion learns one SD1.5 UNet block's feature distribution
with timestep-aware geometric regularization.

Author: AbstractPhil + Claude Sonnet 4.5
Date: 2025-10-21
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from pathlib import Path
import json

# Assuming David imports
from geovocab2.train.model.core.david import (
    David,
    DavidArchitectureConfig,
    RoseLoss,
    CayleyChaosLoss,
    MultiScaleCrystalLoss
)


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class SD15BlockSpec:
    """Specification for one SD1.5 UNet block."""
    name: str
    channels: int
    spatial_size: int  # Approximate spatial resolution
    position: str  # 'down', 'mid', 'up'
    index: int


# SD1.5 UNet block specifications
SD15_BLOCKS = [
    # Down blocks
    SD15BlockSpec('down_0', 320, 64, 'down', 0),
    SD15BlockSpec('down_1', 640, 32, 'down', 1),
    SD15BlockSpec('down_2', 1280, 16, 'down', 2),
    SD15BlockSpec('down_3', 1280, 8, 'down', 3),

    # Mid block
    SD15BlockSpec('mid', 1280, 8, 'mid', 0),

    # Up blocks
    SD15BlockSpec('up_0', 1280, 16, 'up', 0),
    SD15BlockSpec('up_1', 1280, 32, 'up', 1),
    SD15BlockSpec('up_2', 640, 64, 'up', 2),
    SD15BlockSpec('up_3', 320, 64, 'up', 3),
]


@dataclass
class DavidCollectiveConfig:
    """Configuration for entire DavidCollective system."""

    # Which blocks to use
    active_blocks: List[str] = None  # None = all blocks

    # Timestep discretization
    num_timestep_bins: int = 100  # 0-999 → 100 bins of 10 steps each
    num_feature_patterns_per_timestep: int = 10  # Feature clusters per timestep

    # David architecture per block
    david_sharing_mode: str = 'fully_shared'
    david_fusion_mode: str = 'deep_efficiency'
    use_belly: bool = True
    belly_expand: float = 1.5

    # Training
    feature_similarity_weight: float = 0.5
    rose_weight: float = 0.2
    cayley_weight: float = 0.1
    ce_weight: float = 0.2

    # Geometric constraints
    rose_margin: float = 1.0
    rose_temperature: float = 0.07
    cayley_volume_floor: float = 1e-4

    # Progressive training
    progressive_training: bool = True
    warmup_epochs_per_block: int = 5

    # Caching
    cache_dir: str = "./sd15_features_cache"
    max_cache_size_gb: float = 100.0

    def __post_init__(self):
        if self.active_blocks is None:
            self.active_blocks = [block.name for block in SD15_BLOCKS]


# ============================================================================
# DAVID BLOCK COMPANION
# ============================================================================

class DavidBlockCompanion(nn.Module):
    """
    Single David instance for one SD1.5 block.
    Learns timestep-conditioned feature distributions.
    """

    def __init__(
        self,
        block_spec: SD15BlockSpec,
        config: DavidCollectiveConfig
    ):
        super().__init__()

        self.block_spec = block_spec
        self.config = config

        # Determine David scales based on block channels
        # More channels → can support larger internal scales
        if block_spec.channels <= 320:
            scales = [64, 128, 256]
        elif block_spec.channels <= 640:
            scales = [128, 256, 512]
        else:
            scales = [256, 512, 768]

        # David configuration for this block
        self.david_config = DavidArchitectureConfig(
            feature_dim=block_spec.channels,
            num_classes=config.num_timestep_bins * config.num_feature_patterns_per_timestep,
            scales=scales,
            sharing_mode=config.david_sharing_mode,
            fusion_mode=config.david_fusion_mode,
            use_belly=config.use_belly,
            belly_expand=config.belly_expand,
            progressive_training=config.progressive_training,
        )

        # Core David model
        self.david = David.from_config(self.david_config)

        # Spatial → Vector adapter
        self.spatial_adapter = SpatialFeatureAdapter(
            in_channels=block_spec.channels,
            out_features=block_spec.channels,
            spatial_size=block_spec.spatial_size
        )

        # Crystal anchors: [timestep_bins, feature_patterns, 5, max_scale_dim]
        max_scale = max(scales)
        self.crystal_anchors = nn.Parameter(
            torch.randn(
                config.num_timestep_bins,
                config.num_feature_patterns_per_timestep,
                5,  # pentachoron vertices
                max_scale
            )
        )

        # Initialize crystals with good geometry
        with torch.no_grad():
            self._initialize_crystals()

        # Teacher feature projections (match teacher dim to each scale)
        self.teacher_projections = nn.ModuleDict({
            str(scale): nn.Sequential(
                nn.Linear(block_spec.channels, scale),
                nn.LayerNorm(scale)
            )
            for scale in scales
        })

        # Losses
        self.rose_loss = RoseLoss(
            margin=config.rose_margin,
            temperature=config.rose_temperature
        )
        self.cayley_loss = CayleyChaosLoss(
            volume_floor=config.cayley_volume_floor
        )

    def _initialize_crystals(self):
        """Initialize crystal anchors with valid geometric properties."""
        for t_bin in range(self.config.num_timestep_bins):
            for pattern in range(self.config.num_feature_patterns_per_timestep):
                # Sample from unit sphere and ensure minimum volume
                crystal = self.crystal_anchors[t_bin, pattern]
                crystal.normal_()
                crystal = F.normalize(crystal, dim=-1)

                # Add small perturbation to break symmetry
                crystal += 0.1 * torch.randn_like(crystal)

    def get_timestep_crystals(self, timestep_class: torch.Tensor) -> torch.Tensor:
        """
        Get crystal anchors for given timestep classes.

        Args:
            timestep_class: [B] tensor of timestep bin indices

        Returns:
            crystals: [B, num_patterns, 5, scale_dim]
        """
        return self.crystal_anchors[timestep_class]

    def get_class_anchors(self, scale: int) -> torch.Tensor:
        """
        Get 2D anchors for David from pentachoron centroids.

        Args:
            scale: Embedding dimension

        Returns:
            anchors: [num_total_classes, scale_dim]
        """
        # All crystals for all timesteps and patterns
        # [timestep_bins, patterns, 5, max_scale]
        all_crystals = self.crystal_anchors

        # Take appropriate scale dimension
        scale_crystals = all_crystals[..., :scale]  # [bins, patterns, 5, scale]

        # Compute centroid of each pentachoron
        centroids = scale_crystals.mean(dim=2)  # [bins, patterns, scale]

        # Flatten to [num_classes, scale]
        num_classes = centroids.shape[0] * centroids.shape[1]
        anchors = centroids.reshape(num_classes, scale)

        return anchors

    def forward(
        self,
        spatial_features: torch.Tensor,
        timestep: torch.Tensor,
        return_all_scales: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through block companion.

        Args:
            spatial_features: [B, C, H, W] from SD1.5 block
            timestep: [B] timesteps in [0, 1000)

        Returns:
            Dictionary with logits, features, and intermediate outputs
        """
        batch_size = spatial_features.shape[0]

        # 1. Convert spatial features to vector
        feature_vector = self.spatial_adapter(spatial_features)

        # 2. Discretize timesteps
        timestep_class = self.discretize_timestep(timestep)

        # 3. Prepare anchor dict for David
        # David expects {scale: [num_classes, scale_dim]}
        # Use pentachoron centroids as class anchors
        anchors_dict = {}
        for scale in self.david_config.scales:
            anchors_dict[scale] = self.get_class_anchors(scale)

        # 4. Forward through David
        if return_all_scales:
            combined_logits, scale_logits, scale_features, fusion_weights = self.david(
                feature_vector,
                anchors_dict,
                return_all_scales=True
            )
        else:
            combined_logits, features = self.david(
                feature_vector,
                anchors_dict,
                return_all_scales=False
            )
            scale_logits = [combined_logits]
            scale_features = [features]
            fusion_weights = torch.ones(1)

        return {
            'combined_logits': combined_logits,
            'scale_logits': scale_logits,
            'scale_features': scale_features,
            'fusion_weights': fusion_weights,
            'timestep_class': timestep_class,
            'feature_vector': feature_vector,
            'spatial_features': spatial_features
        }

    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        teacher_features: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all losses for this block.

        Args:
            outputs: Output from forward()
            teacher_features: [B, C, H, W] ground truth features from SD1.5

        Returns:
            Dictionary of losses
        """
        losses = {}

        # Get teacher vector for comparison
        teacher_vector = self.spatial_adapter(teacher_features)
        timestep_class = outputs['timestep_class']
        batch_size = teacher_vector.shape[0]

        # Get full pentachoron crystals for this batch
        batch_crystals = self.get_timestep_crystals(timestep_class)
        # batch_crystals: [B, num_patterns, 5, max_scale]

        # 1. Feature similarity loss (per scale)
        feature_sim_losses = []
        for i, scale_features in enumerate(outputs['scale_features']):
            scale = self.david_config.scales[i]

            # Project teacher features to match this scale
            teacher_at_scale = self.teacher_projections[str(scale)](teacher_vector)

            # Cosine similarity loss
            cos_sim = F.cosine_similarity(scale_features, teacher_at_scale, dim=-1)
            sim_loss = (1 - cos_sim).mean()
            feature_sim_losses.append(sim_loss)
            losses[f'feature_sim_scale_{i}'] = sim_loss

        losses['feature_similarity'] = sum(feature_sim_losses) / len(feature_sim_losses)

        # 2. Rose loss (geometric alignment with crystals)
        rose_losses = []
        for i, features in enumerate(outputs['scale_features']):
            scale = self.david_config.scales[i]

            # Get centroids for this scale
            scale_crystals = batch_crystals[..., :scale]  # [B, patterns, 5, scale]
            centroids = scale_crystals.mean(dim=2)  # [B, patterns, scale]

            # Flatten for all classes
            all_centroids = centroids.reshape(batch_size, -1, scale)  # [B, patterns, scale]

            # For Rose loss, we need [num_classes, 5, scale] format
            # Use first pattern's full pentachoron as representative
            first_pattern_crystals = scale_crystals[:, 0, :, :]  # [B, 5, scale]

            # Create pseudo-targets based on timestep class
            # Each sample should be closest to its own timestep's crystals
            targets = torch.arange(batch_size, device=features.device)

            # Simplified Rose loss: cosine similarity to own centroid
            own_centroids = centroids[:, 0, :]  # [B, scale] - use first pattern
            cos_sim = F.cosine_similarity(features, own_centroids, dim=-1)
            rose_loss = (1 - cos_sim).mean()

            rose_losses.append(rose_loss)
            losses[f'rose_scale_{i}'] = rose_loss

        losses['rose'] = sum(rose_losses) / len(rose_losses)

        # 3. Cayley loss (geometric quality)
        cayley_losses = []
        for i, features in enumerate(outputs['scale_features']):
            scale = self.david_config.scales[i]

            # Get pentachora for this scale
            scale_crystals = batch_crystals[..., :scale]  # [B, patterns, 5, scale]

            # Use first pattern for each batch item
            pentachora = scale_crystals[:, 0, :, :]  # [B, 5, scale]

            cayley_loss = self.cayley_loss(pentachora)
            cayley_losses.append(cayley_loss)
            losses[f'cayley_scale_{i}'] = cayley_loss

        losses['cayley'] = sum(cayley_losses) / len(cayley_losses)

        # 4. Classification loss (timestep prediction)
        # The logits predict which class (timestep_bin * num_patterns + pattern_id)
        # For simplicity, just predict timestep bin
        ce_loss = F.cross_entropy(
            outputs['combined_logits'],
            timestep_class
        )
        losses['ce'] = ce_loss

        # Compute accuracy
        pred = outputs['combined_logits'].argmax(dim=-1) // self.config.num_feature_patterns_per_timestep
        accuracy = (pred == timestep_class).float().mean()
        losses['accuracy'] = accuracy

        # 5. Total loss
        losses['total'] = (
            self.config.feature_similarity_weight * losses['feature_similarity'] +
            self.config.rose_weight * losses['rose'] +
            self.config.cayley_weight * losses['cayley'] +
            self.config.ce_weight * losses['ce']
        )

        return losses

    def discretize_timestep(self, timestep: torch.Tensor) -> torch.Tensor:
        """
        Convert continuous timesteps to discrete bins.

        Args:
            timestep: [B] values in [0, 1000)

        Returns:
            timestep_class: [B] bin indices in [0, num_bins)
        """
        return (timestep / 1000.0 * self.config.num_timestep_bins).long().clamp(
            0, self.config.num_timestep_bins - 1
        )


# ============================================================================
# SPATIAL FEATURE ADAPTER
# ============================================================================

class SpatialFeatureAdapter(nn.Module):
    """
    Convert spatial feature maps to vectors for David.
    Uses adaptive pooling + projection.
    """

    def __init__(
        self,
        in_channels: int,
        out_features: int,
        spatial_size: int
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_features = out_features

        # Adaptive pooling to fixed size
        self.pool = nn.AdaptiveAvgPool2d((4, 4))

        # Project to output dimension
        pooled_dim = in_channels * 16  # 4*4 = 16
        self.projection = nn.Sequential(
            nn.Linear(pooled_dim, out_features * 2),
            nn.LayerNorm(out_features * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(out_features * 2, out_features),
            nn.LayerNorm(out_features)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W]
        Returns:
            out: [B, out_features]
        """
        # Pool
        pooled = self.pool(x)  # [B, C, 4, 4]

        # Flatten
        flat = pooled.flatten(1)  # [B, C*16]

        # Project
        out = self.projection(flat)  # [B, out_features]

        return out


# ============================================================================
# DAVID COLLECTIVE
# ============================================================================

class DavidCollective(nn.Module):
    """
    Collection of DavidBlockCompanions for full SD1.5 distillation.
    """

    def __init__(self, config: DavidCollectiveConfig):
        super().__init__()

        self.config = config

        # Filter active blocks
        self.active_blocks = [
            block for block in SD15_BLOCKS
            if block.name in config.active_blocks
        ]

        # Create companions
        self.companions = nn.ModuleDict({
            block.name: DavidBlockCompanion(block, config)
            for block in self.active_blocks
        })

        # Tracking
        self.current_epoch = 0
        self.block_metrics = {block.name: [] for block in self.active_blocks}

    def forward(
        self,
        block_features: Dict[str, torch.Tensor],
        timesteps: torch.Tensor,
        return_all_scales: bool = True
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Forward pass through all companions.

        Args:
            block_features: {block_name: [B, C, H, W]}
            timesteps: [B] timesteps

        Returns:
            {block_name: companion_outputs}
        """
        outputs = {}

        for block_name, companion in self.companions.items():
            if block_name in block_features:
                outputs[block_name] = companion(
                    block_features[block_name],
                    timesteps,
                    return_all_scales=return_all_scales
                )

        return outputs

    def compute_losses(
        self,
        outputs: Dict[str, Dict[str, torch.Tensor]],
        teacher_features: Dict[str, torch.Tensor]
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Compute losses for all companions.

        Args:
            outputs: {block_name: companion_outputs}
            teacher_features: {block_name: [B, C, H, W]}

        Returns:
            {block_name: losses_dict}
        """
        all_losses = {}

        for block_name, companion in self.companions.items():
            if block_name in outputs and block_name in teacher_features:
                losses = companion.compute_loss(
                    outputs[block_name],
                    teacher_features[block_name]
                )
                all_losses[block_name] = losses

        return all_losses

    def get_total_loss(
        self,
        block_losses: Dict[str, Dict[str, torch.Tensor]]
    ) -> torch.Tensor:
        """Aggregate total loss across all blocks."""
        total = sum(
            losses['total']
            for losses in block_losses.values()
        ) / len(block_losses)
        return total

    def update_epoch(self, epoch: int):
        """Update epoch for all companions."""
        self.current_epoch = epoch
        for companion in self.companions.values():
            companion.david.update_epoch(epoch)

    def get_status(self) -> Dict:
        """Get status of all companions."""
        return {
            'epoch': self.current_epoch,
            'num_companions': len(self.companions),
            'active_blocks': [block.name for block in self.active_blocks],
            'total_parameters': sum(
                sum(p.numel() for p in companion.parameters())
                for companion in self.companions.values()
            ),
            'companion_info': {
                name: {
                    'scales': companion.david_config.scales,
                    'channels': companion.block_spec.channels,
                    'parameters': sum(p.numel() for p in companion.parameters())
                }
                for name, companion in self.companions.items()
            }
        }

    def save_checkpoint(self, path: str):
        """Save collective checkpoint."""
        checkpoint = {
            'config': self.config,
            'epoch': self.current_epoch,
            'state_dict': self.state_dict(),
            'block_metrics': self.block_metrics
        }
        torch.save(checkpoint, path)

    @classmethod
    def load_checkpoint(cls, path: str):
        """Load collective from checkpoint."""
        checkpoint = torch.load(path, weights_only=False)
        collective = cls(checkpoint['config'])
        collective.load_state_dict(checkpoint['state_dict'])
        collective.current_epoch = checkpoint['epoch']
        collective.block_metrics = checkpoint['block_metrics']
        return collective


# ============================================================================
# FEATURE CACHING UTILITIES
# ============================================================================

class SD15FeatureCache:
    """
    Utilities for caching SD1.5 features to disk.
    Enables fast training without repeated forward passes.
    """

    def __init__(self, cache_dir: str = "./sd15_features_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.metadata_file = self.cache_dir / "metadata.json"
        self.metadata = self._load_metadata()

    def _load_metadata(self) -> Dict:
        """Load or create metadata."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {
            'num_samples': 0,
            'blocks': [b.name for b in SD15_BLOCKS],
            'timestep_range': [0, 1000],
            'shards': []
        }

    def _save_metadata(self):
        """Save metadata to disk."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)

    def cache_features(
        self,
        features: Dict[str, torch.Tensor],
        timesteps: torch.Tensor,
        prompts: List[str],
        shard_id: int
    ):
        """
        Cache a batch of features to disk.

        Args:
            features: {block_name: [B, C, H, W]}
            timesteps: [B]
            prompts: List of B prompts
            shard_id: Shard identifier
        """
        shard_path = self.cache_dir / f"shard_{shard_id:06d}.pt"

        # Prepare data
        data = {
            'features': features,
            'timesteps': timesteps.cpu(),
            'prompts': prompts,
            'shard_id': shard_id
        }

        # Save
        torch.save(data, shard_path)

        # Update metadata
        if shard_id not in self.metadata['shards']:
            self.metadata['shards'].append(shard_id)
            self.metadata['num_samples'] += len(prompts)
            self._save_metadata()

        return shard_path

    def load_shard(self, shard_id: int) -> Dict:
        """Load a specific shard."""
        shard_path = self.cache_dir / f"shard_{shard_id:06d}.pt"
        if not shard_path.exists():
            raise FileNotFoundError(f"Shard {shard_id} not found")
        return torch.load(shard_path)

    def get_shard_ids(self) -> List[int]:
        """Get all available shard IDs."""
        return sorted(self.metadata['shards'])

    def estimate_size_gb(self) -> float:
        """Estimate total cache size in GB."""
        total_bytes = sum(
            (self.cache_dir / f"shard_{sid:06d}.pt").stat().st_size
            for sid in self.metadata['shards']
            if (self.cache_dir / f"shard_{sid:06d}.pt").exists()
        )
        return total_bytes / (1024 ** 3)


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("DAVID COLLECTIVE - SD1.5 Block-Level Distillation")
    print("=" * 80)

    # Configuration
    config = DavidCollectiveConfig(
        active_blocks=['down_0', 'down_1', 'mid', 'up_0', 'up_1'],
        num_timestep_bins=100,
        num_feature_patterns_per_timestep=10,
        progressive_training=True
    )

    # Create collective
    collective = DavidCollective(config)

    print(f"\n[Status]")
    status = collective.get_status()
    print(f"  Active blocks: {len(status['active_blocks'])}")
    print(f"  Total parameters: {status['total_parameters']:,}")

    # Simulate forward pass
    print(f"\n[Test Forward Pass]")
    batch_size = 4

    # Simulate block features
    block_features = {
        'down_0': torch.randn(batch_size, 320, 64, 64),
        'down_1': torch.randn(batch_size, 640, 32, 32),
        'mid': torch.randn(batch_size, 1280, 8, 8),
        'up_0': torch.randn(batch_size, 1280, 16, 16),
        'up_1': torch.randn(batch_size, 1280, 32, 32),
    }

    timesteps = torch.randint(0, 1000, (batch_size,))

    # Forward
    with torch.no_grad():
        outputs = collective(block_features, timesteps)

    print(f"  Outputs for {len(outputs)} blocks")

    # Test losses
    print(f"\n[Test Loss Computation]")
    teacher_features = block_features  # Use same for test

    with torch.no_grad():
        losses = collective.compute_losses(outputs, teacher_features)
        total_loss = collective.get_total_loss(losses)

    print(f"  Total loss: {total_loss.item():.4f}")
    for block_name, block_losses in losses.items():
        print(f"  {block_name}:")
        print(f"    Feature similarity: {block_losses['feature_similarity'].item():.4f}")
        print(f"    Rose: {block_losses['rose'].item():.4f}")
        print(f"    Cayley: {block_losses['cayley'].item():.4f}")
        print(f"    Accuracy: {block_losses['accuracy'].item():.2%}")

    # Test caching
    print(f"\n[Test Feature Caching]")
    cache = SD15FeatureCache("./test_cache")

    shard_path = cache.cache_features(
        block_features,
        timesteps,
        prompts=['test prompt'] * batch_size,
        shard_id=0
    )

    print(f"  Cached to: {shard_path}")
    print(f"  Cache size: {cache.estimate_size_gb():.4f} GB")

    loaded = cache.load_shard(0)
    print(f"  Loaded shard with {len(loaded['prompts'])} samples")

    print("\n" + "=" * 80)
    print("DAVID COLLECTIVE READY FOR TRAINING")
    print("=" * 80)