"""
DavidCollective: SD1.5 Block-Level Feature Distillation - REFACTORED
====================================================================
Block-parallel geometric flow matching distillation system with
EXTRACTED and CENTRALIZED loss computation.

Each David companion learns one SD1.5 UNet block's feature distribution
with timestep-aware geometric regularization.

Author: AbstractPhil + Claude Sonnet 4.5
Date: 2025-10-27 (Refactored)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from pathlib import Path
import json
from safetensors.torch import save_file, load_file

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
class LossConfig:
    """Centralized loss configuration."""

    # Loss weights
    feature_similarity_weight: float = 0.5
    rose_weight: float = 0.2
    cayley_weight: float = 0.1
    ce_weight: float = 0.2

    # Geometric constraints
    rose_margin: float = 1.0
    rose_temperature: float = 0.07
    cayley_volume_floor: float = 1e-4

    # Loss computation modes
    use_rose_loss: bool = True
    use_cayley_loss: bool = True
    use_ce_loss: bool = True

    # Multi-scale aggregation
    scale_loss_mode: str = 'mean'  # 'mean', 'weighted', 'last'


@dataclass
class PatternSupervisedLossConfig(LossConfig):
    """
    Extended config for pattern-supervised learning.

    This is the BATTLE-TESTED loss configuration that successfully
    trained previous David models with 1000-class supervision
    (100 timestep bins × 10 patterns per bin).

    Key differences from base LossConfig:
    - Soft assignment with temperature smoothing
    - Pattern diversity regularization (prevents mode collapse)
    - Rose loss = feature_sim (not separate geometric loss)
    - Student-driven pattern assignment (uses student features, not teacher)
    - Comparison to learned centroids (not direct teacher features)
    """

    # Pattern supervision parameters
    use_soft_assignment: bool = True
    assignment_temperature: float = 0.1  # Lower = sharper, higher = smoother
    pattern_diversity_weight: float = 0.05  # Entropy regularization

    # Override defaults for pattern supervision
    feature_similarity_weight: float = 0.5
    rose_weight: float = 0.3  # Adds to feature_sim (same computation in this version)
    ce_weight: float = 0.2

    # Disable unused components
    use_cayley_loss: bool = False  # Not used in pattern supervision
    cayley_weight: float = 0.0


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

    # Loss configuration
    loss_config: LossConfig = None
    loss_calculator_type: str = 'pattern'  # 'block' or 'pattern'

    # Progressive training
    progressive_training: bool = True
    warmup_epochs_per_block: int = 5

    # Caching
    cache_dir: str = "./sd15_features_cache"
    max_cache_size_gb: float = 100.0

    def __post_init__(self):
        if self.active_blocks is None:
            self.active_blocks = [block.name for block in SD15_BLOCKS]
        if self.loss_config is None:
            self.loss_config = LossConfig()


# ============================================================================
# BLOCK LOSS CALCULATOR - EXTRACTED SYSTEM
# ============================================================================

class BlockLossCalculator(nn.Module):
    """
    Centralized loss computation for DavidBlockCompanion.

    Extracts ALL loss logic from the companion, creating clean separation:
    - Companion: Forward passes and feature extraction
    - Calculator: Loss computation and optimization targets

    Benefits:
    - Easy to swap loss strategies
    - Centralized loss configuration
    - Clean testability
    - No loss logic leakage into model code
    """

    def __init__(
        self,
        block_spec: SD15BlockSpec,
        scales: List[int],
        loss_config: LossConfig,
        num_timestep_bins: int,
        num_feature_patterns: int
    ):
        super().__init__()

        self.block_spec = block_spec
        self.scales = scales
        self.config = loss_config
        self.num_timestep_bins = num_timestep_bins
        self.num_feature_patterns = num_feature_patterns

        # Teacher feature projections (moved from companion)
        self.teacher_projections = nn.ModuleDict({
            str(scale): nn.Sequential(
                nn.Linear(block_spec.channels, scale),
                nn.LayerNorm(scale)
            )
            for scale in scales
        })

        # Loss function instances (moved from companion)
        if self.config.use_rose_loss:
            self.rose_loss = RoseLoss(
                margin=self.config.rose_margin,
                temperature=self.config.rose_temperature
            )

        if self.config.use_cayley_loss:
            self.cayley_loss = CayleyChaosLoss(
                volume_floor=self.config.cayley_volume_floor
            )

    def compute_losses(
        self,
        outputs: Dict[str, torch.Tensor],
        teacher_features: torch.Tensor,
        batch_crystals: torch.Tensor,
        spatial_adapter: nn.Module
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all losses for a batch.

        Args:
            outputs: Forward outputs from companion
            teacher_features: [B, C, H, W] ground truth
            batch_crystals: [B, num_patterns, 5, max_scale] pentachora
            spatial_adapter: For converting teacher features to vectors

        Returns:
            Dictionary of losses and metrics
        """
        losses = {}

        # Get teacher vector
        teacher_vector = spatial_adapter(teacher_features)
        timestep_class = outputs['timestep_class']
        batch_size = teacher_vector.shape[0]

        # 1. Feature Similarity Loss
        losses.update(self._compute_feature_similarity_loss(
            outputs, teacher_vector
        ))

        # 2. Rose Loss (geometric alignment)
        if self.config.use_rose_loss:
            losses.update(self._compute_rose_loss(
                outputs, batch_crystals, batch_size
            ))

        # 3. Cayley Loss (geometric quality)
        if self.config.use_cayley_loss:
            losses.update(self._compute_cayley_loss(
                outputs, batch_crystals
            ))

        # 4. Classification Loss
        if self.config.use_ce_loss:
            losses.update(self._compute_classification_loss(
                outputs, timestep_class
            ))

        # 5. Total Loss (weighted combination)
        losses['total'] = self._compute_total_loss(losses)

        return losses

    def _compute_feature_similarity_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        teacher_vector: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Feature similarity loss per scale."""
        losses = {}
        scale_losses = []

        for i, scale_features in enumerate(outputs['scale_features']):
            scale = self.scales[i]

            # Project teacher to this scale
            teacher_at_scale = self.teacher_projections[str(scale)](teacher_vector)

            # Cosine similarity loss
            cos_sim = F.cosine_similarity(scale_features, teacher_at_scale, dim=-1)
            sim_loss = (1 - cos_sim).mean()

            scale_losses.append(sim_loss)
            losses[f'feature_sim_scale_{i}'] = sim_loss

        # Aggregate across scales
        if self.config.scale_loss_mode == 'mean':
            losses['feature_similarity'] = sum(scale_losses) / len(scale_losses)
        elif self.config.scale_loss_mode == 'last':
            losses['feature_similarity'] = scale_losses[-1]
        elif self.config.scale_loss_mode == 'weighted':
            # Weight by scale size
            weights = torch.tensor([s / sum(self.scales) for s in self.scales])
            losses['feature_similarity'] = sum(
                w * l for w, l in zip(weights, scale_losses)
            )

        return losses

    def _compute_rose_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        batch_crystals: torch.Tensor,
        batch_size: int
    ) -> Dict[str, torch.Tensor]:
        """Rose loss - geometric alignment with crystals."""
        losses = {}
        rose_losses = []

        for i, features in enumerate(outputs['scale_features']):
            scale = self.scales[i]

            # Get centroids for this scale
            scale_crystals = batch_crystals[..., :scale]  # [B, patterns, 5, scale]
            centroids = scale_crystals.mean(dim=2)  # [B, patterns, scale]

            # Use first pattern as representative
            own_centroids = centroids[:, 0, :]  # [B, scale]

            # Cosine similarity to own centroid
            cos_sim = F.cosine_similarity(features, own_centroids, dim=-1)
            rose_loss = (1 - cos_sim).mean()

            rose_losses.append(rose_loss)
            losses[f'rose_scale_{i}'] = rose_loss

        losses['rose'] = sum(rose_losses) / len(rose_losses)
        return losses

    def _compute_cayley_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        batch_crystals: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Cayley loss - geometric quality of pentachora."""
        losses = {}
        cayley_losses = []

        for i, features in enumerate(outputs['scale_features']):
            scale = self.scales[i]

            # Get pentachora for this scale
            scale_crystals = batch_crystals[..., :scale]  # [B, patterns, 5, scale]
            pentachora = scale_crystals[:, 0, :, :]  # [B, 5, scale]

            cayley_loss = self.cayley_loss(pentachora)
            cayley_losses.append(cayley_loss)
            losses[f'cayley_scale_{i}'] = cayley_loss

        losses['cayley'] = sum(cayley_losses) / len(cayley_losses)
        return losses

    def _compute_classification_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        timestep_class: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Classification loss and accuracy."""
        losses = {}

        # Cross entropy for timestep prediction
        ce_loss = F.cross_entropy(
            outputs['combined_logits'],
            timestep_class
        )
        losses['ce'] = ce_loss

        # Compute accuracy
        pred = outputs['combined_logits'].argmax(dim=-1) // self.num_feature_patterns
        accuracy = (pred == timestep_class).float().mean()
        losses['accuracy'] = accuracy

        return losses

    def _compute_total_loss(self, losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Weighted combination of all losses."""
        total = 0.0

        if 'feature_similarity' in losses:
            total += self.config.feature_similarity_weight * losses['feature_similarity']

        if 'rose' in losses:
            total += self.config.rose_weight * losses['rose']

        if 'cayley' in losses:
            total += self.config.cayley_weight * losses['cayley']

        if 'ce' in losses:
            total += self.config.ce_weight * losses['ce']

        return total


# ============================================================================
# PATTERN-SUPERVISED LOSS CALCULATOR
# ============================================================================

class PatternSupervisedLossCalculator(BlockLossCalculator):
    """
    Pattern-supervised loss calculator for self-organizing pattern learning.

    This is the BATTLE-TESTED approach that successfully trained previous David models.

    Key Features:
    - 1000-class supervision (100 bins × 10 patterns)
    - Student-driven pattern assignment (uses student features, not teacher)
    - Soft assignment with temperature smoothing
    - Pattern diversity regularization (prevents mode collapse)
    - Compares to centroids, not teacher features directly

    Differences from base BlockLossCalculator:
    - assign_patterns(): Explicit pattern assignment via cosine similarity
    - compute_soft_assignment(): Temperature-smoothed soft targets
    - compute_pattern_diversity_loss(): Entropy regularization
    - Rose loss = feature_sim (simplified, not separate geometric loss)
    """

    def __init__(
        self,
        block_spec: SD15BlockSpec,
        scales: List[int],
        loss_config: PatternSupervisedLossConfig,
        num_timestep_bins: int,
        num_feature_patterns: int
    ):
        # Initialize base (creates teacher projections, loss instances)
        super().__init__(
            block_spec=block_spec,
            scales=scales,
            loss_config=loss_config,
            num_timestep_bins=num_timestep_bins,
            num_feature_patterns=num_feature_patterns
        )

        # Store pattern-specific config
        self.pattern_config = loss_config
        self.num_bins = num_timestep_bins
        self.num_patterns = num_feature_patterns
        self.num_classes = num_timestep_bins * num_feature_patterns

    def assign_patterns(
        self,
        features: torch.Tensor,
        timestep_class: torch.Tensor,
        crystal_centroids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Assign samples to nearest pattern within their timestep bin.

        Uses COSINE SIMILARITY (critical for geometric consistency).

        Args:
            features: [B, D] - Student features
            timestep_class: [B] - Timestep bins [0, num_bins)
            crystal_centroids: [num_bins, num_patterns, D] - Pattern centroids

        Returns:
            pattern_ids: [B] - Pattern indices [0, num_patterns)
            full_class_ids: [B] - Full class [0, num_classes)
        """
        B = features.shape[0]

        # Get centroids for each sample's timestep
        batch_centroids = crystal_centroids[timestep_class]  # [B, num_patterns, D]

        # Compute cosine similarities
        features_expanded = features.unsqueeze(1)  # [B, 1, D]
        similarities = F.cosine_similarity(
            features_expanded,
            batch_centroids,
            dim=2
        )  # [B, num_patterns]

        # Assign to nearest (highest similarity)
        pattern_ids = similarities.argmax(dim=1)
        full_class_ids = timestep_class * self.num_patterns + pattern_ids

        return pattern_ids, full_class_ids

    def compute_soft_assignment(
        self,
        features: torch.Tensor,
        timestep_class: torch.Tensor,
        crystal_centroids: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute soft pattern assignment with temperature smoothing.

        Args:
            features: [B, D]
            timestep_class: [B] - Timestep bins
            crystal_centroids: [num_bins, num_patterns, D]

        Returns:
            soft_targets: [B, num_classes] - Soft target distribution
        """
        B, D = features.shape
        device = features.device

        # Get centroids for each sample's timestep bin
        batch_centroids = crystal_centroids[timestep_class]  # [B, num_patterns, D]
        features_expanded = features.unsqueeze(1)  # [B, 1, D]

        # Compute cosine similarities
        similarities = F.cosine_similarity(
            features_expanded,
            batch_centroids,
            dim=2
        )  # [B, num_patterns]

        # Soft assignment with temperature
        pattern_probs = F.softmax(
            similarities / self.pattern_config.assignment_temperature,
            dim=1
        )

        # Create full soft targets [B, num_classes]
        soft_targets = torch.zeros(B, self.num_classes, device=device)
        for i in range(B):
            t = timestep_class[i]
            start_idx = t * self.num_patterns
            end_idx = start_idx + self.num_patterns
            soft_targets[i, start_idx:end_idx] = pattern_probs[i]

        return soft_targets

    def compute_pattern_diversity_loss(
        self,
        logits: torch.Tensor,
        timestep_class: torch.Tensor
    ) -> torch.Tensor:
        """
        Encourage diverse pattern usage (prevent mode collapse).

        Computes entropy of pattern distribution and minimizes negative entropy
        to maximize diversity.

        Args:
            logits: [B, num_classes]
            timestep_class: [B]

        Returns:
            diversity_loss: Scalar (lower = more diverse)
        """
        B = logits.shape[0]

        # For each sample, get pattern probs within its timestep
        pattern_probs_list = []
        for i in range(B):
            t = timestep_class[i]
            start_idx = t * self.num_patterns
            end_idx = start_idx + self.num_patterns
            probs = F.softmax(logits[i, start_idx:end_idx], dim=0)
            pattern_probs_list.append(probs)

        pattern_probs = torch.stack(pattern_probs_list)  # [B, num_patterns]

        # Entropy (higher = more diverse)
        entropy = -(pattern_probs * torch.log(pattern_probs + 1e-8)).sum(dim=1).mean()

        # Minimize negative entropy (maximize diversity)
        return -entropy

    def compute_losses(
        self,
        outputs: Dict[str, torch.Tensor],
        teacher_features: torch.Tensor,
        crystal_centroids: torch.Tensor,
        spatial_adapter: nn.Module
    ) -> Dict[str, torch.Tensor]:
        """
        Compute pattern-supervised losses.

        CRITICAL: This method expects crystal_centroids [num_bins, num_patterns, scale]
        NOT batch_crystals [B, num_patterns, 5, scale].

        Args:
            outputs: Forward outputs from companion
            teacher_features: [B, C, H, W] - Not used directly in this version!
            crystal_centroids: [num_bins, num_patterns, scale] - FULL centroids
            spatial_adapter: For converting features (not used in pattern supervision)

        Returns:
            Dictionary of losses and metrics
        """
        losses = {}

        timestep_class = outputs['timestep_class']
        batch_size = timestep_class.shape[0]

        # Compute losses per scale and aggregate
        scale_feature_losses = []
        scale_rose_losses = []
        scale_ce_losses = []
        scale_diversity_losses = []

        scale_timestep_accs = []
        scale_pattern_accs = []
        scale_full_accs = []

        for i, scale_features in enumerate(outputs['scale_features']):
            scale = self.scales[i]
            scale_logits = outputs['scale_logits'][i]

            # Get centroids for this scale [num_bins, num_patterns, scale]
            scale_centroids = crystal_centroids[:, :, :scale]

            # Pattern assignment using STUDENT features
            pattern_ids, full_class_ids = self.assign_patterns(
                scale_features,
                timestep_class,
                scale_centroids
            )

            # Get target centroids for assigned patterns
            target_centroids = torch.stack([
                scale_centroids[timestep_class[j], pattern_ids[j]]
                for j in range(batch_size)
            ])

            # 1. Feature Similarity Loss
            feature_sim_loss = (1.0 - F.cosine_similarity(
                scale_features,
                target_centroids,
                dim=-1
            )).mean()

            scale_feature_losses.append(feature_sim_loss)
            losses[f'feature_sim_scale_{i}'] = feature_sim_loss

            # 2. Rose Loss (same as feature similarity in this version)
            rose_loss = feature_sim_loss
            scale_rose_losses.append(rose_loss)
            losses[f'rose_scale_{i}'] = rose_loss

            # 3. Cross-Entropy Loss
            if self.pattern_config.use_soft_assignment:
                soft_targets = self.compute_soft_assignment(
                    scale_features,
                    timestep_class,
                    scale_centroids
                )
                log_probs = F.log_softmax(scale_logits, dim=1)
                ce_loss = -(soft_targets * log_probs).sum(dim=1).mean()
            else:
                ce_loss = F.cross_entropy(scale_logits, full_class_ids)

            scale_ce_losses.append(ce_loss)
            losses[f'ce_scale_{i}'] = ce_loss

            # 4. Pattern Diversity Loss
            diversity_loss = self.compute_pattern_diversity_loss(
                scale_logits,
                timestep_class
            )
            scale_diversity_losses.append(diversity_loss)
            losses[f'diversity_scale_{i}'] = diversity_loss

            # Accuracy metrics
            timestep_pred = scale_logits.argmax(dim=-1) // self.num_patterns
            pattern_pred = scale_logits.argmax(dim=-1) % self.num_patterns
            full_pred = scale_logits.argmax(dim=-1)

            timestep_acc = (timestep_pred == timestep_class).float().mean()
            pattern_acc = (pattern_pred == pattern_ids).float().mean()
            full_acc = (full_pred == full_class_ids).float().mean()

            scale_timestep_accs.append(timestep_acc)
            scale_pattern_accs.append(pattern_acc)
            scale_full_accs.append(full_acc)

            losses[f'timestep_acc_scale_{i}'] = timestep_acc
            losses[f'pattern_acc_scale_{i}'] = pattern_acc
            losses[f'full_acc_scale_{i}'] = full_acc

        # Aggregate across scales
        losses['feature_similarity'] = sum(scale_feature_losses) / len(scale_feature_losses)
        losses['rose'] = sum(scale_rose_losses) / len(scale_rose_losses)
        losses['ce'] = sum(scale_ce_losses) / len(scale_ce_losses)
        losses['pattern_diversity'] = sum(scale_diversity_losses) / len(scale_diversity_losses)

        losses['timestep_acc'] = sum(scale_timestep_accs) / len(scale_timestep_accs)
        losses['pattern_acc'] = sum(scale_pattern_accs) / len(scale_pattern_accs)
        losses['full_acc'] = sum(scale_full_accs) / len(scale_full_accs)

        # Also store as 'accuracy' for compatibility
        losses['accuracy'] = losses['full_acc']

        # Total loss
        losses['total'] = self._compute_total_loss(losses)

        return losses

    def _compute_total_loss(self, losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Weighted combination of pattern-supervised losses.

        Note: rose_weight effectively adds to feature_similarity_weight
        since rose = feature_sim in this version.
        """
        total = 0.0

        if 'feature_similarity' in losses:
            total += self.config.feature_similarity_weight * losses['feature_similarity']

        if 'rose' in losses:
            total += self.config.rose_weight * losses['rose']

        if 'ce' in losses:
            total += self.config.ce_weight * losses['ce']

        if 'pattern_diversity' in losses:
            total += self.pattern_config.pattern_diversity_weight * losses['pattern_diversity']

        return total


# ============================================================================
# DAVID BLOCK COMPANION - REFACTORED
# ============================================================================

class DavidBlockCompanion(nn.Module):
    """
    Single David instance for one SD1.5 block.
    Learns timestep-conditioned feature distributions.

    PURE FORWARD PASS MODEL: No loss computation or loss calculator.
    Loss computation handled externally by DavidCollective using BlockLossCalculator.

    Companion responsibilities:
    - Forward passes through David
    - Crystal anchor management
    - Spatial feature adaptation
    - Timestep discretization

    Loss computation is NOT the companion's responsibility.
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
        if block_spec.channels <= 320:
            scales = [64, 128, 256]
        elif block_spec.channels <= 640:
            scales = [128, 256, 512]
        else:
            scales = [256, 512, 768]

        self.scales = scales

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

    def _initialize_crystals(self):
        """Initialize crystal anchors with valid geometric properties."""
        for t_bin in range(self.config.num_timestep_bins):
            for pattern in range(self.config.num_feature_patterns_per_timestep):
                crystal = self.crystal_anchors[t_bin, pattern]
                crystal.normal_()
                crystal = F.normalize(crystal, dim=-1)
                crystal += 0.1 * torch.randn_like(crystal)

    def get_timestep_crystals(self, timestep_class: torch.Tensor) -> torch.Tensor:
        """Get crystal anchors for given timestep classes."""
        return self.crystal_anchors[timestep_class]

    def get_class_anchors(self, scale: int) -> torch.Tensor:
        """Get 2D anchors for David from pentachoron centroids."""
        all_crystals = self.crystal_anchors
        scale_crystals = all_crystals[..., :scale]
        centroids = scale_crystals.mean(dim=2)
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

        CLEAN: Only forward computation, no loss logic.
        """
        batch_size = spatial_features.shape[0]

        # 1. Convert spatial features to vector
        feature_vector = self.spatial_adapter(spatial_features)

        # 2. Discretize timesteps
        timestep_class = self.discretize_timestep(timestep)

        # 3. Prepare anchor dict for David
        anchors_dict = {}
        for scale in self.scales:
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

    def discretize_timestep(self, timestep: torch.Tensor) -> torch.Tensor:
        """Convert continuous timesteps to discrete bins."""
        return (timestep / 1000.0 * self.config.num_timestep_bins).long().clamp(
            0, self.config.num_timestep_bins - 1
        )


# ============================================================================
# SPATIAL FEATURE ADAPTER
# ============================================================================

class SpatialFeatureAdapter(nn.Module):
    """Convert spatial feature maps to vectors for David."""

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
            nn.Linear(pooled_dim, out_features),
            nn.LayerNorm(out_features),
            nn.GELU(),
            nn.Linear(out_features, out_features)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W]
        Returns:
            [B, out_features]
        """
        # Pool
        x = self.pool(x)  # [B, C, 4, 4]

        # Flatten
        x = x.flatten(1)  # [B, C*16]

        # Project
        x = self.projection(x)  # [B, out_features]

        return x


# ============================================================================
# DAVID COLLECTIVE - REFACTORED
# ============================================================================

class DavidCollective(nn.Module):
    """
    Parallel ensemble of DavidBlockCompanions with external loss computation.

    Architecture:
    - Companions: Pure forward pass models (one per SD1.5 UNet block)
    - Loss Calculators: External loss computation (one per companion, owned by collective)

    CLEAN SEPARATION:
    - Companions handle: forward passes, feature extraction, crystal management
    - Collective handles: loss computation via external BlockLossCalculators

    This separation enables:
    - Testable components (test forward passes and losses independently)
    - Flexible loss strategies (swap calculators without touching companions)
    - Clear responsibilities (companions predict, collective evaluates)
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

        # Loss calculators (EXTERNAL to companions)
        # Choose calculator type based on config
        self.loss_calculators = nn.ModuleDict()
        for block in self.active_blocks:
            if config.loss_calculator_type == 'pattern':
                # Pattern-supervised loss (battle-tested)
                calculator = PatternSupervisedLossCalculator(
                    block_spec=block,
                    scales=self.companions[block.name].scales,
                    loss_config=config.loss_config,
                    num_timestep_bins=config.num_timestep_bins,
                    num_feature_patterns=config.num_feature_patterns_per_timestep
                )
            else:
                # Block loss calculator (distillation-style)
                calculator = BlockLossCalculator(
                    block_spec=block,
                    scales=self.companions[block.name].scales,
                    loss_config=config.loss_config,
                    num_timestep_bins=config.num_timestep_bins,
                    num_feature_patterns=config.num_feature_patterns_per_timestep
                )
            self.loss_calculators[block.name] = calculator

        # Tracking
        self.current_epoch = 0
        self.block_metrics = {block.name: {} for block in self.active_blocks}

    def forward(
        self,
        block_features: Dict[str, torch.Tensor],
        timesteps: torch.Tensor
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Forward pass through all active companions.

        CLEAN: Just forward passes, no loss computation.
        """
        outputs = {}

        for block_name, companion in self.companions.items():
            if block_name in block_features:
                outputs[block_name] = companion(
                    block_features[block_name],
                    timesteps
                )

        return outputs

    def compute_losses(
        self,
        outputs: Dict[str, Dict[str, torch.Tensor]],
        teacher_features: Dict[str, torch.Tensor]
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Compute losses for all blocks using EXTERNAL loss calculators.

        Loss calculators are owned by DavidCollective, not companions.
        Companions are pure forward pass models.

        Handles both calculator types:
        - BlockLossCalculator: Uses batch_crystals [B, patterns, 5, scale]
        - PatternSupervisedLossCalculator: Uses crystal_centroids [bins, patterns, scale]
        """
        all_losses = {}

        for block_name in self.companions.keys():
            if block_name in outputs and block_name in teacher_features:
                companion = self.companions[block_name]
                calculator = self.loss_calculators[block_name]
                block_outputs = outputs[block_name]

                timestep_class = block_outputs['timestep_class']

                # Prepare data based on calculator type
                if isinstance(calculator, PatternSupervisedLossCalculator):
                    # Pattern-supervised needs FULL crystal centroids
                    # crystal_anchors: [num_bins, num_patterns, 5, max_scale]
                    # Compute centroids by averaging vertices: [num_bins, num_patterns, max_scale]
                    crystal_centroids = companion.crystal_anchors.mean(dim=2)

                    # Compute losses using crystal_centroids
                    losses = calculator.compute_losses(
                        outputs=block_outputs,
                        teacher_features=teacher_features[block_name],
                        crystal_centroids=crystal_centroids,  # FULL centroids
                        spatial_adapter=companion.spatial_adapter
                    )
                else:
                    # BlockLossCalculator uses batch crystals
                    batch_crystals = companion.get_timestep_crystals(timestep_class)

                    # Compute losses using batch_crystals
                    losses = calculator.compute_losses(
                        outputs=block_outputs,
                        teacher_features=teacher_features[block_name],
                        batch_crystals=batch_crystals,  # Batch only
                        spatial_adapter=companion.spatial_adapter
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
        """Get status of all companions and loss calculators."""
        # Total parameters includes both companions and loss calculators
        companion_params = sum(
            sum(p.numel() for p in companion.parameters())
            for companion in self.companions.values()
        )
        calculator_params = sum(
            sum(p.numel() for p in calculator.parameters())
            for calculator in self.loss_calculators.values()
        )

        return {
            'epoch': self.current_epoch,
            'num_companions': len(self.companions),
            'num_loss_calculators': len(self.loss_calculators),
            'active_blocks': [block.name for block in self.active_blocks],
            'total_parameters': companion_params + calculator_params,
            'companion_parameters': companion_params,
            'calculator_parameters': calculator_params,
            'companion_info': {
                name: {
                    'scales': companion.scales,
                    'channels': companion.block_spec.channels,
                    'parameters': sum(p.numel() for p in companion.parameters())
                }
                for name, companion in self.companions.items()
            },
            'calculator_info': {
                name: {
                    'parameters': sum(p.numel() for p in calculator.parameters())
                }
                for name, calculator in self.loss_calculators.items()
            }
        }

    def save_checkpoint(self, path: str, use_safetensors: bool = True):
        """
        Save collective checkpoint.

        Args:
            path: Path to save checkpoint (extension will be adjusted if needed)
            use_safetensors: If True (default), save as safetensors format.
                           If False, save as PyTorch .pt format.
        """
        path = Path(path)

        if use_safetensors:
            # Ensure .safetensors extension
            if path.suffix != '.safetensors':
                path = path.with_suffix('.safetensors')

            # Save state dict with safetensors
            state_dict = self.state_dict()
            save_file(state_dict, str(path))

            # Save metadata separately as JSON
            metadata = {
                'config': self.config,
                'epoch': self.current_epoch,
                'block_metrics': self.block_metrics
            }
            metadata_path = path.with_suffix('.json')
            with open(metadata_path, 'w') as f:
                # Convert config to dict for JSON serialization
                config_dict = {
                    'active_blocks': self.config.active_blocks,
                    'num_timestep_bins': self.config.num_timestep_bins,
                    'num_feature_patterns_per_timestep': self.config.num_feature_patterns_per_timestep,
                    'david_sharing_mode': self.config.david_sharing_mode,
                    'david_fusion_mode': self.config.david_fusion_mode,
                    'use_belly': self.config.use_belly,
                    'belly_expand': self.config.belly_expand,
                    'progressive_training': self.config.progressive_training,
                    'warmup_epochs_per_block': self.config.warmup_epochs_per_block,
                    'cache_dir': self.config.cache_dir,
                    'max_cache_size_gb': self.config.max_cache_size_gb,
                    'loss_config': {
                        'feature_similarity_weight': self.config.loss_config.feature_similarity_weight,
                        'rose_weight': self.config.loss_config.rose_weight,
                        'cayley_weight': self.config.loss_config.cayley_weight,
                        'ce_weight': self.config.loss_config.ce_weight,
                        'rose_margin': self.config.loss_config.rose_margin,
                        'rose_temperature': self.config.loss_config.rose_temperature,
                        'cayley_volume_floor': self.config.loss_config.cayley_volume_floor,
                        'use_rose_loss': self.config.loss_config.use_rose_loss,
                        'use_cayley_loss': self.config.loss_config.use_cayley_loss,
                        'use_ce_loss': self.config.loss_config.use_ce_loss,
                        'scale_loss_mode': self.config.loss_config.scale_loss_mode,
                    }
                }
                metadata['config'] = config_dict
                json.dump(metadata, f, indent=2)

            print(f"Saved checkpoint to {path}")
            print(f"Saved metadata to {metadata_path}")
        else:
            # Traditional PyTorch format
            if path.suffix not in ['.pt', '.pth']:
                path = path.with_suffix('.pt')

            checkpoint = {
                'config': self.config,
                'epoch': self.current_epoch,
                'state_dict': self.state_dict(),
                'block_metrics': self.block_metrics
            }
            torch.save(checkpoint, str(path))
            print(f"Saved checkpoint to {path}")

    @classmethod
    def load_checkpoint(cls, path: str, use_safetensors: bool = True):
        """
        Load collective from checkpoint.

        Args:
            path: Path to checkpoint file
            use_safetensors: If True (default), load from safetensors format.
                           If False, load from PyTorch .pt format.
                           If None, auto-detect from file extension.

        Returns:
            Loaded DavidCollective instance
        """
        path = Path(path)

        # Auto-detect format if needed
        if use_safetensors is None:
            use_safetensors = path.suffix == '.safetensors'

        if use_safetensors:
            # Ensure correct extension
            if path.suffix != '.safetensors':
                path = path.with_suffix('.safetensors')

            # Load state dict from safetensors
            state_dict = load_file(str(path))

            # Load metadata from JSON
            metadata_path = path.with_suffix('.json')
            if not metadata_path.exists():
                raise FileNotFoundError(
                    f"Metadata file not found: {metadata_path}. "
                    "Safetensors checkpoints require a corresponding .json metadata file."
                )

            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

            # Reconstruct config from dict
            config_dict = metadata['config']
            loss_config_dict = config_dict.pop('loss_config')
            loss_config = LossConfig(**loss_config_dict)
            config = DavidCollectiveConfig(**config_dict, loss_config=loss_config)

            # Create collective and load state
            collective = cls(config)
            collective.load_state_dict(state_dict)
            collective.current_epoch = metadata['epoch']
            collective.block_metrics = metadata['block_metrics']

            print(f"Loaded checkpoint from {path}")
            print(f"Loaded metadata from {metadata_path}")
        else:
            # Traditional PyTorch format
            if path.suffix not in ['.pt', '.pth']:
                path = path.with_suffix('.pt')

            checkpoint = torch.load(str(path), weights_only=False)
            collective = cls(checkpoint['config'])
            collective.load_state_dict(checkpoint['state_dict'])
            collective.current_epoch = checkpoint['epoch']
            collective.block_metrics = checkpoint['block_metrics']

            print(f"Loaded checkpoint from {path}")

        return collective


# ============================================================================
# FEATURE CACHING UTILITIES
# ============================================================================

class SD15FeatureCache:
    """Utilities for caching SD1.5 features to disk."""

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
        """Cache a batch of features to disk."""
        shard_path = self.cache_dir / f"shard_{shard_id:06d}.pt"

        data = {
            'features': features,
            'timesteps': timesteps.cpu(),
            'prompts': prompts,
            'shard_id': shard_id
        }

        torch.save(data, shard_path)

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
    print("DAVID COLLECTIVE - REFACTORED WITH EXTRACTED LOSS SYSTEM")
    print("=" * 80)

    # Loss configuration
    loss_config = LossConfig(
        feature_similarity_weight=0.5,
        rose_weight=0.2,
        cayley_weight=0.1,
        ce_weight=0.2,
        rose_margin=1.0,
        rose_temperature=0.07,
        cayley_volume_floor=1e-4
    )

    # Collective configuration
    config = DavidCollectiveConfig(
        active_blocks=['down_0', 'down_1', 'mid', 'up_0', 'up_1'],
        num_timestep_bins=100,
        num_feature_patterns_per_timestep=10,
        progressive_training=True,
        loss_config=loss_config
    )

    # Create collective
    collective = DavidCollective(config)

    print(f"\n[Refactoring Status]")
    print(f"  ✓ Loss computation FULLY EXTERNAL - not in companions")
    print(f"  ✓ Loss calculators owned by DavidCollective")
    print(f"  ✓ Companions are PURE forward pass models")
    print(f"  ✓ Teacher projections in external calculators")
    print(f"  ✓ Loss instances in external calculators")
    print(f"  ✓ Complete separation of concerns achieved")

    print(f"\n[Architecture]")
    status = collective.get_status()
    print(f"  DavidCollective:")
    print(f"    ├── companions: {status['num_companions']} forward pass models")
    print(f"    └── loss_calculators: {status['num_loss_calculators']} external calculators")
    print(f"  Companion parameters: {status['companion_parameters']:,}")
    print(f"  Calculator parameters: {status['calculator_parameters']:,}")

    print(f"\n[Status]")
    print(f"  Active blocks: {len(status['active_blocks'])}")
    print(f"  Total parameters: {status['total_parameters']:,}")

    # Simulate forward pass
    print(f"\n[Test Forward Pass]")
    batch_size = 4

    block_features = {
        'down_0': torch.randn(batch_size, 320, 64, 64),
        'down_1': torch.randn(batch_size, 640, 32, 32),
        'mid': torch.randn(batch_size, 1280, 8, 8),
        'up_0': torch.randn(batch_size, 1280, 16, 16),
        'up_1': torch.randn(batch_size, 1280, 32, 32),
    }

    timesteps = torch.randint(0, 1000, (batch_size,))

    with torch.no_grad():
        outputs = collective(block_features, timesteps)

    print(f"  ✓ Forward pass successful for {len(outputs)} blocks")

    # Test losses
    print(f"\n[Test Loss Computation]")
    teacher_features = block_features

    with torch.no_grad():
        losses = collective.compute_losses(outputs, teacher_features)
        total_loss = collective.get_total_loss(losses)

    print(f"  ✓ Loss computation successful")
    print(f"  Total loss: {total_loss.item():.4f}")

    for block_name, block_losses in losses.items():
        print(f"\n  {block_name}:")
        print(f"    Feature similarity: {block_losses['feature_similarity'].item():.4f}")
        if 'rose' in block_losses:
            print(f"    Rose: {block_losses['rose'].item():.4f}")
        if 'cayley' in block_losses:
            print(f"    Cayley: {block_losses['cayley'].item():.4f}")
        if 'ce' in block_losses:
            print(f"    CE: {block_losses['ce'].item():.4f}")
        if 'accuracy' in block_losses:
            print(f"    Accuracy: {block_losses['accuracy'].item():.2%}")

    print("\n" + "=" * 80)
    print("EXAMPLE 2: PATTERN-SUPERVISED LOSS (BATTLE-TESTED)")
    print("=" * 80)

    # Pattern-supervised loss configuration
    pattern_loss_config = PatternSupervisedLossConfig(
        feature_similarity_weight=0.5,
        rose_weight=0.3,  # Adds to feature_sim (same computation)
        ce_weight=0.2,
        pattern_diversity_weight=0.05,
        use_soft_assignment=True,
        assignment_temperature=0.1,
        use_cayley_loss=False
    )

    # Collective configuration with pattern-supervised loss
    pattern_config = DavidCollectiveConfig(
        active_blocks=['down_0', 'mid'],
        num_timestep_bins=100,
        num_feature_patterns_per_timestep=10,
        progressive_training=False,
        loss_config=pattern_loss_config,
        loss_calculator_type='pattern'  # Use pattern-supervised calculator
    )

    # Create collective with pattern-supervised loss
    pattern_collective = DavidCollective(pattern_config)

    print(f"\n[Pattern-Supervised Features]")
    print(f"  ✓ 1000-class supervision (100 bins × 10 patterns)")
    print(f"  ✓ Student-driven pattern assignment")
    print(f"  ✓ Soft assignment with temperature: {pattern_loss_config.assignment_temperature}")
    print(f"  ✓ Pattern diversity regularization: {pattern_loss_config.pattern_diversity_weight}")
    print(f"  ✓ Comparison to learned centroids")

    # Test forward and loss
    pattern_features = {
        'down_0': torch.randn(batch_size, 320, 64, 64),
        'mid': torch.randn(batch_size, 1280, 8, 8)
    }

    with torch.no_grad():
        pattern_outputs = pattern_collective(pattern_features, timesteps)
        pattern_losses = pattern_collective.compute_losses(pattern_outputs, pattern_features)
        pattern_total_loss = pattern_collective.get_total_loss(pattern_losses)

    print(f"\n[Pattern-Supervised Loss Results]")
    print(f"  Total loss: {pattern_total_loss.item():.4f}")

    for block_name, block_losses in pattern_losses.items():
        print(f"\n  {block_name}:")
        print(f"    Feature similarity: {block_losses['feature_similarity'].item():.4f}")
        print(f"    Rose: {block_losses['rose'].item():.4f}")
        print(f"    CE: {block_losses['ce'].item():.4f}")
        print(f"    Pattern diversity: {block_losses['pattern_diversity'].item():.4f}")
        if 'timestep_acc' in block_losses:
            print(f"    Timestep accuracy: {block_losses['timestep_acc'].item():.2%}")
        if 'pattern_acc' in block_losses:
            print(f"    Pattern accuracy: {block_losses['pattern_acc'].item():.2%}")
        if 'full_acc' in block_losses:
            print(f"    Full accuracy (1000-class): {block_losses['full_acc'].item():.2%}")

    print("\n" + "=" * 80)
    print("REFACTORING COMPLETE - LOSS SYSTEM EXTRACTED")
    print("Both BlockLossCalculator and PatternSupervisedLossCalculator available!")
    print("=" * 80)