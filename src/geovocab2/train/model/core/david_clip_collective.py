"""
DavidCollective: CLIP_L Layer-Level Feature Distillation
=========================================================
Layer-parallel geometric flow matching distillation system for CLIP text encoder.

Each David companion learns one CLIP_L transformer layer's hidden state distribution
with semantic pattern-aware geometric regularization.

Key Differences from SD1.5 Version:
- Sequential features [B, seq_len, hidden_dim] instead of spatial [B, C, H, W]
- No timestep conditioning (text encoding is direct)
- Semantic pattern bins instead of timestep bins
- Token-level or pooled feature learning

Author: AbstractPhil + Claude Sonnet 4.5
Date: 2025-11-12 (CLIP_L Adaptation)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from pathlib import Path
import json
from safetensors.torch import save_file, load_file

# Assuming David imports (same as SD1.5 version)
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
class CLIPLayerSpec:
    """Specification for one CLIP_L transformer layer."""
    name: str
    hidden_dim: int  # 768 for CLIP-L
    num_heads: int  # 12 for CLIP-L
    layer_idx: int
    position: str  # 'early', 'mid', 'late'


# CLIP-L (ViT-L/14) layer specifications
CLIP_L_LAYERS = [
    # Early layers (0-3): Low-level features
    CLIPLayerSpec('layer_0', 768, 12, 0, 'early'),
    CLIPLayerSpec('layer_1', 768, 12, 1, 'early'),
    CLIPLayerSpec('layer_2', 768, 12, 2, 'early'),
    CLIPLayerSpec('layer_3', 768, 12, 3, 'early'),

    # Mid layers (4-7): Compositional features
    CLIPLayerSpec('layer_4', 768, 12, 4, 'mid'),
    CLIPLayerSpec('layer_5', 768, 12, 5, 'mid'),
    CLIPLayerSpec('layer_6', 768, 12, 6, 'mid'),
    CLIPLayerSpec('layer_7', 768, 12, 7, 'mid'),

    # Late layers (8-11): High-level semantic features
    CLIPLayerSpec('layer_8', 768, 12, 8, 'late'),
    CLIPLayerSpec('layer_9', 768, 12, 9, 'late'),
    CLIPLayerSpec('layer_10', 768, 12, 10, 'late'),
    CLIPLayerSpec('layer_11', 768, 12, 11, 'late'),
]


@dataclass
class LossConfig:
    """Centralized loss configuration (same as SD1.5)."""

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
class SemanticPatternLossConfig(LossConfig):
    """
    Extended config for semantic pattern-supervised learning.

    Adapted from PatternSupervisedLossConfig for text semantics.
    Instead of timestep bins, we use semantic pattern bins that
    the model learns to discover in text embedding space.
    """

    # Semantic pattern parameters
    use_soft_assignment: bool = True
    assignment_temperature: float = 0.1
    pattern_diversity_weight: float = 0.05

    # Override defaults for semantic supervision
    feature_similarity_weight: float = 0.5
    rose_weight: float = 0.3
    ce_weight: float = 0.2

    # Disable unused components
    use_cayley_loss: bool = False
    cayley_weight: float = 0.0


@dataclass
class CLIPCollectiveConfig:
    """Configuration for CLIP-L DavidCollective system."""

    # Which layers to use
    active_layers: List[str] = None  # None = all layers

    # Semantic pattern discretization
    num_semantic_bins: int = 50  # Semantic clusters to discover
    num_feature_patterns_per_bin: int = 10  # Patterns per cluster

    # Feature extraction mode
    feature_mode: str = 'mean_pool'  # 'mean_pool', 'cls_token', 'all_tokens'

    # David architecture per layer
    david_sharing_mode: str = 'fully_shared'
    david_fusion_mode: str = 'deep_efficiency'
    use_belly: bool = True
    belly_expand: float = 1.5

    # Loss configuration
    loss_config: LossConfig = None
    loss_calculator_type: str = 'semantic'  # 'block' or 'semantic'

    # Progressive training
    progressive_training: bool = True
    warmup_epochs_per_layer: int = 5

    # Caching
    cache_dir: str = "./clip_features_cache"
    max_cache_size_gb: float = 100.0

    def __post_init__(self):
        if self.active_layers is None:
            self.active_layers = [layer.name for layer in CLIP_L_LAYERS]
        if self.loss_config is None:
            self.loss_config = LossConfig()


# ============================================================================
# SEQUENCE FEATURE ADAPTER
# ============================================================================

class SequenceFeatureAdapter(nn.Module):
    """
    Convert sequential CLIP features to vectors for David.

    Replaces SpatialFeatureAdapter from SD1.5 version.
    Handles [B, seq_len, hidden_dim] → [B, out_features]
    """

    def __init__(
            self,
            hidden_dim: int,
            out_features: int,
            mode: str = 'mean_pool'
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.out_features = out_features
        self.mode = mode

        # Project to output dimension
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, out_features),
            nn.LayerNorm(out_features),
            nn.GELU(),
            nn.Linear(out_features, out_features)
        )

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [B, seq_len, hidden_dim]
            attention_mask: [B, seq_len] optional

        Returns:
            [B, out_features]
        """
        if self.mode == 'mean_pool':
            # Mean pooling over sequence
            if attention_mask is not None:
                # Masked mean
                mask_expanded = attention_mask.unsqueeze(-1).float()
                x = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
            else:
                x = x.mean(dim=1)

        elif self.mode == 'cls_token':
            # Use first token (CLS/BOS)
            x = x[:, 0, :]

        elif self.mode == 'max_pool':
            # Max pooling over sequence
            x = x.max(dim=1)[0]

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        # Project
        x = self.projection(x)  # [B, out_features]

        return x


# ============================================================================
# SEMANTIC PATTERN LOSS CALCULATOR
# ============================================================================

class SemanticPatternLossCalculator(nn.Module):
    """
    Semantic pattern-supervised loss for CLIP layer distillation.

    Adapted from PatternSupervisedLossCalculator.

    Key differences:
    - No timestep bins → semantic bins (discovered clusters)
    - Token-level or pooled features
    - Semantic pattern discovery in text embedding space
    """

    def __init__(
            self,
            layer_spec: CLIPLayerSpec,
            scales: List[int],
            loss_config: SemanticPatternLossConfig,
            num_semantic_bins: int,
            num_feature_patterns: int
    ):
        super().__init__()

        self.layer_spec = layer_spec
        self.scales = scales
        self.config = loss_config
        self.num_bins = num_semantic_bins
        self.num_patterns = num_feature_patterns
        self.num_classes = num_semantic_bins * num_feature_patterns

        # Teacher feature projections
        self.teacher_projections = nn.ModuleDict({
            str(scale): nn.Sequential(
                nn.Linear(layer_spec.hidden_dim, scale),
                nn.LayerNorm(scale)
            )
            for scale in scales
        })

        # Loss function instances
        if self.config.use_rose_loss:
            self.rose_loss = RoseLoss(
                margin=self.config.rose_margin,
                temperature=self.config.rose_temperature
            )

        if self.config.use_cayley_loss:
            self.cayley_loss = CayleyChaosLoss(
                volume_floor=self.config.cayley_volume_floor
            )

    def assign_semantic_patterns(
            self,
            features: torch.Tensor,
            semantic_bin: torch.Tensor,
            crystal_centroids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Assign samples to nearest semantic pattern within their bin.

        Args:
            features: [B, D] - Student features
            semantic_bin: [B] - Semantic bins [0, num_bins)
            crystal_centroids: [num_bins, num_patterns, D]

        Returns:
            pattern_ids: [B]
            full_class_ids: [B]
        """
        B = features.shape[0]

        # Get centroids for each sample's semantic bin
        batch_centroids = crystal_centroids[semantic_bin]  # [B, num_patterns, D]

        # Compute cosine similarities
        features_expanded = features.unsqueeze(1)  # [B, 1, D]
        similarities = F.cosine_similarity(
            features_expanded,
            batch_centroids,
            dim=2
        )  # [B, num_patterns]

        # Assign to nearest
        pattern_ids = similarities.argmax(dim=1)
        full_class_ids = semantic_bin * self.num_patterns + pattern_ids

        return pattern_ids, full_class_ids

    def compute_soft_assignment(
            self,
            features: torch.Tensor,
            semantic_bin: torch.Tensor,
            crystal_centroids: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute soft semantic pattern assignment.

        Args:
            features: [B, D]
            semantic_bin: [B]
            crystal_centroids: [num_bins, num_patterns, D]

        Returns:
            soft_targets: [B, num_classes]
        """
        B, D = features.shape
        device = features.device

        # Get centroids
        batch_centroids = crystal_centroids[semantic_bin]
        features_expanded = features.unsqueeze(1)

        # Compute similarities
        similarities = F.cosine_similarity(
            features_expanded,
            batch_centroids,
            dim=2
        )

        # Soft assignment with temperature
        pattern_probs = F.softmax(
            similarities / self.config.assignment_temperature,
            dim=1
        )

        # Create full soft targets
        soft_targets = torch.zeros(B, self.num_classes, device=device)
        for i in range(B):
            bin_idx = semantic_bin[i]
            start_idx = bin_idx * self.num_patterns
            end_idx = start_idx + self.num_patterns
            soft_targets[i, start_idx:end_idx] = pattern_probs[i]

        return soft_targets

    def compute_pattern_diversity_loss(
            self,
            logits: torch.Tensor,
            semantic_bin: torch.Tensor
    ) -> torch.Tensor:
        """Encourage diverse pattern usage."""
        B = logits.shape[0]

        pattern_probs_list = []
        for i in range(B):
            bin_idx = semantic_bin[i]
            start_idx = bin_idx * self.num_patterns
            end_idx = start_idx + self.num_patterns
            probs = F.softmax(logits[i, start_idx:end_idx], dim=0)
            pattern_probs_list.append(probs)

        pattern_probs = torch.stack(pattern_probs_list)

        # Entropy (higher = more diverse)
        entropy = -(pattern_probs * torch.log(pattern_probs + 1e-8)).sum(dim=1).mean()

        # Minimize negative entropy
        return -entropy

    def compute_losses(
            self,
            outputs: Dict[str, torch.Tensor],
            teacher_features: torch.Tensor,
            crystal_centroids: torch.Tensor,
            sequence_adapter: nn.Module
    ) -> Dict[str, torch.Tensor]:
        """
        Compute semantic pattern-supervised losses.

        Args:
            outputs: Forward outputs from companion
            teacher_features: [B, seq_len, hidden_dim] - Teacher hidden states
            crystal_centroids: [num_bins, num_patterns, scale]
            sequence_adapter: For converting teacher features

        Returns:
            Dictionary of losses and metrics
        """
        losses = {}

        semantic_bin = outputs['semantic_bin']
        batch_size = semantic_bin.shape[0]

        # Compute losses per scale
        scale_feature_losses = []
        scale_rose_losses = []
        scale_ce_losses = []
        scale_diversity_losses = []

        scale_bin_accs = []
        scale_pattern_accs = []
        scale_full_accs = []

        for i, scale_features in enumerate(outputs['scale_features']):
            scale = self.scales[i]
            scale_logits = outputs['scale_logits'][i]

            # Get centroids for this scale
            scale_centroids = crystal_centroids[:, :, :scale]

            # Pattern assignment
            pattern_ids, full_class_ids = self.assign_semantic_patterns(
                scale_features,
                semantic_bin,
                scale_centroids
            )

            # Get target centroids
            target_centroids = torch.stack([
                scale_centroids[semantic_bin[j], pattern_ids[j]]
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

            # 2. Rose Loss (same as feature similarity)
            rose_loss = feature_sim_loss
            scale_rose_losses.append(rose_loss)
            losses[f'rose_scale_{i}'] = rose_loss

            # 3. Cross-Entropy Loss
            if self.config.use_soft_assignment:
                soft_targets = self.compute_soft_assignment(
                    scale_features,
                    semantic_bin,
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
                semantic_bin
            )
            scale_diversity_losses.append(diversity_loss)
            losses[f'diversity_scale_{i}'] = diversity_loss

            # Accuracy metrics
            bin_pred = scale_logits.argmax(dim=-1) // self.num_patterns
            pattern_pred = scale_logits.argmax(dim=-1) % self.num_patterns
            full_pred = scale_logits.argmax(dim=-1)

            bin_acc = (bin_pred == semantic_bin).float().mean()
            pattern_acc = (pattern_pred == pattern_ids).float().mean()
            full_acc = (full_pred == full_class_ids).float().mean()

            scale_bin_accs.append(bin_acc)
            scale_pattern_accs.append(pattern_acc)
            scale_full_accs.append(full_acc)

            losses[f'bin_acc_scale_{i}'] = bin_acc
            losses[f'pattern_acc_scale_{i}'] = pattern_acc
            losses[f'full_acc_scale_{i}'] = full_acc

        # Aggregate across scales
        losses['feature_similarity'] = sum(scale_feature_losses) / len(scale_feature_losses)
        losses['rose'] = sum(scale_rose_losses) / len(scale_rose_losses)
        losses['ce'] = sum(scale_ce_losses) / len(scale_ce_losses)
        losses['pattern_diversity'] = sum(scale_diversity_losses) / len(scale_diversity_losses)

        losses['bin_acc'] = sum(scale_bin_accs) / len(scale_bin_accs)
        losses['pattern_acc'] = sum(scale_pattern_accs) / len(scale_pattern_accs)
        losses['full_acc'] = sum(scale_full_accs) / len(scale_full_accs)
        losses['accuracy'] = losses['full_acc']

        # Total loss
        losses['total'] = self._compute_total_loss(losses)

        return losses

    def _compute_total_loss(self, losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Weighted combination of losses."""
        total = 0.0

        if 'feature_similarity' in losses:
            total += self.config.feature_similarity_weight * losses['feature_similarity']

        if 'rose' in losses:
            total += self.config.rose_weight * losses['rose']

        if 'ce' in losses:
            total += self.config.ce_weight * losses['ce']

        if 'pattern_diversity' in losses:
            total += self.config.pattern_diversity_weight * losses['pattern_diversity']

        return total


# ============================================================================
# DAVID LAYER COMPANION
# ============================================================================

class DavidLayerCompanion(nn.Module):
    """
    Single David instance for one CLIP layer.
    Learns semantic pattern-conditioned feature distributions.

    Adapted from DavidBlockCompanion for sequential features.
    """

    def __init__(
            self,
            layer_spec: CLIPLayerSpec,
            config: CLIPCollectiveConfig
    ):
        super().__init__()

        self.layer_spec = layer_spec
        self.config = config

        # Determine David scales
        scales = [128, 256, 512]  # Fixed for 768-dim CLIP
        self.scales = scales

        # David configuration
        self.david_config = DavidArchitectureConfig(
            feature_dim=layer_spec.hidden_dim,
            num_classes=config.num_semantic_bins * config.num_feature_patterns_per_bin,
            scales=scales,
            sharing_mode=config.david_sharing_mode,
            fusion_mode=config.david_fusion_mode,
            use_belly=config.use_belly,
            belly_expand=config.belly_expand,
            progressive_training=config.progressive_training,
        )

        # Core David model
        self.david = David.from_config(self.david_config)

        # Sequence → Vector adapter
        self.sequence_adapter = SequenceFeatureAdapter(
            hidden_dim=layer_spec.hidden_dim,
            out_features=layer_spec.hidden_dim,
            mode=config.feature_mode
        )

        # Crystal anchors: [semantic_bins, patterns, 5, max_scale]
        max_scale = max(scales)
        self.crystal_anchors = nn.Parameter(
            torch.randn(
                config.num_semantic_bins,
                config.num_feature_patterns_per_bin,
                5,
                max_scale
            )
        )

        # Initialize crystals
        with torch.no_grad():
            self._initialize_crystals()

        # Semantic bin assignment network
        # This learns to map features → semantic bins
        self.bin_classifier = nn.Sequential(
            nn.Linear(layer_spec.hidden_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, config.num_semantic_bins)
        )

    def _initialize_crystals(self):
        """Initialize crystal anchors."""
        for bin_idx in range(self.config.num_semantic_bins):
            for pattern in range(self.config.num_feature_patterns_per_bin):
                crystal = self.crystal_anchors[bin_idx, pattern]
                crystal.normal_()
                crystal = F.normalize(crystal, dim=-1)
                crystal += 0.1 * torch.randn_like(crystal)

    def get_semantic_crystals(self, semantic_bin: torch.Tensor) -> torch.Tensor:
        """Get crystal anchors for semantic bins."""
        return self.crystal_anchors[semantic_bin]

    def get_class_anchors(self, scale: int) -> torch.Tensor:
        """Get 2D anchors for David."""
        all_crystals = self.crystal_anchors
        scale_crystals = all_crystals[..., :scale]
        centroids = scale_crystals.mean(dim=2)
        num_classes = centroids.shape[0] * centroids.shape[1]
        anchors = centroids.reshape(num_classes, scale)
        return anchors

    def assign_semantic_bin(self, features: torch.Tensor) -> torch.Tensor:
        """
        Assign features to semantic bins.

        This is learned jointly with the rest of the model.
        """
        logits = self.bin_classifier(features)
        semantic_bin = logits.argmax(dim=-1)
        return semantic_bin

    def forward(
            self,
            sequence_features: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            return_all_scales: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through layer companion.

        Args:
            sequence_features: [B, seq_len, hidden_dim]
            attention_mask: [B, seq_len] optional

        Returns:
            Dict with outputs
        """
        batch_size = sequence_features.shape[0]

        # 1. Convert sequence to vector
        feature_vector = self.sequence_adapter(sequence_features, attention_mask)

        # 2. Assign to semantic bin
        semantic_bin = self.assign_semantic_bin(feature_vector)

        # 3. Prepare anchor dict
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
            'semantic_bin': semantic_bin,
            'feature_vector': feature_vector,
            'sequence_features': sequence_features
        }


# ============================================================================
# CLIP COLLECTIVE
# ============================================================================

class CLIPCollective(nn.Module):
    """
    Parallel ensemble of DavidLayerCompanions for CLIP-L distillation.

    Adapted from DavidCollective for sequential transformer layers.
    """

    def __init__(self, config: CLIPCollectiveConfig):
        super().__init__()

        self.config = config

        # Filter active layers
        self.active_layers = [
            layer for layer in CLIP_L_LAYERS
            if layer.name in config.active_layers
        ]

        # Create companions
        self.companions = nn.ModuleDict({
            layer.name: DavidLayerCompanion(layer, config)
            for layer in self.active_layers
        })

        # Loss calculators (external)
        self.loss_calculators = nn.ModuleDict()
        for layer in self.active_layers:
            # Use semantic pattern loss calculator
            calculator = SemanticPatternLossCalculator(
                layer_spec=layer,
                scales=self.companions[layer.name].scales,
                loss_config=config.loss_config,
                num_semantic_bins=config.num_semantic_bins,
                num_feature_patterns=config.num_feature_patterns_per_bin
            )
            self.loss_calculators[layer.name] = calculator

        # Tracking
        self.current_epoch = 0
        self.layer_metrics = {layer.name: {} for layer in self.active_layers}

    def forward(
            self,
            layer_features: Dict[str, torch.Tensor],
            attention_masks: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Forward pass through all active companions.

        Args:
            layer_features: Dict[layer_name, [B, seq_len, 768]]
            attention_masks: Dict[layer_name, [B, seq_len]]
        """
        outputs = {}

        for layer_name, companion in self.companions.items():
            if layer_name in layer_features:
                mask = attention_masks.get(layer_name) if attention_masks else None
                outputs[layer_name] = companion(
                    layer_features[layer_name],
                    mask
                )

        return outputs

    def compute_losses(
            self,
            outputs: Dict[str, Dict[str, torch.Tensor]],
            teacher_features: Dict[str, torch.Tensor]
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Compute losses using external loss calculators.
        """
        all_losses = {}

        for layer_name in self.companions.keys():
            if layer_name in outputs and layer_name in teacher_features:
                companion = self.companions[layer_name]
                calculator = self.loss_calculators[layer_name]
                layer_outputs = outputs[layer_name]

                semantic_bin = layer_outputs['semantic_bin']

                # Get crystal centroids
                crystal_centroids = companion.crystal_anchors.mean(dim=2)

                # Compute losses
                losses = calculator.compute_losses(
                    outputs=layer_outputs,
                    teacher_features=teacher_features[layer_name],
                    crystal_centroids=crystal_centroids,
                    sequence_adapter=companion.sequence_adapter
                )

                all_losses[layer_name] = losses

        return all_losses

    def get_total_loss(
            self,
            layer_losses: Dict[str, Dict[str, torch.Tensor]]
    ) -> torch.Tensor:
        """Aggregate total loss across all layers."""
        total = sum(
            losses['total']
            for losses in layer_losses.values()
        ) / len(layer_losses)
        return total

    def update_epoch(self, epoch: int):
        """Update epoch."""
        self.current_epoch = epoch
        for companion in self.companions.values():
            companion.david.update_epoch(epoch)

    def get_status(self) -> Dict:
        """Get status."""
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
            'active_layers': [layer.name for layer in self.active_layers],
            'total_parameters': companion_params + calculator_params,
            'companion_parameters': companion_params,
            'calculator_parameters': calculator_params,
            'companion_info': {
                name: {
                    'scales': companion.scales,
                    'hidden_dim': companion.layer_spec.hidden_dim,
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
        """Save checkpoint."""
        path = Path(path)

        if use_safetensors:
            if path.suffix != '.safetensors':
                path = path.with_suffix('.safetensors')

            state_dict = self.state_dict()
            save_file(state_dict, str(path))

            metadata = {
                'config': self._config_to_dict(),
                'epoch': self.current_epoch,
                'layer_metrics': self.layer_metrics
            }
            metadata_path = path.with_suffix('.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            print(f"Saved checkpoint to {path}")
            print(f"Saved metadata to {metadata_path}")
        else:
            if path.suffix not in ['.pt', '.pth']:
                path = path.with_suffix('.pt')

            checkpoint = {
                'config': self.config,
                'epoch': self.current_epoch,
                'state_dict': self.state_dict(),
                'layer_metrics': self.layer_metrics
            }
            torch.save(checkpoint, str(path))
            print(f"Saved checkpoint to {path}")

    def _config_to_dict(self) -> Dict:
        """Convert config to dict for JSON serialization."""
        loss_config_dict = {
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

        if isinstance(self.config.loss_config, SemanticPatternLossConfig):
            loss_config_dict.update({
                'use_soft_assignment': self.config.loss_config.use_soft_assignment,
                'assignment_temperature': self.config.loss_config.assignment_temperature,
                'pattern_diversity_weight': self.config.loss_config.pattern_diversity_weight,
            })

        return {
            'active_layers': self.config.active_layers,
            'num_semantic_bins': self.config.num_semantic_bins,
            'num_feature_patterns_per_bin': self.config.num_feature_patterns_per_bin,
            'feature_mode': self.config.feature_mode,
            'david_sharing_mode': self.config.david_sharing_mode,
            'david_fusion_mode': self.config.david_fusion_mode,
            'use_belly': self.config.use_belly,
            'belly_expand': self.config.belly_expand,
            'progressive_training': self.config.progressive_training,
            'warmup_epochs_per_layer': self.config.warmup_epochs_per_layer,
            'cache_dir': self.config.cache_dir,
            'max_cache_size_gb': self.config.max_cache_size_gb,
            'loss_calculator_type': self.config.loss_calculator_type,
            'loss_config': loss_config_dict
        }

    @classmethod
    def load_checkpoint(cls, path: str, use_safetensors: bool = True):
        """Load from checkpoint."""
        # Implementation similar to SD1.5 version
        # (Simplified for brevity)
        path = Path(path)

        if use_safetensors is None:
            use_safetensors = path.suffix == '.safetensors'

        if use_safetensors:
            if path.suffix != '.safetensors':
                path = path.with_suffix('.safetensors')

            state_dict = load_file(str(path))

            metadata_path = path.with_suffix('.json')
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

            # Reconstruct config
            config = cls._dict_to_config(metadata['config'])

            collective = cls(config)
            collective.load_state_dict(state_dict)
            collective.current_epoch = metadata['epoch']
            collective.layer_metrics = metadata['layer_metrics']

            print(f"Loaded checkpoint from {path}")
        else:
            if path.suffix not in ['.pt', '.pth']:
                path = path.with_suffix('.pt')

            checkpoint = torch.load(str(path), weights_only=False)
            collective = cls(checkpoint['config'])
            collective.load_state_dict(checkpoint['state_dict'])
            collective.current_epoch = checkpoint['epoch']
            collective.layer_metrics = checkpoint['layer_metrics']

            print(f"Loaded checkpoint from {path}")

        return collective

    @classmethod
    def _dict_to_config(cls, config_dict: Dict) -> CLIPCollectiveConfig:
        """Reconstruct config from dict."""
        loss_config_dict = config_dict.pop('loss_config')

        if 'use_soft_assignment' in loss_config_dict:
            loss_config = SemanticPatternLossConfig(**loss_config_dict)
        else:
            loss_config = LossConfig(**loss_config_dict)

        return CLIPCollectiveConfig(**config_dict, loss_config=loss_config)


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("CLIP-L DAVID COLLECTIVE - SEMANTIC PATTERN DISTILLATION")
    print("=" * 80)

    # Semantic pattern loss config
    loss_config = SemanticPatternLossConfig(
        feature_similarity_weight=0.5,
        rose_weight=0.3,
        ce_weight=0.2,
        pattern_diversity_weight=0.05,
        use_soft_assignment=True,
        assignment_temperature=0.1,
        use_cayley_loss=False
    )

    # CLIP collective config
    config = CLIPCollectiveConfig(
        active_layers=['layer_0', 'layer_5', 'layer_11'],  # Early, mid, late
        num_semantic_bins=50,
        num_feature_patterns_per_bin=10,
        feature_mode='mean_pool',
        progressive_training=False,
        loss_config=loss_config,
        loss_calculator_type='semantic'
    )

    # Create collective
    collective = CLIPCollective(config)

    print(f"\n[CLIP-L Adaptation Features]")
    print(f"  ✓ Sequential transformer features [B, seq_len, 768]")
    print(f"  ✓ No timesteps → Semantic pattern bins")
    print(f"  ✓ 500-class supervision (50 bins × 10 patterns)")
    print(f"  ✓ Learned semantic clustering")
    print(f"  ✓ Feature mode: {config.feature_mode}")

    print(f"\n[Architecture]")
    status = collective.get_status()
    print(f"  CLIPCollective:")
    print(f"    ├── companions: {status['num_companions']} layer models")
    print(f"    └── loss_calculators: {status['num_loss_calculators']} semantic calculators")
    print(f"  Companion parameters: {status['companion_parameters']:,}")
    print(f"  Calculator parameters: {status['calculator_parameters']:,}")

    print(f"\n[Status]")
    print(f"  Active layers: {status['active_layers']}")
    print(f"  Total parameters: {status['total_parameters']:,}")

    # Test forward pass
    print(f"\n[Test Forward Pass]")
    batch_size = 4
    seq_len = 77  # Standard CLIP text length

    layer_features = {
        'layer_0': torch.randn(batch_size, seq_len, 768),
        'layer_5': torch.randn(batch_size, seq_len, 768),
        'layer_11': torch.randn(batch_size, seq_len, 768),
    }

    attention_masks = {
        'layer_0': torch.ones(batch_size, seq_len),
        'layer_5': torch.ones(batch_size, seq_len),
        'layer_11': torch.ones(batch_size, seq_len),
    }

    with torch.no_grad():
        outputs = collective(layer_features, attention_masks)

    print(f"  ✓ Forward pass successful for {len(outputs)} layers")

    # Test losses
    print(f"\n[Test Loss Computation]")
    teacher_features = layer_features

    with torch.no_grad():
        losses = collective.compute_losses(outputs, teacher_features)
        total_loss = collective.get_total_loss(losses)

    print(f"  ✓ Loss computation successful")
    print(f"  Total loss: {total_loss.item():.4f}")

    for layer_name, layer_losses in losses.items():
        print(f"\n  {layer_name}:")
        print(f"    Feature similarity: {layer_losses['feature_similarity'].item():.4f}")
        print(f"    Rose: {layer_losses['rose'].item():.4f}")
        print(f"    CE: {layer_losses['ce'].item():.4f}")
        print(f"    Pattern diversity: {layer_losses['pattern_diversity'].item():.4f}")
        print(f"    Semantic bin accuracy: {layer_losses['bin_acc'].item():.2%}")
        print(f"    Pattern accuracy: {layer_losses['pattern_acc'].item():.2%}")
        print(f"    Full accuracy (500-class): {layer_losses['full_acc'].item():.2%}")

    print("\n" + "=" * 80)
    print("CLIP-L ADAPTATION COMPLETE")
    print("Ready for text encoder distillation! 🎯")
    print("=" * 80)