"""
David - The Multi-Scale Crystal Classifier - Core AI Modules
=================================================
Modular components for multi-scale deep learning with crystal-based representations.

Author: AbstractPhil
- Assistant: GPT-4o Mirel
- Refactor: Claude Sonnet 4.5


MIT Licensed
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union
from enum import Enum
import math


# ============================================================================
# CONFIGURATION ENUMS
# ============================================================================

class SharingMode(Enum):
    """Parameter sharing strategies across scales."""
    FULLY_SHARED = "fully_shared"      # All scales share feature extractor
    PARTIAL_SHARED = "partial_shared"  # Shared base + scale-specific heads
    DECOUPLED = "decoupled"            # Independent parameters per scale
    HIERARCHICAL = "hierarchical"      # Progressive refinement across scales


class FusionMode(Enum):
    """Multi-scale prediction fusion strategies."""
    WEIGHTED_SUM = "weighted_sum"          # Learnable weighted combination
    ATTENTION = "attention"                # Attention-based fusion
    GATED = "gated"                        # Gated mixture of experts
    HIERARCHICAL_TREE = "hierarchical_tree"  # Tree-structured gating
    DEEP_EFFICIENCY = "deep_efficiency"    # Efficient multi-expert gating
    MAX_CONFIDENCE = "max_confidence"      # Select highest confidence scale
    PROGRESSIVE = "progressive"            # Fixed progressive weighting


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

class RoseLoss(nn.Module):
    """
    Rose Loss with pentachora role weighting.

    Implements a margin-based loss using crystal vertex roles (anchor, need,
    relation, purpose, observer) with temperature scaling.
    """

    def __init__(self, margin: float = 3.0, temperature: float = 0.71,
                 role_weights: Optional[Dict[str, float]] = None):
        super().__init__()
        self.margin = margin
        self.temperature = temperature

        default_weights = {
            "anchor": 1.0,
            "need": -0.75,
            "relation": 0.75,
            "purpose": 0.75,
            "observer": -0.75,
        }
        weights = role_weights or default_weights
        role_vec = torch.tensor([
            weights["anchor"],
            weights["need"],
            weights["relation"],
            weights["purpose"],
            weights["observer"],
        ], dtype=torch.float32)

        self.register_buffer("role_weights", role_vec)

    def forward(self, z: torch.Tensor, crystals: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: Normalized embeddings [B, D]
            crystals: Crystal vertices [C, 5, D]
            targets: Target class indices [B]

        Returns:
            Scalar loss value
        """
        crystals = crystals.to(z.device)
        role_weights = self.role_weights.to(z.device)

        # Normalize and compute cosine similarities
        crystals_norm = F.normalize(crystals, dim=-1)
        cos_sim = torch.einsum("bd,cvd->bcv", z, crystals_norm)

        # Apply role weighting and temperature
        rose_scores = (cos_sim * role_weights.view(1, 1, 5)).sum(dim=-1)
        rose_scores = rose_scores / self.temperature

        # Margin-based loss
        true_scores = rose_scores.gather(1, targets.view(-1, 1)).squeeze(1)
        mask = torch.zeros_like(rose_scores, dtype=torch.bool)
        mask.scatter_(1, targets.view(-1, 1), True)
        hard_neg = rose_scores.masked_fill(mask, float("-inf")).max(dim=1).values

        loss = F.relu(self.margin - (true_scores - hard_neg))
        return loss.mean()


class CayleyChaosLoss(nn.Module):
    """
    Cayley-Menger chaos loss for geometric regularization.

    Penalizes degenerate or chaotic pentachora by measuring volume,
    edge uniformity, and Gram matrix properties.
    """

    def __init__(self, volume_floor: float = 1e-4, chaos_scale: float = 1.0,
                 edge_dev_weight: float = 0.5, gram_weight: float = 0.1):
        super().__init__()
        self.volume_floor = volume_floor
        self.chaos_scale = chaos_scale
        self.edge_dev_weight = edge_dev_weight
        self.gram_weight = gram_weight

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Args:
            X: Pentachora vertices [B, 5, D]

        Returns:
            Scalar chaos penalty
        """
        B, N, D = X.shape
        assert N == 5, "Input must be [B, 5, D] for pentachora"

        # Compute pairwise squared distances
        diff = X.unsqueeze(2) - X.unsqueeze(1)
        distsq = (diff ** 2).sum(dim=-1)

        # Build Cayley-Menger determinant matrix
        ones_row = torch.ones((B, 1, 5), device=X.device)
        ones_col = torch.ones((B, 5, 1), device=X.device)
        zeros = torch.zeros((B, 1, 1), device=X.device)

        top_row = torch.cat([zeros, ones_row], dim=2)
        bottom_rows = torch.cat([ones_col, distsq], dim=2)
        M = torch.cat([top_row, bottom_rows], dim=1)

        # Volume from determinant
        det = torch.linalg.det(M)
        volume_sq = (-det / 9216.0).clamp(min=0.0)
        volume = volume_sq.sqrt()

        # Penalize small volumes (degeneracy)
        chaos_penalty = F.relu(self.volume_floor - volume)
        chaos_loss = chaos_penalty.mean()

        # Edge length uniformity
        triu_indices = torch.triu_indices(5, 5, offset=1)
        edge_lengths = distsq[:, triu_indices[0], triu_indices[1]]
        edge_mean = edge_lengths.mean(dim=1)
        edge_std = edge_lengths.std(dim=1)
        edge_dev = (edge_std / edge_mean.clamp(min=1e-6)).mean()

        # Gram matrix condition (optional)
        gram_penalty = 0.0
        if self.gram_weight > 0:
            centered = X - X.mean(dim=1, keepdim=True)
            gram = torch.bmm(centered, centered.transpose(1, 2))
            gram_trace = torch.diagonal(gram, dim1=1, dim2=2).sum(dim=1)
            gram_det = torch.linalg.det(gram)
            gram_penalty = F.relu(1.0 - gram_det / gram_trace.clamp(min=1e-6)).mean()

        total_loss = (
            self.chaos_scale * chaos_loss +
            self.edge_dev_weight * edge_dev +
            self.gram_weight * gram_penalty
        )
        return total_loss


# ============================================================================
# MODEL BUILDING BLOCKS
# ============================================================================

class SharedFeatureExtractor(nn.Module):
    """Shared feature extraction layers for multi-scale processing."""

    def __init__(self, input_dim: int, output_dim: int, num_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()

        layers = []
        hidden_dim = (input_dim + output_dim) // 2

        # Input layer
        layers.extend([
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        ])

        # Hidden layers
        for _ in range(num_layers - 2):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])

        # Output layer
        if num_layers > 1:
            layers.extend([
                nn.Linear(hidden_dim, output_dim),
                nn.LayerNorm(output_dim)
            ])

        self.extractor = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.extractor(x)


class ScaleSpecificHead(nn.Module):
    """
    Scale-specific projection head with optional bottleneck.

    Projects features to crystal embedding space at a specific scale.
    """

    def __init__(self, input_dim: int, crystal_dim: int, use_belly: bool = True):
        super().__init__()
        self.crystal_dim = crystal_dim

        if use_belly:
            # Bottleneck projection
            belly_dim = int(crystal_dim * 2.0)
            self.projection = nn.Sequential(
                nn.Linear(input_dim, belly_dim),
                nn.ReLU(),
                nn.Dropout(1.0 / math.sqrt(crystal_dim)),
                nn.Linear(belly_dim, crystal_dim, bias=False)
            )
        else:
            # Direct projection
            self.projection = nn.Linear(input_dim, crystal_dim, bias=False)

        self._init_weights()

    def _init_weights(self):
        for layer in self.projection.modules():
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, std=0.02)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, features: torch.Tensor,
                anchors: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            features: Input features [B, D_in]
            anchors: Crystal anchors [C, D_crystal]

        Returns:
            logits: Classification logits [B, C]
            z: Normalized embeddings [B, D_crystal]
        """
        z = self.projection(features)
        z = F.normalize(z, dim=-1)
        logits = (z @ anchors.T) / 0.03  # Temperature scaling
        return logits, z


# ============================================================================
# FUSION MECHANISMS
# ============================================================================

class AttentionFusion(nn.Module):
    """Attention-based fusion of multi-scale predictions."""

    def __init__(self, feature_dim: int, num_scales: int,
                 temperature: float = 1.0, dropout: float = 0.1):
        super().__init__()
        self.query = nn.Linear(feature_dim, num_scales)
        self.temperature = temperature
        self.dropout = nn.Dropout(dropout)

    def forward(self, features: torch.Tensor,
                logits_list: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            features: Shared features [B, D]
            logits_list: List of scale logits, each [B, C]

        Returns:
            combined: Fused logits [B, C]
            weights: Attention weights [B, num_scales]
        """
        attn_logits = self.query(features)
        attn_weights = F.softmax(attn_logits / self.temperature, dim=-1)
        attn_weights = self.dropout(attn_weights)

        combined = torch.zeros_like(logits_list[0])
        for i, logits in enumerate(logits_list):
            combined += attn_weights[:, i:i+1] * logits

        return combined, attn_weights


class GatedFusion(nn.Module):
    """Gated mixture of experts fusion."""

    def __init__(self, feature_dim: int, num_scales: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, num_scales),
            nn.Softmax(dim=-1)
        )

    def forward(self, features: torch.Tensor,
                logits_list: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            features: Shared features [B, D]
            logits_list: List of scale logits, each [B, C]

        Returns:
            combined: Fused logits [B, C]
            gates: Gate weights [B, num_scales]
        """
        gates = self.gate(features)
        combined = torch.zeros_like(logits_list[0])
        for i, logits in enumerate(logits_list):
            combined += gates[:, i:i+1] * logits
        return combined, gates


class HierarchicalTreeGating(nn.Module):
    """
    Tree-based hierarchical gating for multi-scale fusion.

    Uses a binary tree structure to learn hierarchical scale selection
    with a depth-controlled architecture.
    """

    def __init__(self, feature_dim: int, num_scales: int = 5, depth: int = 3):
        super().__init__()
        self.num_scales = num_scales
        self.depth = depth

        # Build tree nodes (binary decisions at each level)
        self.tree_nodes = nn.ModuleList()
        for level in range(depth):
            num_nodes = 2 ** level
            level_nodes = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(feature_dim, 64),
                    nn.LayerNorm(64),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Linear(64, 2)
                ) for _ in range(num_nodes)
            ])
            self.tree_nodes.append(level_nodes)

        # Map leaf probabilities to scale weights
        num_leaves = 2 ** depth
        self.leaf_to_scale = nn.Sequential(
            nn.Linear(num_leaves, num_scales * 2),
            nn.GELU(),
            nn.Linear(num_scales * 2, num_scales)
        )

        # Direct gating path (shortcut)
        self.direct_gate = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 4),
            nn.GELU(),
            nn.Linear(feature_dim // 4, num_scales)
        )

        self.combine_weight = nn.Parameter(torch.tensor(0.7))

    def forward(self, features: torch.Tensor,
                logits_list: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            features: Input features [B, D]
            logits_list: List of scale logits, each [B, C]

        Returns:
            combined: Fused logits [B, C]
            gates: Final gate weights [B, num_scales]
        """
        batch_size = features.shape[0]
        device = features.device

        # Traverse tree to compute path probabilities
        path_probs = [torch.ones(batch_size, 1, device=device)]

        for level in range(self.depth):
            next_probs = []
            num_nodes = 2 ** level

            for node_idx in range(num_nodes):
                parent_prob = path_probs[0] if level == 0 else path_probs[node_idx]

                # Binary decision at this node
                node_logits = self.tree_nodes[level][node_idx](features)
                node_probs = F.softmax(node_logits / 0.5, dim=-1)

                left_prob = parent_prob * node_probs[:, 0:1]
                right_prob = parent_prob * node_probs[:, 1:2]
                next_probs.extend([left_prob, right_prob])

            path_probs = next_probs

        # Convert leaf probabilities to scale gates
        leaf_probs = torch.cat(path_probs, dim=1)
        tree_gates = F.softmax(self.leaf_to_scale(leaf_probs), dim=-1)

        # Combine with direct gating
        direct_gates = F.softmax(self.direct_gate(features), dim=-1)
        gates = torch.sigmoid(self.combine_weight) * tree_gates + \
                (1 - torch.sigmoid(self.combine_weight)) * direct_gates
        gates = gates / gates.sum(dim=-1, keepdim=True)

        # Fuse predictions
        combined = torch.zeros_like(logits_list[0])
        for i, logits in enumerate(logits_list):
            if i < gates.shape[1]:
                combined += gates[:, i:i+1] * logits

        return combined, gates


class DeepEfficiencyGating(nn.Module):
    """
    Ultra-efficient gating with cross-attention and sparsity.

    Uses multiple expert pathways with cross-attention aggregation
    and optional sparse gating for inference efficiency.
    """

    def __init__(self, feature_dim: int, num_scales: int = 5,
                 compression_ratio: int = 4, num_experts: int = 3):
        super().__init__()
        self.num_scales = num_scales
        self.num_experts = num_experts

        self.temperature = nn.Parameter(torch.ones(1) * 0.5)

        bottleneck_dim = max(num_scales * 8, feature_dim // compression_ratio)

        # Expert pathways
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim, bottleneck_dim),
                nn.LayerNorm(bottleneck_dim),
                nn.GELU(),
                nn.Dropout(0.1)
            ) for _ in range(num_experts)
        ])

        # Cross-attention aggregation
        self.cross_attention = nn.MultiheadAttention(
            bottleneck_dim,
            num_heads=min(8, bottleneck_dim // 8),
            batch_first=True,
            dropout=0.1
        )

        # Multiple gate heads for ensemble
        self.gate_heads = nn.ModuleList([
            nn.Linear(bottleneck_dim * num_experts, num_scales)
            for _ in range(3)
        ])

        self.head_weights = nn.Parameter(torch.ones(3) / 3)
        self.scale_bias = nn.Parameter(torch.zeros(num_scales))
        self.sparsity_threshold = nn.Parameter(torch.tensor(0.1))

    def forward(self, features: torch.Tensor,
                logits_list: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            features: Input features [B, D]
            logits_list: List of scale logits, each [B, C]

        Returns:
            combined: Fused logits [B, C]
            gates: Gate weights [B, num_scales] (sparse in eval mode)
        """
        # Process through expert pathways
        expert_outputs = [expert(features) for expert in self.experts]
        stacked = torch.stack(expert_outputs, dim=1)

        # Cross-attention aggregation
        attended, _ = self.cross_attention(stacked, stacked, stacked)
        flattened = attended.reshape(features.shape[0], -1)

        # Ensemble of gate heads
        gate_outputs = [head(flattened) for head in self.gate_heads]
        head_weights = F.softmax(self.head_weights, dim=0)
        combined_gates = sum(w * g for w, g in zip(head_weights, gate_outputs))

        # Temperature-scaled softmax
        gate_logits = (combined_gates + self.scale_bias) / self.temperature.abs()
        gates = F.softmax(gate_logits, dim=-1)

        # Sparse gating during inference
        if not self.training:
            mask = gates > self.sparsity_threshold
            sparse_gates = gates * mask
            sparse_gates = sparse_gates / (sparse_gates.sum(dim=-1, keepdim=True) + 1e-8)
        else:
            sparse_gates = gates

        # Fuse predictions
        combined = torch.zeros_like(logits_list[0])
        for i, logits in enumerate(logits_list):
            if i < sparse_gates.shape[1]:
                combined += sparse_gates[:, i:i+1] * logits

        return combined, sparse_gates


# ============================================================================
# MULTI-SCALE LOSS AGGREGATION
# ============================================================================

class MultiScaleCrystalLoss(nn.Module):
    """
    Aggregate loss across scales with configurable weighting.

    Combines cross-entropy, Rose loss, and Cayley chaos penalties
    with per-scale balancing and adaptive scheduling.
    """

    def __init__(self, scales: List[int], num_classes: int = 1000,
                 use_rose_loss: bool = True, use_cayley_loss: bool = True,
                 rose_initial_weight: float = 0.1, rose_max_weight: float = 0.5,
                 cayley_weight: float = 0.05, scale_loss_balance: Optional[Dict[int, float]] = None):
        super().__init__()

        self.scales = scales
        self.num_classes = num_classes
        self.use_rose_loss = use_rose_loss
        self.use_cayley_loss = use_cayley_loss
        self.rose_weight = rose_initial_weight
        self.rose_max_weight = rose_max_weight
        self.cayley_weight = cayley_weight

        self.scale_balance = scale_loss_balance or {s: 1.0 for s in scales}

        self.ce_loss = nn.CrossEntropyLoss()

        if use_rose_loss:
            self.rose_losses = nn.ModuleDict({
                str(scale): RoseLoss(margin=3, temperature=0.71)
                for scale in scales
            })

        if use_cayley_loss:
            self.cayley_loss = CayleyChaosLoss()

    def forward(self, combined_logits: torch.Tensor,
                scale_logits: List[torch.Tensor],
                scale_features: List[torch.Tensor],
                targets: torch.Tensor,
                crystals_dict: Dict[int, torch.Tensor],
                epoch: int = 0) -> Dict[str, torch.Tensor]:
        """
        Args:
            combined_logits: Fused predictions [B, C]
            scale_logits: Per-scale predictions, list of [B, C]
            scale_features: Per-scale embeddings, list of [B, D_scale]
            targets: Ground truth labels [B]
            crystals_dict: Crystal vertices per scale {scale: [C, 5, D_scale]}
            epoch: Current training epoch

        Returns:
            Dictionary of loss components
        """
        losses = {}

        # Main cross-entropy
        ce_main = self.ce_loss(combined_logits, targets)
        losses['ce_main'] = ce_main

        # Per-scale losses
        total_scale_loss = 0
        for i, (scale, logits, features) in enumerate(zip(self.scales, scale_logits, scale_features)):
            scale_weight = self.scale_balance.get(scale, 1.0)

            # Scale CE loss
            ce_scale = self.ce_loss(logits, targets)
            losses[f'ce_{scale}'] = ce_scale
            total_scale_loss += scale_weight * ce_scale

            # Rose loss
            if self.use_rose_loss and scale in crystals_dict:
                rose_loss = self.rose_losses[str(scale)](
                    features, crystals_dict[scale], targets
                )
                losses[f'rose_{scale}'] = rose_loss

                # Adaptive weighting
                progress = epoch / 100  # Assumes max 100 epochs
                current_weight = self.rose_weight + \
                                (self.rose_max_weight - self.rose_weight) * progress
                total_scale_loss += current_weight * rose_loss

            # Cayley chaos loss
            if self.use_cayley_loss:
                batch_crystals = crystals_dict[scale][targets]
                cayley_loss = self.cayley_loss(batch_crystals)
                losses[f'cayley_{scale}'] = cayley_loss
                total_scale_loss += self.cayley_weight * cayley_loss

        # Total loss
        total_loss = ce_main + total_scale_loss / len(self.scales)
        losses['total'] = total_loss

        return losses


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def resize_crystal_anchors(anchors: torch.Tensor, target_dim: int,
                          method: str = "linear") -> torch.Tensor:
    """
    Resize crystal anchors to target dimension via interpolation.

    Args:
        anchors: Input anchors [C, D_in]
        target_dim: Target dimension D_out
        method: Interpolation method ('linear')

    Returns:
        Resized anchors [C, D_out]
    """
    C, vocab_dim = anchors.shape
    device = anchors.device

    if vocab_dim == target_dim:
        return anchors

    if method == "linear":
        if target_dim < vocab_dim:
            # Subsample
            indices = torch.linspace(0, vocab_dim - 1, target_dim, dtype=torch.long)
            resized = anchors[:, indices]
        else:
            # Interpolate
            resized = F.interpolate(
                anchors.unsqueeze(1),
                size=target_dim,
                mode='linear',
                align_corners=True
            ).squeeze(1)

    # Re-normalize
    resized = F.normalize(resized, dim=-1)
    return resized


# ============================================================================
# DAVID - THE MULTI-SCALE ORCHESTRATOR
# ============================================================================

class David(nn.Module):
    """
    David: Multi-Scale Crystal Classifier
    ======================================

    The behavioral curator for multi-scale deep learning. David orchestrates
    parallel processing across multiple embedding scales, dynamically fusing
    their insights through learned attention or gating mechanisms.

    Personality traits:
    - Progressive: Scales activate gradually during training
    - Adaptive: Can freeze high-performing scales
    - Collaborative: Shares knowledge across scales (configurable)
    - Strategic: Chooses fusion strategy based on task complexity
    """

    def __init__(
        self,
        feature_dim: int = 512,
        num_classes: int = 1000,
        scales: List[int] = None,
        sharing_mode: SharingMode = SharingMode.PARTIAL_SHARED,
        fusion_mode: FusionMode = FusionMode.GATED,
        shared_feature_dim: int = 768,
        shared_layers: int = 2,
        shared_dropout: float = 0.1,
        fusion_temperature: float = 1.0,
        fusion_dropout: float = 0.1,
        tree_depth: int = 3,
        num_experts: int = 3,
        compression_ratio: int = 4,
        progressive_training: bool = True,
        scale_warmup_epochs: Optional[Dict[int, int]] = None
    ):
        """
        Initialize David with multi-scale architecture.

        Args:
            feature_dim: Input feature dimension
            num_classes: Number of output classes
            scales: List of crystal embedding dimensions (e.g., [256, 512, 768, 1024])
            sharing_mode: How to share parameters across scales
            fusion_mode: How to combine multi-scale predictions
            shared_feature_dim: Dimension of shared feature space
            shared_layers: Number of shared extraction layers
            shared_dropout: Dropout in shared layers
            fusion_temperature: Temperature for fusion softmax
            fusion_dropout: Dropout in fusion module
            tree_depth: Depth for hierarchical tree gating
            num_experts: Number of experts for deep efficiency gating
            compression_ratio: Compression for deep efficiency gating
            progressive_training: Whether to progressively activate scales
            scale_warmup_epochs: When each scale becomes active {scale: epoch}
        """
        super().__init__()

        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.scales = scales or [256, 512, 768, 1024]
        self.sharing_mode = sharing_mode
        self.fusion_mode = fusion_mode
        self.progressive_training = progressive_training
        self.scale_warmup_epochs = scale_warmup_epochs or {s: 0 for s in self.scales}

        # David's memory
        self.current_epoch = 0
        self.scale_accuracies = {s: [] for s in self.scales}

        # Build David's neural architecture
        self._build_architecture(
            shared_feature_dim=shared_feature_dim,
            shared_layers=shared_layers,
            shared_dropout=shared_dropout
        )

        # Build David's fusion strategy
        self._build_fusion(
            shared_feature_dim=shared_feature_dim,
            temperature=fusion_temperature,
            dropout=fusion_dropout,
            tree_depth=tree_depth,
            num_experts=num_experts,
            compression_ratio=compression_ratio
        )

        # David's confidence in each scale
        self.register_buffer(
            "scale_weights",
            torch.tensor([1.0 for _ in self.scales])
        )

    def _build_architecture(self, shared_feature_dim: int,
                           shared_layers: int, shared_dropout: float):
        """Build David's processing architecture based on sharing mode."""

        if self.sharing_mode == SharingMode.FULLY_SHARED:
            # David shares everything - maximum parameter efficiency
            self.shared_extractor = SharedFeatureExtractor(
                self.feature_dim,
                shared_feature_dim,
                shared_layers,
                shared_dropout
            )
            self.heads = nn.ModuleDict({
                str(scale): ScaleSpecificHead(
                    shared_feature_dim,
                    scale,
                    use_belly=False
                )
                for scale in self.scales
            })

        elif self.sharing_mode == SharingMode.PARTIAL_SHARED:
            # David shares a base, then specializes - balanced approach
            self.shared_base = nn.Linear(self.feature_dim, shared_feature_dim)
            self.heads = nn.ModuleDict({
                str(scale): ScaleSpecificHead(
                    shared_feature_dim,
                    scale,
                    use_belly=True
                )
                for scale in self.scales
            })

        elif self.sharing_mode == SharingMode.DECOUPLED:
            # David keeps scales independent - maximum flexibility
            self.heads = nn.ModuleDict({
                str(scale): ScaleSpecificHead(
                    self.feature_dim,
                    scale,
                    use_belly=True
                )
                for scale in self.scales
            })

        elif self.sharing_mode == SharingMode.HIERARCHICAL:
            # David refines progressively - coarse to fine
            for i, scale in enumerate(self.scales):
                if i == 0:
                    # First scale processes directly
                    setattr(self, f'head_{scale}',
                           ScaleSpecificHead(self.feature_dim, scale))
                else:
                    # Later scales refine previous outputs
                    prev_scale = self.scales[i-1]
                    setattr(self, f'refine_{scale}', nn.Sequential(
                        nn.Linear(prev_scale + self.feature_dim, scale),
                        nn.ReLU()
                    ))
                    setattr(self, f'head_{scale}',
                           ScaleSpecificHead(scale, scale))

    def _build_fusion(self, shared_feature_dim: int, temperature: float,
                     dropout: float, tree_depth: int, num_experts: int,
                     compression_ratio: int):
        """Build David's fusion strategy."""

        fusion_input_dim = (
            shared_feature_dim
            if self.sharing_mode in [SharingMode.FULLY_SHARED, SharingMode.PARTIAL_SHARED]
            else self.feature_dim
        )

        if self.fusion_mode == FusionMode.ATTENTION:
            self.fusion = AttentionFusion(
                fusion_input_dim, len(self.scales), temperature, dropout
            )

        elif self.fusion_mode == FusionMode.GATED:
            self.fusion = GatedFusion(fusion_input_dim, len(self.scales))

        elif self.fusion_mode == FusionMode.HIERARCHICAL_TREE:
            self.fusion = HierarchicalTreeGating(
                fusion_input_dim, len(self.scales), depth=tree_depth
            )

        elif self.fusion_mode == FusionMode.DEEP_EFFICIENCY:
            self.fusion = DeepEfficiencyGating(
                fusion_input_dim, len(self.scales),
                compression_ratio=compression_ratio,
                num_experts=num_experts
            )

        elif self.fusion_mode == FusionMode.WEIGHTED_SUM:
            self.fusion_weights = nn.Parameter(
                torch.ones(len(self.scales)) / len(self.scales)
            )

        elif self.fusion_mode == FusionMode.PROGRESSIVE:
            # Fixed progressive weighting: earlier scales get less weight
            weights = torch.tensor([0.1, 0.15, 0.25, 0.25, 0.25])
            self.register_buffer("progressive_weights", weights[:len(self.scales)])

    def _should_use_scale(self, scale: int) -> bool:
        """David decides if this scale should be active yet."""
        if not self.progressive_training:
            return True
        warmup_epoch = self.scale_warmup_epochs.get(scale, 0)
        return self.current_epoch >= warmup_epoch

    def _extract_base_features(self, x: torch.Tensor) -> torch.Tensor:
        """David extracts base features according to sharing mode."""
        if self.sharing_mode == SharingMode.FULLY_SHARED:
            return self.shared_extractor(x)
        elif self.sharing_mode == SharingMode.PARTIAL_SHARED:
            return F.relu(self.shared_base(x))
        else:
            return x

    def _hierarchical_forward(
        self,
        x: torch.Tensor,
        anchors_dict: Dict[int, torch.Tensor]
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """David's hierarchical refinement process."""
        logits_list = []
        features_list = []
        prev_features = None

        for i, scale in enumerate(self.scales):
            if not self._should_use_scale(scale):
                continue

            if i == 0:
                # First scale processes directly
                head = getattr(self, f'head_{scale}')
                logits, features = head(x, anchors_dict[scale])
                prev_features = features
            else:
                # Refine using previous scale's output
                refine = getattr(self, f'refine_{scale}')
                head = getattr(self, f'head_{scale}')
                refined = refine(torch.cat([prev_features, x], dim=-1))
                logits, features = head(refined, anchors_dict[scale])
                prev_features = features

            logits_list.append(logits)
            features_list.append(features)

        return logits_list, features_list

    def _parallel_forward(
        self,
        base_features: torch.Tensor,
        anchors_dict: Dict[int, torch.Tensor]
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """David's parallel multi-scale processing."""
        logits_list = []
        features_list = []

        for scale in self.scales:
            if self._should_use_scale(scale):
                logits, features = self.heads[str(scale)](
                    base_features,
                    anchors_dict[scale]
                )
                logits_list.append(logits)
                features_list.append(features)

        return logits_list, features_list

    def _fuse_predictions(
        self,
        features: torch.Tensor,
        logits_list: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """David combines multi-scale predictions."""

        if len(logits_list) == 1:
            # Single scale active - no fusion needed
            return logits_list[0], torch.ones(1, device=features.device)

        # Use configured fusion strategy
        if self.fusion_mode in [FusionMode.ATTENTION, FusionMode.GATED,
                                FusionMode.HIERARCHICAL_TREE, FusionMode.DEEP_EFFICIENCY]:
            return self.fusion(features, logits_list)

        elif self.fusion_mode == FusionMode.WEIGHTED_SUM:
            weights = F.softmax(self.fusion_weights, dim=0)
            combined = sum(w * logits for w, logits in zip(weights, logits_list))
            return combined, weights

        elif self.fusion_mode == FusionMode.MAX_CONFIDENCE:
            # Choose the most confident scale
            confidences = [
                F.softmax(logits, dim=-1).max(dim=-1)[0].mean()
                for logits in logits_list
            ]
            best_idx = torch.stack(confidences).argmax()
            weights = F.one_hot(best_idx, len(logits_list)).float()
            return logits_list[best_idx], weights

        elif self.fusion_mode == FusionMode.PROGRESSIVE:
            weights = self.progressive_weights[:len(logits_list)]
            weights = weights / weights.sum()
            combined = sum(w * logits for w, logits in zip(weights, logits_list))
            return combined, weights

    def forward(
        self,
        x: torch.Tensor,
        anchors_dict: Dict[int, torch.Tensor],
        return_all_scales: bool = False
    ) -> Union[Tuple[torch.Tensor, torch.Tensor],
               Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor], torch.Tensor]]:
        """
        David's forward pass - the behavioral curation of multi-scale processing.

        Args:
            x: Input features [B, D]
            anchors_dict: Crystal anchors for each scale {scale: [C, D_scale]}
            return_all_scales: Whether to return all intermediate outputs

        Returns:
            If return_all_scales=False:
                combined_logits: Fused predictions [B, C]
                features: Primary scale features [B, D_scale]

            If return_all_scales=True:
                combined_logits: Fused predictions [B, C]
                logits_list: Per-scale predictions, list of [B, C]
                features_list: Per-scale embeddings, list of [B, D_scale]
                fusion_weights: Scale contribution weights [B, num_scales]
        """

        if self.sharing_mode == SharingMode.HIERARCHICAL:
            # David refines progressively
            logits_list, features_list = self._hierarchical_forward(x, anchors_dict)
            base_features = x  # Use original features for fusion
        else:
            # David processes in parallel
            base_features = self._extract_base_features(x)
            logits_list, features_list = self._parallel_forward(base_features, anchors_dict)

        # David combines the multi-scale insights
        combined, fusion_weights = self._fuse_predictions(base_features, logits_list)

        if return_all_scales:
            return combined, logits_list, features_list, fusion_weights
        else:
            return combined, features_list[0] if features_list else base_features

    # ========================================================================
    # DAVID'S ADAPTIVE BEHAVIORS
    # ========================================================================

    def update_epoch(self, epoch: int):
        """Update David's internal clock for progressive training."""
        self.current_epoch = epoch

    def get_active_scales(self) -> List[int]:
        """Get list of scales David is currently using."""
        return [s for s in self.scales if self._should_use_scale(s)]

    def get_scale_parameters(self, scale: int) -> List[nn.Parameter]:
        """Get parameters for a specific scale."""
        if self.sharing_mode == SharingMode.HIERARCHICAL:
            if scale not in self.scales:
                return []

            scale_idx = self.scales.index(scale)
            params = []

            head = getattr(self, f'head_{scale}', None)
            if head:
                params.extend(head.parameters())

            if scale_idx > 0:
                refine = getattr(self, f'refine_{scale}', None)
                if refine:
                    params.extend(refine.parameters())

            return params
        else:
            return list(self.heads[str(scale)].parameters())

    def freeze_scale(self, scale: int):
        """Freeze a scale - David stops learning for this scale."""
        for param in self.get_scale_parameters(scale):
            param.requires_grad = False
        print(f"[David] ❄️  Frozen scale {scale}")

    def unfreeze_scale(self, scale: int):
        """Unfreeze a scale - David resumes learning for this scale."""
        for param in self.get_scale_parameters(scale):
            param.requires_grad = True
        print(f"[David] 🔥 Unfrozen scale {scale}")

    def freeze_all_scales(self):
        """David freezes all scale-specific parameters."""
        for scale in self.scales:
            self.freeze_scale(scale)

    def unfreeze_all_scales(self):
        """David unfreezes all scale-specific parameters."""
        for scale in self.scales:
            self.unfreeze_scale(scale)

    def get_model_info(self) -> Dict[str, any]:
        """Get David's configuration and status."""
        return {
            "name": "David",
            "feature_dim": self.feature_dim,
            "num_classes": self.num_classes,
            "scales": self.scales,
            "sharing_mode": self.sharing_mode.value,
            "fusion_mode": self.fusion_mode.value,
            "active_scales": self.get_active_scales(),
            "current_epoch": self.current_epoch,
            "total_parameters": sum(p.numel() for p in self.parameters()),
            "trainable_parameters": sum(p.numel() for p in self.parameters() if p.requires_grad)
        }

    def __repr__(self):
        info = self.get_model_info()
        return (
            f"David(Multi-Scale Crystal Classifier)\n"
            f"  Scales: {info['scales']}\n"
            f"  Sharing: {info['sharing_mode']}\n"
            f"  Fusion: {info['fusion_mode']}\n"
            f"  Active: {info['active_scales']}\n"
            f"  Parameters: {info['total_parameters']:,}"
        )


# ============================================================================
# TEST SUITE - DAVID'S BEHAVIORAL VALIDATION
# ============================================================================

if __name__ == "__main__":
    import sys

    print("="*80)
    print("🧪 DAVID - MULTI-SCALE CRYSTAL CLASSIFIER TEST SUITE")
    print("="*80)

    # Test configuration
    batch_size = 4
    feature_dim = 512
    num_classes = 1000
    scales = [256, 512, 768, 1024]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n📍 Running on: {device}")
    print(f"   Batch size: {batch_size}")
    print(f"   Feature dim: {feature_dim}")
    print(f"   Scales: {scales}")

    # ========================================================================
    # TEST 1: Crystal Anchor Resizing
    # ========================================================================
    print("\n" + "-"*80)
    print("TEST 1: Crystal Anchor Resizing")
    print("-"*80)

    try:
        base_anchors = torch.randn(num_classes, 512).to(device)
        base_anchors = F.normalize(base_anchors, dim=-1)

        for target_scale in scales:
            resized = resize_crystal_anchors(base_anchors, target_scale)
            assert resized.shape == (num_classes, target_scale), \
                f"Expected shape ({num_classes}, {target_scale}), got {resized.shape}"

            # Check normalization
            norms = torch.norm(resized, dim=-1)
            assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), \
                "Anchors not properly normalized"

        print("✅ Crystal anchor resizing works correctly")
        print(f"   Tested scales: {scales}")
    except Exception as e:
        print(f"❌ FAILED: {e}")
        sys.exit(1)

    # ========================================================================
    # TEST 2: Rose Loss
    # ========================================================================
    print("\n" + "-"*80)
    print("TEST 2: Rose Loss Computation")
    print("-"*80)

    try:
        rose_loss = RoseLoss(margin=3, temperature=0.71).to(device)

        # Create test data
        z = F.normalize(torch.randn(batch_size, 256).to(device), dim=-1)
        crystals = F.normalize(torch.randn(num_classes, 5, 256).to(device), dim=-1)
        targets = torch.randint(0, num_classes, (batch_size,)).to(device)

        loss = rose_loss(z, crystals, targets)

        assert loss.ndim == 0, "Loss should be scalar"
        assert loss.item() >= 0, "Loss should be non-negative"
        assert not torch.isnan(loss), "Loss contains NaN"

        print("✅ Rose loss computation successful")
        print(f"   Loss value: {loss.item():.4f}")
    except Exception as e:
        print(f"❌ FAILED: {e}")
        sys.exit(1)

    # ========================================================================
    # TEST 3: Cayley Chaos Loss
    # ========================================================================
    print("\n" + "-"*80)
    print("TEST 3: Cayley Chaos Loss Computation")
    print("-"*80)

    try:
        cayley_loss = CayleyChaosLoss(volume_floor=1e-4).to(device)

        # Create test pentachora
        pentachora = torch.randn(batch_size, 5, 256).to(device)

        loss = cayley_loss(pentachora)

        assert loss.ndim == 0, "Loss should be scalar"
        assert loss.item() >= 0, "Loss should be non-negative"
        assert not torch.isnan(loss), "Loss contains NaN"

        print("✅ Cayley chaos loss computation successful")
        print(f"   Loss value: {loss.item():.4f}")
    except Exception as e:
        print(f"❌ FAILED: {e}")
        sys.exit(1)

    # ========================================================================
    # TEST 4: David with Different Sharing Modes
    # ========================================================================
    print("\n" + "-"*80)
    print("TEST 4: David's Sharing Modes")
    print("-"*80)

    sharing_modes = [
        SharingMode.FULLY_SHARED,
        SharingMode.PARTIAL_SHARED,
        SharingMode.DECOUPLED,
        SharingMode.HIERARCHICAL
    ]

    for sharing_mode in sharing_modes:
        try:
            david = David(
                feature_dim=feature_dim,
                num_classes=num_classes,
                scales=scales,
                sharing_mode=sharing_mode,
                fusion_mode=FusionMode.GATED,
                progressive_training=False
            ).to(device)

            # Create dummy input and anchors
            x = torch.randn(batch_size, feature_dim).to(device)
            anchors_dict = {
                scale: F.normalize(torch.randn(num_classes, scale).to(device), dim=-1)
                for scale in scales
            }

            # Forward pass
            logits, features = david(x, anchors_dict, return_all_scales=False)

            assert logits.shape == (batch_size, num_classes), \
                f"Expected logits shape ({batch_size}, {num_classes}), got {logits.shape}"

            print(f"✅ {sharing_mode.value:20s} - Parameters: {sum(p.numel() for p in david.parameters()):,}")

        except Exception as e:
            print(f"❌ FAILED ({sharing_mode.value}): {e}")
            sys.exit(1)

    # ========================================================================
    # TEST 5: David with Different Fusion Modes
    # ========================================================================
    print("\n" + "-"*80)
    print("TEST 5: David's Fusion Strategies")
    print("-"*80)

    fusion_modes = [
        FusionMode.ATTENTION,
        FusionMode.GATED,
        FusionMode.HIERARCHICAL_TREE,
        FusionMode.DEEP_EFFICIENCY,
        FusionMode.WEIGHTED_SUM,
        FusionMode.MAX_CONFIDENCE,
        FusionMode.PROGRESSIVE
    ]

    x = torch.randn(batch_size, feature_dim).to(device)
    anchors_dict = {
        scale: F.normalize(torch.randn(num_classes, scale).to(device), dim=-1)
        for scale in scales
    }

    for fusion_mode in fusion_modes:
        try:
            david = David(
                feature_dim=feature_dim,
                num_classes=num_classes,
                scales=scales,
                sharing_mode=SharingMode.PARTIAL_SHARED,
                fusion_mode=fusion_mode,
                progressive_training=False
            ).to(device)

            # Forward pass with all scales
            combined, logits_list, features_list, fusion_weights = david(
                x, anchors_dict, return_all_scales=True
            )

            assert combined.shape == (batch_size, num_classes), \
                f"Expected combined shape ({batch_size}, {num_classes})"
            assert len(logits_list) == len(scales), \
                f"Expected {len(scales)} scale outputs, got {len(logits_list)}"

            print(f"✅ {fusion_mode.value:25s} - Weights shape: {fusion_weights.shape}")

        except Exception as e:
            print(f"❌ FAILED ({fusion_mode.value}): {e}")
            sys.exit(1)

    # ========================================================================
    # TEST 6: Progressive Training
    # ========================================================================
    print("\n" + "-"*80)
    print("TEST 6: David's Progressive Scale Activation")
    print("-"*80)

    try:
        scale_warmup = {256: 0, 512: 5, 768: 10, 1024: 15}

        david = David(
            feature_dim=feature_dim,
            num_classes=num_classes,
            scales=scales,
            sharing_mode=SharingMode.DECOUPLED,
            fusion_mode=FusionMode.GATED,
            progressive_training=True,
            scale_warmup_epochs=scale_warmup
        ).to(device)

        test_epochs = [0, 5, 10, 15, 20]

        for epoch in test_epochs:
            david.update_epoch(epoch)
            active_scales = david.get_active_scales()

            # Verify correct scales are active
            expected_active = [s for s in scales if scale_warmup[s] <= epoch]
            assert active_scales == expected_active, \
                f"Epoch {epoch}: Expected {expected_active}, got {active_scales}"

            print(f"✅ Epoch {epoch:2d}: Active scales {active_scales}")

    except Exception as e:
        print(f"❌ FAILED: {e}")
        sys.exit(1)

    # ========================================================================
    # TEST 7: Freeze/Unfreeze Mechanics
    # ========================================================================
    print("\n" + "-"*80)
    print("TEST 7: David's Adaptive Freeze/Unfreeze")
    print("-"*80)

    try:
        david = David(
            feature_dim=feature_dim,
            num_classes=num_classes,
            scales=scales,
            sharing_mode=SharingMode.DECOUPLED,
            fusion_mode=FusionMode.GATED,
            progressive_training=False
        ).to(device)

        # Get initial trainable params
        initial_trainable = sum(p.numel() for p in david.parameters() if p.requires_grad)

        # Freeze scale 512
        david.freeze_scale(512)
        after_freeze = sum(p.numel() for p in david.parameters() if p.requires_grad)

        assert after_freeze < initial_trainable, "Freezing should reduce trainable params"

        # Unfreeze scale 512
        david.unfreeze_scale(512)
        after_unfreeze = sum(p.numel() for p in david.parameters() if p.requires_grad)

        assert after_unfreeze == initial_trainable, "Unfreezing should restore trainable params"

        print(f"✅ Freeze/unfreeze mechanics working correctly")
        print(f"   Initial: {initial_trainable:,} params")
        print(f"   After freeze: {after_freeze:,} params")
        print(f"   After unfreeze: {after_unfreeze:,} params")

    except Exception as e:
        print(f"❌ FAILED: {e}")
        sys.exit(1)

    # ========================================================================
    # TEST 8: Multi-Scale Loss Integration
    # ========================================================================
    print("\n" + "-"*80)
    print("TEST 8: Multi-Scale Loss Integration")
    print("-"*80)

    try:
        david = David(
            feature_dim=feature_dim,
            num_classes=num_classes,
            scales=scales,
            sharing_mode=SharingMode.PARTIAL_SHARED,
            fusion_mode=FusionMode.GATED,
            progressive_training=False
        ).to(device)

        loss_fn = MultiScaleCrystalLoss(
            scales=scales,
            num_classes=num_classes,
            use_rose_loss=True,
            use_cayley_loss=True
        ).to(device)

        # Create test data
        x = torch.randn(batch_size, feature_dim).to(device)
        targets = torch.randint(0, num_classes, (batch_size,)).to(device)

        anchors_dict = {}
        crystals_dict = {}

        for scale in scales:
            anchors_dict[scale] = F.normalize(
                torch.randn(num_classes, scale).to(device), dim=-1
            )
            crystals_dict[scale] = F.normalize(
                torch.randn(num_classes, 5, scale).to(device), dim=-1
            )

        # Forward pass
        combined, logits_list, features_list, _ = david(
            x, anchors_dict, return_all_scales=True
        )

        # Compute loss
        losses = loss_fn(
            combined, logits_list, features_list, targets, crystals_dict, epoch=10
        )

        assert 'total' in losses, "Loss dict should contain 'total'"
        assert 'ce_main' in losses, "Loss dict should contain 'ce_main'"

        total_loss = losses['total']
        assert not torch.isnan(total_loss), "Total loss contains NaN"
        assert total_loss.item() >= 0, "Total loss should be non-negative"

        print(f"✅ Multi-scale loss computation successful")
        print(f"   Total loss: {total_loss.item():.4f}")
        print(f"   Loss components: {list(losses.keys())}")

    except Exception as e:
        print(f"❌ FAILED: {e}")
        sys.exit(1)

    # ========================================================================
    # TEST 9: David's Self-Awareness
    # ========================================================================
    print("\n" + "-"*80)
    print("TEST 9: David's Introspection")
    print("-"*80)

    try:
        david = David(
            feature_dim=feature_dim,
            num_classes=num_classes,
            scales=[256, 512, 768],
            sharing_mode=SharingMode.HIERARCHICAL,
            fusion_mode=FusionMode.DEEP_EFFICIENCY,
            progressive_training=True,
            scale_warmup_epochs={256: 0, 512: 5, 768: 10}
        ).to(device)

        david.update_epoch(7)

        info = david.get_model_info()

        assert info['name'] == 'David', "Name should be David"
        assert info['feature_dim'] == feature_dim, "Feature dim mismatch"
        assert info['num_classes'] == num_classes, "Num classes mismatch"
        assert info['current_epoch'] == 7, "Epoch tracking broken"
        assert info['sharing_mode'] == 'hierarchical', "Sharing mode mismatch"
        assert info['fusion_mode'] == 'deep_efficiency', "Fusion mode mismatch"

        print("✅ David's introspection working correctly")
        print(f"   {david}")

    except Exception as e:
        print(f"❌ FAILED: {e}")
        sys.exit(1)

    # ========================================================================
    # TEST 10: Full Training Step Simulation
    # ========================================================================
    print("\n" + "-"*80)
    print("TEST 10: Full Training Step Simulation")
    print("-"*80)

    try:
        david = David(
            feature_dim=feature_dim,
            num_classes=num_classes,
            scales=scales,
            sharing_mode=SharingMode.PARTIAL_SHARED,
            fusion_mode=FusionMode.HIERARCHICAL_TREE,
            progressive_training=False,
            tree_depth=2
        ).to(device)

        optimizer = torch.optim.AdamW(david.parameters(), lr=1e-4)

        loss_fn = MultiScaleCrystalLoss(
            scales=scales,
            num_classes=num_classes,
            use_rose_loss=True,
            use_cayley_loss=True,
            scale_loss_balance={256: 1.5, 512: 1.2, 768: 1.0, 1024: 0.8}
        ).to(device)

        # Simulate one training step
        david.train()

        x = torch.randn(batch_size, feature_dim).to(device)
        targets = torch.randint(0, num_classes, (batch_size,)).to(device)

        anchors_dict = {}
        crystals_dict = {}
        for scale in scales:
            anchors_dict[scale] = F.normalize(
                torch.randn(num_classes, scale).to(device), dim=-1
            )
            crystals_dict[scale] = F.normalize(
                torch.randn(num_classes, 5, scale).to(device), dim=-1
            )

        # Forward
        combined, logits_list, features_list, fusion_weights = david(
            x, anchors_dict, return_all_scales=True
        )

        # Loss
        losses = loss_fn(
            combined, logits_list, features_list, targets, crystals_dict, epoch=0
        )

        # Backward
        optimizer.zero_grad()
        losses['total'].backward()

        # Check gradients exist
        has_grads = any(p.grad is not None for p in david.parameters() if p.requires_grad)
        assert has_grads, "No gradients computed"

        # Optimizer step
        optimizer.step()

        # Check predictions
        _, predicted = torch.max(combined, 1)
        accuracy = (predicted == targets).float().mean().item() * 100

        print(f"✅ Full training step successful")
        print(f"   Loss: {losses['total'].item():.4f}")
        print(f"   Accuracy: {accuracy:.2f}%")
        print(f"   Fusion weights: {fusion_weights.mean(0).cpu().tolist()}")

    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # ========================================================================
    # FINAL REPORT
    # ========================================================================
    print("\n" + "="*80)
    print("🎉 ALL TESTS PASSED - DAVID IS READY FOR DEPLOYMENT")
    print("="*80)

    final_david = David(
        feature_dim=512,
        num_classes=1000,
        scales=[256, 512, 768, 1024],
        sharing_mode=SharingMode.PARTIAL_SHARED,
        fusion_mode=FusionMode.GATED,
        progressive_training=True
    ).to(device)

    print("\n📊 Final Configuration:")
    print(final_david)
    print(f"\n✨ David is operational and ready to classify!")
    print(f"   Device: {device}")
    print(f"   Total tests: 10")
    print(f"   All passed: ✅")
    print("="*80)