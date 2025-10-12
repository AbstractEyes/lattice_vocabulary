"""
David - Multi-Scale Crystal Classifier
========================================
Model implementation for multi-scale deep learning with crystal representations.

Author: AbstractPhil
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union
import math

from geovocab2.train.config.david_config import (
    DavidArchitectureConfig,
    DavidPresets,
    SharingMode,
    FusionMode
)


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

class RoseLoss(nn.Module):
    """Rose Loss with pentachora role weighting."""

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
        crystals = crystals.to(z.device)
        role_weights = self.role_weights.to(z.device)

        crystals_norm = F.normalize(crystals, dim=-1)
        cos_sim = torch.einsum("bd,cvd->bcv", z, crystals_norm)

        rose_scores = (cos_sim * role_weights.view(1, 1, 5)).sum(dim=-1)
        rose_scores = rose_scores / self.temperature

        true_scores = rose_scores.gather(1, targets.view(-1, 1)).squeeze(1)
        mask = torch.zeros_like(rose_scores, dtype=torch.bool)
        mask.scatter_(1, targets.view(-1, 1), True)
        hard_neg = rose_scores.masked_fill(mask, float("-inf")).max(dim=1).values

        loss = F.relu(self.margin - (true_scores - hard_neg))
        return loss.mean()


class CayleyChaosLoss(nn.Module):
    """Batched Cayley-Menger chaos loss for geometric regularization."""

    def __init__(self, volume_floor: float = 1e-4, chaos_scale: float = 1.0,
                 edge_dev_weight: float = 0.5, gram_weight: float = 0.1,
                 use_sqrt_volume: bool = True):
        super().__init__()
        self.volume_floor = volume_floor
        self.chaos_scale = chaos_scale
        self.edge_dev_weight = edge_dev_weight
        self.gram_weight = gram_weight
        self.use_sqrt_volume = use_sqrt_volume

        self.register_buffer('_triu_i', None)
        self.register_buffer('_triu_j', None)

    def _get_triu_indices(self, device: torch.device):
        if self._triu_i is None or self._triu_i.device != device:
            indices = torch.triu_indices(5, 5, offset=1, device=device)
            self._triu_i = indices[0]
            self._triu_j = indices[1]
        return self._triu_i, self._triu_j

    def compute_cayley_menger_volume(self, X: torch.Tensor) -> torch.Tensor:
        B, N, D = X.shape
        diff = X.unsqueeze(2) - X.unsqueeze(1)
        distsq = (diff * diff).sum(dim=-1)

        M = torch.zeros((B, 6, 6), dtype=X.dtype, device=X.device)
        M[:, 0, 1:] = 1.0
        M[:, 1:, 0] = 1.0
        M[:, 1:, 1:] = distsq

        det = torch.linalg.det(M)
        volume_sq = (-det / 9216.0).clamp(min=0.0)

        return volume_sq.sqrt() if self.use_sqrt_volume else volume_sq

    def compute_edge_uniformity(self, X: torch.Tensor) -> torch.Tensor:
        diff = X.unsqueeze(2) - X.unsqueeze(1)
        distsq = (diff * diff).sum(dim=-1)

        triu_i, triu_j = self._get_triu_indices(X.device)
        edge_lengths = distsq[:, triu_i, triu_j]

        edge_mean = edge_lengths.mean(dim=1)
        edge_std = edge_lengths.std(dim=1)
        edge_dev = edge_std / edge_mean.clamp(min=1e-6)

        return edge_dev

    def compute_gram_condition(self, X: torch.Tensor) -> torch.Tensor:
        centered = X - X.mean(dim=1, keepdim=True)
        gram = torch.bmm(centered, centered.transpose(1, 2))

        gram_trace = torch.diagonal(gram, dim1=1, dim2=2).sum(dim=1)
        gram_det = torch.linalg.det(gram)

        condition = gram_det / gram_trace.clamp(min=1e-6)
        gram_penalty = F.relu(1.0 - condition)

        return gram_penalty

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        B, N, D = X.shape
        assert N == 5, f"Expected 5 vertices, got {N}"

        volumes = self.compute_cayley_menger_volume(X)
        chaos_penalty = F.relu(self.volume_floor - volumes)
        chaos_loss = chaos_penalty.mean()

        edge_dev = self.compute_edge_uniformity(X)
        edge_loss = edge_dev.mean()

        gram_loss = 0.0
        if self.gram_weight > 0:
            gram_penalty = self.compute_gram_condition(X)
            gram_loss = gram_penalty.mean()

        total_loss = (
            self.chaos_scale * chaos_loss +
            self.edge_dev_weight * edge_loss +
            self.gram_weight * gram_loss
        )

        return total_loss


class MultiScaleCrystalLoss(nn.Module):
    """Aggregate loss across scales."""

    def __init__(self, scales: List[int], num_classes: int = 1000,
                 use_rose_loss: bool = True, use_cayley_loss: bool = True,
                 rose_initial_weight: float = 0.1, rose_max_weight: float = 0.5,
                 cayley_weight: float = 0.05,
                 scale_loss_balance: Optional[Dict[int, float]] = None):
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
        losses = {}

        ce_main = self.ce_loss(combined_logits, targets)
        losses['ce_main'] = ce_main

        total_scale_loss = 0
        for i, (scale, logits, features) in enumerate(
            zip(self.scales, scale_logits, scale_features)
        ):
            scale_weight = self.scale_balance.get(scale, 1.0)

            ce_scale = self.ce_loss(logits, targets)
            losses[f'ce_{scale}'] = ce_scale
            total_scale_loss += scale_weight * ce_scale

            if self.use_rose_loss and scale in crystals_dict:
                rose_loss = self.rose_losses[str(scale)](
                    features, crystals_dict[scale], targets
                )
                losses[f'rose_{scale}'] = rose_loss

                progress = epoch / 100
                current_weight = self.rose_weight + \
                    (self.rose_max_weight - self.rose_weight) * progress
                total_scale_loss += current_weight * rose_loss

            if self.use_cayley_loss:
                batch_crystals = crystals_dict[scale][targets]
                cayley_loss = self.cayley_loss(batch_crystals)
                losses[f'cayley_{scale}'] = cayley_loss
                total_scale_loss += self.cayley_weight * cayley_loss

        total_loss = ce_main + total_scale_loss / len(self.scales)
        losses['total'] = total_loss

        return losses


# ============================================================================
# MODEL COMPONENTS
# ============================================================================

class SharedFeatureExtractor(nn.Module):
    """Shared feature extraction layers."""

    def __init__(self, input_dim: int, output_dim: int,
                 num_layers: int = 2, dropout: float = 0.1):
        super().__init__()

        layers = []
        hidden_dim = (input_dim + output_dim) // 2

        layers.extend([
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        ])

        for _ in range(num_layers - 2):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])

        if num_layers > 1:
            layers.extend([
                nn.Linear(hidden_dim, output_dim),
                nn.LayerNorm(output_dim)
            ])

        self.extractor = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.extractor(x)


class ScaleSpecificHead(nn.Module):
    """Scale-specific projection head with optional bottleneck."""

    def __init__(self, input_dim: int, crystal_dim: int,
                 use_belly: bool = True, belly_expand: float = 2.0):
        super().__init__()
        self.crystal_dim = crystal_dim

        if use_belly:
            belly_dim = int(crystal_dim * belly_expand)
            self.projection = nn.Sequential(
                nn.Linear(input_dim, belly_dim),
                nn.ReLU(),
                nn.Dropout(1.0 / math.sqrt(crystal_dim)),
                nn.Linear(belly_dim, crystal_dim, bias=False)
            )
        else:
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
        z = self.projection(features)
        z = F.normalize(z, dim=-1)
        logits = (z @ anchors.T) / 0.03
        return logits, z


# ============================================================================
# FUSION MECHANISMS
# ============================================================================

class AttentionFusion(nn.Module):
    """Attention-based fusion."""

    def __init__(self, feature_dim: int, num_scales: int,
                 temperature: float = 1.0, dropout: float = 0.1):
        super().__init__()
        self.query = nn.Linear(feature_dim, num_scales)
        self.temperature = temperature
        self.dropout = nn.Dropout(dropout)

    def forward(self, features: torch.Tensor,
                logits_list: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        attn_logits = self.query(features)
        attn_weights = F.softmax(attn_logits / self.temperature, dim=-1)
        attn_weights = self.dropout(attn_weights)

        combined = torch.zeros_like(logits_list[0])
        for i, logits in enumerate(logits_list):
            combined += attn_weights[:, i:i+1] * logits

        return combined, attn_weights


class GatedFusion(nn.Module):
    """Gated mixture of experts."""

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
        gates = self.gate(features)
        combined = torch.zeros_like(logits_list[0])
        for i, logits in enumerate(logits_list):
            combined += gates[:, i:i+1] * logits
        return combined, gates


class HierarchicalTreeGating(nn.Module):
    """Tree-based hierarchical gating."""

    def __init__(self, feature_dim: int, num_scales: int = 5, depth: int = 3):
        super().__init__()
        self.num_scales = num_scales
        self.depth = depth

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

        num_leaves = 2 ** depth
        self.leaf_to_scale = nn.Sequential(
            nn.Linear(num_leaves, num_scales * 2),
            nn.GELU(),
            nn.Linear(num_scales * 2, num_scales)
        )

        self.direct_gate = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 4),
            nn.GELU(),
            nn.Linear(feature_dim // 4, num_scales)
        )

        self.combine_weight = nn.Parameter(torch.tensor(0.7))

    def forward(self, features: torch.Tensor,
                logits_list: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = features.shape[0]
        device = features.device

        path_probs = [torch.ones(batch_size, 1, device=device)]

        for level in range(self.depth):
            next_probs = []
            num_nodes = 2 ** level

            for node_idx in range(num_nodes):
                parent_prob = path_probs[0] if level == 0 else path_probs[node_idx]

                node_logits = self.tree_nodes[level][node_idx](features)
                node_probs = F.softmax(node_logits / 0.5, dim=-1)

                left_prob = parent_prob * node_probs[:, 0:1]
                right_prob = parent_prob * node_probs[:, 1:2]
                next_probs.extend([left_prob, right_prob])

            path_probs = next_probs

        leaf_probs = torch.cat(path_probs, dim=1)
        tree_gates = F.softmax(self.leaf_to_scale(leaf_probs), dim=-1)

        direct_gates = F.softmax(self.direct_gate(features), dim=-1)
        gates = torch.sigmoid(self.combine_weight) * tree_gates + \
                (1 - torch.sigmoid(self.combine_weight)) * direct_gates
        gates = gates / gates.sum(dim=-1, keepdim=True)

        combined = torch.zeros_like(logits_list[0])
        for i, logits in enumerate(logits_list):
            if i < gates.shape[1]:
                combined += gates[:, i:i+1] * logits

        return combined, gates


class DeepEfficiencyGating(nn.Module):
    """Ultra-efficient gating with cross-attention."""

    def __init__(self, feature_dim: int, num_scales: int = 5,
                 compression_ratio: int = 4, num_experts: int = 3,
                 expert_dropout: float = 0.1, attention_dropout: float = 0.1):
        super().__init__()
        self.num_scales = num_scales
        self.num_experts = num_experts

        self.temperature = nn.Parameter(torch.ones(1) * 0.5)
        bottleneck_dim = max(num_scales * 8, feature_dim // compression_ratio)

        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim, bottleneck_dim),
                nn.LayerNorm(bottleneck_dim),
                nn.GELU(),
                nn.Dropout(expert_dropout)
            ) for _ in range(num_experts)
        ])

        self.cross_attention = nn.MultiheadAttention(
            bottleneck_dim,
            num_heads=min(8, bottleneck_dim // 8),
            batch_first=True,
            dropout=attention_dropout
        )

        self.gate_heads = nn.ModuleList([
            nn.Linear(bottleneck_dim * num_experts, num_scales)
            for _ in range(3)
        ])

        self.head_weights = nn.Parameter(torch.ones(3) / 3)
        self.scale_bias = nn.Parameter(torch.zeros(num_scales))
        self.sparsity_threshold = nn.Parameter(torch.tensor(0.1))

    def forward(self, features: torch.Tensor,
                logits_list: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        expert_outputs = [expert(features) for expert in self.experts]
        stacked = torch.stack(expert_outputs, dim=1)

        attended, _ = self.cross_attention(stacked, stacked, stacked)
        flattened = attended.reshape(features.shape[0], -1)

        gate_outputs = [head(flattened) for head in self.gate_heads]
        head_weights = F.softmax(self.head_weights, dim=0)
        combined_gates = sum(w * g for w, g in zip(head_weights, gate_outputs))

        gate_logits = (combined_gates + self.scale_bias) / self.temperature.abs()
        gates = F.softmax(gate_logits, dim=-1)

        if not self.training:
            mask = gates > self.sparsity_threshold
            sparse_gates = gates * mask
            sparse_gates = sparse_gates / (sparse_gates.sum(dim=-1, keepdim=True) + 1e-8)
        else:
            sparse_gates = gates

        combined = torch.zeros_like(logits_list[0])
        for i, logits in enumerate(logits_list):
            if i < sparse_gates.shape[1]:
                combined += sparse_gates[:, i:i+1] * logits

        return combined, sparse_gates


# ============================================================================
# DAVID MODEL
# ============================================================================

class David(nn.Module):
    """
    David: Multi-Scale Crystal Classifier

    Orchestrates parallel processing across multiple embedding scales.
    """

    def __init__(
        self,
        config: Optional[DavidArchitectureConfig] = None,
        # Optional overrides
        feature_dim: Optional[int] = None,
        num_classes: Optional[int] = None,
        scales: Optional[List[int]] = None,
        sharing_mode: Optional[str] = None,
        fusion_mode: Optional[str] = None,
        use_belly: Optional[bool] = None,
        belly_expand: Optional[float] = None,
        shared_feature_dim: Optional[int] = None,
        shared_layers: Optional[int] = None,
        shared_dropout: Optional[float] = None,
        fusion_temperature: Optional[float] = None,
        fusion_dropout: Optional[float] = None,
        tree_depth: Optional[int] = None,
        num_experts: Optional[int] = None,
        compression_ratio: Optional[int] = None,
        expert_dropout: Optional[float] = None,
        attention_dropout: Optional[float] = None,
        progressive_training: Optional[bool] = None,
        scale_warmup_epochs: Optional[Dict[int, int]] = None
    ):
        """
        Initialize David from config with optional parameter overrides.

        Args:
            config: DavidArchitectureConfig (if None, uses defaults)
            All other args: Optional overrides for config values
        """
        super().__init__()

        # Use config as base, override with explicit params
        if config is None:
            config = DavidArchitectureConfig()

        self.feature_dim = feature_dim or config.feature_dim
        self.num_classes = num_classes or config.num_classes
        self.scales = scales or config.scales
        self.use_belly = use_belly if use_belly is not None else config.use_belly
        self.belly_expand = belly_expand or config.belly_expand
        self.progressive_training = progressive_training if progressive_training is not None else config.progressive_training
        self.scale_warmup_epochs = scale_warmup_epochs or config.scale_warmup_epochs

        # Convert string modes to enums
        sharing_str = sharing_mode or config.sharing_mode
        self.sharing_mode = SharingMode(sharing_str)

        fusion_str = fusion_mode or config.fusion_mode
        self.fusion_mode = FusionMode(fusion_str)

        # Extract other params
        _shared_feature_dim = shared_feature_dim or config.shared_feature_dim
        _shared_layers = shared_layers or config.shared_layers
        _shared_dropout = shared_dropout or config.shared_dropout
        _fusion_temperature = fusion_temperature or config.fusion_temperature
        _fusion_dropout = fusion_dropout or config.fusion_dropout
        _tree_depth = tree_depth or config.tree_depth
        _num_experts = num_experts or config.num_experts
        _compression_ratio = compression_ratio or config.compression_ratio
        _expert_dropout = expert_dropout or config.expert_dropout
        _attention_dropout = attention_dropout or config.attention_dropout

        # State tracking
        self.current_epoch = 0
        self.scale_accuracies = {s: [] for s in self.scales}

        # Build architecture
        self._build_architecture(_shared_feature_dim, _shared_layers, _shared_dropout)
        self._build_fusion(_shared_feature_dim, _fusion_temperature, _fusion_dropout,
                          _tree_depth, _num_experts, _compression_ratio,
                          _expert_dropout, _attention_dropout)

        self.register_buffer("scale_weights", torch.tensor([1.0 for _ in self.scales]))

    def _build_architecture(self, shared_feature_dim: int,
                           shared_layers: int, shared_dropout: float):
        """Build processing architecture based on sharing mode."""

        if self.sharing_mode == SharingMode.FULLY_SHARED:
            self.shared_extractor = SharedFeatureExtractor(
                self.feature_dim, shared_feature_dim, shared_layers, shared_dropout
            )
            self.heads = nn.ModuleDict({
                str(scale): ScaleSpecificHead(
                    shared_feature_dim, scale,
                    use_belly=self.use_belly,
                    belly_expand=self.belly_expand
                )
                for scale in self.scales
            })

        elif self.sharing_mode == SharingMode.PARTIAL_SHARED:
            self.shared_base = nn.Linear(self.feature_dim, shared_feature_dim)
            self.heads = nn.ModuleDict({
                str(scale): ScaleSpecificHead(
                    shared_feature_dim, scale,
                    use_belly=self.use_belly,
                    belly_expand=self.belly_expand
                )
                for scale in self.scales
            })

        elif self.sharing_mode == SharingMode.DECOUPLED:
            self.heads = nn.ModuleDict({
                str(scale): ScaleSpecificHead(
                    self.feature_dim, scale,
                    use_belly=self.use_belly,
                    belly_expand=self.belly_expand
                )
                for scale in self.scales
            })

        elif self.sharing_mode == SharingMode.HIERARCHICAL:
            for i, scale in enumerate(self.scales):
                if i == 0:
                    setattr(self, f'head_{scale}',
                           ScaleSpecificHead(
                               self.feature_dim, scale,
                               use_belly=self.use_belly,
                               belly_expand=self.belly_expand
                           ))
                else:
                    prev_scale = self.scales[i-1]
                    setattr(self, f'refine_{scale}', nn.Sequential(
                        nn.Linear(prev_scale + self.feature_dim, scale),
                        nn.ReLU()
                    ))
                    setattr(self, f'head_{scale}',
                           ScaleSpecificHead(
                               scale, scale,
                               use_belly=self.use_belly,
                               belly_expand=self.belly_expand
                           ))

    def _build_fusion(self, shared_feature_dim: int, temperature: float,
                     dropout: float, tree_depth: int, num_experts: int,
                     compression_ratio: int, expert_dropout: float,
                     attention_dropout: float):
        """Build fusion strategy."""

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
                num_experts=num_experts,
                expert_dropout=expert_dropout,
                attention_dropout=attention_dropout
            )

        elif self.fusion_mode == FusionMode.WEIGHTED_SUM:
            self.fusion_weights = nn.Parameter(
                torch.ones(len(self.scales)) / len(self.scales)
            )

        elif self.fusion_mode == FusionMode.PROGRESSIVE:
            weights = torch.tensor([0.1, 0.15, 0.25, 0.25, 0.25])
            self.register_buffer("progressive_weights", weights[:len(self.scales)])

    def _should_use_scale(self, scale: int) -> bool:
        """Check if scale should be active."""
        if not self.progressive_training:
            return True
        warmup_epoch = self.scale_warmup_epochs.get(scale, 0)
        return self.current_epoch >= warmup_epoch

    def _extract_base_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract base features according to sharing mode."""
        if self.sharing_mode == SharingMode.FULLY_SHARED:
            return self.shared_extractor(x)
        elif self.sharing_mode == SharingMode.PARTIAL_SHARED:
            return F.relu(self.shared_base(x))
        else:
            return x

    def _hierarchical_forward(
        self, x: torch.Tensor, anchors_dict: Dict[int, torch.Tensor]
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Hierarchical refinement process."""
        logits_list = []
        features_list = []
        prev_features = None

        for i, scale in enumerate(self.scales):
            if not self._should_use_scale(scale):
                continue

            if i == 0:
                head = getattr(self, f'head_{scale}')
                logits, features = head(x, anchors_dict[scale])
                prev_features = features
            else:
                refine = getattr(self, f'refine_{scale}')
                head = getattr(self, f'head_{scale}')
                refined = refine(torch.cat([prev_features, x], dim=-1))
                logits, features = head(refined, anchors_dict[scale])
                prev_features = features

            logits_list.append(logits)
            features_list.append(features)

        return logits_list, features_list

    def _parallel_forward(
        self, base_features: torch.Tensor, anchors_dict: Dict[int, torch.Tensor]
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Parallel multi-scale processing."""
        logits_list = []
        features_list = []

        for scale in self.scales:
            if self._should_use_scale(scale):
                logits, features = self.heads[str(scale)](
                    base_features, anchors_dict[scale]
                )
                logits_list.append(logits)
                features_list.append(features)

        return logits_list, features_list

    def _fuse_predictions(
        self, features: torch.Tensor, logits_list: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Combine multi-scale predictions."""

        if len(logits_list) == 1:
            return logits_list[0], torch.ones(1, device=features.device)

        if self.fusion_mode in [FusionMode.ATTENTION, FusionMode.GATED,
                                FusionMode.HIERARCHICAL_TREE, FusionMode.DEEP_EFFICIENCY]:
            return self.fusion(features, logits_list)

        elif self.fusion_mode == FusionMode.WEIGHTED_SUM:
            weights = F.softmax(self.fusion_weights, dim=0)
            combined = sum(w * logits for w, logits in zip(weights, logits_list))
            return combined, weights

        elif self.fusion_mode == FusionMode.MAX_CONFIDENCE:
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
        Forward pass.

        Args:
            x: Input features [B, D]
            anchors_dict: Crystal anchors {scale: [C, D_scale]}
            return_all_scales: Return all intermediate outputs

        Returns:
            If return_all_scales=False:
                combined_logits, features
            If return_all_scales=True:
                combined_logits, logits_list, features_list, fusion_weights
        """

        if self.sharing_mode == SharingMode.HIERARCHICAL:
            logits_list, features_list = self._hierarchical_forward(x, anchors_dict)
            base_features = x
        else:
            base_features = self._extract_base_features(x)
            logits_list, features_list = self._parallel_forward(base_features, anchors_dict)

        combined, fusion_weights = self._fuse_predictions(base_features, logits_list)

        if return_all_scales:
            return combined, logits_list, features_list, fusion_weights
        else:
            return combined, features_list[0] if features_list else base_features

    # ========================================================================
    # CLASS METHODS
    # ========================================================================

    @classmethod
    def from_config(cls, config: DavidArchitectureConfig, **kwargs) -> 'David':
        """
        Create David from config with optional overrides.

        Args:
            config: DavidArchitectureConfig instance
            **kwargs: Optional parameter overrides

        Example:
            config = DavidPresets.balanced()
            david = David.from_config(config, num_classes=100)
        """
        return cls(config=config, **kwargs)

    @classmethod
    def from_preset(cls, preset_name: str, **kwargs) -> 'David':
        """
        Create David from preset name with optional overrides.

        Args:
            preset_name: Name of preset ('balanced', 'small_fast', etc.)
            **kwargs: Optional parameter overrides

        Example:
            david = David.from_preset('balanced', num_classes=100)
        """
        config = DavidPresets.get_preset(preset_name)
        return cls(config=config, **kwargs)

    @classmethod
    def from_json(cls, path: str, **kwargs) -> 'David':
        """
        Load David from JSON config file with optional overrides.

        Args:
            path: Path to JSON config file
            **kwargs: Optional parameter overrides

        Example:
            david = David.from_json('my_config.json', num_classes=100)
        """
        config = DavidArchitectureConfig.from_json(path)
        return cls(config=config, **kwargs)

    def get_config(self) -> DavidArchitectureConfig:
        """Extract current configuration from model."""
        return DavidArchitectureConfig(
            feature_dim=self.feature_dim,
            num_classes=self.num_classes,
            scales=self.scales,
            sharing_mode=self.sharing_mode.value,
            fusion_mode=self.fusion_mode.value,
            use_belly=self.use_belly,
            belly_expand=self.belly_expand,
            progressive_training=self.progressive_training,
            scale_warmup_epochs=self.scale_warmup_epochs,
        )

    # ========================================================================
    # ADAPTIVE BEHAVIORS
    # ========================================================================

    def update_epoch(self, epoch: int):
        """Update internal clock for progressive training."""
        self.current_epoch = epoch

    def get_active_scales(self) -> List[int]:
        """Get currently active scales."""
        return [s for s in self.scales if self._should_use_scale(s)]

    def freeze_scale(self, scale: int):
        """Freeze a specific scale."""
        if self.sharing_mode == SharingMode.HIERARCHICAL:
            head = getattr(self, f'head_{scale}', None)
            if head:
                for param in head.parameters():
                    param.requires_grad = False
        else:
            for param in self.heads[str(scale)].parameters():
                param.requires_grad = False

    def unfreeze_scale(self, scale: int):
        """Unfreeze a specific scale."""
        if self.sharing_mode == SharingMode.HIERARCHICAL:
            head = getattr(self, f'head_{scale}', None)
            if head:
                for param in head.parameters():
                    param.requires_grad = True
        else:
            for param in self.heads[str(scale)].parameters():
                param.requires_grad = True

    def get_model_info(self) -> Dict[str, any]:
        """Get model configuration and status."""
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
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("David Model - Usage Examples")
    print("="*80)

    # Method 1: From preset
    print("\n[1] From Preset:")
    david = David.from_preset('balanced')
    print(f"  {david}")

    # Method 2: From preset with overrides
    print("\n[2] From Preset with Overrides:")
    david = David.from_preset('balanced', num_classes=100, use_belly=False)
    print(f"  Num classes: {david.num_classes}")
    print(f"  Use belly: {david.use_belly}")

    # Method 3: From config object
    print("\n[3] From Config Object:")
    config = DavidPresets.get_preset('small_fast')
    david = David.from_config(config, num_classes=10)
    print(f"  {david}")

    # Method 4: From JSON
    print("\n[4] From JSON:")
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        path = f.name

    config = DavidPresets.balanced()
    config.to_json(path)
    david = David.from_json(path, num_classes=50)
    print(f"  Loaded from: {path}")
    print(f"  Num classes: {david.num_classes}")

    import os
    os.unlink(path)

    print("\n" + "="*80)