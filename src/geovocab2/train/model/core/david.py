"""
David - Multi-Scale Crystal Classifier
========================================
Model implementation for multi-scale deep learning with crystal representations.

Should be placed at: geovocab2/train/model/core/david.py

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

    def __init__(self, margin: float = 1.0, temperature: float = 0.07,
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
                 rose_margin: float = 1.0, rose_temperature: float = 0.07,
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
                str(scale): RoseLoss(margin=rose_margin, temperature=rose_temperature)
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
# PATTERN-SUPERVISED LOSS
# ============================================================================
class PatternSupervisedLoss(nn.Module):
    """Pattern-supervised loss with full 1000-class supervision."""

    def __init__(
            self,
            num_timestep_bins: int = 100,
            num_patterns_per_timestep: int = 10,
            feature_similarity_weight: float = 0.5,
            rose_weight: float = 0.3,
            ce_weight: float = 0.2,
            pattern_diversity_weight: float = 0.05,
            use_soft_assignment: bool = True,
            temperature: float = 0.1
    ):
        super().__init__()

        self.num_bins = num_timestep_bins
        self.num_patterns = num_patterns_per_timestep
        self.num_classes = num_timestep_bins * num_patterns_per_timestep

        self.feature_sim_weight = feature_similarity_weight
        self.rose_weight = rose_weight
        self.ce_weight = ce_weight
        self.pattern_diversity_weight = pattern_diversity_weight

        self.use_soft_assignment = use_soft_assignment
        self.temperature = temperature

    def assign_patterns(
            self,
            features: torch.Tensor,
            timestep_class: torch.Tensor,
            crystal_centroids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Assign samples to nearest pattern within their timestep bin."""
        B = features.shape[0]

        batch_centroids = crystal_centroids[timestep_class]
        features_expanded = features.unsqueeze(1)
        similarities = F.cosine_similarity(
            features_expanded,
            batch_centroids,
            dim=2
        )

        pattern_ids = similarities.argmax(dim=1)
        full_class_ids = timestep_class * self.num_patterns + pattern_ids

        return pattern_ids, full_class_ids

    def compute_soft_assignment(
            self,
            features: torch.Tensor,
            timestep_class: torch.Tensor,
            crystal_centroids: torch.Tensor
    ) -> torch.Tensor:
        """Compute soft pattern assignment with temperature smoothing."""
        B, D = features.shape
        device = features.device

        batch_centroids = crystal_centroids[timestep_class]
        features_expanded = features.unsqueeze(1)

        similarities = F.cosine_similarity(
            features_expanded,
            batch_centroids,
            dim=2
        )

        pattern_probs = F.softmax(similarities / self.temperature, dim=1)

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
        """Encourage diverse pattern usage."""
        B = logits.shape[0]

        pattern_probs_list = []
        for i in range(B):
            t = timestep_class[i]
            start_idx = t * self.num_patterns
            end_idx = start_idx + self.num_patterns
            probs = F.softmax(logits[i, start_idx:end_idx], dim=0)
            pattern_probs_list.append(probs)

        pattern_probs = torch.stack(pattern_probs_list)
        entropy = -(pattern_probs * torch.log(pattern_probs + 1e-8)).sum(dim=1).mean()

        return -entropy

    def forward(
            self,
            student_features: torch.Tensor,
            teacher_features: torch.Tensor,
            student_logits: torch.Tensor,
            crystal_centroids: torch.Tensor,
            timesteps: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict]:
        """Compute full loss with pattern supervision."""

        timestep_class = (timesteps // 10).clamp(0, self.num_bins - 1)

        pattern_ids, full_class_ids = self.assign_patterns(
            student_features,
            timestep_class,
            crystal_centroids
        )

        target_centroids = torch.stack([
            crystal_centroids[timestep_class[j], pattern_ids[j]]
            for j in range(len(timestep_class))
        ])

        feature_sim_loss = 1.0 - F.cosine_similarity(
            student_features,
            target_centroids,
            dim=-1
        ).mean()

        rose_loss = feature_sim_loss

        if self.use_soft_assignment:
            soft_targets = self.compute_soft_assignment(
                student_features, timestep_class, crystal_centroids
            )
            log_probs = F.log_softmax(student_logits, dim=1)
            ce_loss = -(soft_targets * log_probs).sum(dim=1).mean()
        else:
            ce_loss = F.cross_entropy(student_logits, full_class_ids)

        diversity_loss = self.compute_pattern_diversity_loss(
            student_logits, timestep_class
        )

        total_loss = (
                self.feature_sim_weight * feature_sim_loss +
                self.rose_weight * rose_loss +
                self.ce_weight * ce_loss +
                self.pattern_diversity_weight * diversity_loss
        )

        timestep_pred = student_logits.argmax(dim=-1) // self.num_patterns
        pattern_pred = student_logits.argmax(dim=-1) % self.num_patterns
        full_pred = student_logits.argmax(dim=-1)

        timestep_acc = (timestep_pred == timestep_class).float().mean()
        pattern_acc = (pattern_pred == pattern_ids).float().mean()
        full_acc = (full_pred == full_class_ids).float().mean()

        metrics = {
            'feature_sim': feature_sim_loss.item(),
            'rose': rose_loss.item(),
            'ce': ce_loss.item(),
            'pattern_diversity': diversity_loss.item(),
            'timestep_acc': timestep_acc.item(),
            'pattern_acc': pattern_acc.item(),
            'full_acc': full_acc.item()
        }

        return total_loss, metrics


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
                 use_belly: bool = True, belly_expand: float = 2.0,
                 temperature: float = 0.07):
        super().__init__()
        self.crystal_dim = crystal_dim
        self.temperature = temperature

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
                nn.init.xavier_uniform_(layer.weight, gain=0.5)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, features: torch.Tensor,
                anchors: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.projection(features)
        z = F.normalize(z, dim=-1)
        logits = (z @ anchors.T) / self.temperature
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


class CantorScaleFusion(nn.Module):
    """
    Cantor-based multi-scale fusion using fractal geometry for scale routing.
    """

    def __init__(
            self,
            feature_dim: int,
            scales: List[int],
            num_heads: int = 4,
            cantor_depth: int = 8,
            local_window: int = 3,
            temperature: float = 0.07,
            dropout: float = 0.1,
            use_scale_embeddings: bool = True
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.scales = scales
        self.num_scales = len(scales)
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        self.cantor_depth = cantor_depth
        self.local_window = min(local_window, self.num_scales)
        self.temperature = nn.Parameter(torch.tensor(temperature))

        # ✓ FIXED: Per-scale projections to handle different input dimensions
        self.scale_to_common = nn.ModuleList([
            nn.Linear(scale, feature_dim)
            for scale in scales
        ])

        # QKV projections (now all work on feature_dim)
        self.q_proj = nn.Linear(feature_dim, feature_dim)
        self.k_proj = nn.Linear(feature_dim, feature_dim)
        self.v_proj = nn.Linear(feature_dim, feature_dim)

        # Scale embeddings (optional, now in common space)
        if use_scale_embeddings:
            self.scale_embeddings = nn.Parameter(
                torch.randn(self.num_scales, feature_dim) * 0.02
            )
        else:
            self.scale_embeddings = None

        # Pre-compute Cantor coordinates for scales
        self.register_buffer(
            'scale_cantor_coords',
            self._compute_scale_cantor_coordinates()
        )

        # Pre-compute scale routing
        self.register_buffer(
            'scale_routes',
            self._build_scale_routes()
        )

        # Output projection
        self.out_proj = nn.Linear(feature_dim, feature_dim)
        self.dropout = nn.Dropout(dropout)

        # Gate network
        self.gate_net = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.LayerNorm(feature_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim // 2, self.num_scales)
        )

    def _cantor_coordinate(self, position: int, max_len: int, depth: int) -> float:
        """Compute Cantor set coordinate for a position."""
        x = position / max(1, max_len - 1)
        x = max(1e-6, min(x, 1.0 - 1e-6))

        cantor_val = 0.0
        factor = 0.5

        for _ in range(depth):
            x *= 3.0
            digit = int(x)
            x -= digit

            if digit == 2:
                cantor_val += factor

            factor *= 0.5

        return cantor_val

    def _compute_scale_cantor_coordinates(self) -> torch.Tensor:
        """Map each scale to a Cantor coordinate."""
        coords = torch.tensor([
            self._cantor_coordinate(i, self.num_scales, self.cantor_depth)
            for i in range(self.num_scales)
        ], dtype=torch.float32)

        return coords

    def _build_scale_routes(self) -> torch.Tensor:
        """Build routing table: which scales attend to which."""
        routes = torch.zeros(self.num_scales, self.local_window, dtype=torch.long)

        for i in range(self.num_scales):
            distances = torch.abs(self.scale_cantor_coords - self.scale_cantor_coords[i])
            _, nearest = torch.topk(distances, self.local_window, largest=False)
            routes[i] = nearest

        return routes

    def _sparse_scale_attention(
            self,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor
    ) -> torch.Tensor:
        """Sparse attention over scales using Cantor routing."""
        B, H, _, D = q.shape
        device = q.device

        routes = self.scale_routes.to(device)
        routes_exp = routes.unsqueeze(0).unsqueeze(0).expand(B, H, -1, -1)

        batch_idx = torch.arange(B, device=device).view(-1, 1, 1, 1)
        head_idx = torch.arange(H, device=device).view(1, -1, 1, 1)

        batch_idx = batch_idx.expand(B, H, self.num_scales, self.local_window)
        head_idx = head_idx.expand(B, H, self.num_scales, self.local_window)

        k_gathered = k[batch_idx, head_idx, routes_exp, :]
        v_gathered = v[batch_idx, head_idx, routes_exp, :]

        q_exp = q.expand(-1, -1, self.num_scales, -1)

        scores = torch.einsum('bhsd,bhskd->bhsk', q_exp, k_gathered) / math.sqrt(D)

        attn_weights_sparse = F.softmax(scores / self.temperature.abs(), dim=-1)
        attn_weights_sparse = self.dropout(attn_weights_sparse)

        output = torch.einsum('bhsk,bhskd->bhsd', attn_weights_sparse, v_gathered)

        scale_importance = output.norm(dim=-1).mean(dim=1)

        return scale_importance

    def forward(
            self,
            features: torch.Tensor,
            logits_list: List[torch.Tensor],
            scale_features: Optional[List[torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            features: Base features [B, feature_dim]
            logits_list: List of scale logits, each [B, num_classes]
            scale_features: List of scale-specific features [B, scale_i]

        Returns:
            combined_logits, attention_weights
        """
        B = features.shape[0]
        device = features.device

        if scale_features is None:
            scale_features = [features] * self.num_scales

        # ✓ FIXED: Project each scale to common dimension first
        scale_features_common = [
            self.scale_to_common[i](feat)
            for i, feat in enumerate(scale_features)
        ]

        # Add scale embeddings (now all in common space)
        if self.scale_embeddings is not None:
            scale_features_common = [
                feat + self.scale_embeddings[i]
                for i, feat in enumerate(scale_features_common)
            ]

        # Project to Q, K, V (now all same dimension)
        Q = self.q_proj(features).view(B, self.num_heads, 1, self.head_dim)

        K_list, V_list = [], []
        for scale_feat in scale_features_common:
            K = self.k_proj(scale_feat).view(B, self.num_heads, 1, self.head_dim)
            V = self.v_proj(scale_feat).view(B, self.num_heads, 1, self.head_dim)
            K_list.append(K)
            V_list.append(V)

        K = torch.cat(K_list, dim=2)  # [B, H, num_scales, D]
        V = torch.cat(V_list, dim=2)  # [B, H, num_scales, D]

        # Sparse attention using Cantor routing
        scale_importance = self._sparse_scale_attention(Q, K, V)

        # Combine with learned gating
        gate_logits = self.gate_net(features)

        # 70% Cantor routing, 30% learned gate
        alpha = 0.7
        combined_scores = (
                alpha * scale_importance +
                (1 - alpha) * gate_logits
        )

        # Final attention weights
        attention_weights = F.softmax(combined_scores, dim=-1)

        # Apply to logits
        combined_logits = torch.zeros_like(logits_list[0])
        for i, logits in enumerate(logits_list):
            combined_logits += attention_weights[:, i:i + 1] * logits

        return combined_logits, attention_weights


class GeometricAttentionGate(nn.Module):
    """
    Geometric attention gate using pentachoron-inspired multi-scale fusion.
    """

    def __init__(
            self,
            feature_dim: int,
            num_scales: int = 5,
            scales: Optional[List[int]] = None,
            num_heads: int = 4,
            use_cayley_attention: bool = True,
            use_angular_attention: bool = True,
            temperature: float = 0.07,
            dropout: float = 0.1,
            scale_dim_aware: bool = True
    ):
        super().__init__()
        self.num_scales = num_scales
        self.scales = scales
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        self.use_cayley = use_cayley_attention
        self.use_angular = use_angular_attention
        self.temperature = nn.Parameter(torch.tensor(temperature))
        self.scale_dim_aware = scale_dim_aware

        # ✓ FIXED: Per-scale projections to common dimension
        if scales is not None:
            self.scale_to_common = nn.ModuleList([
                nn.Linear(scale, feature_dim)
                for scale in scales
            ])
        else:
            self.scale_to_common = None

        # Query/Key/Value projections (now all work on feature_dim)
        self.q_proj = nn.Linear(feature_dim, feature_dim)
        self.k_proj = nn.Linear(feature_dim, feature_dim)
        self.v_proj = nn.Linear(feature_dim, feature_dim)

        # Scale embeddings (now in common space)
        self.scale_embeddings = nn.Parameter(
            torch.randn(num_scales, feature_dim) * 0.02
        )

        # Pentachoron role weights
        if self.use_angular:
            role_weights = torch.tensor([1.0, -0.75, 0.75, 0.75, -0.75])
            self.register_buffer("role_weights", role_weights)

        # Output projection and gating
        self.out_proj = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Learnable combination weights for different attention types
        self.attention_weights = nn.Parameter(torch.ones(3) / 3)

        self.dropout = nn.Dropout(dropout)

    def _compute_geometric_attention(
            self,
            features: torch.Tensor,
            scale_features: List[torch.Tensor]
    ) -> torch.Tensor:
        """Compute attention based on geometric relationships."""
        features_norm = F.normalize(features, dim=-1)

        geometric_scores = []
        for i, scale_feat in enumerate(scale_features):
            scale_feat_norm = F.normalize(scale_feat, dim=-1)

            cos_sim = (features_norm * scale_feat_norm).sum(dim=-1, keepdim=True)
            angles = torch.acos(cos_sim.clamp(-1 + 1e-7, 1 - 1e-7))

            geometric_scores.append(angles)

        angles_stack = torch.cat(geometric_scores, dim=1)
        attention = torch.exp(-angles_stack / self.temperature.abs())

        return attention

    def _compute_cayley_attention(
            self,
            features: torch.Tensor,
            scale_features: List[torch.Tensor]
    ) -> torch.Tensor:
        """Compute attention based on Cayley-Menger volumes."""
        volume_scores = []

        for scale_feat in scale_features:
            points = [features, scale_feat]

            for j in range(3):
                angle = (j + 1) * math.pi / 4
                rot_feat = features * math.cos(angle) + scale_feat * math.sin(angle)
                points.append(rot_feat)

            simplex = torch.stack(points, dim=1)
            diff = simplex.unsqueeze(2) - simplex.unsqueeze(1)
            distsq = (diff * diff).sum(dim=-1)

            volume_proxy = distsq.mean(dim=(1, 2))
            volume_scores.append(volume_proxy.unsqueeze(1))

        volumes = torch.cat(volume_scores, dim=1)
        attention = F.softmax(volumes / self.temperature.abs(), dim=-1)

        return attention

    def _compute_multihead_attention(
            self,
            features: torch.Tensor,
            scale_features: List[torch.Tensor]
    ) -> torch.Tensor:
        """Standard multi-head attention over scales."""
        B = features.shape[0]

        Q = self.q_proj(features)
        Q = Q.view(B, self.num_heads, self.head_dim)

        K_list, V_list = [], []
        for i, scale_feat in enumerate(scale_features):
            # Add embedding
            scale_feat_emb = scale_feat + self.scale_embeddings[i]

            K = self.k_proj(scale_feat_emb).view(B, self.num_heads, self.head_dim)
            V = self.v_proj(scale_feat_emb).view(B, self.num_heads, self.head_dim)

            K_list.append(K.unsqueeze(2))
            V_list.append(V.unsqueeze(2))

        K = torch.cat(K_list, dim=2)
        V = torch.cat(V_list, dim=2)

        scores = torch.einsum('bhd,bhsd->bhs', Q, K) / math.sqrt(self.head_dim)
        attn = F.softmax(scores / self.temperature.abs(), dim=-1)
        attn = self.dropout(attn)

        attn_avg = attn.mean(dim=1)

        return attn_avg

    def forward(
            self,
            features: torch.Tensor,
            logits_list: List[torch.Tensor],
            scale_features: Optional[List[torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            features: Base features [B, feature_dim]
            logits_list: List of scale logits [B, num_classes]
            scale_features: List of scale-specific features [B, scale_i]

        Returns:
            combined_logits, attention_weights
        """
        B = features.shape[0]

        if scale_features is None:
            scale_features = [features] * len(logits_list)

        # ✓ FIXED: Project all scales to common dimension
        if self.scale_to_common is not None:
            scale_features = [
                self.scale_to_common[i](feat)
                for i, feat in enumerate(scale_features)
            ]

        attention_types = []

        # 1. Standard multi-head attention
        mha_attention = self._compute_multihead_attention(features, scale_features)
        attention_types.append(mha_attention)

        # 2. Geometric attention
        if self.use_angular:
            geo_attention = self._compute_geometric_attention(features, scale_features)
            geo_attention = F.softmax(geo_attention, dim=-1)
            attention_types.append(geo_attention)

        # 3. Cayley-Menger volume attention
        if self.use_cayley:
            cayley_attention = self._compute_cayley_attention(features, scale_features)
            attention_types.append(cayley_attention)

        # Combine attention types with learnable weights
        attn_weights = F.softmax(self.attention_weights[:len(attention_types)], dim=0)
        combined_attention = sum(
            w * attn for w, attn in zip(attn_weights, attention_types)
        )

        # Optional: Apply scale-dimensional awareness
        if self.scale_dim_aware and self.scales is not None:
            scale_dims = torch.tensor(
                self.scales,
                device=features.device, dtype=torch.float32
            )
            dim_weights = scale_dims / scale_dims.sum()
            combined_attention = combined_attention * dim_weights.unsqueeze(0)

        # Normalize
        combined_attention = combined_attention / combined_attention.sum(dim=-1, keepdim=True)

        # Apply attention to logits
        combined_logits = torch.zeros_like(logits_list[0])
        for i, logits in enumerate(logits_list):
            combined_logits += combined_attention[:, i:i + 1] * logits

        return combined_logits, combined_attention


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
        """Initialize David from config with optional parameter overrides."""
        super().__init__()

        # Use config as base, override with explicit params
        if config is None:
            config = DavidArchitectureConfig()

        self.feature_dim = feature_dim or config.feature_dim
        self.num_classes = num_classes or config.num_classes
        self.scales = scales or config.scales
        self.use_belly = use_belly if use_belly is not None else config.use_belly
        self.belly_expand = belly_expand or config.belly_expand
        self.projection_temperature = config.projection_temperature
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
                    belly_expand=self.belly_expand,
                    temperature=self.projection_temperature
                )
                for scale in self.scales
            })

        elif self.sharing_mode == SharingMode.PARTIAL_SHARED:
            self.shared_base = nn.Linear(self.feature_dim, shared_feature_dim)
            self.heads = nn.ModuleDict({
                str(scale): ScaleSpecificHead(
                    shared_feature_dim, scale,
                    use_belly=self.use_belly,
                    belly_expand=self.belly_expand,
                    temperature=self.projection_temperature
                )
                for scale in self.scales
            })

        elif self.sharing_mode == SharingMode.DECOUPLED:
            self.heads = nn.ModuleDict({
                str(scale): ScaleSpecificHead(
                    self.feature_dim, scale,
                    use_belly=self.use_belly,
                    belly_expand=self.belly_expand,
                    temperature=self.projection_temperature
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
                               belly_expand=self.belly_expand,
                               temperature=self.projection_temperature
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
                               belly_expand=self.belly_expand,
                               temperature=self.projection_temperature
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

        elif self.fusion_mode == FusionMode.GEOMETRIC_ATTENTION:
            self.fusion = GeometricAttentionGate(
                fusion_input_dim,
                len(self.scales),
                scales=self.scales,  # ✓ ADDED: Pass actual scale dimensions
                num_heads=4,
                use_cayley_attention=True,
                use_angular_attention=True,
                temperature=temperature,
                dropout=dropout
            )

        elif self.fusion_mode == FusionMode.CANTOR_SCALE:
            self.fusion = CantorScaleFusion(
                fusion_input_dim,
                self.scales,  # ✓ Already passing scales
                num_heads=4,
                cantor_depth=8,
                local_window=3,
                temperature=temperature,
                dropout=dropout
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
        self, features: torch.Tensor, logits_list: List[torch.Tensor],
        features_list: Optional[List[torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Combine multi-scale predictions."""

        if len(logits_list) == 1:
            return logits_list[0], torch.ones(1, device=features.device)

        if self.fusion_mode in [FusionMode.ATTENTION, FusionMode.GATED,
                                FusionMode.HIERARCHICAL_TREE, FusionMode.DEEP_EFFICIENCY]:
            return self.fusion(features, logits_list)

        elif self.fusion_mode in [FusionMode.GEOMETRIC_ATTENTION, FusionMode.CANTOR_SCALE]:
            # These need scale features
            return self.fusion(features, logits_list, features_list)

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

        combined, fusion_weights = self._fuse_predictions(
            base_features, logits_list, features_list
        )

        if return_all_scales:
            return combined, logits_list, features_list, fusion_weights
        else:
            return combined, features_list[0] if features_list else base_features

    # ========================================================================
    # CLASS METHODS
    # ========================================================================

    @classmethod
    def from_config(cls, config: DavidArchitectureConfig, **kwargs) -> 'David':
        """Create David from config with optional overrides."""
        return cls(config=config, **kwargs)

    @classmethod
    def from_preset(cls, preset_name: str, **kwargs) -> 'David':
        """Create David from preset name with optional overrides."""
        config = DavidPresets.get_preset(preset_name)
        return cls(config=config, **kwargs)

    @classmethod
    def from_json(cls, path: str, **kwargs) -> 'David':
        """Load David from JSON config file with optional overrides."""
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

    # Method 2: Test GeometricAttentionGate
    print("\n[2] With GeometricAttentionGate:")
    david = David.from_preset('balanced', fusion_mode='geometric_attention')
    print(f"  Fusion: {david.fusion_mode.value}")

    # Method 3: Test CantorScaleFusion
    print("\n[3] With CantorScaleFusion:")
    david = David.from_preset('balanced', fusion_mode='cantor_scale')
    print(f"  Fusion: {david.fusion_mode.value}")
    print(f"  Cantor coords: {david.fusion.scale_cantor_coords}")
    print(f"  Scale routes:\n{david.fusion.scale_routes}")

    print("\n" + "="*80)
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