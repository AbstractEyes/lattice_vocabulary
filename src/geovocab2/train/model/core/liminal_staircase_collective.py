"""
Liminal Staircase - Vision-to-Text Token Prediction Collective
==========================================================

Multi-expert collective predicting text token sequences from images through
democratic consensus between SigLIP vision and Illustrious CLIP text encoders.

ARCHITECTURAL PHILOSOPHY:
-------------------------
This is NOT a hierarchical system. This is a DEMOCRACY of experts.

Each layer from each encoder = One independent expert with an opinion
- SigLIP Vision layers 0-23: 24 vision experts (PRIMARY, STRONG)
- CLIP Text layers 0-11: 12 text experts (AUXILIARY, NOISY)
- Total: ~36 independent voters in the collective

CONSENSUS PROCESS:
------------------
1. Each expert receives layer features
2. Tokens match to nearest pentachoron anchor points (semantic space)
3. Cantor attention uses POSITIONAL routing (not pentachoron-based)
4. Each expert forms multi-scale opinion vectors
5. Shallow fusion aggregates all expert opinions (DEMOCRATIC)
6. Multi-level Cantor attention scaffolds output sequence
7. Shared vocabulary projection → predict 77 tokens
8. Collective votes → Final token sequence

TASK:
-----
Predict text tokens that describe an image.

Input:  Image (→ SigLIP vision) + optional text prompt (→ CLIP text)
Output: 77 text tokens [B, 77, vocab_size]

KEY INNOVATIONS:
----------------
- Multi-level Cantor attention (positional routing)
- Geometric opinion anchors (pentachora for semantic space)
- Multi-expert consensus: All layers vote independently
- Shallow fusion: Democratic aggregation, NOT hierarchical
- Shared vocabulary projection (compact!)
- O(n) complexity with pre-computed routes

Author: AbstractPhil + Claude Sonnet 4.5
Date: 2025-11-12
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import math


# ============================================================================
# CANTOR ATTENTION (Positional Routing)
# ============================================================================

@dataclass
class CantorAttentionConfig:
    """Configuration for Cantor Global Attention."""
    dim: int = 512
    num_heads: int = 8
    head_dim: Optional[int] = None
    local_window: int = 64
    adaptive_window: bool = True
    min_window: int = 16
    max_window: int = 128
    sparsity_target: float = 0.15
    dropout: float = 0.1
    causal: bool = False
    qkv_bias: bool = True
    out_bias: bool = True

    def __post_init__(self):
        if self.head_dim is None:
            assert self.dim % self.num_heads == 0
            self.head_dim = self.dim // self.num_heads

    def get_window_size(self, seq_len: int) -> int:
        if not self.adaptive_window:
            return self.local_window
        adaptive_k = int(seq_len * self.sparsity_target)
        return max(self.min_window, min(adaptive_k, self.max_window))


class CantorAttention(nn.Module):
    """
    O(n) attention with POSITIONAL Cantor routing.

    Routes are based on sequence positions (0, 1, 2, ...), NOT pentachoron IDs.
    Pentachoron matching is separate (for semantic opinion formation only).
    """

    def __init__(
        self,
        config: CantorAttentionConfig,
        max_seq_len: int = 512,
        k: int = 64
    ):
        super().__init__()
        self.config = config
        self.dim = config.dim
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.k = k

        self.qkv = nn.Linear(config.dim, 3 * config.dim, bias=config.qkv_bias)
        self.out_proj = nn.Linear(config.dim, config.dim, bias=config.out_bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Pre-compute routes for common sequence lengths
        self.routes_cache = {}
        common_lengths = [64, 77, 128, 196, 256, 384, 512]
        for seq_len in common_lengths:
            if seq_len <= max_seq_len:
                routes = self._build_positional_routes(seq_len, k)
                self.register_buffer(f'routes_{seq_len}', routes)
                self.routes_cache[seq_len] = routes

    def _compute_cantor_coord(self, position: int, seq_len: int, depth: int = 8) -> float:
        """Compute Cantor coordinate for a sequence position."""
        x = position / max(1, seq_len - 1)
        x = max(1e-6, min(x, 1.0 - 1e-6))

        cantor_val = 0.0
        factor = 0.5

        for _ in range(depth):
            x_scaled = x * 3.0
            digit = int(x_scaled)
            x_frac = x_scaled - digit

            if digit == 2:
                cantor_val += factor

            x = x_frac
            factor *= 0.5

        return cantor_val

    def _build_positional_routes(self, seq_len: int, k: int) -> torch.Tensor:
        """Build k-NN routes based on positional Cantor coordinates."""
        coords = torch.tensor([
            self._compute_cantor_coord(pos, seq_len)
            for pos in range(seq_len)
        ], dtype=torch.float32)

        distances = torch.abs(
            coords.unsqueeze(1) - coords.unsqueeze(0)
        )

        _, routes = torch.topk(distances, k, dim=1, largest=False)
        return routes

    def _get_routes_for_seq_len(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Get or build routes for sequence length."""
        if seq_len in self.routes_cache:
            return self.routes_cache[seq_len].to(device)

        # Build on-demand for non-cached lengths
        routes = self._build_positional_routes(seq_len, self.k)
        return routes.to(device)

    def _sparse_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        routes: torch.Tensor
    ) -> torch.Tensor:
        """Sparse attention using positional routes."""
        batch_size, num_heads, seq_len, head_dim = q.shape
        k_neighbors = routes.shape[-1]
        device = q.device

        if routes.dim() == 2:
            routes = routes.unsqueeze(0).expand(batch_size, -1, -1)

        batch_idx = torch.arange(batch_size, device=device).view(-1, 1, 1, 1)
        head_idx = torch.arange(num_heads, device=device).view(1, -1, 1, 1)
        routes_bc = routes.unsqueeze(1)

        batch_idx = batch_idx.expand(batch_size, num_heads, seq_len, k_neighbors)
        head_idx = head_idx.expand(batch_size, num_heads, seq_len, k_neighbors)
        routes_bc = routes_bc.expand(batch_size, num_heads, seq_len, k_neighbors)

        k_gathered = k[batch_idx, head_idx, routes_bc, :]
        v_gathered = v[batch_idx, head_idx, routes_bc, :]

        scores = torch.einsum('bhqd,bhqkd->bhqk', q, k_gathered) * self.scale

        if self.config.causal:
            position_idx = torch.arange(seq_len, device=device).unsqueeze(1)
            causal_mask = routes > position_idx.unsqueeze(0)
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, num_heads, -1, -1)
            scores = scores.masked_fill(causal_mask, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        output = torch.einsum('bhqk,bhqkd->bhqd', attn_weights, v_gathered)
        return output

    def forward(
        self,
        x: torch.Tensor,
        anchor_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward with positional Cantor routing.

        Args:
            x: Input features [B, seq_len, dim]
            anchor_ids: Ignored (kept for API compatibility)

        Returns:
            output: [B, seq_len, dim]
        """
        batch_size, seq_len, _ = x.shape

        qkv = self.qkv(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        routes = self._get_routes_for_seq_len(seq_len, x.device)
        attn_output = self._sparse_attention(q, k, v, routes)

        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.dim)
        output = self.out_proj(attn_output)
        output = self.resid_dropout(output)

        return output


# ============================================================================
# GEOMETRIC POSITIONAL FINGERPRINTING (for pentachora)
# ============================================================================

class GeometricPositionalFingerprinter(nn.Module):
    """Computes deterministic Cantor positions from pentachoron geometry."""

    def __init__(
        self,
        cantor_depth: int = 8,
        volume_scale: float = 10.0,
        volume_weight: float = 0.4,
        edge_weight: float = 0.3,
        spread_weight: float = 0.3,
        epsilon: float = 1e-6
    ):
        super().__init__()
        self.cantor_depth = cantor_depth
        self.volume_scale = volume_scale
        self.volume_weight = volume_weight
        self.edge_weight = edge_weight
        self.spread_weight = spread_weight
        self.epsilon = epsilon

        # Validate weights
        total_weight = volume_weight + edge_weight + spread_weight
        assert abs(total_weight - 1.0) < 1e-5, f"Weights must sum to 1.0, got {total_weight}"

    def compute_cayley_menger_volume(self, vertices: torch.Tensor) -> torch.Tensor:
        """
        Compute pentachoron (4-simplex) volume via Cayley-Menger determinant.

        For n-dimensional simplex: V² = (-1)^(n+1) / (2^n × (n!)²) × det(M)
        For 4-simplex (pentachoron): 1 / (2^4 × 4!²) = 1 / (16 × 576) = 1 / 9216
        """
        diff = vertices.unsqueeze(0) - vertices.unsqueeze(1)
        dist_sq = (diff ** 2).sum(dim=-1)

        # Cayley-Menger matrix: bordered matrix of squared distances
        M = torch.zeros(6, 6, device=vertices.device, dtype=vertices.dtype)
        M[0, 1:] = 1.0
        M[1:, 0] = 1.0
        M[1:, 1:] = dist_sq

        det = torch.linalg.det(M)
        # 9216 = 2^4 × (4!)² for 4-simplex volume formula
        volume_sq = (-det / 9216.0).clamp(min=0.0)
        return volume_sq.sqrt()

    def compute_edge_statistics(self, vertices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute edge length statistics."""
        diff = vertices.unsqueeze(0) - vertices.unsqueeze(1)
        dist_sq = (diff ** 2).sum(dim=-1)

        triu_indices = torch.triu_indices(5, 5, offset=1, device=vertices.device)
        edge_lengths = dist_sq[triu_indices[0], triu_indices[1]].sqrt()

        return edge_lengths.mean(), edge_lengths.std()

    def compute_vertex_spread(self, vertices: torch.Tensor) -> torch.Tensor:
        """Compute vertex spatial distribution."""
        centroid = vertices.mean(dim=0)
        distances = torch.norm(vertices - centroid, dim=-1)
        return distances.std()

    def geometry_to_cantor_position(self, vertices: torch.Tensor) -> torch.Tensor:
        """
        Convert pentachoron geometry to Cantor position [0,1].

        Uses PURE ternary Cantor set iteration (no perturbations) for:
        - Deterministic mapping from geometry to position
        - O(n) global attentiveness via fractal self-similarity
        - Multi-scale semantic clustering in containment zones

        Returns:
            Cantor coordinate in [0, 1] representing position in fractal structure
        """
        # Extract geometric features
        volume = self.compute_cayley_menger_volume(vertices)
        mean_edge, std_edge = self.compute_edge_statistics(vertices)
        spread = self.compute_vertex_spread(vertices)

        # Normalize features to [0, 1] via sigmoid
        volume_norm = torch.sigmoid(volume * self.volume_scale)
        edge_ratio = torch.sigmoid(std_edge / (mean_edge + self.epsilon))
        spread_norm = torch.sigmoid(spread)

        # Weighted combination as seed for Cantor iteration
        seed = (
            volume_norm * self.volume_weight +
            edge_ratio * self.edge_weight +
            spread_norm * self.spread_weight
        ).clamp(self.epsilon, 1.0 - self.epsilon)

        # PURE ternary Cantor set iteration (NO perturbations)
        # Each iteration: divide [0,1] into thirds, keep left/right, remove middle
        # This creates fractal structure with 2^depth containment zones
        x = seed
        cantor_val = 0.0
        factor = 0.5

        for _ in range(self.cantor_depth):
            # Ternary expansion: x ∈ [0,1] → digit ∈ {0,1,2}
            x_scaled = x * 3.0
            digit = x_scaled.long()
            x_frac = x_scaled - digit.float()

            # Cantor set: keep segments where digit ∈ {0, 2}, remove middle (digit=1)
            # Encode position: 0 → left branch, 2 → right branch
            middle_bit = (digit == 2).float()
            cantor_val = cantor_val + middle_bit * factor

            # Pure iteration: only use fractional part (no perturbations!)
            x = x_frac
            factor *= 0.5

        return cantor_val.clamp(0.0, 1.0)

    def compute_vocabulary_positions(
        self,
        pentachora: torch.Tensor,
        batch_size: int = 256
    ) -> torch.Tensor:
        """
        Compute positional fingerprints for all pentachora (VECTORIZED).

        Args:
            pentachora: [vocab_size, 5, dim] pentachoron vertices
            batch_size: Process in batches to avoid OOM (default: 256)

        Returns:
            positions: [vocab_size] Cantor coordinates
        """
        vocab_size = pentachora.shape[0]
        positions = torch.zeros(vocab_size, device=pentachora.device)

        print(f"Computing {vocab_size} positional fingerprints (batched)...")

        # Process in batches to avoid OOM for large vocabularies
        num_batches = (vocab_size + batch_size - 1) // batch_size

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, vocab_size)

            # Process batch
            batch_positions = torch.stack([
                self.geometry_to_cantor_position(pentachora[i])
                for i in range(start_idx, end_idx)
            ])

            positions[start_idx:end_idx] = batch_positions

            # Progress reporting
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == num_batches:
                print(f"  Batch {batch_idx + 1}/{num_batches} ({end_idx}/{vocab_size})")

        return positions


# ============================================================================
# MULTI-SCALE EXPERT COMPANION
# ============================================================================

class MultiScaleExpertCompanion(nn.Module):
    """Single expert companion for one encoder layer."""

    def __init__(
        self,
        layer_name: str,
        input_dim: int,
        pentachoron_dim: int,
        scales: List[int],
        num_heads: int,
        dropout: float,
        shared_pentachora: torch.Tensor,
        max_seq_len: int = 512
    ):
        super().__init__()

        self.layer_name = layer_name
        self.input_dim = input_dim
        self.pentachoron_dim = pentachoron_dim
        self.scales = scales

        self.register_buffer('shared_pentachora', shared_pentachora)

        pentachora_centroids = shared_pentachora.mean(dim=1)
        self.register_buffer('pentachora_centroids', F.normalize(pentachora_centroids, dim=-1))

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, pentachoron_dim),
            nn.LayerNorm(pentachoron_dim),
            nn.GELU()
        )

        # Cantor attention with POSITIONAL routing
        cantor_config = CantorAttentionConfig(
            dim=pentachoron_dim,
            num_heads=num_heads,
            adaptive_window=True,
            sparsity_target=0.15,
            dropout=dropout
        )

        k_neighbors = int(225 * 0.15)  # 15% sparsity of anchors (for reference)
        k_neighbors = max(16, min(k_neighbors, 64))

        self.cantor_attention = CantorAttention(
            cantor_config,
            max_seq_len=max_seq_len,
            k=k_neighbors
        )

        self.scale_projectors = nn.ModuleDict()
        for scale in scales:
            self.scale_projectors[str(scale)] = nn.Sequential(
                nn.Linear(pentachoron_dim, scale * 2),
                nn.LayerNorm(scale * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(scale * 2, scale)
            )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def match_to_opinion_anchors(self, features: torch.Tensor) -> torch.Tensor:
        """Match tokens to nearest pentachoron anchors (semantic space)."""
        similarities = torch.matmul(features, self.pentachora_centroids.T)
        return similarities.argmax(dim=-1)

    def forward(
        self,
        sequence_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Form expert opinion through geometric routing."""
        B, seq_len, _ = sequence_features.shape

        z = self.input_proj(sequence_features)
        z = F.normalize(z, dim=-1)

        # Match to opinion anchors (semantic space - for reference only)
        anchor_ids = self.match_to_opinion_anchors(z)

        # Cantor attention with POSITIONAL routing (anchor_ids not used)
        z_attended = self.cantor_attention(z, anchor_ids)

        # Pool for opinion formation
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).float()
            z_pooled = (z_attended * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        else:
            z_pooled = z_attended.mean(dim=1)

        # Multi-scale opinion extraction
        scale_opinions = {}
        for scale in self.scales:
            opinion = self.scale_projectors[str(scale)](z_pooled)
            scale_opinions[scale] = opinion

        return {
            'scale_opinions': scale_opinions,
            'pooled_features': z_pooled
        }


# ============================================================================
# SHALLOW TOKEN PREDICTION FUSION (Multi-level Cantor)
# ============================================================================

class ShallowTokenPredictionFusion(nn.Module):
    """
    Shallow fusion with MULTI-LEVEL Cantor attention scaffolding.

    Multi-level Cantor architecture:
    1. Expert level: Cantor attention within input sequences ✓
    2. Fusion level: Democratic aggregation to collective opinion ✓
    3. OUTPUT LEVEL: Cantor attention between 77 token positions ✓
    4. Classification: Shared vocabulary head per position
    """

    def __init__(
        self,
        num_experts: int,
        scales: List[int],
        vocab_size: int,
        max_seq_len: int = 77,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()

        self.num_experts = num_experts
        self.scales = scales
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len

        self.expert_weights = nn.ParameterDict({
            str(scale): nn.Parameter(torch.ones(num_experts) / num_experts)
            for scale in scales
        })

        self.scale_weights = nn.Parameter(torch.ones(len(scales)) / len(scales))

        # Output sequence modules per scale
        self.scale_output_modules = nn.ModuleDict()

        print("\nInitializing multi-level Cantor output scaffolding...")
        for scale in scales:
            print(f"  Scale {scale}:")

            # Position embeddings
            position_embeds = nn.Parameter(torch.randn(max_seq_len, scale))
            nn.init.normal_(position_embeds, std=0.02)
            self.register_parameter(f'position_embeds_{scale}', position_embeds)

            # Cantor attention for output sequence (POSITIONAL routing)
            cantor_config = CantorAttentionConfig(
                dim=scale,
                num_heads=num_heads,
                adaptive_window=False,
                local_window=32,
                dropout=dropout
            )

            k_neighbors = min(32, max_seq_len // 2)
            cantor_attn = CantorAttention(
                cantor_config,
                max_seq_len=max_seq_len,
                k=k_neighbors
            )
            self.scale_output_modules[f'cantor_{scale}'] = cantor_attn
            print(f"    ✓ Cantor attention: k={k_neighbors} neighbors")

            # Shared vocabulary projection
            vocab_head = nn.Sequential(
                nn.LayerNorm(scale),
                nn.Linear(scale, vocab_size)
            )
            self.scale_output_modules[f'vocab_{scale}'] = vocab_head

            vocab_params = scale * vocab_size
            print(f"    ✓ Shared vocab head: {vocab_params:,} params")

        self._init_weights()

    def _init_weights(self):
        for module in self.scale_output_modules.values():
            if isinstance(module, nn.Sequential):
                for m in module.modules():
                    if isinstance(m, nn.Linear):
                        nn.init.xavier_uniform_(m.weight, gain=0.5)
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)

    def forward(
        self,
        expert_opinions: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """Multi-level Cantor fusion to token predictions."""
        batch_size = list(expert_opinions[0]['scale_opinions'].values())[0].shape[0]
        device = list(expert_opinions[0]['scale_opinions'].values())[0].device

        scale_token_logits = {}

        for scale in self.scales:
            # 1. Collect expert opinions
            expert_ops = torch.stack([
                exp['scale_opinions'][scale]
                for exp in expert_opinions
            ], dim=0)

            # 2. Democratic aggregation
            weights = F.softmax(self.expert_weights[str(scale)], dim=0)
            weights = weights.view(-1, 1, 1)
            collective_opinion = (expert_ops * weights).sum(dim=0)

            # 3. Expand to 77 positions
            position_embeds = getattr(self, f'position_embeds_{scale}')
            token_features = collective_opinion.unsqueeze(1) + position_embeds.unsqueeze(0)

            # 4. Cantor attention between positions (POSITIONAL routing)
            cantor_attn = self.scale_output_modules[f'cantor_{scale}']
            attended_features = cantor_attn(token_features)

            # 5. Shared vocabulary projection
            vocab_head = self.scale_output_modules[f'vocab_{scale}']
            token_logits = vocab_head(attended_features)

            scale_token_logits[scale] = token_logits

        # 6. Fuse across scales
        scale_weights = F.softmax(self.scale_weights, dim=0)

        logits_list = [scale_token_logits[scale] for scale in self.scales]
        logits_stacked = torch.stack(logits_list, dim=0)

        weights_expanded = scale_weights.view(-1, 1, 1, 1)
        final_token_logits = (logits_stacked * weights_expanded).sum(dim=0)

        return {
            'token_logits': final_token_logits,
            'scale_token_logits': scale_token_logits
        }


# ============================================================================
# LIMINAL STAIRCASE - MAIN COLLECTIVE
# ============================================================================

@dataclass
class LiminalStaircaseConfig:
    """Configuration for Liminal Staircase."""

    num_opinion_anchors: int = 225
    pentachoron_dim: int = 512
    cantor_depth: int = 8

    scales: List[int] = None

    siglip_hidden_dim: int = 1664
    siglip_num_layers: int = 24

    clip_hidden_dim: int = 768
    clip_num_layers: int = 12
    clip_skip: int = 2

    vocab_size: int = 49408
    max_seq_len: int = 77

    num_heads: int = 8
    dropout: float = 0.1

    # Geometric fingerprinting parameters
    geometry_volume_scale: float = 10.0
    geometry_volume_weight: float = 0.4
    geometry_edge_weight: float = 0.3
    geometry_spread_weight: float = 0.3
    geometry_epsilon: float = 1e-6  # Numerical stability for division

    def __post_init__(self):
        if self.scales is None:
            self.scales = [128, 256, 512]

        # Validate geometry weights sum to 1.0
        total_weight = self.geometry_volume_weight + self.geometry_edge_weight + self.geometry_spread_weight
        assert abs(total_weight - 1.0) < 1e-5, f"Geometry weights must sum to 1.0, got {total_weight}"


class LiminalStaircase(nn.Module):
    """
    Liminal Staircase - Vision-to-Text Token Prediction Collective

    Multi-expert democratic system with multi-level Cantor attention.
    """

    def __init__(self, config: LiminalStaircaseConfig):
        super().__init__()

        self.config = config

        print("=" * 80)
        print("LIMINAL STAIRCASE - Vision-to-Text Token Prediction")
        print("=" * 80)
        print(f"Opinion anchors: {config.num_opinion_anchors}")
        print(f"SigLIP Vision experts: {config.siglip_num_layers} (PRIMARY)")

        active_clip_layers = config.clip_num_layers - config.clip_skip
        print(f"CLIP Text experts: {active_clip_layers} (AUXILIARY, skip={config.clip_skip})")
        print(f"Total experts: {config.siglip_num_layers + active_clip_layers}")
        print(f"Token prediction: {config.max_seq_len} tokens, vocab={config.vocab_size}")

        # Initialize opinion anchors
        print("\nInitializing opinion anchors...")
        self.opinion_anchors = self._init_opinion_anchors()
        print(f"✓ {config.num_opinion_anchors} pentachora created")

        # Compute positional fingerprints (for semantic reference)
        print("\nComputing positional fingerprints...")
        self.fingerprinter = GeometricPositionalFingerprinter(
            cantor_depth=config.cantor_depth,
            volume_scale=config.geometry_volume_scale,
            volume_weight=config.geometry_volume_weight,
            edge_weight=config.geometry_edge_weight,
            spread_weight=config.geometry_spread_weight,
            epsilon=config.geometry_epsilon
        )
        self.anchor_positions = self.fingerprinter.compute_vocabulary_positions(
            self.opinion_anchors
        )
        print(f"✓ Positions: [{self.anchor_positions.min():.4f}, {self.anchor_positions.max():.4f}]")

        # SigLIP Vision experts
        print("\nCreating SigLIP Vision experts...")
        self.siglip_experts = nn.ModuleDict()
        for i in range(config.siglip_num_layers):
            expert = MultiScaleExpertCompanion(
                layer_name=f'siglip_layer_{i}',
                input_dim=config.siglip_hidden_dim,
                pentachoron_dim=config.pentachoron_dim,
                scales=config.scales,
                num_heads=config.num_heads,
                dropout=config.dropout,
                shared_pentachora=self.opinion_anchors,
                max_seq_len=512
            )
            self.siglip_experts[f'siglip_layer_{i}'] = expert
            if (i + 1) % 6 == 0:
                print(f"  ✓ Layers 0-{i}")

        # CLIP Text experts
        print(f"\nCreating CLIP Text experts (skip last {config.clip_skip})...")
        self.clip_experts = nn.ModuleDict()
        for i in range(active_clip_layers):
            expert = MultiScaleExpertCompanion(
                layer_name=f'clip_layer_{i}',
                input_dim=config.clip_hidden_dim,
                pentachoron_dim=config.pentachoron_dim,
                scales=config.scales,
                num_heads=config.num_heads,
                dropout=config.dropout,
                shared_pentachora=self.opinion_anchors,
                max_seq_len=512
            )
            self.clip_experts[f'clip_layer_{i}'] = expert
        print(f"  ✓ {len(self.clip_experts)} CLIP text experts")

        # Shallow fusion
        print("\nCreating shallow token prediction fusion...")
        total_experts = len(self.siglip_experts) + len(self.clip_experts)
        self.fusion = ShallowTokenPredictionFusion(
            num_experts=total_experts,
            scales=config.scales,
            vocab_size=config.vocab_size,
            max_seq_len=config.max_seq_len,
            dropout=config.dropout
        )
        print(f"✓ Fusion: {total_experts} experts → {config.max_seq_len} tokens (SHALLOW)")

        total_params = sum(p.numel() for p in self.parameters())
        print(f"\n{'='*80}")
        print(f"Total parameters: {total_params:,}")
        print(f"{'='*80}\n")

    def _init_opinion_anchors(self) -> nn.Parameter:
        """Initialize pentachoron opinion anchors."""
        pentachora = torch.randn(
            self.config.num_opinion_anchors,
            5,
            self.config.pentachoron_dim
        )

        pentachora = F.normalize(pentachora, dim=-1)

        for i in range(self.config.num_opinion_anchors):
            perturbation = torch.randn_like(pentachora[i]) * 0.1
            pentachora[i] = pentachora[i] + perturbation
            pentachora[i] = F.normalize(pentachora[i], dim=-1)

        return nn.Parameter(pentachora, requires_grad=True)

    def forward(
        self,
        siglip_features: Dict[str, torch.Tensor],
        clip_features: Optional[Dict[str, torch.Tensor]] = None,
        siglip_masks: Optional[Dict[str, torch.Tensor]] = None,
        clip_masks: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """Predict tokens from vision and optional text."""
        expert_opinions = []

        # Collect SigLIP opinions
        for layer_name, features in siglip_features.items():
            if layer_name in self.siglip_experts:
                mask = siglip_masks.get(layer_name) if siglip_masks else None
                opinion = self.siglip_experts[layer_name](features, mask)
                expert_opinions.append(opinion)

        # Collect CLIP opinions
        if clip_features is not None:
            for layer_name, features in clip_features.items():
                if layer_name in self.clip_experts:
                    mask = clip_masks.get(layer_name) if clip_masks else None
                    opinion = self.clip_experts[layer_name](features, mask)
                    expert_opinions.append(opinion)

        # Shallow fusion → token prediction
        output = self.fusion(expert_opinions)

        return output

    def get_info(self) -> Dict:
        """Get model info."""
        return {
            'num_opinion_anchors': self.config.num_opinion_anchors,
            'siglip_experts': len(self.siglip_experts),
            'clip_experts': len(self.clip_experts),
            'total_experts': len(self.siglip_experts) + len(self.clip_experts),
            'vocab_size': self.config.vocab_size,
            'max_seq_len': self.config.max_seq_len,
            'clip_skip': self.config.clip_skip,
            'scales': self.config.scales,
            'total_parameters': sum(p.numel() for p in self.parameters())
        }


# ============================================================================
# DEMO
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("LIMINAL STAIRCASE - DEMO")
    print("=" * 80 + "\n")

    config = LiminalStaircaseConfig(
        num_opinion_anchors=225,
        pentachoron_dim=512,
        scales=[128, 256, 512],
        siglip_num_layers=24,
        clip_num_layers=12,
        clip_skip=2,
        vocab_size=49408,
        max_seq_len=77
    )

    liminal_staircase_model = LiminalStaircase(config)

    # Test data
    batch_size = 4
    siglip_seq_len = 256
    clip_seq_len = 77

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    liminal_staircase_model = liminal_staircase_model.to(device)

    siglip_features = {
        f'siglip_layer_{i}': torch.randn(batch_size, siglip_seq_len, 1664, device=device)
        for i in range(24)
    }

    clip_features = {
        f'clip_layer_{i}': torch.randn(batch_size, clip_seq_len, 768, device=device)
        for i in range(10)
    }

    print(f"\n[TEST] Forward pass:")
    print(f"  SigLIP: {len(siglip_features)} layers × [B={batch_size}, N={siglip_seq_len}, D=1664]")
    print(f"  CLIP: {len(clip_features)} layers × [B={batch_size}, N={clip_seq_len}, D=768]")

    with torch.no_grad():
        output = liminal_staircase_model(siglip_features, clip_features)

    print(f"\n✓ Token prediction complete!")
    print(f"  Token logits: {output['token_logits'].shape}")
    print(f"  Expected: [B={batch_size}, seq={config.max_seq_len}, vocab={config.vocab_size}]")

    predicted_tokens = output['token_logits'].argmax(dim=-1)
    print(f"\n[EXAMPLE] Predicted tokens (first sample):")
    print(f"  First 10 tokens: {predicted_tokens[0, :10].tolist()}")

    print(f"\n{'='*80}")
    print("✅ LIMINAL STAIRCASE - Multi-Level Cantor Token Prediction")
    print(f"{'='*80}")
    print("\nKey Features:")
    print(f"  • {len(siglip_features) + len(clip_features)} independent experts")
    print(f"  • Multi-level Cantor attention (positional routing)")
    print(f"  • Geometric opinion anchors (semantic space)")
    print(f"  • O(n) complexity with pre-computed routes")
    print(f"  • Multi-scale opinions: {config.scales}")
    print(f"  • Shared vocabulary projection (compact!)")
    print(f"  • Output: {config.max_seq_len} tokens")
    print(f"{'='*80}\n")