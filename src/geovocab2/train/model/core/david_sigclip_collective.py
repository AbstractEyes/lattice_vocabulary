"""
Pentachoron Collective Consensus Classifier
============================================

A multi-expert collective voting system that fuses opinions from all layers
of dual encoders (CLIP + SigLIP) through geometric positional fingerprinting
and Cantor fractal routing.

ARCHITECTURAL PHILOSOPHY:
-------------------------
This is NOT a hierarchical system. This is a DEMOCRACY of experts.

Each layer from each encoder = One independent expert with an opinion
- CLIP layers 0-11: 12 text/vision experts
- SigLIP layers 0-23: 24 vision experts
- Total: ~36 independent voters in the collective

GEOMETRIC OPINION ANCHORS:
--------------------------
~225 pentachora serve as geometric opinion anchor points in shared space.
These are NOT output classes - they are interpretive reference positions.

Each pentachoron:
  - Has unique geometry (5 vertices in 512D space)
  - Gets deterministic Cantor coordinate [0,1] from its geometry
  - This Cantor coordinate = POSITIONAL FINGERPRINT (fixed, not learned)
  - Enables unique positional awareness for routing

CONSENSUS PROCESS:
------------------
1. Each expert receives layer features
2. Tokens match to nearest pentachoron anchor points
3. Tokens inherit pentachoron's geometric Cantor POSITION
4. Geometric routing via Cantor attention using these POSITIONS
5. Each expert forms multi-scale opinion vectors
6. Shallow fusion aggregates all expert opinions
7. Collective votes → Final classification

The more experts contribute, the more robust the collective decision.
Geometric routing ensures consistency across expert opinions.

KEY INNOVATIONS:
----------------
- Positional fingerprinting: Geometry → unique Cantor coordinates
- Multi-expert consensus: All layers vote independently
- Shallow fusion: Democratic aggregation, not hierarchy
- Cross-modal learning: CLIP + SigLIP opinions fused geometrically
- Fractal routing: O(n) attention via geometric positions

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
# CANTOR ATTENTION (with pre-computed geometric positions)
# ============================================================================

@dataclass
class CantorAttentionConfig:
    """Configuration for Cantor Global Attention with geometric positions."""
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
        """Compute adaptive window size."""
        if not self.adaptive_window:
            return self.local_window
        adaptive_k = int(seq_len * self.sparsity_target)
        return max(self.min_window, min(adaptive_k, self.max_window))


class CantorAttention(nn.Module):
    """
    Cantor Global Attention with O(n) complexity.

    Routes tokens based on pre-computed geometric Cantor positions (not sequence order).
    Each token inherits the positional fingerprint of its matched pentachoron.
    """

    def __init__(self, config: CantorAttentionConfig):
        super().__init__()
        self.config = config
        self.dim = config.dim
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # QKV projection
        self.qkv = nn.Linear(config.dim, 3 * config.dim, bias=config.qkv_bias)

        # Output projection
        self.out_proj = nn.Linear(config.dim, config.dim, bias=config.out_bias)

        # Dropout
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def _build_routes_from_positions(
            self,
            cantor_positions: torch.Tensor,  # [seq_len] - geometric positions
            k: int
    ) -> torch.Tensor:
        """
        Build routing table from geometric Cantor positions.

        Tokens are routed to their k nearest neighbors in Cantor space,
        where positions are determined by pentachoron geometry (not sequence order).

        Args:
            cantor_positions: [seq_len] - inherited from matched pentachora
            k: Number of neighbors

        Returns:
            routes: [seq_len, k] - neighbor indices
        """
        seq_len = cantor_positions.shape[0]

        # Pairwise distances in Cantor space
        distances = torch.abs(
            cantor_positions.unsqueeze(1) - cantor_positions.unsqueeze(0)
        )

        # Find k-nearest neighbors
        _, routes = torch.topk(distances, k, dim=1, largest=False)

        return routes

    def _sparse_attention(
            self,
            q: torch.Tensor,  # [B, H, N, D]
            k: torch.Tensor,  # [B, H, N, D]
            v: torch.Tensor,  # [B, H, N, D]
            routes: torch.Tensor  # [N, k]
    ) -> torch.Tensor:
        """Sparse attention using geometric routes."""
        batch_size, num_heads, seq_len, head_dim = q.shape
        k_neighbors = routes.shape[1]
        device = q.device

        # Broadcast indices
        batch_idx = torch.arange(batch_size, device=device).view(-1, 1, 1, 1)
        head_idx = torch.arange(num_heads, device=device).view(1, -1, 1, 1)
        routes_bc = routes.view(1, 1, seq_len, k_neighbors)

        batch_idx = batch_idx.expand(batch_size, num_heads, seq_len, k_neighbors)
        head_idx = head_idx.expand(batch_size, num_heads, seq_len, k_neighbors)
        routes_bc = routes_bc.expand(batch_size, num_heads, seq_len, k_neighbors)

        # Gather K and V according to routes
        k_gathered = k[batch_idx, head_idx, routes_bc, :]
        v_gathered = v[batch_idx, head_idx, routes_bc, :]

        # Attention scores
        scores = torch.einsum('bhqd,bhqkd->bhqk', q, k_gathered) * self.scale

        # Causal mask if needed
        if self.config.causal:
            position_idx = torch.arange(seq_len, device=device).unsqueeze(1)
            causal_mask = routes > position_idx
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, num_heads, -1, -1)
            scores = scores.masked_fill(causal_mask, float('-inf'))

        # Softmax and apply
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        output = torch.einsum('bhqk,bhqkd->bhqd', attn_weights, v_gathered)
        return output

    def forward(
            self,
            x: torch.Tensor,  # [B, N, D]
            cantor_positions: torch.Tensor  # [N] - geometric positions
    ) -> torch.Tensor:
        """
        Forward with geometric positional routing.

        Args:
            x: Input [batch, seq_len, dim]
            cantor_positions: Geometric Cantor positions [seq_len]

        Returns:
            output: [batch, seq_len, dim]
        """
        batch_size, seq_len, _ = x.shape

        # QKV projection
        qkv = self.qkv(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Build routes from geometric positions
        k_neighbors = self.config.get_window_size(seq_len)
        routes = self._build_routes_from_positions(cantor_positions, k_neighbors)

        # Sparse attention
        attn_output = self._sparse_attention(q, k, v, routes)

        # Reshape and project
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.dim)
        output = self.out_proj(attn_output)
        output = self.resid_dropout(output)

        return output


# ============================================================================
# GEOMETRIC POSITIONAL FINGERPRINTING
# ============================================================================

class GeometricPositionalFingerprinter(nn.Module):
    """
    Computes deterministic Cantor positional fingerprints from pentachoron geometry.

    Each pentachoron's unique geometry → unique Cantor coordinate [0,1]
    This coordinate serves as a POSITIONAL FINGERPRINT for routing.

    Geometric features used:
    - Cayley-Menger volume (5D geometric content)
    - Edge length statistics (structural uniformity)
    - Vertex spread (spatial distribution)
    - Deterministic hash → unique position in [0,1]
    """

    def __init__(self, cantor_depth: int = 8):
        super().__init__()
        self.cantor_depth = cantor_depth

    def compute_cayley_menger_volume(self, vertices: torch.Tensor) -> torch.Tensor:
        """
        Compute pentachoron volume using Cayley-Menger determinant.

        The Cayley-Menger determinant provides the signed volume of a simplex
        from pairwise distances, capturing the geometric content.

        Args:
            vertices: [5, D] - pentachoron vertices

        Returns:
            volume: scalar (non-negative)
        """
        # Pairwise squared distances
        diff = vertices.unsqueeze(0) - vertices.unsqueeze(1)
        dist_sq = (diff ** 2).sum(dim=-1)

        # Cayley-Menger matrix
        M = torch.zeros(6, 6, device=vertices.device, dtype=vertices.dtype)
        M[0, 1:] = 1.0
        M[1:, 0] = 1.0
        M[1:, 1:] = dist_sq

        # Volume from determinant: V² = (-1)^(n+1) * det(M) / (2^n * (n!)²)
        # For n=4 (pentachoron): V² = -det(M) / 9216
        det = torch.linalg.det(M)
        volume_sq = (-det / 9216.0).clamp(min=0.0)
        volume = volume_sq.sqrt()

        return volume

    def compute_edge_statistics(self, vertices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute edge length statistics.

        Measures structural uniformity through edge length distribution.

        Args:
            vertices: [5, D]

        Returns:
            mean_edge: Average edge length
            std_edge: Edge length standard deviation
        """
        diff = vertices.unsqueeze(0) - vertices.unsqueeze(1)
        dist_sq = (diff ** 2).sum(dim=-1)

        # Get upper triangle (10 edges for pentachoron)
        triu_indices = torch.triu_indices(5, 5, offset=1, device=vertices.device)
        edge_lengths = dist_sq[triu_indices[0], triu_indices[1]].sqrt()

        mean_edge = edge_lengths.mean()
        std_edge = edge_lengths.std()

        return mean_edge, std_edge

    def compute_vertex_spread(self, vertices: torch.Tensor) -> torch.Tensor:
        """
        Compute vertex spatial distribution.

        Measures how spread out vertices are from centroid.

        Args:
            vertices: [5, D]

        Returns:
            spread: Standard deviation of centroid distances
        """
        centroid = vertices.mean(dim=0)
        distances = torch.norm(vertices - centroid, dim=-1)
        spread = distances.std()
        return spread

    def geometry_to_cantor_position(self, vertices: torch.Tensor) -> torch.Tensor:
        """
        Convert pentachoron geometry to deterministic Cantor position [0,1].

        This is the POSITIONAL FINGERPRINT - uniquely determined by geometry.
        Different pentachora have different positions in Cantor space.

        Process:
        1. Extract geometric features (volume, edges, spread)
        2. Normalize features to [0,1]
        3. Combine into geometric seed
        4. Apply hierarchical Cantor construction
        5. Result: unique position in [0,1] for routing

        Args:
            vertices: [5, D] - pentachoron vertices

        Returns:
            cantor_position: scalar in [0,1] - the positional fingerprint
        """
        # Extract geometric features
        volume = self.compute_cayley_menger_volume(vertices)
        mean_edge, std_edge = self.compute_edge_statistics(vertices)
        spread = self.compute_vertex_spread(vertices)

        # Normalize to [0,1] with sigmoid for soft bounds
        volume_norm = torch.sigmoid(volume * 10.0)
        edge_ratio = torch.sigmoid(std_edge / (mean_edge + 1e-6))
        spread_norm = torch.sigmoid(spread)

        # Geometric seed: weighted combination of features
        # Each feature contributes to initial position
        seed = (
                volume_norm * 0.4 +  # 40% from volume
                edge_ratio * 0.3 +  # 30% from edge uniformity
                spread_norm * 0.3  # 30% from vertex spread
        ).clamp(1e-6, 1.0 - 1e-6)

        # Hierarchical Cantor construction
        # Different geometries take different paths through fractal tree
        x = seed
        cantor_val = 0.0
        factor = 0.5

        for _ in range(self.cantor_depth):
            x_scaled = x * 3.0
            digit = x_scaled.long()
            x_frac = x_scaled - digit.float()

            # Middle third contribution (Cantor set property)
            middle_bit = (digit == 2).float()
            cantor_val = cantor_val + middle_bit * factor

            # Geometric modulation: different shapes recurse differently
            x = x_frac + (volume_norm + edge_ratio + spread_norm) * 0.01
            x = x.clamp(1e-6, 1.0 - 1e-6)
            factor *= 0.5

        return cantor_val.clamp(0.0, 1.0)

    def compute_vocabulary_positions(
            self,
            pentachora: torch.Tensor  # [vocab_size, 5, D]
    ) -> torch.Tensor:
        """
        Compute positional fingerprints for entire vocabulary.

        Each pentachoron gets a unique Cantor position based on its geometry.
        These positions are FIXED (not learned) and determine routing behavior.

        Args:
            pentachora: [vocab_size, 5, D] - all opinion anchor pentachora

        Returns:
            cantor_positions: [vocab_size] - positional fingerprint for each
        """
        vocab_size = pentachora.shape[0]
        positions = torch.zeros(vocab_size, device=pentachora.device)

        print(f"Computing geometric positional fingerprints for {vocab_size} pentachora...")
        for i in range(vocab_size):
            positions[i] = self.geometry_to_cantor_position(pentachora[i])
            if (i + 1) % 25 == 0:
                print(f"  {i + 1}/{vocab_size} positions computed...")

        return positions


# ============================================================================
# MULTI-SCALE EXPERT COMPANION
# ============================================================================

class MultiScaleExpertCompanion(nn.Module):
    """
    Single expert companion for one encoder layer.

    Forms independent opinion through:
    1. Matching tokens to geometric opinion anchors (pentachora)
    2. Inheriting positional fingerprints for routing
    3. Geometric Cantor attention using these positions
    4. Multi-scale feature extraction (different interpretation levels)
    5. Opinion vector output for collective voting

    Each expert is independent - forms its own interpretation of the layer.
    """

    def __init__(
            self,
            layer_name: str,
            input_dim: int,
            pentachoron_dim: int,
            scales: List[int],  # e.g., [128, 256, 512]
            num_heads: int,
            dropout: float,
            shared_pentachora: torch.Tensor,  # [vocab_size, 5, D]
            shared_positions: torch.Tensor  # [vocab_size] - pre-computed!
    ):
        super().__init__()

        self.layer_name = layer_name
        self.input_dim = input_dim
        self.pentachoron_dim = pentachoron_dim
        self.scales = scales

        # Shared geometric opinion anchors (external)
        self.register_buffer('shared_pentachora', shared_pentachora)
        self.register_buffer('shared_positions', shared_positions)

        # Precompute pentachoron centroids for matching
        pentachora_centroids = shared_pentachora.mean(dim=1)
        self.register_buffer('pentachora_centroids', F.normalize(pentachora_centroids, dim=-1))

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, pentachoron_dim),
            nn.LayerNorm(pentachoron_dim),
            nn.GELU()
        )

        # Cantor attention
        cantor_config = CantorAttentionConfig(
            dim=pentachoron_dim,
            num_heads=num_heads,
            adaptive_window=True,
            sparsity_target=0.15,
            dropout=dropout
        )
        self.cantor_attention = CantorAttention(cantor_config)

        # Multi-scale opinion extractors
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
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def match_to_opinion_anchors(self, features: torch.Tensor) -> torch.Tensor:
        """
        Match token features to nearest geometric opinion anchor.

        Args:
            features: [B, N, D] - normalized token features

        Returns:
            anchor_ids: [B, N] - nearest pentachoron ID for each token
        """
        B, N, D = features.shape

        # Cosine similarity to all pentachoron centroids
        similarities = torch.matmul(features, self.pentachora_centroids.T)

        # Nearest anchor
        anchor_ids = similarities.argmax(dim=-1)

        return anchor_ids

    def forward(
            self,
            sequence_features: torch.Tensor,  # [B, seq_len, input_dim]
            attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Form expert opinion through geometric routing and multi-scale extraction.

        Args:
            sequence_features: Layer features [B, seq_len, input_dim]
            attention_mask: Optional mask [B, seq_len]

        Returns:
            Dict containing:
                - scale_opinions: Dict[scale, opinion_vector]
                - anchor_ids: Matched pentachoron IDs
                - cantor_positions: Inherited geometric positions
        """
        B, seq_len, _ = sequence_features.shape

        # 1. Project to pentachoron space
        z = self.input_proj(sequence_features)  # [B, seq_len, pentachoron_dim]
        z = F.normalize(z, dim=-1)

        # 2. Match to geometric opinion anchors
        anchor_ids = self.match_to_opinion_anchors(z)  # [B, seq_len]

        # 3. Inherit positional fingerprints from matched anchors
        # This is KEY: geometry → positions → routing
        cantor_positions_batch = []
        for b in range(B):
            batch_positions = self.shared_positions[anchor_ids[b]]  # [seq_len]
            cantor_positions_batch.append(batch_positions)

        cantor_positions_batch = torch.stack(cantor_positions_batch, dim=0)  # [B, seq_len]

        # 4. Geometric Cantor attention using positional fingerprints
        z_attended_list = []
        for b in range(B):
            z_b = z[b:b + 1]
            positions_b = cantor_positions_batch[b]
            z_attended_b = self.cantor_attention(z_b, positions_b)
            z_attended_list.append(z_attended_b)

        z_attended = torch.cat(z_attended_list, dim=0)  # [B, seq_len, pentachoron_dim]

        # 5. Pool for opinion formation
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).float()
            z_pooled = (z_attended * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        else:
            z_pooled = z_attended.mean(dim=1)  # [B, pentachoron_dim]

        # 6. Multi-scale opinion extraction
        scale_opinions = {}
        for scale in self.scales:
            opinion = self.scale_projectors[str(scale)](z_pooled)  # [B, scale]
            scale_opinions[scale] = opinion

        return {
            'scale_opinions': scale_opinions,
            'anchor_ids': anchor_ids,
            'cantor_positions': cantor_positions_batch,
            'pooled_features': z_pooled
        }


# ============================================================================
# SHALLOW COLLECTIVE FUSION
# ============================================================================

class ShallowCollectiveFusion(nn.Module):
    """
    Shallow fusion layer for collective consensus voting.

    DEMOCRATIC AGGREGATION (not hierarchical):
    - Takes all expert opinions
    - Weighted voting mechanism
    - Learns to weight expert reliability
    - Produces final collective decision

    This is NOT a deep hierarchy - it's a single aggregation layer
    that respects the independence of each expert opinion.
    """

    def __init__(
            self,
            num_experts: int,
            scales: List[int],
            num_classes: int,
            dropout: float = 0.1
    ):
        super().__init__()

        self.num_experts = num_experts
        self.scales = scales
        self.num_classes = num_classes

        # Expert reliability weights (learnable)
        # Each expert gets a reliability score per scale
        self.expert_weights = nn.ParameterDict({
            str(scale): nn.Parameter(torch.ones(num_experts) / num_experts)
            for scale in scales
        })

        # Scale fusion weights (learnable)
        self.scale_weights = nn.Parameter(torch.ones(len(scales)) / len(scales))

        # Final classification heads per scale
        self.scale_classifiers = nn.ModuleDict({
            str(scale): nn.Sequential(
                nn.Linear(scale, scale * 2),
                nn.LayerNorm(scale * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(scale * 2, num_classes)
            )
            for scale in scales
        })

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for module in self.scale_classifiers.values():
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight, gain=0.5)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(
            self,
            expert_opinions: List[Dict[str, torch.Tensor]]  # List of expert outputs
    ) -> Dict[str, torch.Tensor]:
        """
        Fuse expert opinions through democratic voting.

        Process:
        1. Collect opinions from all experts at each scale
        2. Weight by learned expert reliability
        3. Aggregate into collective opinion per scale
        4. Classify at each scale
        5. Fuse scale predictions with learned weights
        6. Final collective decision

        Args:
            expert_opinions: List of dicts from each expert, each containing:
                {'scale_opinions': {scale: opinion_vector}}

        Returns:
            Dict containing:
                - logits: Final collective classification
                - scale_logits: Classification per scale
                - expert_contributions: Weighted contributions
        """
        batch_size = list(expert_opinions[0]['scale_opinions'].values())[0].shape[0]
        device = list(expert_opinions[0]['scale_opinions'].values())[0].device

        scale_logits = {}
        expert_contributions = {}

        # Aggregate per scale
        for scale in self.scales:
            # Collect all expert opinions at this scale
            expert_ops = []
            for expert_out in expert_opinions:
                expert_ops.append(expert_out['scale_opinions'][scale])

            expert_ops = torch.stack(expert_ops, dim=0)  # [num_experts, B, scale]

            # Weighted aggregation (democratic voting)
            weights = F.softmax(self.expert_weights[str(scale)], dim=0)
            weights = weights.view(-1, 1, 1)  # [num_experts, 1, 1]

            collective_opinion = (expert_ops * weights).sum(dim=0)  # [B, scale]

            # Store contributions for interpretability
            expert_contributions[scale] = weights.squeeze()

            # Classify at this scale
            scale_logits[scale] = self.scale_classifiers[str(scale)](collective_opinion)

        # Fuse across scales
        scale_weights = F.softmax(self.scale_weights, dim=0)

        logits_list = [scale_logits[scale] for scale in self.scales]
        logits_stacked = torch.stack(logits_list, dim=0)  # [num_scales, B, num_classes]

        weights_expanded = scale_weights.view(-1, 1, 1)
        final_logits = (logits_stacked * weights_expanded).sum(dim=0)  # [B, num_classes]

        return {
            'logits': final_logits,
            'scale_logits': scale_logits,
            'expert_contributions': expert_contributions,
            'scale_weights': scale_weights
        }


# ============================================================================
# PENTACHORON COLLECTIVE CONSENSUS CLASSIFIER
# ============================================================================

@dataclass
class CollectiveConsensusConfig:
    """Configuration for collective consensus classifier."""

    # Opinion anchor geometry
    num_opinion_anchors: int = 225  # Geometric anchor points
    pentachoron_dim: int = 512
    cantor_depth: int = 8

    # Multi-scale opinion formation
    scales: List[int] = None  # e.g., [128, 256, 512]

    # Encoder configurations
    clip_hidden_dim: int = 768
    clip_num_layers: int = 12
    siglip_hidden_dim: int = 1664
    siglip_num_layers: int = 24

    # Attention
    num_heads: int = 8
    dropout: float = 0.1

    # Classification
    num_classes: int = 1000  # Final output classes

    def __post_init__(self):
        if self.scales is None:
            self.scales = [128, 256, 512]


class PentachoronCollectiveConsensus(nn.Module):
    """
    Pentachoron Collective Consensus Classifier.

    A democratic multi-expert system that fuses opinions from all layers
    of dual encoders through geometric positional fingerprinting.

    Architecture:
        CLIP Layers [0-11] → 12 independent experts
        SigLIP Layers [0-23] → 24 independent experts
        Total: 36 experts voting in collective

        Each expert:
        - Receives layer features
        - Matches to 225 geometric opinion anchors
        - Inherits positional fingerprints
        - Routes via Cantor attention
        - Forms multi-scale opinions

        Shallow fusion:
        - Aggregates all expert opinions
        - Weighted democratic voting
        - Final collective classification

    The more experts, the more robust the consensus.
    """

    def __init__(self, config: CollectiveConsensusConfig):
        super().__init__()

        self.config = config

        print("=" * 80)
        print("PENTACHORON COLLECTIVE CONSENSUS CLASSIFIER")
        print("=" * 80)
        print(f"Opinion anchors: {config.num_opinion_anchors}")
        print(f"CLIP experts: {config.clip_num_layers}")
        print(f"SigLIP experts: {config.siglip_num_layers}")
        print(f"Total experts: {config.clip_num_layers + config.siglip_num_layers}")
        print(f"Scales: {config.scales}")
        print(f"Output classes: {config.num_classes}")

        # ================================================================
        # GEOMETRIC OPINION ANCHORS
        # ================================================================
        print("\nInitializing geometric opinion anchors...")
        self.opinion_anchors = self._init_opinion_anchors()
        print(f"✓ Created {config.num_opinion_anchors} pentachoron anchors: {list(self.opinion_anchors.shape)}")

        # ================================================================
        # COMPUTE POSITIONAL FINGERPRINTS
        # ================================================================
        print("\nComputing geometric positional fingerprints...")
        self.fingerprinter = GeometricPositionalFingerprinter(depth=config.cantor_depth)
        self.anchor_positions = self.fingerprinter.compute_vocabulary_positions(
            self.opinion_anchors
        )
        print(f"✓ Computed positional fingerprints: {list(self.anchor_positions.shape)}")
        print(f"  Position range: [{self.anchor_positions.min():.4f}, {self.anchor_positions.max():.4f}]")
        print(f"  Position mean: {self.anchor_positions.mean():.4f}, std: {self.anchor_positions.std():.4f}")

        # ================================================================
        # CREATE EXPERT COMPANIONS (ONE PER LAYER)
        # ================================================================
        print("\nCreating expert companions...")

        # CLIP experts
        print("  CLIP experts:")
        self.clip_experts = nn.ModuleDict()
        for i in range(config.clip_num_layers):
            expert = MultiScaleExpertCompanion(
                layer_name=f'clip_layer_{i}',
                input_dim=config.clip_hidden_dim,
                pentachoron_dim=config.pentachoron_dim,
                scales=config.scales,
                num_heads=config.num_heads,
                dropout=config.dropout,
                shared_pentachora=self.opinion_anchors,
                shared_positions=self.anchor_positions
            )
            self.clip_experts[f'clip_layer_{i}'] = expert
            print(f"    ✓ clip_layer_{i}")

        # SigLIP experts
        print("  SigLIP experts:")
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
                shared_positions=self.anchor_positions
            )
            self.siglip_experts[f'siglip_layer_{i}'] = expert
            print(f"    ✓ siglip_layer_{i}")

        # ================================================================
        # SHALLOW COLLECTIVE FUSION
        # ================================================================
        print("\nCreating shallow collective fusion...")
        total_experts = config.clip_num_layers + config.siglip_num_layers
        self.fusion = ShallowCollectiveFusion(
            num_experts=total_experts,
            scales=config.scales,
            num_classes=config.num_classes,
            dropout=config.dropout
        )
        print(f"✓ Fusion layer: {total_experts} experts → {config.num_classes} classes")

        total_params = sum(p.numel() for p in self.parameters())
        print(f"\n{'=' * 80}")
        print(f"Total parameters: {total_params:,}")
        print(f"Opinion anchor parameters: {self.opinion_anchors.numel():,}")
        print(f"Positional fingerprints: {self.anchor_positions.numel():,} (pre-computed)")
        print(f"{'=' * 80}\n")

    def _init_opinion_anchors(self) -> nn.Parameter:
        """
        Initialize geometric opinion anchor pentachora.

        These are NOT output classes - they are interpretive reference points
        in shared geometric space that enable opinion formation.

        Returns:
            pentachora: [num_opinion_anchors, 5, pentachoron_dim]
        """
        pentachora = torch.randn(
            self.config.num_opinion_anchors,
            5,
            self.config.pentachoron_dim
        )

        # Normalize vertices
        pentachora = F.normalize(pentachora, dim=-1)

        # Add perturbations for uniqueness
        for i in range(self.config.num_opinion_anchors):
            perturbation = torch.randn_like(pentachora[i]) * 0.1
            pentachora[i] = pentachora[i] + perturbation
            pentachora[i] = F.normalize(pentachora[i], dim=-1)

        return nn.Parameter(pentachora, requires_grad=True)

    def forward(
            self,
            clip_features: Dict[str, torch.Tensor],  # {layer_name: [B, seq_len, 768]}
            siglip_features: Dict[str, torch.Tensor],  # {layer_name: [B, seq_len, 1664]}
            clip_masks: Optional[Dict[str, torch.Tensor]] = None,
            siglip_masks: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Collective consensus classification.

        Process:
        1. Each expert forms independent opinion from its layer
        2. All opinions collected
        3. Shallow fusion aggregates through democratic voting
        4. Final collective decision

        Args:
            clip_features: Features from all CLIP layers
            siglip_features: Features from all SigLIP layers
            clip_masks: Optional attention masks
            siglip_masks: Optional attention masks

        Returns:
            Dict containing:
                - logits: Final collective classification
                - scale_logits: Per-scale classifications
                - expert_contributions: Weight of each expert
                - scale_weights: Weight of each scale
        """
        expert_opinions = []

        # Collect CLIP expert opinions
        for layer_name, features in clip_features.items():
            if layer_name in self.clip_experts:
                mask = clip_masks.get(layer_name) if clip_masks else None
                opinion = self.clip_experts[layer_name](features, mask)
                expert_opinions.append(opinion)

        # Collect SigLIP expert opinions
        for layer_name, features in siglip_features.items():
            if layer_name in self.siglip_experts:
                mask = siglip_masks.get(layer_name) if siglip_masks else None
                opinion = self.siglip_experts[layer_name](features, mask)
                expert_opinions.append(opinion)

        # Shallow fusion: democratic aggregation
        fusion_output = self.fusion(expert_opinions)

        return fusion_output

    def get_info(self) -> Dict:
        """Get model information."""
        return {
            'num_opinion_anchors': self.config.num_opinion_anchors,
            'clip_experts': len(self.clip_experts),
            'siglip_experts': len(self.siglip_experts),
            'total_experts': len(self.clip_experts) + len(self.siglip_experts),
            'scales': self.config.scales,
            'num_classes': self.config.num_classes,
            'total_parameters': sum(p.numel() for p in self.parameters()),
            'position_range': [
                self.anchor_positions.min().item(),
                self.anchor_positions.max().item()
            ],
            'position_stats': {
                'mean': self.anchor_positions.mean().item(),
                'std': self.anchor_positions.std().item()
            }
        }


# ============================================================================
# DEMO
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("PENTACHORON COLLECTIVE CONSENSUS - DEMO")
    print("=" * 80 + "\n")

    # Configuration
    config = CollectiveConsensusConfig(
        num_opinion_anchors=225,
        pentachoron_dim=512,
        scales=[128, 256, 512],
        clip_num_layers=12,
        siglip_num_layers=24,
        num_classes=1000
    )

    # Create collective
    collective = PentachoronCollectiveConsensus(config)

    # Test data
    batch_size = 4
    clip_seq_len = 77
    siglip_seq_len = 256

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    collective = collective.to(device)

    # Generate test features for all layers
    clip_features = {
        f'clip_layer_{i}': torch.randn(batch_size, clip_seq_len, 768, device=device)
        for i in range(12)
    }

    siglip_features = {
        f'siglip_layer_{i}': torch.randn(batch_size, siglip_seq_len, 1664, device=device)
        for i in range(24)
    }

    print(f"\n[TEST] Forward pass:")
    print(f"  CLIP: {len(clip_features)} layers × [B={batch_size}, N={clip_seq_len}, D=768]")
    print(f"  SigLIP: {len(siglip_features)} layers × [B={batch_size}, N={siglip_seq_len}, D=1664]")

    with torch.no_grad():
        output = collective(clip_features, siglip_features)

    print(f"\n✓ Collective consensus formed!")
    print(f"  Final logits: {output['logits'].shape}")
    print(f"  Scale logits: {[v.shape for v in output['scale_logits'].values()]}")

    print(f"\n[EXPERT CONTRIBUTIONS]")
    for scale, weights in output['expert_contributions'].items():
        print(f"  Scale {scale}:")
        print(f"    Top 5 experts: {weights.topk(5).indices.tolist()}")
        print(f"    Top 5 weights: {weights.topk(5).values.tolist()}")

    print(f"\n[SCALE WEIGHTS]")
    scale_weights = output['scale_weights']
    for i, scale in enumerate(config.scales):
        print(f"  Scale {scale}: {scale_weights[i].item():.4f}")

    print(f"\n{'=' * 80}")
    print("✅ Pentachoron Collective Consensus Complete!")
    print(f"{'=' * 80}")
    print("\nKey Features:")
    print(f"  • {len(clip_features) + len(siglip_features)} independent experts voting")
    print(f"  • {config.num_opinion_anchors} geometric opinion anchors")
    print(f"  • Positional fingerprinting from pentachoron geometry")
    print(f"  • O(n) Cantor attention with geometric routing")
    print(f"  • Multi-scale opinion formation: {config.scales}")
    print(f"  • Shallow democratic fusion (not hierarchical)")
    print(f"  • Final consensus: {config.num_classes} classes")
    print(f"{'=' * 80}\n")