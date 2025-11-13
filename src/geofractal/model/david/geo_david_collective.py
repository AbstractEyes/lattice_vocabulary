"""
GeoDavidCollective: Geometric Multi-Block Diffusion Collective - ENHANCED
==========================================================================
Complete geometric architecture for SD1.5 block-level distillation
with pentachoron structure, Cantor hierarchical encoding, and
geometric solidity enforcement.

ENHANCEMENT: Replaces simple linear heads with DeepEfficiencyGating-inspired
ProjectiveHead architecture for richer projective representations.

Each block companion is a multi-scale geometric structure that learns
to classify diffusion timesteps and patterns using validated k-simplices
and hierarchical Cantor position encoding.

Key Features:
- Pentachoron structure (5-vertex 4-simplices) per pattern
- SimplexFactory-validated initialization for stability
- Cantor staircase hierarchical position encoding
- ProjectiveHead multi-expert classification heads
- 6 geometric loss components (feature, Rose, CE, diversity, Cayley, Cantor)
- Multi-block ensemble architecture
- Per-block scale weighting

Author: AbstractPhil
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from collections import OrderedDict
import math

try:
    from geovocab2.shapes.factory import SimplexFactory

    HAS_SIMPLEX_FACTORY = True
except ImportError:
    HAS_SIMPLEX_FACTORY = False
    print("WARNING: SimplexFactory not available. Using fallback initialization.")
    print("Training may be less stable without validated geometry.")


@dataclass
class GeoDavidConfig:
    """Configuration for GeoDavidBlockCompanion."""
    feature_dim: int = 256          # Input feature dimension
    num_classes: int = 1000         # Number of diffusion patterns/classes
    scale_dim: int = 256            # Scale dimension for block
    projective_head_config: dict = None  # Config for ProjectiveHead
    cantor_levels: int = 12         # Levels in Cantor staircase
    cantor_tau: float = 0.25        # Temperature for Cantor soft assignment
    cantor_alpha_init: float = 0.5  # Initial alpha for Cantor middle weighting
    use_simplex_factory: bool = True  # Whether to use SimplexFactory for initialization
    # Block configurations (mimicking SD1.5 UNet structure)
    block_configs: dict = field(default_factory=lambda: block_config_overrides.copy().get("sd15_full_blocks"))



block_config_overrides = {
    "sd15_full_blocks": {
        'down_0': {
            'input_dim': 320,
            'scale_dim': 64,
            'use_belly': True,
            'belly_expand': 2.0,
            'weight': 0.5,
        },
        'down_1': {
            'input_dim': 640,
            'scale_dim': 96,
            'use_belly': True,
            'belly_expand': 2.0,
        },
        'down_2': {
            'input_dim': 1280,
            'scale_dim': 128,
            'use_belly': True,
            'belly_expand': 2.0,
        },
        'down_3': {
            'input_dim': 1280,
            'scale_dim': 128,
            'use_belly': True,
            'belly_expand': 2.0,
        },
        'mid': {
            'input_dim': 1280,
            'scale_dim': 256,
            'use_belly': True,
            'belly_expand': 4,
            'num_experts': 4,
            'num_gate_heads': 4,
        },
        'up_0': {
            'input_dim': 1280,
            'scale_dim': 128,
            'use_belly': True,
            'belly_expand': 2.0,
        },
        'up_1': {
            'input_dim': 1280,
            'scale_dim': 128,
            'use_belly': True,
            'belly_expand': 2.0,
        },
        'up_2': {
            'input_dim': 640,
            'scale_dim': 96,
            'use_belly': True,
            'belly_expand': 2.0,
        },
        'up_3': {
            'input_dim': 320,
            'scale_dim': 64,
            'use_belly': True,
            'belly_expand': 1.5,
        }
    }
}

# ============================================================================
# PROJECTIVE HEAD - DeepEfficiencyGating Inspired
# ============================================================================
class CantorStaircase(nn.Module):
    """
    Learnable soft Cantor staircase with alpha-normalized middle weighting.

    Provides:
    - Deterministic hierarchical lattice structure
    - Feature-driven position modulation
    - Learnable alpha parameter for middle-interval weighting
    - Triadic decomposition (base-3 intervals)

    This creates the "deterministic Cantor step offset" that features
    traverse through, providing positional encoding via geometric structure.
    """

    def __init__(
            self,
            feature_dim: int,
            alpha_init: float = 0.5,
            tau: float = 0.25,
            base: int = 3,
            levels: int = 12
    ):
        """
        Args:
            feature_dim: Dimension of input features
            alpha_init: Initial alpha value (middle-interval weight)
            tau: Temperature for soft assignment
            base: Base for decomposition (3 for triadic Cantor)
            levels: Number of hierarchical levels
        """
        super().__init__()

        self.feature_dim = feature_dim
        self.tau = tau
        self.base = base
        self.levels = levels

        # Learnable alpha parameter (middle-interval weighting)
        self.alpha = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32))

        # Fixed triadic centers [left, middle, right]
        self.register_buffer(
            'centers',
            torch.tensor([0.5, 1.5, 2.5], dtype=torch.float32)
        )

        # Feature-driven position modulation
        # Features influence traversal through lattice
        self.feature_to_position = nn.Linear(feature_dim, 1)

    def forward(
            self,
            features: torch.Tensor,
            positions: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute soft Cantor values with hierarchical structure.

        Args:
            features: [B, D] - input features
            positions: [B] - base position indices (e.g., timestep bins)

        Returns:
            cantor_values: [B] - Cantor values in [0, 1]
        """
        batch_size = features.size(0)
        device = features.device

        # Normalize base positions to [0, 1] (deterministic lattice)
        max_pos = positions.max().item() + 1
        if max_pos > 1:
            x_base = positions.float() / float(max_pos - 1)
        else:
            x_base = positions.float()

        x_base = x_base.clamp(1e-6, 1.0 - 1e-6)

        # Feature-driven modulation (learned trajectories through lattice)
        feature_shift = self.feature_to_position(features).squeeze(-1)  # [B]
        feature_shift = torch.tanh(feature_shift) * 0.3  # Bounded shift ±0.3

        # Combine: deterministic base + learned modulation
        x = (x_base + feature_shift).clamp(1e-6, 1.0 - 1e-6)

        # HIERARCHICAL TRIADIC DECOMPOSITION
        # Build Cantor value through recursive subdivision
        Cx = torch.zeros_like(x)
        w = 0.5  # Weight for current level

        for level in range(self.levels):
            # Map to triadic interval [0, 3)
            y = x * float(self.base)

            # Compute distances to triadic centers
            d2 = (y.unsqueeze(-1) - self.centers) ** 2  # [B, 3]

            # Soft assignment via temperature-scaled softmax
            logits = -d2 / (self.tau + 1e-8)
            p = F.softmax(logits, dim=-1)  # [B, 3]

            # Alpha-normalized middle weighting (Beatrix paradigm)
            # bit_k = p[left]*0 + p[middle]*alpha + p[right]*1
            bit_k = p[:, 1] * self.alpha + p[:, 2]

            # Accumulate weighted contribution
            Cx = Cx + bit_k * w

            # Recurse into selected interval
            t = y.floor()
            x = y - t  # Fractional part for next level
            w *= 0.5  # Halve weight for next level

        return Cx.clamp(0.0, 1.0)

    def get_alpha(self) -> float:
        """Get current alpha value."""
        return self.alpha.item()

    def get_interval_distribution(
            self,
            features: torch.Tensor,
            positions: torch.Tensor
    ) -> dict:
        """
        Get soft interval distribution for diagnostics.

        Returns:
            Dict with average probabilities for [left, middle, right] intervals
        """
        batch_size = features.size(0)

        # Get base position
        max_pos = positions.max().item() + 1
        if max_pos > 1:
            x_base = positions.float() / float(max_pos - 1)
        else:
            x_base = positions.float()
        x_base = x_base.clamp(1e-6, 1.0 - 1e-6)

        # Feature modulation
        feature_shift = self.feature_to_position(features).squeeze(-1)
        feature_shift = torch.tanh(feature_shift) * 0.3

        # Combined position
        x = (x_base + feature_shift).clamp(1e-6, 1.0 - 1e-6)

        # First-level triadic position
        y = x * float(self.base)
        d2 = (y.unsqueeze(-1) - self.centers) ** 2
        logits = -d2 / (self.tau + 1e-8)
        p = F.softmax(logits, dim=-1)  # [B, 3]

        # Average across batch
        avg_probs = p.mean(dim=0)

        return {
            'left': avg_probs[0].item(),
            'middle': avg_probs[1].item(),
            'right': avg_probs[2].item(),
            'alpha': self.alpha.item()
        }

class ProjectiveHead(nn.Module):
    """
    Deep efficiency gating-inspired projective head for classification.

    Multi-expert architecture with cross-attention and multi-head gating
    for building rich projective representations.

    Architecture:
        Input → Multi-Expert Pathways → Cross-Attention → Multi-Head Gating → Output

    Features:
    - Multiple expert pathways for diverse feature representations
    - Cross-attention for expert refinement
    - Multi-head gating for ensemble classification
    - Learnable temperature scaling
    - Optional sparsity for inference efficiency
    """

    def __init__(
            self,
            input_dim: int,
            num_classes: int,
            num_attention_heads: int = 4,
            num_experts: int = 3,
            compression_ratio: int = 4,
            num_gate_heads: int = 3,
            expert_dropout: float = 0.1,
            attention_dropout: float = 0.1,
            temperature_init: float = 0.5,
            use_sparsity: bool = True,
            sparsity_threshold: float = 0.1
    ):
        """
        Args:
            input_dim: Input feature dimension
            num_classes: Number of output classes
            num_attention_heads: Number of attention heads in cross-attention
            num_experts: Number of expert pathways (2-4 recommended)
            compression_ratio: Bottleneck compression factor (3-6 recommended)
            num_gate_heads: Number of gating heads (2-4 recommended)
            expert_dropout: Dropout rate for expert pathways
            attention_dropout: Dropout rate for cross-attention
            temperature_init: Initial temperature for scaling
            use_sparsity: Enable sparsity during inference
            sparsity_threshold: Threshold for sparse predictions
        """
        super().__init__()

        self.input_dim = input_dim
        self.num_classes = num_classes
        self.num_experts = num_experts
        self.num_gate_heads = num_gate_heads
        self.use_sparsity = use_sparsity

        # Learnable temperature (constrained positive via abs())
        self.temperature = nn.Parameter(torch.ones(1) * temperature_init)

        # Learnable sparsity threshold
        self.sparsity_threshold = nn.Parameter(torch.tensor(sparsity_threshold))

        # Calculate bottleneck dimension
        # Ensure it's large enough for classification but compressed from input
        self.bottleneck_dim = max(
            num_classes * 2,  # At least 2x num_classes
            input_dim // compression_ratio
        )

        # Multi-expert pathways
        # Each expert learns a different projection of the input
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, self.bottleneck_dim),
                nn.LayerNorm(self.bottleneck_dim),
                nn.GELU(),
                nn.Dropout(expert_dropout)
            )
            for _ in range(num_experts)
        ])

        # Cross-attention mechanism
        # Allows experts to refine each other's representations
        #num_attention_heads = max(1, min(8, self.bottleneck_dim // 64))
        self.cross_attention = nn.MultiheadAttention(
            self.bottleneck_dim,
            num_heads=num_attention_heads,
            batch_first=True,
            dropout=attention_dropout
        )

        # Multi-head gating
        # Multiple classification heads vote on the final prediction
        expert_combined_dim = self.bottleneck_dim * num_experts
        self.gate_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(expert_combined_dim, expert_combined_dim // 2),
                nn.GELU(),
                nn.Dropout(expert_dropout),
                nn.Linear(expert_combined_dim // 2, num_classes)
            )
            for _ in range(num_gate_heads)
        ])

        # Learnable weights for combining gate heads
        self.head_weights = nn.Parameter(torch.ones(num_gate_heads) / num_gate_heads)

        # Per-class bias (learnable class preferences)
        self.class_bias = nn.Parameter(torch.zeros(num_classes))

        # Final projection layer (direct path)
        self.final_projection = nn.Sequential(
            nn.Linear(expert_combined_dim, num_classes),
            nn.LayerNorm(num_classes)
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights for better gradient flow and stability."""
        # Initialize expert pathways
        for expert in self.experts:
            for module in expert.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight, gain=0.5)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

        # Initialize gate heads
        for gate_head in self.gate_heads:
            for module in gate_head.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight, gain=0.5)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

        # Initialize final projection
        for module in self.final_projection.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
            self,
            features: torch.Tensor,
            return_gates: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through projective head.

        Args:
            features: [B, D] - input features
            return_gates: Whether to return attention weights (for analysis)

        Returns:
            logits: [B, num_classes] - classification logits
            attention_weights: [B, num_experts] - attention weights (if return_gates=True, else None)
        """
        B, D = features.shape

        # Step 1: Multi-expert processing
        # Each expert creates its own representation
        expert_outputs = [expert(features) for expert in self.experts]
        stacked_experts = torch.stack(expert_outputs, dim=1)  # [B, num_experts, bottleneck_dim]

        # Step 2: Cross-attention refinement
        # Experts attend to each other, refining their representations
        attended_experts, attention_weights = self.cross_attention(
            stacked_experts,  # query
            stacked_experts,  # key
            stacked_experts   # value
        )  # [B, num_experts, bottleneck_dim], [B, num_experts, num_experts]

        # Step 3: Flatten attended representations
        flattened = attended_experts.reshape(B, -1)  # [B, bottleneck_dim * num_experts]

        # Step 4: Multi-head gating
        # Multiple heads vote on the classification
        gate_outputs = [head(flattened) for head in self.gate_heads]

        # Step 5: Combine gates with learnable weights
        head_weights_normalized = F.softmax(self.head_weights, dim=0)
        combined_gates = sum(
            w * g for w, g in zip(head_weights_normalized, gate_outputs)
        )

        # Step 6: Add class bias and apply temperature scaling
        gate_logits = (combined_gates + self.class_bias) / self.temperature.abs()

        # Step 7: Final projection refinement (direct path)
        final_logits = self.final_projection(flattened)

        # Step 8: Combine gated and projected logits
        # Weighted combination favors the gated path but includes direct path
        alpha = 0.7  # Weight towards gated path
        logits = alpha * gate_logits + (1 - alpha) * final_logits

        # Step 9: Optional sparsity (during inference only)
        # Zero out low-probability predictions for efficiency
        if not self.training and self.use_sparsity:
            probs = F.softmax(logits, dim=-1)
            mask = probs > self.sparsity_threshold
            sparse_logits = logits * mask
            if mask.any():
                logits = sparse_logits

        if return_gates:
            # Return mean attention weights across attention heads
            return logits, attention_weights.mean(dim=1)
        else:
            return logits, None


# ============================================================================
# CONFIGURATION HELPERS
# ============================================================================

# Default ProjectiveHead configurations for different model scales
PROJECTIVE_HEAD_CONFIGS = {
    'small': {  # scale_dim < 128
        'num_experts': 2,
        'compression_ratio': 6,
        'num_gate_heads': 2,
        'expert_dropout': 0.1,
        'attention_dropout': 0.1,
        'head_temperature': 0.5,
        'use_head_sparsity': True,
        'head_sparsity_threshold': 0.1
    },
    'medium': {  # 128 <= scale_dim < 512 (DEFAULT)
        'num_experts': 3,
        'compression_ratio': 4,
        'num_gate_heads': 3,
        'expert_dropout': 0.1,
        'attention_dropout': 0.1,
        'head_temperature': 0.5,
        'use_head_sparsity': True,
        'head_sparsity_threshold': 0.1
    },
    'large': {  # scale_dim >= 512
        'num_experts': 4,
        'compression_ratio': 3,
        'num_gate_heads': 4,
        'expert_dropout': 0.15,
        'attention_dropout': 0.15,
        'head_temperature': 0.5,
        'use_head_sparsity': True,
        'head_sparsity_threshold': 0.1
    }
}


def get_projective_head_config(scale_dim: int) -> dict:
    """
    Get recommended ProjectiveHead config based on scale_dim.

    Auto-selects appropriate complexity based on feature dimension.
    Smaller models get fewer experts/heads, larger models get more.

    Args:
        scale_dim: Feature dimension

    Returns:
        Config dict with ProjectiveHead parameters
    """
    if scale_dim < 128:
        return PROJECTIVE_HEAD_CONFIGS['small'].copy()
    elif scale_dim < 512:
        return PROJECTIVE_HEAD_CONFIGS['medium'].copy()
    else:
        return PROJECTIVE_HEAD_CONFIGS['large'].copy()


# ============================================================================
# GEOMETRIC LOSS FUNCTIONS
# ============================================================================

class CayleyChaosLoss(nn.Module):
    """
    Cayley-Menger chaos loss for geometric regularization of pentachora.

    Enforces three geometric constraints:
    1. Volume preservation (prevent collapse to lower dimensions)
    2. Edge uniformity (encourage regular simplex structure)
    3. Gram matrix conditioning (proper high-D embedding)

    Uses classical Cayley-Menger determinant formula for 4-simplex volume.
    """

    def __init__(
            self,
            volume_floor: float = 1e-4,
            chaos_scale: float = 1.0,
            edge_dev_weight: float = 0.5,
            gram_weight: float = 0.1,
            use_sqrt_volume: bool = True
    ):
        """
        Args:
            volume_floor: Minimum allowed volume (prevents collapse)
            chaos_scale: Weight for volume chaos penalty
            edge_dev_weight: Weight for edge uniformity loss
            gram_weight: Weight for Gram matrix conditioning
            use_sqrt_volume: Use sqrt(volume) instead of volume²
        """
        super().__init__()

        self.volume_floor = volume_floor
        self.chaos_scale = chaos_scale
        self.edge_dev_weight = edge_dev_weight
        self.gram_weight = gram_weight
        self.use_sqrt_volume = use_sqrt_volume

        # Cache for upper triangular indices (edge pairs)
        self.register_buffer('_triu_i', None)
        self.register_buffer('_triu_j', None)

    def _get_triu_indices(self, device: torch.device):
        """Get cached upper triangular indices for 5x5 matrix."""
        if self._triu_i is None or self._triu_i.device != device:
            indices = torch.triu_indices(5, 5, offset=1, device=device)
            self._triu_i = indices[0]
            self._triu_j = indices[1]
        return self._triu_i, self._triu_j

    def compute_cayley_menger_volume(self, X: torch.Tensor) -> torch.Tensor:
        """
        Compute 4-simplex (pentachoron) volume using Cayley-Menger determinant.

        The Cayley-Menger determinant for 5 vertices (4-simplex) is:

        | 0   1   1   1   1   1  |
        | 1   0  d01 d02 d03 d04 |
        | 1  d10  0  d12 d13 d14 |
        | 1  d20 d21  0  d23 d24 |
        | 1  d30 d31 d32  0  d34 |
        | 1  d40 d41 d42 d43  0  |

        Volume² = -det(M) / 288²

        Args:
            X: [B, 5, D] - batch of pentachora (5 vertices each)

        Returns:
            volumes: [B] - volume of each pentachoron
        """
        B, N, D = X.shape
        assert N == 5, f"Expected 5 vertices (pentachoron), got {N}"

        # Compute pairwise squared distances
        diff = X.unsqueeze(2) - X.unsqueeze(1)  # [B, 5, 5, D]
        distsq = (diff * diff).sum(dim=-1)  # [B, 5, 5]

        # Build Cayley-Menger matrix: [B, 6, 6]
        M = torch.zeros((B, 6, 6), dtype=X.dtype, device=X.device)
        M[:, 0, 1:] = 1.0  # First row
        M[:, 1:, 0] = 1.0  # First column
        M[:, 1:, 1:] = distsq  # Distance matrix

        # Compute determinant
        det = torch.linalg.det(M)

        # Volume² = -det / (2^8 * 3^2) = -det / 9216
        volume_sq = (-det / 9216.0).clamp(min=0.0)

        if self.use_sqrt_volume:
            return volume_sq.sqrt()
        else:
            return volume_sq

    def compute_edge_uniformity(self, X: torch.Tensor) -> torch.Tensor:
        """
        Compute edge uniformity metric (coefficient of variation).

        For a regular 4-simplex, all 10 edges should have equal length.
        We measure: std(edge_lengths) / mean(edge_lengths)

        Args:
            X: [B, 5, D] - batch of pentachora

        Returns:
            edge_dev: [B] - edge deviation (lower is more uniform)
        """
        # Compute pairwise squared distances
        diff = X.unsqueeze(2) - X.unsqueeze(1)  # [B, 5, 5, D]
        distsq = (diff * diff).sum(dim=-1)  # [B, 5, 5]

        # Extract upper triangular (10 unique edges)
        triu_i, triu_j = self._get_triu_indices(X.device)
        edge_lengths = distsq[:, triu_i, triu_j]  # [B, 10]

        # Compute coefficient of variation
        edge_mean = edge_lengths.mean(dim=1)  # [B]
        edge_std = edge_lengths.std(dim=1)  # [B]
        edge_dev = edge_std / edge_mean.clamp(min=1e-6)

        return edge_dev

    def compute_gram_condition(self, X: torch.Tensor) -> torch.Tensor:
        """
        Compute Gram matrix conditioning penalty.

        The Gram matrix G = X^T X (after centering) should be well-conditioned
        for proper embedding in high-D space. We use:

        condition = det(G) / trace(G)

        Well-conditioned → condition close to 1
        Poorly conditioned → condition close to 0

        Args:
            X: [B, 5, D] - batch of pentachora

        Returns:
            gram_penalty: [B] - penalty for poor conditioning
        """
        # Center the vertices (zero mean)
        centered = X - X.mean(dim=1, keepdim=True)  # [B, 5, D]

        # Compute Gram matrix: G = X X^T
        gram = torch.bmm(centered, centered.transpose(1, 2))  # [B, 5, 5]

        # Trace and determinant
        gram_trace = torch.diagonal(gram, dim1=1, dim2=2).sum(dim=1)  # [B]
        gram_det = torch.linalg.det(gram)  # [B]

        # Condition number proxy
        condition = gram_det / gram_trace.clamp(min=1e-6)

        # Penalize poor conditioning (want condition close to 1)
        gram_penalty = F.relu(1.0 - condition)

        return gram_penalty

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Compute total geometric chaos loss.

        Args:
            X: [B, 5, D] - batch of pentachora

        Returns:
            total_loss: scalar - combined geometric loss
        """
        B, N, D = X.shape
        assert N == 5, f"CayleyChaosLoss requires 5 vertices (pentachoron), got {N}"

        # 1. Volume chaos loss (prevent collapse)
        volumes = self.compute_cayley_menger_volume(X)
        chaos_penalty = F.relu(self.volume_floor - volumes)
        chaos_loss = chaos_penalty.mean()

        # 2. Edge uniformity loss (encourage regularity)
        edge_dev = self.compute_edge_uniformity(X)
        edge_loss = edge_dev.mean()

        # 3. Gram matrix conditioning (optional)
        gram_loss = 0.0
        if self.gram_weight > 0:
            gram_penalty = self.compute_gram_condition(X)
            gram_loss = gram_penalty.mean()

        # Total weighted loss
        total_loss = (
                self.chaos_scale * chaos_loss +
                self.edge_dev_weight * edge_loss +
                self.gram_weight * gram_loss
        )

        return total_loss


class RoseLoss(nn.Module):
    """
    Rose Loss with pentachoron role weighting.

    Dream-inspired geometric alignment loss that uses semantic roles
    for the 5 vertices of each pentachoron:
    - Anchor: positive (what IS)
    - Need: negative (what's LACKING)
    - Relation: positive (CONNECTION)
    - Purpose: positive (GOAL/INTENT)
    - Observer: negative (EXTERNAL perspective)

    This creates a multi-dimensional projection space where features
    align with their corresponding pentachora through weighted cosine similarity.
    """

    def __init__(
            self,
            margin: float = 1.0,
            temperature: float = 0.07,
            role_weights: Optional[Dict[str, float]] = None
    ):
        """
        Args:
            margin: Margin for triplet-style loss
            temperature: Temperature for scaling similarities
            role_weights: Optional custom role weights (default uses standard geometry)
        """
        super().__init__()

        self.margin = margin
        self.temperature = temperature

        # Default semantic role weights
        default_weights = {
            "anchor": 1.0,      # Core identity
            "need": -0.75,      # What's lacking
            "relation": 0.75,   # Connections
            "purpose": 0.75,    # Goal/intent
            "observer": -0.75,  # External view
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

    def forward(
            self,
            z: torch.Tensor,
            pentachora: torch.Tensor,
            targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Rose loss.

        Args:
            z: [B, D] - normalized feature vectors
            pentachora: [C, 5, D] - pentachora vertices (C classes, 5 vertices each)
            targets: [B] - target class indices

        Returns:
            loss: scalar - Rose loss value
        """
        # Ensure everything on same device
        pentachora = pentachora.to(z.device)
        role_weights = self.role_weights.to(z.device)

        B, D = z.shape
        C, V, _ = pentachora.shape
        assert V == 5, f"Expected 5 vertices per pentachoron, got {V}"

        # Normalize pentachora
        pentachora_norm = F.normalize(pentachora, dim=-1)

        # Compute cosine similarity to all vertices
        # [B, D] @ [C, 5, D] -> [B, C, 5]
        cos_sim = torch.einsum("bd,cvd->bcv", z, pentachora_norm)

        # Weight by semantic roles and sum across vertices
        # [B, C, 5] * [5] -> [B, C]
        rose_scores = (cos_sim * role_weights.view(1, 1, 5)).sum(dim=-1)

        # Scale by temperature
        rose_scores = rose_scores / self.temperature

        # Get scores for true classes
        true_scores = rose_scores.gather(1, targets.view(-1, 1)).squeeze(1)

        # Get hard negative (best non-target score)
        mask = torch.zeros_like(rose_scores, dtype=torch.bool)
        mask.scatter_(1, targets.view(-1, 1), True)
        hard_neg = rose_scores.masked_fill(mask, float("-inf")).max(dim=1).values

        # Triplet-style margin loss
        loss = F.relu(self.margin - (true_scores - hard_neg))

        return loss.mean()


# ============================================================================
# GEO DAVID BLOCK COMPANION - ENHANCED
# ============================================================================

class GeoDavidBlockCompanion(nn.Module):
    """
    Geometric companion for a single SD1.5 block - ENHANCED VERSION.

    Uses ProjectiveHead instead of simple linear layers for richer
    projective representations in both timestep and pattern classification.

    Architecture:
        Input Features → Projection (with optional "belly") → Normalized Features
        → Cantor Hierarchical Encoding
        → Timestep ProjectiveHead → timestep logits
        → Pattern ProjectiveHead → pattern logits

    The pentachora serve as geometric anchors for the feature space,
    with Rose loss aligning features to their appropriate pentachora
    using semantic role weighting.
    """

    def __init__(
            self,
            block_name: str,
            input_dim: int,
            scale_dim: int,
            num_timestep_bins: int = 100,
            num_patterns_per_bin: int = 10,
            num_timestep_steps: int = 1000,
            use_belly: bool = True,
            belly_expand: float = 2.0,
            temperature: float = 0.07,
            cantor_alpha_init: float = 0.5,
            cantor_tau: float = 0.25,
            cantor_levels: int = 12,
            cantor_base: int = 3,
            simplex_k: int = 4,
            simplex_seed_base: int = 42,
            # ProjectiveHead parameters
            num_experts: int = 3,
            compression_ratio: int = 4,
            num_gate_heads: int = 3,
            expert_dropout: float = 0.1,
            attention_dropout: float = 0.1,
            head_temperature: float = 0.5,
            use_head_sparsity: bool = True,
            head_sparsity_threshold: float = 0.1
    ):
        """
        Args:
            block_name: Identifier for this block (e.g., "down_0", "up_3")
            input_dim: Dimension of input features from SD1.5 block
            scale_dim: Dimension of geometric feature space
            num_timestep_bins: Number of timestep buckets
            num_patterns_per_bin: Patterns per timestep bucket
            num_timestep_steps: Total diffusion steps (usually 1000)
            use_belly: Use expanded intermediate projection
            belly_expand: Expansion factor for belly layer
            temperature: Temperature for Rose loss
            cantor_alpha_init: Initial alpha for Cantor staircase
            cantor_tau: Tau parameter for Cantor staircase
            cantor_levels: Number of hierarchical levels in Cantor encoding
            cantor_base: Base for Cantor staircase (usually 3)
            simplex_k: Simplex dimension (4 for pentachoron)
            simplex_seed_base: Base seed for simplex initialization

            ProjectiveHead parameters:
            num_experts: Number of expert pathways in ProjectiveHead
            compression_ratio: Bottleneck compression in ProjectiveHead
            num_gate_heads: Number of gating heads in ProjectiveHead
            expert_dropout: Dropout for expert pathways
            attention_dropout: Dropout for cross-attention
            head_temperature: Initial temperature for ProjectiveHead
            use_head_sparsity: Enable sparsity in ProjectiveHead
            head_sparsity_threshold: Threshold for sparse predictions
        """
        super().__init__()

        self.block_name = block_name
        self.input_dim = input_dim
        self.scale_dim = scale_dim
        self.num_bins = num_timestep_bins
        self.num_patterns = num_patterns_per_bin
        self.temperature = temperature
        self.simplex_k = simplex_k
        self.num_vertices = simplex_k + 1  # 5 for pentachoron
        self.num_timestep_steps = num_timestep_steps

        # Feature projection (with optional "belly" expansion)
        if use_belly:
            belly_dim = int(scale_dim * belly_expand)
            # Adaptive dropout based on scale
            dropout_rate = min(0.5, max(1.0 / math.sqrt(scale_dim), 0.2))
            self.projection = nn.Sequential(
                nn.Linear(input_dim, belly_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(belly_dim, scale_dim, bias=False)
            )
        else:
            self.projection = nn.Linear(input_dim, scale_dim, bias=False)

        self._init_projection_weights()

        # PENTACHORON STRUCTURE (geometric core)
        print(f"  [{block_name}] Initializing geometric pentachora (scale={scale_dim})...")
        self.crystal_pentachora = self._init_pentachora_validated(
            num_timestep_bins,
            num_patterns_per_bin,
            scale_dim,
            simplex_k,
            simplex_seed_base
        )

        # Semantic role weights for Rose loss
        role_weights = torch.tensor([
            1.0,    # Anchor
            -0.75,  # Need
            0.75,   # Relation
            0.75,   # Purpose
            -0.75   # Observer
        ], dtype=torch.float32)
        self.register_buffer("role_weights", role_weights)

        # Cantor staircase (hierarchical position encoding)
        self.cantor_stairs = CantorStaircase(
            feature_dim=scale_dim,
            alpha_init=cantor_alpha_init,
            tau=cantor_tau,
            base=cantor_base,
            levels=cantor_levels
        )

        # ENHANCED CLASSIFICATION HEADS - ProjectiveHead instead of Linear
        total_patterns = num_timestep_bins * num_patterns_per_bin

        print(f"  [{block_name}] Creating ProjectiveHead for timestep classification...")
        self.timestep_head = ProjectiveHead(
            input_dim=scale_dim,
            num_classes=num_timestep_bins,
            num_experts=num_experts,
            compression_ratio=compression_ratio,
            num_gate_heads=num_gate_heads,
            expert_dropout=expert_dropout,
            attention_dropout=attention_dropout,
            temperature_init=head_temperature,
            use_sparsity=use_head_sparsity,
            sparsity_threshold=head_sparsity_threshold
        )

        print(f"  [{block_name}] Creating ProjectiveHead for pattern classification...")
        self.pattern_head = ProjectiveHead(
            input_dim=scale_dim,
            num_classes=total_patterns,
            num_experts=num_experts,
            compression_ratio=compression_ratio,
            num_gate_heads=num_gate_heads,
            expert_dropout=expert_dropout,
            attention_dropout=attention_dropout,
            temperature_init=head_temperature,
            use_sparsity=use_head_sparsity,
            sparsity_threshold=head_sparsity_threshold
        )

    def _init_projection_weights(self):
        """Initialize projection weights for stable training."""
        for module in self.projection.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _init_pentachora_validated(
            self,
            num_bins: int,
            num_patterns: int,
            dim: int,
            k: int,
            seed: int
    ) -> nn.Parameter:
        """
        Initialize pentachora with SimplexFactory validation.

        Creates validated k-simplices that are geometrically valid and
        well-distributed in the feature space.

        Args:
            num_bins: Number of timestep bins
            num_patterns: Patterns per bin
            dim: Feature dimension
            k: Simplex dimension (4 for pentachoron)
            seed: Base seed for reproducibility

        Returns:
            Initialized pentachora parameter [bins, patterns, k+1, dim]
        """
        if not HAS_SIMPLEX_FACTORY:
            # Fallback: random initialization with normalization
            print(f"    WARNING: Using fallback initialization (SimplexFactory unavailable)")
            pentachora = torch.randn(num_bins, num_patterns, k + 1, dim)
            pentachora = F.normalize(pentachora, dim=-1)
            return nn.Parameter(pentachora, requires_grad=True)

        # Use SimplexFactory for validated initialization
        factory = SimplexFactory(
            embed_dim=dim,
            k=k,
            method='random',
            seed=seed,
        )

        pentachora_list = []
        for bin_idx in range(num_bins):
            bin_patterns = []
            for pattern_idx in range(num_patterns):
                # Unique seed for each simplex
                subseed = seed + bin_idx * num_patterns + pattern_idx
                simplex = factory.build(backend="torch", validate=True, seed=subseed)
                bin_patterns.append(simplex)
            pentachora_list.append(torch.stack(bin_patterns, dim=0))

        pentachora = torch.stack(pentachora_list, dim=0)
        print(f"    ✓ Created {num_bins * num_patterns} validated pentachora")
        return nn.Parameter(pentachora, requires_grad=True)

    def forward(
            self,
            features: torch.Tensor,
            timesteps: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with enhanced projective heads.

        Args:
            features: [B, input_dim] or [B, C, H, W] (spatial)
            timesteps: [B] - diffusion timesteps [0, 999]

        Returns:
            Dict containing:
                'features': [B, scale_dim] - projected & normalized features
                'timestep_logits': [B, num_bins] - timestep classification logits
                'pattern_logits': [B, num_bins * num_patterns] - pattern logits
                'timestep_class': [B] - timestep bin indices
                'cantor_values': [B] - Cantor hierarchical encoding values
        """
        # Flatten spatial features if needed
        if features.dim() > 2:
            features = features.flatten(1)

        # Project and normalize features
        z = self.projection(features)
        z = F.normalize(z, dim=-1)

        # Compute timestep class (bin index)
        timestep_class = (timesteps / self.num_timestep_steps * self.num_bins).long().clamp(
            0, self.num_bins - 1
        )

        # Cantor hierarchical position encoding
        cantor_values = self.cantor_stairs(z, timestep_class)

        # Enhanced classification heads
        # (ignore gate outputs during training - they're for analysis only)
        timestep_logits, _ = self.timestep_head(z, return_gates=False)
        pattern_logits, _ = self.pattern_head(z, return_gates=False)

        return {
            'features': z,
            'timestep_logits': timestep_logits,
            'pattern_logits': pattern_logits,
            'timestep_class': timestep_class,
            'cantor_values': cantor_values
        }


# ============================================================================
# GEOMETRIC MULTI-SCALE LOSS
# ============================================================================

class GeometricMultiScaleLoss(nn.Module):
    """
    BATCHED geometric loss calculator for multi-block ensemble.

    Computes 6 loss components per block:
    1. Feature similarity - alignment with teacher features
    2. Rose loss - geometric pentachoron alignment
    3. Cross-entropy - pattern classification
    4. Pattern diversity - mode collapse prevention
    5. Cayley-Menger - geometric solidity (CRITICAL for stability)
    6. Cantor coherence - hierarchical clustering

    All components are balanced to:
    - Preserve geometric structure (Cayley + Rose)
    - Learn accurate classifications (CE)
    - Prevent collapse (diversity + Cayley)
    - Maintain hierarchical organization (Cantor)
    """

    def __init__(
            self,
            num_timestep_bins: int = 100,
            num_patterns_per_bin: int = 10,
            feature_similarity_weight: float = 0.4,
            rose_weight: float = 0.25,
            ce_weight: float = 0.15,
            pattern_diversity_weight: float = 0.05,
            cayley_weight: float = 0.10,
            cantor_coherence_weight: float = 0.05,
            use_soft_assignment: bool = True,
            temperature: float = 0.1,
            cayley_volume_floor: float = 1e-4,
            cayley_chaos_scale: float = 1.0,
            cayley_edge_weight: float = 0.5,
            cayley_gram_weight: float = 0.1,
            rose_margin: float = 1.0,
            rose_temperature: float = 0.07,
            cantor_bandwidth: float = 0.1
    ):
        """
        Args:
            num_timestep_bins: Number of timestep buckets
            num_patterns_per_bin: Patterns per timestep
            feature_similarity_weight: Weight for teacher alignment (0.4 default)
            rose_weight: Weight for Rose loss (0.25 default)
            ce_weight: Weight for cross-entropy (0.15 default)
            pattern_diversity_weight: Weight for diversity (0.05 default)
            cayley_weight: Weight for Cayley-Menger (0.10 default - CRITICAL)
            cantor_coherence_weight: Weight for Cantor coherence (0.05 default)
            use_soft_assignment: Use soft pattern assignment
            temperature: Temperature for soft assignment
            cayley_volume_floor: Minimum pentachoron volume
            cayley_chaos_scale: Scale for volume chaos penalty
            cayley_edge_weight: Weight for edge uniformity
            cayley_gram_weight: Weight for Gram conditioning
            rose_margin: Margin for Rose loss
            rose_temperature: Temperature for Rose loss
            cantor_bandwidth: Bandwidth for Cantor coherence
        """
        super().__init__()

        self.num_bins = num_timestep_bins
        self.num_patterns = num_patterns_per_bin
        self.num_classes = num_timestep_bins * num_patterns_per_bin

        # Loss weights (must sum to ~1.0 for balanced training)
        self.feature_sim_weight = feature_similarity_weight
        self.rose_weight = rose_weight
        self.ce_weight = ce_weight
        self.pattern_diversity_weight = pattern_diversity_weight
        self.cayley_weight = cayley_weight
        self.cantor_coherence_weight = cantor_coherence_weight
        self.cantor_bandwidth = cantor_bandwidth

        self.use_soft_assignment = use_soft_assignment
        self.temperature = temperature

        # Geometric losses
        self.cayley_loss = CayleyChaosLoss(
            volume_floor=cayley_volume_floor,
            chaos_scale=cayley_chaos_scale,
            edge_dev_weight=cayley_edge_weight,
            gram_weight=cayley_gram_weight
        )

        self.rose_loss = RoseLoss(
            margin=rose_margin,
            temperature=rose_temperature
        )

    def compute_pattern_diversity(
            self,
            pentachora: torch.Tensor,
            pattern_probs: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute pattern diversity loss to prevent mode collapse.

        Encourages the model to use all patterns rather than
        collapsing to a few dominant ones.

        Args:
            pentachora: [num_classes, 5, D] - pentachora for each pattern
            pattern_probs: [B, num_classes] - pattern probabilities

        Returns:
            diversity_loss: scalar - diversity penalty
        """
        # Average usage across batch
        pattern_usage = pattern_probs.mean(dim=0)  # [num_classes]

        # Entropy of pattern usage (higher = more diverse)
        entropy = -(pattern_usage * torch.log(pattern_usage + 1e-8)).sum()
        max_entropy = torch.log(torch.tensor(float(self.num_classes)))

        # Diversity loss: encourage high entropy
        diversity_loss = max_entropy - entropy

        return diversity_loss

    def compute_cantor_coherence(
            self,
            cantor_values: torch.Tensor,
            timestep_class: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Cantor coherence loss for hierarchical organization.

        Encourages samples in the same timestep bin to have similar
        Cantor values (hierarchical clustering).

        Args:
            cantor_values: [B] - Cantor position encodings
            timestep_class: [B] - timestep bin indices

        Returns:
            coherence_loss: scalar - coherence penalty
        """
        B = cantor_values.shape[0]

        # Create pairwise mask: same timestep = 1, different = 0
        same_timestep = (timestep_class.unsqueeze(0) == timestep_class.unsqueeze(1)).float()

        # Pairwise Cantor distances
        cantor_diff = (cantor_values.unsqueeze(0) - cantor_values.unsqueeze(1)).abs()

        # Weighted distance (penalize large distances within same timestep)
        weighted_dist = (cantor_diff * same_timestep).sum() / (same_timestep.sum() + 1e-8)

        # Apply bandwidth
        coherence_loss = weighted_dist / self.cantor_bandwidth

        return coherence_loss

    def forward(
            self,
            companions_outputs: Dict[str, Dict[str, torch.Tensor]],
            teacher_features_dict: Dict[str, torch.Tensor],
            timesteps: torch.Tensor,
            companions: nn.ModuleDict,
            block_weights: Dict[str, float]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute multi-block geometric loss.

        Args:
            companions_outputs: Dict[block_name, outputs] from all companions
            teacher_features_dict: Dict[block_name, teacher_features] targets
            timesteps: [B] - diffusion timesteps
            companions: ModuleDict of all companions (for accessing pentachora)
            block_weights: Dict[block_name, importance_weight]

        Returns:
            total_loss: scalar - weighted sum of all losses
            metrics: Dict[str, float] - detailed metrics for logging
        """
        total_loss = 0.0
        metrics = {}

        # Process each block
        for block_name, outputs in companions_outputs.items():
            companion = companions[block_name]
            block_weight = block_weights[block_name]
            teacher_features = teacher_features_dict[block_name]

            # Extract outputs
            z = outputs['features']  # [B, D]
            timestep_logits = outputs['timestep_logits']  # [B, num_bins]
            pattern_logits = outputs['pattern_logits']  # [B, num_classes]
            timestep_class = outputs['timestep_class']  # [B]
            cantor_values = outputs['cantor_values']  # [B]

            B, D = z.shape

            # Get pentachora
            pentachora_bins = companion.crystal_pentachora  # [bins, patterns, 5, D]
            pentachora_flat = pentachora_bins.view(-1, 5, D)  # [num_classes, 5, D]

            # ================================================================
            # 1. FEATURE SIMILARITY (teacher alignment)
            # ================================================================
            teacher_z = companion.projection(teacher_features)
            teacher_z = F.normalize(teacher_z, dim=-1)
            feature_sim_loss = 1.0 - F.cosine_similarity(z, teacher_z, dim=-1).mean()


            # ================================================================
            # 2. ROSE LOSS (geometric alignment)
            # ================================================================
            # Compute pattern targets based on timestep and local pattern index
            # For simplicity: use pattern 0 for each timestep
            pattern_targets = timestep_class * companion.num_patterns

            rose_loss_value = self.rose_loss(z, pentachora_flat, pattern_targets)

            # ================================================================
            # 3. CROSS-ENTROPY (classification)
            # ================================================================
            ce_timestep = F.cross_entropy(timestep_logits, timestep_class)
            ce_pattern = F.cross_entropy(pattern_logits, pattern_targets)
            ce_loss = (ce_timestep + ce_pattern) / 2.0

            # ================================================================
            # 4. PATTERN DIVERSITY (anti-collapse)
            # ================================================================
            pattern_probs = F.softmax(pattern_logits, dim=-1)
            diversity_loss = self.compute_pattern_diversity(pentachora_flat, pattern_probs)

            # ================================================================
            # 5. CAYLEY-MENGER (geometric stability) - CRITICAL
            # ================================================================
            # Sample pentachora for Cayley loss (avoid full batch computation)
            sample_size = min(B * 5, pentachora_flat.shape[0])
            sample_indices = torch.randperm(pentachora_flat.shape[0], device=z.device)[:sample_size]
            pentachora_sample = pentachora_flat[sample_indices]

            cayley_loss_value = self.cayley_loss(pentachora_sample)

            # ================================================================
            # 6. CANTOR COHERENCE (hierarchical organization)
            # ================================================================
            cantor_coherence_loss = self.compute_cantor_coherence(cantor_values, timestep_class)

            # ================================================================
            # COMBINE LOSSES
            # ================================================================
            block_loss = (
                    self.feature_sim_weight * feature_sim_loss +
                    self.rose_weight * rose_loss_value +
                    self.ce_weight * ce_loss +
                    self.pattern_diversity_weight * diversity_loss +
                    self.cayley_weight * cayley_loss_value +
                    self.cantor_coherence_weight * cantor_coherence_loss
            )

            # Weight by block importance
            total_loss += block_weight * block_loss

            # ================================================================
            # METRICS
            # ================================================================
            with torch.no_grad():
                # Accuracies
                timestep_acc = (timestep_logits.argmax(dim=-1) == timestep_class).float().mean()
                pattern_acc = (pattern_logits.argmax(dim=-1) == pattern_targets).float().mean()

                # Joint accuracy (both correct)
                timestep_correct = (timestep_logits.argmax(dim=-1) == timestep_class)
                pattern_correct = (pattern_logits.argmax(dim=-1) == pattern_targets)
                full_acc = (timestep_correct & pattern_correct).float().mean()

            # Store metrics
            metrics[f'{block_name}/feature_sim'] = feature_sim_loss.item()
            metrics[f'{block_name}/rose'] = rose_loss_value.item()
            metrics[f'{block_name}/ce'] = ce_loss.item()
            metrics[f'{block_name}/diversity'] = diversity_loss.item()
            metrics[f'{block_name}/cayley'] = cayley_loss_value.item()
            metrics[f'{block_name}/cantor'] = cantor_coherence_loss.item()
            metrics[f'{block_name}/total'] = block_loss.item()
            metrics[f'{block_name}/timestep_acc'] = timestep_acc.item()
            metrics[f'{block_name}/pattern_acc'] = pattern_acc.item()
            metrics[f'{block_name}/full_acc'] = full_acc.item()
            metrics[f'{block_name}/cantor_alpha'] = companion.cantor_stairs.get_alpha()

        # Aggregate metrics
        num_blocks = len(companions_outputs)
        metrics['avg/cayley'] = sum(
            metrics[f'{b}/cayley'] for b in companions_outputs.keys()
        ) / num_blocks
        metrics['avg/timestep_acc'] = sum(
            metrics[f'{b}/timestep_acc'] for b in companions_outputs.keys()
        ) / num_blocks
        metrics['avg/pattern_acc'] = sum(
            metrics[f'{b}/pattern_acc'] for b in companions_outputs.keys()
        ) / num_blocks
        metrics['avg/full_acc'] = sum(
            metrics[f'{b}/full_acc'] for b in companions_outputs.keys()
        ) / num_blocks

        metrics['total_loss'] = total_loss.item()

        return total_loss, metrics


# ============================================================================
# GEODAVID COLLECTIVE - ENHANCED
# ============================================================================

class GeoDavidCollective(nn.Module):
    """
    Multi-block geometric ensemble for SD1.5 distillation - ENHANCED VERSION.

    Complete geometric architecture for learning SD1.5 block representations
    using pentachoron structures, Cantor hierarchical encoding, and geometric
    solidity enforcement.

    ENHANCEMENT: Uses ProjectiveHead architecture for richer projective
    representations in both timestep and pattern classification heads.

    Each block companion learns:
    - Geometric feature projection to pentachoron space
    - Timestep classification (which diffusion step)
    - Pattern classification (which sub-pattern within timestep)
    - Hierarchical Cantor position encoding

    The multi-block ensemble combines knowledge from different SD1.5
    scales/blocks with learnable importance weights.
    """

    def __init__(
            self,
            block_configs: Dict[str, Dict],
            num_timestep_bins: int = 100,
            num_patterns_per_bin: int = 10,
            block_weights: Optional[Dict[str, float]] = None,
            loss_config: Optional[Dict] = None
    ):
        """
        Args:
            block_configs: Dict[block_name, config_dict]
                Required in config_dict:
                    - input_dim: Input feature dimension
                    - scale_dim: Geometric feature space dimension

                Optional in config_dict (with defaults):
                    - use_belly: Use expanded projection (default: True)
                    - belly_expand: Belly expansion factor (default: 2.0)
                    - temperature: Rose loss temperature (default: 0.07)
                    - cantor_alpha_init: Initial Cantor alpha (default: 0.5)
                    - cantor_tau: Cantor tau parameter (default: 0.25)
                    - cantor_levels: Hierarchical levels (default: 12)
                    - simplex_k: Simplex dimension (default: 4)
                    - simplex_seed_base: Seed for initialization (default: 42)

                ProjectiveHead config (auto-selected by scale_dim if not provided):
                    - num_experts: Number of expert pathways
                    - compression_ratio: Bottleneck compression
                    - num_gate_heads: Number of gating heads
                    - expert_dropout: Dropout for experts
                    - attention_dropout: Dropout for attention
                    - head_temperature: Temperature for head
                    - use_head_sparsity: Enable sparsity
                    - head_sparsity_threshold: Sparsity threshold

            num_timestep_bins: Number of timestep buckets (default: 100)
            num_patterns_per_bin: Patterns per timestep (default: 10)
            block_weights: Optional importance weights per block (default: uniform)
            loss_config: Optional loss configuration overrides
        """
        super().__init__()

        self.num_bins = num_timestep_bins
        self.num_patterns = num_patterns_per_bin
        self.block_names = list(block_configs.keys())
        self.block_weights = block_weights or {
            name: 1.0 for name in self.block_names
        }

        print("=" * 80)
        print("GeoDavidCollective: Initializing ENHANCED Geometric Multi-Block System")
        print("=" * 80)
        print(f"  Architecture: ProjectiveHead with multi-expert gating")
        print(f"  Timestep bins: {num_timestep_bins}")
        print(f"  Patterns per bin: {num_patterns_per_bin}")
        print(f"  Total classes: {num_timestep_bins * num_patterns_per_bin}")

        # Build companions (one per block)
        self.companions = nn.ModuleDict()
        for block_name, config in block_configs.items():
            print(f"\n[{block_name}] Creating geometric companion...")
            print(f"  Input dim: {config['input_dim']}")
            print(f"  Scale dim: {config['scale_dim']}")

            # Auto-select ProjectiveHead config if not specified
            scale_dim = config['scale_dim']
            head_config = get_projective_head_config(scale_dim)

            # Override with user-provided values
            for key in head_config.keys():
                if key in config:
                    head_config[key] = config[key]

            print(f"  ProjectiveHead config:")
            print(f"    Experts: {head_config['num_experts']}")
            print(f"    Compression: {head_config['compression_ratio']}x")
            print(f"    Gate heads: {head_config['num_gate_heads']}")

            # Create companion with ProjectiveHead
            companion = GeoDavidBlockCompanion(
                block_name=block_name,
                input_dim=config['input_dim'],
                scale_dim=scale_dim,
                num_timestep_bins=num_timestep_bins,
                num_patterns_per_bin=num_patterns_per_bin,
                use_belly=config.get('use_belly', True),
                belly_expand=config.get('belly_expand', 2.0),
                temperature=config.get('temperature', 0.07),
                cantor_alpha_init=config.get('cantor_alpha_init', 0.5),
                cantor_tau=config.get('cantor_tau', 0.25),
                cantor_levels=config.get('cantor_levels', 12),
                simplex_k=config.get('simplex_k', 4),
                simplex_seed_base=config.get('simplex_seed_base', 42),
                # ProjectiveHead parameters
                **head_config
            )
            self.companions[block_name] = companion

        # Geometric loss calculator
        print("\nInitializing geometric multi-scale loss system...")
        loss_config = loss_config or {}
        self.loss_calculator = GeometricMultiScaleLoss(
            num_timestep_bins=num_timestep_bins,
            num_patterns_per_bin=num_patterns_per_bin,
            **loss_config
        )

        # Calculate total parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )

        print("\n" + "=" * 80)
        print(f"GeoDavidCollective Initialized Successfully (ENHANCED)")
        print(f"  Blocks: {len(self.companions)}")
        print(f"  Timestep bins: {num_timestep_bins}")
        print(f"  Patterns per bin: {num_patterns_per_bin}")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print("=" * 80)

    def forward(
            self,
            features_dict: Dict[str, torch.Tensor],
            timesteps: torch.Tensor
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Forward pass through all block companions.

        Args:
            features_dict: Dict[block_name, features] - features from each block
            timesteps: [B] - diffusion timesteps

        Returns:
            Dict[block_name, outputs] - outputs from each companion
        """
        outputs = {}
        for block_name, features in features_dict.items():
            if block_name in self.companions:
                outputs[block_name] = self.companions[block_name](
                    features, timesteps
                )
        return outputs

    def compute_loss(
            self,
            companions_outputs: Dict[str, Dict[str, torch.Tensor]],
            teacher_features_dict: Dict[str, torch.Tensor],
            timesteps: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute geometric multi-block loss.

        Args:
            companions_outputs: Outputs from forward()
            teacher_features_dict: Teacher feature targets
            timesteps: Diffusion timesteps

        Returns:
            loss: Total weighted loss
            metrics: Detailed metrics dict
        """
        return self.loss_calculator(
            companions_outputs,
            teacher_features_dict,
            timesteps,
            self.companions,
            self.block_weights
        )

    def get_companion(self, block_name: str) -> GeoDavidBlockCompanion:
        """Get specific block companion."""
        return self.companions[block_name]

    def get_all_companions(self) -> Dict[str, GeoDavidBlockCompanion]:
        """Get all companions."""
        return dict(self.companions)

    def get_pentachora(self, block_name: str) -> torch.Tensor:
        """Get pentachora for a specific block."""
        return self.companions[block_name].crystal_pentachora

    def get_cantor_alphas(self) -> Dict[str, float]:
        """Get current alpha values from all Cantor staircases."""
        return {
            name: companion.cantor_stairs.get_alpha()
            for name, companion in self.companions.items()
        }

    def set_block_weight(self, block_name: str, weight: float):
        """Set importance weight for a specific block."""
        self.block_weights[block_name] = weight

    def get_model_info(self) -> Dict:
        """Get comprehensive model information."""
        info = {
            'architecture': 'GeoDavidCollective (ENHANCED with ProjectiveHead)',
            'num_blocks': len(self.companions),
            'blocks': list(self.block_names),
            'num_timestep_bins': self.num_bins,
            'num_patterns_per_bin': self.num_patterns,
            'total_classes': self.num_bins * self.num_patterns,
            'block_weights': self.block_weights,
            'total_parameters': sum(p.numel() for p in self.parameters()),
            'trainable_parameters': sum(
                p.numel() for p in self.parameters() if p.requires_grad
            ),
            'cantor_alphas': self.get_cantor_alphas(),
            'companions': {}
        }

        for name, companion in self.companions.items():
            info['companions'][name] = {
                'input_dim': companion.input_dim,
                'scale_dim': companion.scale_dim,
                'simplex_k': companion.simplex_k,
                'num_vertices': companion.num_vertices,
                'timestep_head': {
                    'type': 'ProjectiveHead',
                    'num_experts': companion.timestep_head.num_experts,
                    'bottleneck_dim': companion.timestep_head.bottleneck_dim,
                    'num_gate_heads': companion.timestep_head.num_gate_heads
                },
                'pattern_head': {
                    'type': 'ProjectiveHead',
                    'num_experts': companion.pattern_head.num_experts,
                    'bottleneck_dim': companion.pattern_head.bottleneck_dim,
                    'num_gate_heads': companion.pattern_head.num_gate_heads
                }
            }

        return info

    def __repr__(self):
        info = self.get_model_info()
        return (
            f"GeoDavidCollective(ENHANCED with ProjectiveHead)\n"
            f"  Blocks: {info['num_blocks']}\n"
            f"  Timestep bins: {info['num_timestep_bins']}\n"
            f"  Patterns: {info['num_patterns_per_bin']}/bin\n"
            f"  Parameters: {info['total_parameters']:,}"
        )


# ============================================================================
# DEMO / TESTING CODE
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("GeoDavidCollective Demo - ENHANCED Version")
    print("Testing multi-block geometric architecture with ProjectiveHead")
    print("=" * 80 + "\n")

    # ========================================================================
    # 1. CONFIGURATION
    # ========================================================================

    print("Step 1: Setting up configuration...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")

    # Block configurations (mimicking SD1.5 UNet structure)
    block_configs = {
        'down_0': {
            'input_dim': 320,
            'scale_dim': 128,
            # ProjectiveHead params will be auto-selected
        },
        'down_1': {
            'input_dim': 640,
            'scale_dim': 192,
        },
        'mid': {
            'input_dim': 1280,
            'scale_dim': 256,
            # Can override ProjectiveHead params if desired
            'num_experts': 4,  # Custom: use 4 experts for mid block
        },
        'up_3': {
            'input_dim': 640,
            'scale_dim': 192,
        }
    }

    # Block importance weights
    block_weights = {
        'down_0': 1.0,
        'down_1': 1.2,
        'mid': 1.5,  # Mid block is most important
        'up_3': 1.0
    }

    # Loss configuration
    loss_config = {
        'feature_similarity_weight': 0.4,
        'rose_weight': 0.25,
        'ce_weight': 0.15,
        'pattern_diversity_weight': 0.05,
        'cayley_weight': 0.10,  # Critical for stability
        'cantor_coherence_weight': 0.05
    }

    print(f"✓ Configured {len(block_configs)} blocks")
    print(f"  Blocks: {list(block_configs.keys())}")
    print(f"  Loss weights: {list(loss_config.keys())}\n")

    # ========================================================================
    # 2. MODEL INSTANTIATION
    # ========================================================================

    print("Step 2: Creating GeoDavidCollective model (ENHANCED)...")

    model = GeoDavidCollective(
        block_configs=block_configs,
        num_timestep_bins=100,
        num_patterns_per_bin=10,
        block_weights=block_weights,
        loss_config=loss_config
    )

    print()

    # ========================================================================
    # 3. SYNTHETIC DATA GENERATION
    # ========================================================================

    print("Step 3: Generating synthetic data...")

    batch_size = 16
    model = model.to(device)

    # Create synthetic teacher features (would normally come from SD1.5)
    teacher_features_dict = {}
    features_dict = {}

    for block_name, config in block_configs.items():
        # Teacher features (target)
        teacher_features_dict[block_name] = torch.randn(
            batch_size, config['input_dim'], device=device
        )
        # Student features (input) - perturbed teacher
        features_dict[block_name] = teacher_features_dict[block_name] + \
                                    0.1 * torch.randn_like(teacher_features_dict[block_name])

    # Random timesteps [0, 1000]
    timesteps = torch.randint(0, 1000, (batch_size,), device=device).float()

    print(f"✓ Generated synthetic data:")
    print(f"  Batch size: {batch_size}")
    print(f"  Blocks: {list(block_configs.keys())}")
    print(f"  Timestep range: [{timesteps.min():.0f}, {timesteps.max():.0f}]")
    print(f"  Device: {device}\n")

    # ========================================================================
    # 4. FORWARD PASS
    # ========================================================================

    print("Step 4: Running forward pass through all blocks...")

    with torch.no_grad():
        outputs = model(features_dict, timesteps)

    print("✓ Forward pass complete!")
    print(f"  Blocks processed: {len(outputs)}")

    # Show output structure for one block
    example_block = list(outputs.keys())[0]
    example_output = outputs[example_block]
    print(f"\n  Example block '{example_block}' outputs:")
    for key, value in example_output.items():
        if isinstance(value, torch.Tensor):
            print(f"    {key}: {list(value.shape)}")
    print()

    # ========================================================================
    # 5. GEOMETRIC PROPERTIES
    # ========================================================================

    print("Step 5: Checking geometric properties...")

    # Check pentachoron volumes for each block
    print("\n  Pentachoron volumes per block:")
    for block_name in block_configs.keys():
        pentachora = model.get_pentachora(block_name)  # [bins, patterns, 5, dim]

        # Compute volumes for a sample (first bin, first pattern)
        sample_pentachoron = pentachora[0, 0].unsqueeze(0)  # [1, 5, dim]
        cayley_loss = CayleyChaosLoss()
        volume = cayley_loss.compute_cayley_menger_volume(sample_pentachoron)

        print(f"    {block_name}: volume={volume.item():.6f} "
              f"(shape={list(pentachora.shape)})")

    # Check Cantor alpha values
    print("\n  Cantor staircase alpha values:")
    alphas = model.get_cantor_alphas()
    for block_name, alpha in alphas.items():
        print(f"    {block_name}: α={alpha:.4f}")

    print()

    # ========================================================================
    # 6. LOSS COMPUTATION
    # ========================================================================

    print("Step 6: Computing geometric multi-scale loss...")

    with torch.no_grad():
        loss, metrics = model.compute_loss(
            outputs,
            teacher_features_dict,
            timesteps
        )

    print("✓ Loss computation complete!\n")

    # ========================================================================
    # 7. RESULTS & METRICS
    # ========================================================================

    print("=" * 80)
    print("RESULTS: Loss Components & Metrics")
    print("=" * 80)

    # Overall metrics
    print("\n📊 Overall Metrics:")
    print(f"  Total Loss:          {metrics['total_loss']:.4f}")
    print(f"  Avg Cayley Loss:     {metrics['avg/cayley']:.4f} (geometry stability)")
    print(f"  Avg Timestep Acc:    {metrics['avg/timestep_acc']:.1%} (timestep classification)")
    print(f"  Avg Pattern Acc:     {metrics['avg/pattern_acc']:.1%} (pattern classification)")
    print(f"  Avg Full Acc:        {metrics['avg/full_acc']:.1%} (timestep+pattern joint)")

    # Per-block breakdown
    print("\n📋 Per-Block Breakdown:")
    for block_name in block_configs.keys():
        weight = block_weights[block_name]
        cayley = metrics.get(f'{block_name}/cayley', 0.0)
        full_acc = metrics.get(f'{block_name}/full_acc', 0.0)
        alpha = metrics.get(f'{block_name}/cantor_alpha', 0.0)

        print(f"\n  {block_name} (weight={weight}):")
        print(f"    Cayley:      {cayley:.4f}")
        print(f"    Full Acc:    {full_acc:.1%}")
        print(f"    Cantor α:    {alpha:.4f}")

    # Loss components
    print("\n🔧 Loss Component Weights:")
    loss_components = [
        ('Feature Similarity', 'feature_similarity_weight'),
        ('Rose', 'rose_weight'),
        ('Cross-Entropy', 'ce_weight'),
        ('Pattern Diversity', 'pattern_diversity_weight'),
        ('Cayley', 'cayley_weight'),
        ('Cantor Coherence', 'cantor_coherence_weight')
    ]
    for display_name, param_name in loss_components:
        weight = loss_config.get(param_name, 0.0)
        print(f"  {display_name:18s}: {weight:.2f}")

    # ========================================================================
    # 8. HEALTH CHECK
    # ========================================================================

    print("\n" + "=" * 80)
    print("HEALTH CHECK: Geometric Stability")
    print("=" * 80)

    cayley_threshold = 0.1
    cayley_ok = metrics['avg/cayley'] < cayley_threshold

    print(f"\n✓ Cayley Loss: {metrics['avg/cayley']:.4f} ", end="")
    if cayley_ok:
        print(f"(< {cayley_threshold}) ✅ HEALTHY")
    else:
        print(f"(>= {cayley_threshold}) ⚠️  NEEDS TUNING")

    # Check for volume collapse
    volumes_ok = True
    for block_name in block_configs.keys():
        pentachora = model.get_pentachora(block_name)
        sample = pentachora[0, 0].unsqueeze(0)
        cayley_loss = CayleyChaosLoss()
        volume = cayley_loss.compute_cayley_menger_volume(sample)
        if volume.item() < 1e-4:
            volumes_ok = False
            break

    print(f"✓ Pentachoron Volumes: ", end="")
    if volumes_ok:
        print("> 1e-4 ✅ NO COLLAPSE")
    else:
        print("<= 1e-4 ⚠️  POTENTIAL COLLAPSE")

    # Accuracy check
    acc_threshold = 0.3
    acc_ok = metrics['avg/full_acc'] > acc_threshold

    print(f"✓ Full Accuracy: {metrics['avg/full_acc']:.1%} ", end="")
    if acc_ok:
        print(f"(> {acc_threshold*100:.0f}%) ✅ LEARNING")
    else:
        print(f"(<= {acc_threshold*100:.0f}%) ⚠️  EARLY TRAINING")

    # ========================================================================
    # 9. SUMMARY
    # ========================================================================

    print("\n" + "=" * 80)
    print("SUMMARY: GeoDavidCollective Demo Complete (ENHANCED)")
    print("=" * 80)

    print("\n✅ All systems operational!")

    print("\n🎯 Key Features Demonstrated:")
    print("  ✓ Multi-block ensemble (4 blocks)")
    print("  ✓ Pentachoron structure (5-vertex simplices)")
    print("  ✓ ProjectiveHead classification heads")
    print("  ✓ Multi-expert pathways with cross-attention")
    print("  ✓ Multi-head gating for ensemble predictions")
    print("  ✓ Cantor staircase hierarchical encoding")
    print("  ✓ 6 geometric loss components")
    print("  ✓ Per-block importance weighting")
    print("  ✓ Automatic multi-scale aggregation")

    print("\n📈 Next Steps for Real Training:")
    print("  1. Replace synthetic data with SD1.5 extracted features")
    print("  2. Train for 100+ epochs with real diffusion data")
    print("  3. Monitor Cayley loss (keep < 0.1)")
    print("  4. Watch accuracies improve (target: >90%)")
    print("  5. Tune block weights based on performance")
    print("  6. Adjust loss component weights if needed")
    print("  7. Experiment with ProjectiveHead configurations")

    print("\n💡 Architecture Highlights:")
    model_info = model.get_model_info()
    print(f"  Total parameters:    {model_info['total_parameters']:,}")
    print(f"  Trainable params:    {model_info['trainable_parameters']:,}")
    print(f"  Total classes:       {model_info['total_classes']}")
    print(f"  Timestep bins:       {model_info['num_timestep_bins']}")
    print(f"  Patterns per bin:    {model_info['num_patterns_per_bin']}")

    print("\n  ProjectiveHead Info:")
    for block_name, info in model_info['companions'].items():
        print(f"    {block_name}:")
        print(f"      Timestep: {info['timestep_head']['num_experts']} experts, "
              f"{info['timestep_head']['num_gate_heads']} gates")
        print(f"      Pattern:  {info['pattern_head']['num_experts']} experts, "
              f"{info['pattern_head']['num_gate_heads']} gates")

    print("\n" + "=" * 80)
    print("Demo finished successfully! 🚀")
    print("=" * 80)

    # Show model architecture
    print("\n" + str(model))

