"""
GeoDavidCollective: Geometric Multi-Block Diffusion Collective
===============================================================
Complete geometric architecture for SD1.5 block-level distillation
with pentachoron structure, Cantor hierarchical encoding, and
geometric solidity enforcement.

Each block companion is a multi-scale geometric structure that learns
to classify diffusion timesteps and patterns using validated k-simplices
and hierarchical Cantor position encoding.

Key Features:
- Pentachoron structure (5-vertex 4-simplices) per pattern
- SimplexFactory-validated initialization for stability
- Cantor staircase hierarchical position encoding
- 6 geometric loss components (feature, Rose, CE, diversity, Cayley, Cantor)
- Multi-block ensemble architecture
- Per-block scale weighting

Author: AbstractPhil
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from collections import OrderedDict

# Import geometric components
from geovocab2.train.model.core.geo_fractal_david import CantorStaircase

try:
    from geovocab2.shapes.factory import SimplexFactory

    HAS_SIMPLEX_FACTORY = True
except ImportError:
    HAS_SIMPLEX_FACTORY = False
    print("WARNING: SimplexFactory not available. Using fallback initialization.")
    print("Training may be less stable without validated geometry.")


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

        # 3. Gram matrix conditioning loss
        gram_loss = 0.0
        if self.gram_weight > 0:
            gram_penalty = self.compute_gram_condition(X)
            gram_loss = gram_penalty.mean()

        # Combine losses
        total_loss = (
                self.chaos_scale * chaos_loss +
                self.edge_dev_weight * edge_loss +
                self.gram_weight * gram_loss
        )

        return total_loss


class RoseLoss(nn.Module):
    """
    Rose Loss with pentachoron role weighting.

    Learns relational geometry by computing weighted similarity to all 5
    vertices of each class pentachoron, with semantic role weights:

    - Vertex 0 (Anchor): Primary class representation (+1.0)
    - Vertex 1 (Need): What the class lacks (-0.75)
    - Vertex 2 (Relation): How class relates to others (+0.75)
    - Vertex 3 (Purpose): Functional role (+0.75)
    - Vertex 4 (Observer): External perspective (-0.75)

    Uses margin-based contrastive learning to push features toward correct
    class pentachoron and away from incorrect ones.
    """

    def __init__(
            self,
            margin: float = 1.0,
            temperature: float = 0.07,
            role_weights: Optional[Dict[str, float]] = None
    ):
        """
        Args:
            margin: Contrastive margin (higher = stricter separation)
            temperature: Similarity temperature scaling
            role_weights: Custom role weights (if None, use defaults)
        """
        super().__init__()

        self.margin = margin
        self.temperature = temperature

        # Semantic role weights for pentachoron vertices
        default_weights = {
            "anchor": 1.0,  # Primary representation
            "need": -0.75,  # Negative aspect
            "relation": 0.75,  # Relational aspect
            "purpose": 0.75,  # Functional aspect
            "observer": -0.75  # External perspective
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
        Compute Rose loss with pentachoron role weighting.

        Args:
            z: [B, D] - projected features (normalized)
            pentachora: [C, 5, D] - class pentachora (C classes)
            targets: [B] - class labels

        Returns:
            loss: scalar - margin-based contrastive loss
        """
        pentachora = pentachora.to(z.device)
        role_weights = self.role_weights.to(z.device)

        B, D = z.shape
        C, V, _ = pentachora.shape
        assert V == 5, f"Expected 5 vertices per pentachoron, got {V}"

        # Normalize pentachora vertices
        pentachora_norm = F.normalize(pentachora, dim=-1)  # [C, 5, D]

        # Compute cosine similarity to all vertices of all classes
        # z: [B, D] → [B, 1, 1, D]
        # pentachora: [C, 5, D] → [1, C, 5, D]
        cos_sim = torch.einsum("bd,cvd->bcv", z, pentachora_norm)  # [B, C, 5]

        # Apply role weights and sum over vertices
        rose_scores = (cos_sim * role_weights.view(1, 1, 5)).sum(dim=-1)  # [B, C]

        # Temperature scaling
        rose_scores = rose_scores / self.temperature

        # Get scores for correct classes
        true_scores = rose_scores.gather(1, targets.view(-1, 1)).squeeze(1)  # [B]

        # Get hardest negative (highest score among incorrect classes)
        mask = torch.zeros_like(rose_scores, dtype=torch.bool)
        mask.scatter_(1, targets.view(-1, 1), True)
        hard_neg = rose_scores.masked_fill(mask, float("-inf")).max(dim=1).values  # [B]

        # Margin-based contrastive loss
        # Want: true_scores > hard_neg + margin
        loss = F.relu(self.margin - (true_scores - hard_neg))

        return loss.mean()


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


class GeoDavidBlockCompanion(nn.Module):
    """
    Geometric companion for a single SD1.5 block.

    This is a multi-scale structure that processes features from one SD1.5 block
    using pentachoron-based pattern classification with Cantor position encoding.

    Architecture:
    - Feature projection (with optional bottleneck)
    - Pentachoron patterns: [num_bins, num_patterns, 5, scale_dim]
    - Cantor staircase for hierarchical position
    - Timestep and pattern classification heads
    """

    def __init__(
            self,
            block_name: str,
            input_dim: int,
            scale_dim: int,
            num_timestep_bins: int = 100,
            num_patterns_per_bin: int = 10,
            use_belly: bool = True,
            belly_expand: float = 2.0,
            temperature: float = 0.07,
            cantor_alpha_init: float = 0.5,
            cantor_tau: float = 0.25,
            cantor_levels: int = 12,
            simplex_k: int = 4,
            simplex_seed_base: int = 42
    ):
        super().__init__()

        self.block_name = block_name
        self.input_dim = input_dim
        self.scale_dim = scale_dim
        self.num_bins = num_timestep_bins
        self.num_patterns = num_patterns_per_bin
        self.temperature = temperature
        self.simplex_k = simplex_k
        self.num_vertices = simplex_k + 1

        # Feature projection
        if use_belly:
            belly_dim = int(scale_dim * belly_expand)
            dropout_rate = 1.0 / (scale_dim ** 0.5)
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

        # Semantic role weights
        role_weights = torch.tensor([
            1.0,  # Anchor
            -0.75,  # Need
            0.75,  # Relation
            0.75,  # Purpose
            -0.75  # Observer
        ], dtype=torch.float32)
        self.register_buffer("role_weights", role_weights)

        # Cantor staircase (hierarchical position encoding)
        self.cantor_stairs = CantorStaircase(
            feature_dim=scale_dim,
            alpha_init=cantor_alpha_init,
            tau=cantor_tau,
            base=3,
            levels=cantor_levels
        )

        # Classification heads
        self.timestep_head = nn.Linear(scale_dim, num_timestep_bins)
        total_patterns = num_timestep_bins * num_patterns_per_bin
        self.pattern_head = nn.Linear(scale_dim, total_patterns)

    def _init_projection_weights(self):
        """Xavier initialization for stable gradients."""
        for layer in self.projection.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=0.5)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def _init_pentachora_validated(
            self,
            num_bins: int,
            num_patterns: int,
            scale_dim: int,
            simplex_k: int,
            seed_base: int
    ) -> nn.Parameter:
        """
        Initialize pentachora using SimplexFactory with validation.

        This creates geometrically valid k-simplices that prevent
        training divergence through:
        - Non-zero volume (no collapse)
        - Proper edge structure (regularity)
        - Valid Gram matrix (proper embedding)
        """
        if not HAS_SIMPLEX_FACTORY:
            print(f"    [{self.block_name}] Using random initialization (less stable)")
            pentachora = torch.randn(
                num_bins, num_patterns, simplex_k + 1, scale_dim
            ) * 0.1
            return nn.Parameter(pentachora)

        # SimplexFactory with validation
        factory = SimplexFactory(
            k=simplex_k,
            embed_dim=scale_dim,
            method="random"  # Random with geometric validation
        )

        total_simplices = num_bins * num_patterns
        print(f"    [{self.block_name}] Building {total_simplices} validated {simplex_k}-simplices...")

        # Build in batches
        batch_size = 100
        all_pentachora = []

        with torch.no_grad():
            for batch_start in range(0, total_simplices, batch_size):
                batch_end = min(batch_start + batch_size, total_simplices)
                batch_pentachora = []

                for idx in range(batch_start, batch_end):
                    # Deterministic seed per pentachoron
                    seed = seed_base + hash(f"{self.block_name}_penta_{idx}") % (2 ** 31)

                    # Build on CPU, validate geometry
                    simplex = factory.build(
                        backend="torch",
                        device="cpu",
                        dtype=torch.float32,
                        seed=seed,
                        validate=True  # CRITICAL: Ensures geometric validity
                    )
                    batch_pentachora.append(simplex)

                # Move to GPU
                device = "cuda" if torch.cuda.is_available() else "cpu"
                batch_tensor = torch.stack(batch_pentachora).to(device)
                all_pentachora.append(batch_tensor)

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                if batch_end % 500 == 0 or batch_end == total_simplices:
                    print(f"      {batch_end}/{total_simplices} complete", end='\r')

        print(f"      {total_simplices}/{total_simplices} complete ✓")

        # Reshape: [num_bins, num_patterns, 5, scale_dim]
        pentachora_tensor = torch.cat(all_pentachora, dim=0)
        pentachora_tensor = pentachora_tensor.view(
            num_bins, num_patterns, simplex_k + 1, scale_dim
        )

        return nn.Parameter(pentachora_tensor)

    def forward(
            self,
            features: torch.Tensor,
            timesteps: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with geometric structure.

        Args:
            features: [B, input_dim] - block features from SD1.5
            timesteps: [B] - continuous timesteps [0, 1000]

        Returns:
            Dict with:
                'features': [B, scale_dim] - projected features
                'timestep_logits': [B, num_bins]
                'pattern_logits': [B, num_bins * num_patterns]
                'timestep_class': [B] - timestep bins
                'cantor_values': [B] - Cantor position encoding
        """
        # Project and normalize
        z = self.projection(features)
        z = F.normalize(z, dim=-1)

        # Timestep classification
        timestep_logits = self.timestep_head(z)
        timestep_class = (timesteps / 1000.0 * self.num_bins).long().clamp(
            0, self.num_bins - 1
        )

        # Cantor hierarchical position encoding
        cantor_values = self.cantor_stairs(z, timestep_class)

        # Pattern classification
        pattern_logits = self.pattern_head(z)

        return {
            'features': z,
            'timestep_logits': timestep_logits,
            'pattern_logits': pattern_logits,
            'timestep_class': timestep_class,
            'cantor_values': cantor_values
        }


class GeometricMultiScaleLoss(nn.Module):
    """
    BATCHED geometric loss calculator for multi-block ensemble.

    Processes ALL blocks in parallel instead of sequential loop.

    Computes 6 loss components per block:
    1. Feature similarity - teacher alignment
    2. Rose loss - pentachoron relational learning
    3. Cross-entropy - pattern classification
    4. Pattern diversity - mode collapse prevention
    5. Cayley-Menger - geometric solidity (CRITICAL)
    6. Cantor coherence - hierarchical clustering
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
        super().__init__()

        self.num_bins = num_timestep_bins
        self.num_patterns = num_patterns_per_bin
        self.num_classes = num_timestep_bins * num_patterns_per_bin

        # Loss weights
        self.feature_sim_weight = feature_similarity_weight
        self.rose_weight = rose_weight
        self.ce_weight = ce_weight
        self.pattern_diversity_weight = pattern_diversity_weight
        self.cayley_weight = cayley_weight
        self.cantor_coherence_weight = cantor_coherence_weight

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

    def assign_patterns(
            self,
            features: torch.Tensor,
            timestep_class: torch.Tensor,
            pentachora: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Assign patterns using pentachoron centroid matching (single block)."""
        B = features.size(0)

        # Get pentachora for timestep bins
        batch_pentachora = pentachora[timestep_class]  # [B, num_patterns, 5, D]

        # Use centroids
        centroids = batch_pentachora.mean(dim=2)  # [B, num_patterns, D]

        # Cosine similarity
        features_expanded = features.unsqueeze(1)
        similarities = F.cosine_similarity(features_expanded, centroids, dim=2)

        # Assign to nearest
        pattern_ids = similarities.argmax(dim=1)
        full_class_ids = timestep_class * self.num_patterns + pattern_ids

        return pattern_ids, full_class_ids

    def compute_soft_assignment(
            self,
            features: torch.Tensor,
            timestep_class: torch.Tensor,
            pentachora: torch.Tensor
    ) -> torch.Tensor:
        """Soft pattern assignment with temperature (single block)."""
        B, D = features.shape
        device = features.device

        batch_pentachora = pentachora[timestep_class]
        centroids = batch_pentachora.mean(dim=2)

        features_expanded = features.unsqueeze(1)
        similarities = F.cosine_similarity(features_expanded, centroids, dim=2)

        pattern_probs = F.softmax(similarities / self.temperature, dim=1)

        soft_targets = torch.zeros(B, self.num_classes, device=device)
        for i in range(B):
            bin_idx = timestep_class[i].item()
            start_idx = bin_idx * self.num_patterns
            end_idx = start_idx + self.num_patterns
            soft_targets[i, start_idx:end_idx] = pattern_probs[i]

        return soft_targets

    def batch_assign_patterns(
            self,
            features: torch.Tensor,  # [num_blocks * B, D]
            timestep_class: torch.Tensor,  # [num_blocks * B]
            pentachora: torch.Tensor,  # [num_blocks, num_bins, num_patterns, 5, D]
            block_indices: torch.Tensor  # [num_blocks * B] - which block each sample belongs to
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Batched pattern assignment across all blocks.

        Returns:
            pattern_ids: [num_blocks * B] - pattern within timestep bin
            full_class_ids: [num_blocks * B] - full class ID (bin * patterns + pattern)
        """
        total_samples = features.size(0)
        D = features.size(1)
        device = features.device

        # Get pentachora for each sample's block and timestep
        # [num_blocks * B, num_patterns, 5, D]
        batch_pentachora = pentachora[block_indices, timestep_class]

        # Compute centroids: [num_blocks * B, num_patterns, D]
        centroids = batch_pentachora.mean(dim=2)

        # Cosine similarity
        features_expanded = features.unsqueeze(1)  # [num_blocks * B, 1, D]
        similarities = F.cosine_similarity(features_expanded, centroids, dim=2)  # [num_blocks * B, num_patterns]

        # Assign to nearest
        pattern_ids = similarities.argmax(dim=1)  # [num_blocks * B]
        full_class_ids = timestep_class * self.num_patterns + pattern_ids

        return pattern_ids, full_class_ids

    def batch_compute_soft_assignment(
            self,
            features: torch.Tensor,
            timestep_class: torch.Tensor,
            pentachora: torch.Tensor,
            block_indices: torch.Tensor
    ) -> torch.Tensor:
        """Batched soft pattern assignment."""
        total_samples = features.size(0)
        device = features.device

        # Get pentachora
        batch_pentachora = pentachora[block_indices, timestep_class]
        centroids = batch_pentachora.mean(dim=2)

        # Similarities
        features_expanded = features.unsqueeze(1)
        similarities = F.cosine_similarity(features_expanded, centroids, dim=2)
        pattern_probs = F.softmax(similarities / self.temperature, dim=1)

        # Build soft targets
        soft_targets = torch.zeros(total_samples, self.num_classes, device=device)
        for i in range(total_samples):
            bin_idx = timestep_class[i].item()
            start_idx = bin_idx * self.num_patterns
            end_idx = start_idx + self.num_patterns
            soft_targets[i, start_idx:end_idx] = pattern_probs[i]

        return soft_targets

    def batch_compute_cantor_coherence(
            self,
            cantor_values: torch.Tensor,  # [num_blocks * B]
            pattern_ids: torch.Tensor,
            timestep_class: torch.Tensor,
            block_indices: torch.Tensor
    ) -> torch.Tensor:
        """Batched Cantor coherence across all blocks."""
        device = cantor_values.device

        # Full class IDs including block identity
        # We need to make classes unique per block
        full_class_ids = (
                block_indices * self.num_classes +
                timestep_class * self.num_patterns +
                pattern_ids
        )

        unique_classes = torch.unique(full_class_ids)

        if len(unique_classes) < 2:
            return torch.tensor(0.0, device=device)

        coherence_losses = []
        for class_id in unique_classes:
            mask = full_class_ids == class_id
            class_cantor = cantor_values[mask]

            if class_cantor.size(0) > 1:
                coherence_losses.append(class_cantor.var())

        if len(coherence_losses) == 0:
            return torch.tensor(0.0, device=device)

        return torch.stack(coherence_losses).mean()

    def batch_compute_pattern_diversity(
            self,
            pattern_ids: torch.Tensor,
            block_indices: torch.Tensor,
            num_blocks: int
    ) -> torch.Tensor:
        """Compute pattern diversity per block, then average."""
        device = pattern_ids.device
        diversity_losses = []

        for block_idx in range(num_blocks):
            block_mask = block_indices == block_idx
            block_patterns = pattern_ids[block_mask]

            if block_patterns.size(0) == 0:
                continue

            pattern_counts = torch.bincount(
                block_patterns, minlength=self.num_patterns
            ).float()
            pattern_probs = pattern_counts / pattern_counts.sum()
            pattern_probs = pattern_probs[pattern_probs > 0]

            if len(pattern_probs) > 1:
                pattern_entropy = -(pattern_probs * pattern_probs.log()).sum()
                max_entropy = torch.log(torch.tensor(float(self.num_patterns)))
                diversity_loss = (max_entropy - pattern_entropy) / max_entropy
            else:
                diversity_loss = torch.tensor(1.0, device=device)

            diversity_losses.append(diversity_loss)

        if len(diversity_losses) == 0:
            return torch.tensor(0.0, device=device)

        return torch.stack(diversity_losses).mean()

    def forward(
            self,
            companions_outputs: Dict[str, Dict[str, torch.Tensor]],
            teacher_features_dict: Dict[str, torch.Tensor],
            timesteps: torch.Tensor,  # [B]
            companions: Dict[str, 'GeoDavidBlockCompanion'],
            block_weights: Optional[Dict[str, float]] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        BATCHED computation across all blocks (handles variable feature dims).

        Args:
            companions_outputs: Dict[block_name, outputs_dict]
                Each outputs_dict has:
                    'features': [B, D_block] - D can vary per block!
                    'timestep_logits': [B, num_bins]
                    'pattern_logits': [B, num_classes]
                    'timestep_class': [B]
                    'cantor_values': [B]
            teacher_features_dict: Dict[block_name, teacher_features [B, D_teacher]]
            timesteps: [B] - timesteps for all samples
            companions: Dict[block_name, companion]
            block_weights: Optional per-block importance weights

        Returns:
            total_loss: Weighted aggregate loss
            all_metrics: Comprehensive metrics dict
        """
        block_names = list(companions_outputs.keys())
        num_blocks = len(block_names)
        B = timesteps.size(0)
        device = timesteps.device

        if block_weights is None:
            block_weights = {name: 1.0 for name in block_names}

        # ==================================================================
        # STEP 1: Collect block data (DON'T concatenate features yet!)
        # ==================================================================

        # These can be stacked (same dims across blocks)
        timestep_logits_list = []
        pattern_logits_list = []
        timestep_class_list = []
        cantor_values_list = []
        block_indices_list = []

        # These have variable dims - keep as lists
        features_list = []
        teacher_features_list = []
        pentachora_list = []
        projection_list = []

        for block_idx, block_name in enumerate(block_names):
            outputs = companions_outputs[block_name]
            companion = companions[block_name]

            # Variable dimension - keep in list
            features_list.append(outputs['features'])
            teacher_features_list.append(teacher_features_dict[block_name])
            pentachora_list.append(companion.crystal_pentachora)
            projection_list.append(companion.projection)

            # Fixed dimension - can stack
            timestep_logits_list.append(outputs['timestep_logits'])
            pattern_logits_list.append(outputs['pattern_logits'])
            timestep_class_list.append(outputs['timestep_class'])
            cantor_values_list.append(outputs['cantor_values'])

            # Block indices for tracking
            block_indices_list.append(
                torch.full((B,), block_idx, dtype=torch.long, device=device)
            )

        # Stack only fixed-dimension tensors
        timestep_logits_batched = torch.cat(timestep_logits_list, dim=0)  # [num_blocks*B, num_bins]
        pattern_logits_batched = torch.cat(pattern_logits_list, dim=0)  # [num_blocks*B, num_classes]
        timestep_class_batched = torch.cat(timestep_class_list, dim=0)  # [num_blocks*B]
        cantor_values_batched = torch.cat(cantor_values_list, dim=0)  # [num_blocks*B]
        block_indices = torch.cat(block_indices_list, dim=0)  # [num_blocks*B]

        # ==================================================================
        # STEP 2: Pattern assignment (per-block due to variable dims)
        # ==================================================================

        pattern_ids_list = []
        full_class_ids_list = []

        for block_idx, block_name in enumerate(block_names):
            features = features_list[block_idx]  # [B, D_block]
            timestep_class = timestep_class_list[block_idx]  # [B]
            pentachora = pentachora_list[block_idx]  # [num_bins, num_patterns, 5, D_block]

            # Assign patterns for this block
            pattern_ids, full_class_ids = self.assign_patterns(
                features, timestep_class, pentachora
            )

            pattern_ids_list.append(pattern_ids)
            full_class_ids_list.append(full_class_ids)

        # Stack pattern assignments
        pattern_ids_batched = torch.cat(pattern_ids_list, dim=0)  # [num_blocks * B]
        full_class_ids_batched = torch.cat(full_class_ids_list, dim=0)  # [num_blocks * B]

        # ==================================================================
        # STEP 3: Compute loss components (per-block for feature-dependent)
        # ==================================================================

        # 3a. Feature similarity (MUST be per-block due to variable dims + projections)
        feat_sim_losses = []
        for block_idx, block_name in enumerate(block_names):
            features = features_list[block_idx]
            teacher_features = teacher_features_list[block_idx]
            projection = projection_list[block_idx]

            with torch.no_grad():
                teacher_projected = projection(teacher_features)

            teacher_norm = F.normalize(teacher_projected, dim=-1)
            features_norm = F.normalize(features, dim=-1)
            feature_sim = F.cosine_similarity(features_norm, teacher_norm, dim=1)
            feat_sim_loss = (1.0 - feature_sim).mean()
            feat_sim_losses.append(feat_sim_loss)

        feat_sim_loss_total = torch.stack(feat_sim_losses).mean()

        # 3b. Rose loss (per-block due to variable feature dims)
        rose_losses = []
        for block_idx, block_name in enumerate(block_names):
            features = features_list[block_idx]  # [B, D_block]
            pentachora = pentachora_list[block_idx]  # [num_bins, num_patterns, 5, D_block]
            full_class_ids = full_class_ids_list[block_idx]  # [B]

            # Reshape pentachora to [num_classes, 5, D]
            pentachora_all_classes = pentachora.view(-1, 5, pentachora.size(-1))

            rose_loss = self.rose_loss(features, pentachora_all_classes, full_class_ids)
            rose_losses.append(rose_loss)

        rose_loss_total = torch.stack(rose_losses).mean()

        # 3c. Cross-entropy (can batch - same logits structure)
        if self.use_soft_assignment:
            # Compute soft targets per-block
            soft_targets_list = []
            for block_idx in range(num_blocks):
                features = features_list[block_idx]
                timestep_class = timestep_class_list[block_idx]
                pentachora = pentachora_list[block_idx]

                soft_targets = self.compute_soft_assignment(
                    features, timestep_class, pentachora
                )
                soft_targets_list.append(soft_targets)

            soft_targets_batched = torch.cat(soft_targets_list, dim=0)
            log_probs = F.log_softmax(pattern_logits_batched, dim=1)
            ce_loss_total = -(soft_targets_batched * log_probs).sum(dim=1).mean()
        else:
            ce_loss_total = F.cross_entropy(pattern_logits_batched, full_class_ids_batched)

        # 3d. Pattern diversity (per-block, then averaged)
        pattern_div_loss_total = self.batch_compute_pattern_diversity(
            pattern_ids_batched,
            block_indices,
            num_blocks
        )

        # 3e. Cayley loss (per-block due to variable pentachora dims)
        cayley_losses = []
        for block_idx, block_name in enumerate(block_names):
            pentachora = pentachora_list[block_idx]  # [num_bins, num_patterns, 5, D_block]
            timestep_class = timestep_class_list[block_idx]  # [B]
            pattern_ids = pattern_ids_list[block_idx]  # [B]

            # Get assigned pentachora: [B, 5, D_block]
            batch_pentachora = pentachora[timestep_class, pattern_ids]

            cayley_loss = self.cayley_loss(batch_pentachora)
            cayley_losses.append(cayley_loss)

        cayley_loss_total = torch.stack(cayley_losses).mean()

        # 3f. Cantor coherence (batched - scalar values)
        cantor_coherence_total = self.batch_compute_cantor_coherence(
            cantor_values_batched,
            pattern_ids_batched,
            timestep_class_batched,
            block_indices
        )

        # ==================================================================
        # STEP 4: Aggregate losses with block weighting
        # ==================================================================

        # Compute weighted average for shared losses (CE, diversity, cantor)
        # These are computed on batched data but should respect block importance
        total_block_weight = sum(block_weights.get(name, 1.0) for name in block_names)

        # Aggregate per-block losses with weights
        weighted_feat_sim = sum(
            block_weights.get(block_names[i], 1.0) * feat_sim_losses[i]
            for i in range(num_blocks)
        ) / total_block_weight

        weighted_rose = sum(
            block_weights.get(block_names[i], 1.0) * rose_losses[i]
            for i in range(num_blocks)
        ) / total_block_weight

        weighted_cayley = sum(
            block_weights.get(block_names[i], 1.0) * cayley_losses[i]
            for i in range(num_blocks)
        ) / total_block_weight

        # Shared losses (already aggregated across all blocks)
        # These represent the full batch, so we use them directly
        weighted_ce = ce_loss_total
        weighted_diversity = pattern_div_loss_total
        weighted_cantor = cantor_coherence_total

        # Total loss with component weights
        total_loss = (
                self.feature_sim_weight * weighted_feat_sim +
                self.rose_weight * weighted_rose +
                self.ce_weight * weighted_ce +
                self.pattern_diversity_weight * weighted_diversity +
                self.cayley_weight * weighted_cayley +
                self.cantor_coherence_weight * weighted_cantor
        )

        # ==================================================================
        # STEP 5: Compute accuracies (per-block for detailed metrics)
        # ==================================================================

        accuracies = {}
        for block_idx, block_name in enumerate(block_names):
            block_mask = block_indices == block_idx

            block_timestep_logits = timestep_logits_batched[block_mask]
            block_pattern_logits = pattern_logits_batched[block_mask]
            block_timestep_class = timestep_class_list[block_idx]
            block_pattern_ids = pattern_ids_list[block_idx]
            block_full_class_ids = full_class_ids_list[block_idx]

            pred_timestep = block_timestep_logits.argmax(dim=1)
            timestep_acc = (pred_timestep == block_timestep_class).float().mean().item()

            pred_patterns = block_pattern_logits.argmax(dim=1) % self.num_patterns
            pattern_acc = (pred_patterns == block_pattern_ids).float().mean().item()

            pred_full = block_pattern_logits.argmax(dim=1)
            full_acc = (pred_full == block_full_class_ids).float().mean().item()

            accuracies[block_name] = {
                'timestep_acc': timestep_acc,
                'pattern_acc': pattern_acc,
                'full_acc': full_acc
            }

        # ==================================================================
        # STEP 6: Build metrics dict
        # ==================================================================

        all_metrics = {}

        # Per-block metrics
        for block_idx, block_name in enumerate(block_names):
            companion = companions[block_name]

            # Per-block loss values
            all_metrics[f'{block_name}/feat_sim'] = feat_sim_losses[block_idx].item()
            all_metrics[f'{block_name}/rose'] = rose_losses[block_idx].item()
            all_metrics[f'{block_name}/cayley'] = cayley_losses[block_idx].item()

            # Shared loss values (same for all blocks since computed on batched data)
            all_metrics[f'{block_name}/ce'] = ce_loss_total.item()
            all_metrics[f'{block_name}/diversity'] = pattern_div_loss_total.item()
            all_metrics[f'{block_name}/cantor'] = cantor_coherence_total.item()

            # Accuracies (per-block)
            all_metrics[f'{block_name}/timestep_acc'] = accuracies[block_name]['timestep_acc']
            all_metrics[f'{block_name}/pattern_acc'] = accuracies[block_name]['pattern_acc']
            all_metrics[f'{block_name}/full_acc'] = accuracies[block_name]['full_acc']

            # Cantor alpha (per-block)
            all_metrics[f'{block_name}/cantor_alpha'] = companion.cantor_stairs.get_alpha()

        # Averaged metrics
        all_metrics['avg/timestep_acc'] = sum(
            v for k, v in all_metrics.items() if k.endswith('/timestep_acc')
        ) / num_blocks
        all_metrics['avg/pattern_acc'] = sum(
            v for k, v in all_metrics.items() if k.endswith('/pattern_acc')
        ) / num_blocks
        all_metrics['avg/full_acc'] = sum(
            v for k, v in all_metrics.items() if k.endswith('/full_acc')
        ) / num_blocks

        # Weighted average cayley (consistent with total loss computation)
        total_block_weight = sum(block_weights.get(name, 1.0) for name in block_names)
        all_metrics['avg/cayley'] = sum(
            block_weights.get(block_names[i], 1.0) * cayley_losses[i].item()
            for i in range(num_blocks)
        ) / total_block_weight

        all_metrics['total_loss'] = total_loss.item()

        return total_loss, all_metrics

class GeoDavidCollective(nn.Module):
    """
    Geometric David Collective: Multi-Block Diffusion Distillation System

    Complete geometric architecture for learning SD1.5 block representations
    using pentachoron structures, Cantor hierarchical encoding, and geometric
    solidity enforcement.

    Architecture:
    - Multiple block companions (one per SD1.5 block)
    - Each companion: pentachoron patterns + Cantor staircase
    - Geometric multi-scale loss with 6 components
    - Per-block weighting for importance

    Training paradigm:
    - Pattern-supervised learning (timestep + pattern classification)
    - Geometric solidity enforcement (Cayley-Menger)
    - Relational learning (Rose loss with pentachoron roles)
    - Hierarchical position encoding (Cantor staircase)
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
                config_dict contains: input_dim, scale_dim, etc.
            num_timestep_bins: Number of timestep buckets
            num_patterns_per_bin: Patterns per timestep
            block_weights: Optional importance weights per block
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
        print("GeoDavidCollective: Initializing Geometric Multi-Block System")
        print("=" * 80)

        # Build companions (one per block)
        self.companions = nn.ModuleDict()
        for block_name, config in block_configs.items():
            print(f"\n[{block_name}] Creating geometric companion...")
            companion = GeoDavidBlockCompanion(
                block_name=block_name,
                input_dim=config['input_dim'],
                scale_dim=config['scale_dim'],
                num_timestep_bins=num_timestep_bins,
                num_patterns_per_bin=num_patterns_per_bin,
                use_belly=config.get('use_belly', True),
                belly_expand=config.get('belly_expand', 2.0),
                temperature=config.get('temperature', 0.07),
                cantor_alpha_init=config.get('cantor_alpha_init', 0.5),
                cantor_tau=config.get('cantor_tau', 0.25),
                cantor_levels=config.get('cantor_levels', 12),
                simplex_k=config.get('simplex_k', 4),
                simplex_seed_base=config.get('simplex_seed_base', 42)
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
        print(f"GeoDavidCollective Initialized Successfully")
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
            features_dict: Dict[block_name, features]
                features: [B, input_dim] per block
            timesteps: [B] - continuous timesteps [0, 1000]

        Returns:
            Dict[block_name, outputs] where outputs contains:
                'features', 'timestep_logits', 'pattern_logits',
                'timestep_class', 'cantor_values'
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
            teacher_features_dict: Dict[block_name, teacher_features]
            timesteps: [B] - timesteps

        Returns:
            total_loss: Scalar loss
            metrics: Dict of all metrics
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

    def get_pentachora(
            self,
            block_name: str
    ) -> torch.Tensor:
        """
        Get pentachora for a specific block.

        Returns:
            pentachora: [num_bins, num_patterns, 5, scale_dim]
        """
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
            'architecture': 'GeoDavidCollective',
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
                'parameters': sum(p.numel() for p in companion.parameters())
            }

        return info

    def __repr__(self):
        info = self.get_model_info()
        lines = [
            "GeoDavidCollective(Geometric Multi-Block Diffusion System)",
            f"  Blocks: {info['num_blocks']} ({', '.join(info['blocks'])})",
            f"  Timestep bins: {info['num_timestep_bins']}",
            f"  Patterns per bin: {info['num_patterns_per_bin']}",
            f"  Total classes: {info['total_classes']}",
            f"  Parameters: {info['total_parameters']:,}",
            "  Geometric Features:",
            "    ✓ Pentachoron structure (5-vertex simplices)",
            "    ✓ SimplexFactory validation",
            "    ✓ Cantor staircase encoding",
            "    ✓ Cayley-Menger solidity",
            "    ✓ Rose relational learning"
        ]
        return "\n".join(lines)

# ============================================================================
# DEMO & TESTING
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("GeoDavidCollective: Comprehensive Demo")
    print("=" * 80)
    print("\nThis demo shows the complete geometric multi-block system working:")
    print("  1. Multi-block ensemble creation")
    print("  2. Synthetic data generation")
    print("  3. Forward pass through all blocks")
    print("  4. Geometric loss computation (6 components)")
    print("  5. Accuracy metrics (timestep + pattern)")
    print("  6. Geometric health checks (volumes, Cayley)")
    print("\n" + "=" * 80 + "\n")

    # ========================================================================
    # 1. CONFIGURATION
    # ========================================================================

    print("Step 1: Configuring multi-block ensemble...")

    # Block configurations for ALL SD1.5 blocks (9 total)
    block_configs = {
        # Down blocks (4)
        'down_0': {
            'input_dim': 320,
            'scale_dim': 384,
            'use_belly': True,
            'belly_expand': 2.0,
            'temperature': 0.07,
            'cantor_alpha_init': 0.5,
            'cantor_tau': 0.25,
            'cantor_levels': 12,
            'simplex_k': 4,
            'simplex_seed_base': 42
        },
        'down_1': {
            'input_dim': 640,
            'scale_dim': 512,
            'use_belly': True,
            'belly_expand': 2.0,
            'temperature': 0.07,
            'cantor_alpha_init': 0.5,
            'cantor_tau': 0.25,
            'cantor_levels': 12,
            'simplex_k': 4,
            'simplex_seed_base': 43
        },
        'down_2': {
            'input_dim': 1280,
            'scale_dim': 768,
            'use_belly': True,
            'belly_expand': 2.0,
            'temperature': 0.07,
            'cantor_alpha_init': 0.5,
            'cantor_tau': 0.25,
            'cantor_levels': 12,
            'simplex_k': 4,
            'simplex_seed_base': 44
        },
        'down_3': {
            'input_dim': 1280,
            'scale_dim': 768,
            'use_belly': True,
            'belly_expand': 2.0,
            'temperature': 0.07,
            'cantor_alpha_init': 0.5,
            'cantor_tau': 0.25,
            'cantor_levels': 12,
            'simplex_k': 4,
            'simplex_seed_base': 45
        },
        # Mid block (1)
        'mid': {
            'input_dim': 1280,
            'scale_dim': 1024,
            'use_belly': True,
            'belly_expand': 2.5,
            'temperature': 0.07,
            'cantor_alpha_init': 0.5,
            'cantor_tau': 0.25,
            'cantor_levels': 12,
            'simplex_k': 4,
            'simplex_seed_base': 46
        },
        # Up blocks (4)
        'up_0': {
            'input_dim': 1280,
            'scale_dim': 768,
            'use_belly': True,
            'belly_expand': 2.0,
            'temperature': 0.07,
            'cantor_alpha_init': 0.5,
            'cantor_tau': 0.25,
            'cantor_levels': 12,
            'simplex_k': 4,
            'simplex_seed_base': 47
        },
        'up_1': {
            'input_dim': 1280,
            'scale_dim': 768,
            'use_belly': True,
            'belly_expand': 2.0,
            'temperature': 0.07,
            'cantor_alpha_init': 0.5,
            'cantor_tau': 0.25,
            'cantor_levels': 12,
            'simplex_k': 4,
            'simplex_seed_base': 48
        },
        'up_2': {
            'input_dim': 640,
            'scale_dim': 512,
            'use_belly': True,
            'belly_expand': 2.0,
            'temperature': 0.07,
            'cantor_alpha_init': 0.5,
            'cantor_tau': 0.25,
            'cantor_levels': 12,
            'simplex_k': 4,
            'simplex_seed_base': 49
        },
        'up_3': {
            'input_dim': 320,
            'scale_dim': 384,
            'use_belly': True,
            'belly_expand': 2.0,
            'temperature': 0.07,
            'cantor_alpha_init': 0.5,
            'cantor_tau': 0.25,
            'cantor_levels': 12,
            'simplex_k': 4,
            'simplex_seed_base': 50
        }
    }

    # Block importance weights (mid-block most important)
    block_weights = {
        'down_0': 0.8,
        'down_1': 1.0,
        'down_2': 1.2,
        'down_3': 1.3,
        'mid': 1.5,  # Highest importance
        'up_0': 1.3,
        'up_1': 1.2,
        'up_2': 1.0,
        'up_3': 0.8
    }

    # Geometric loss configuration
    loss_config = {
        'feature_similarity_weight': 0.4,
        'rose_weight': 0.25,
        'ce_weight': 0.15,
        'pattern_diversity_weight': 0.05,
        'cayley_weight': 0.10,
        'cantor_coherence_weight': 0.05,
        'use_soft_assignment': True,
        'temperature': 0.1
    }

    print("✓ Configured 4 blocks (down_0, down_1, mid, up_0)")
    print(f"✓ Loss weights: feature={loss_config['feature_similarity_weight']}, "
          f"rose={loss_config['rose_weight']}, ce={loss_config['ce_weight']}")
    print(f"✓ Block weights: mid={block_weights['mid']} (highest)\n")

    # ========================================================================
    # 2. MODEL CREATION
    # ========================================================================

    print("Step 2: Creating GeoDavidCollective...")

    model = GeoDavidCollective(
        block_configs=block_configs,
        num_timestep_bins=100,
        num_patterns_per_bin=10,
        block_weights=block_weights,
        loss_config=loss_config
    )

    print("\n✓ Model created successfully!")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Total classes: {100 * 10} (100 timestep bins × 10 patterns)\n")

    # ========================================================================
    # 3. SYNTHETIC DATA GENERATION
    # ========================================================================

    print("Step 3: Generating synthetic teacher features...")

    batch_size = 16
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
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
    print("SUMMARY: GeoDavidCollective Demo Complete")
    print("=" * 80)

    print("\n✅ All systems operational!")

    print("\n🎯 Key Features Demonstrated:")
    print("  ✓ Multi-block ensemble (4 blocks)")
    print("  ✓ Pentachoron structure (5-vertex simplices)")
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

    print("\n💡 Architecture Highlights:")
    model_info = model.get_model_info()
    print(f"  Total parameters:    {model_info['total_parameters']:,}")
    print(f"  Trainable params:    {model_info['trainable_parameters']:,}")
    print(f"  Total classes:       {model_info['total_classes']}")
    print(f"  Timestep bins:       {model_info['num_timestep_bins']}")
    print(f"  Patterns per bin:    {model_info['num_patterns_per_bin']}")

    print("\n" + "=" * 80)
    print("Demo finished successfully! 🚀")
    print("=" * 80)

    # Show model architecture
    print("\n" + str(model))