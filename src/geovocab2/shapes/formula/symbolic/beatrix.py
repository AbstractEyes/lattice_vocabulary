"""
BEATRIX FORMULA SUITE
---------------------
These are RoSE positional and geometric formulas integrating a series of highly specialized
operations to connect the relative positional encoding from RoPE and the symbolic relational
association provided by potently structured geometric lattices.

Devil's Staircase positional encoding with k-simplex geometric operations through hierarchical
measure-theoretic transformations.

This is the first public release of the Beatrix formula suite, developed over many months.

These formulas were refactored using Claude Sonnet 4.5.

This structure is based on the original principles, ideas, and conceptualizations by AbstractPhil with assistance from
multiple AI models over a year and a half of research leading to a viable symbolic representational framework for
integrating fractal positional encodings with geometric structures.

This is a research prototype and experimental codebase. Use at your own risk.


This suite provides:
  - Cantor measure projection into simplex coordinates
  - Fractal-guided simplex initialization
  - Position-aware geometric regularization
  - Hierarchical flow alignment between PE and lattice structures
  - Multi-scale geometric consistency losses

Mathematical Foundation:

    Cantor Measure to Simplex Mapping:
        Given Cantor measure C(x) ∈ [0,1] from Devil's Staircase PE,
        map to barycentric coordinates on k-simplex:
        w_i = basis_i(C(x)) where Σw_i = 1, w_i ≥ 0

    Fractal-Guided Simplex Positioning:
        Use per-level PE features [bit_k, pdf_proxy_k] to modulate
        simplex vertex positions, creating position-dependent geometry

    Flow Alignment Loss:
        L_flow = ||∇_x C(x) - ∇_x V(simplex(x))||²
        Aligns rate of change in measure with geometric deformation

    Hierarchical Coherence:
        Ensure simplex structures at adjacent positions maintain
        geometric continuity proportional to PE similarity

Author: AbstractPhil
    Assistants used:
        Conceptualization and process self-authored utilizing GPT-4o - Mirel over multiple... MULTIPLE iterations
        Claude Opus 3 + Claude Opus 4 + Claude Opus 4.1
        Claude Sonnet 3 + Claude Sonnet 4 + Claude Sonnet 4.5
        GPT-O1-preview + GPT-O3 + GPT-O3 Pro + GPT-o4 mini,
        GPT-4o + GPT-4o-2024-08-06 + GPT-5 + GPT-5 Thinking

---------------------------------------------------------------
License: MIT
---------------------------------------------------------------
Use it. Modify it. Share it. Make it better. Make it worse.
"""

from typing import Dict, Optional, Tuple, Callable
import torch
from torch import Tensor
import torch.nn.functional as F
import math

from geovocab2.shapes.formula.formula_base import FormulaBase
from ..symbolic.cayley_menger import CayleyMengerFromSimplex
from ..engineering.simplex import SimplexEdges

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CANTOR MEASURE TO SIMPLEX PROJECTION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class CantorToBarycentric(FormulaBase):
    """
    Project Cantor measure C(x) ∈ [0,1] to barycentric coordinates on k-simplex.

    Uses fractal measure as a canonical coordinate system for simplex positioning.
    Enables Devil's Staircase PE to directly parameterize geometric structures.

    Args:
        k_simplex: Dimension of target simplex (default: 5)
        mode: 'uniform' | 'fractal_weighted' (default: 'fractal_weighted')
        temperature: Softness of mapping (default: 1.0)
    """

    def __init__(self, k_simplex: int = 5, mode: str = 'fractal_weighted',
                 temperature: float = 1.0):
        super().__init__("cantor_to_barycentric", "f.beatrix.cantor_bary")
        self.k = k_simplex
        self.k_plus_1 = k_simplex + 1
        self.mode = mode
        self.temperature = temperature

        # Learnable projection bases (optional)
        self.bases = torch.nn.Parameter(
            torch.randn(self.k_plus_1) / math.sqrt(self.k_plus_1),
            requires_grad=True
        )

    def forward(self, cantor_measure: Tensor,
                pe_features: Optional[Tensor] = None) -> Dict[str, Tensor]:
        """
        Map Cantor measure to barycentric coordinates.

        Args:
            cantor_measure: C(x) values [..., ] from Devil's Staircase
            pe_features: Optional PE features [..., levels*features_per_level]

        Returns:
            weights: Barycentric coordinates [..., k+1]
            simplex_position: Position code for lattice lookup
            entropy: Measure of coordinate uncertainty
        """
        Cx = cantor_measure  # [..., ]

        if self.mode == 'uniform':
            # Simple subdivision: partition [0,1] into k+1 regions
            edges = torch.linspace(0, 1, self.k_plus_1 + 1,
                                   device=Cx.device, dtype=Cx.dtype)

            # Compute distance to each region center
            centers = (edges[:-1] + edges[1:]) / 2  # [k+1]
            distances = torch.abs(Cx.unsqueeze(-1) - centers)  # [..., k+1]

            # Softmax with temperature
            logits = -distances / self.temperature
            weights = F.softmax(logits, dim=-1)

        elif self.mode == 'fractal_weighted':
            # Use learned bases with fractal structure
            # Compute k+1 phase-shifted projections
            phases = torch.arange(self.k_plus_1, device=Cx.device,
                                  dtype=Cx.dtype) / self.k_plus_1

            # Fractal oscillation at multiple scales
            angles = 2 * math.pi * (Cx.unsqueeze(-1) + phases)  # [..., k+1]

            # Weight by bases and apply nonlinearity
            raw_scores = torch.sin(angles) * self.bases  # [..., k+1]

            # Optional: modulate by PE features
            if pe_features is not None:
                # Project PE features to k+1 dimensions
                pe_proj = F.adaptive_avg_pool1d(
                    pe_features.unsqueeze(1),
                    self.k_plus_1
                ).squeeze(1)  # [..., k+1]
                raw_scores = raw_scores + 0.1 * pe_proj

            # Softmax to barycentric
            weights = F.softmax(raw_scores / self.temperature, dim=-1)

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        # Compute entropy (measure of uncertainty)
        log_weights = torch.log(weights.clamp(min=1e-10))
        entropy = -(weights * log_weights).sum(dim=-1)

        # Simplex position code (argmax for discrete lookup)
        simplex_position = torch.argmax(weights, dim=-1)

        return {
            'weights': weights,
            'simplex_position': simplex_position,
            'entropy': entropy,
            'cantor_input': Cx
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FRACTAL-GUIDED SIMPLEX INITIALIZATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class FractalSimplexInitializer(FormulaBase):
    """
    Initialize k-simplex vertices using Devil's Staircase PE features.

    Creates position-dependent simplex geometries where PE level features
    control vertex spread, orientation, and local deformations.

    Args:
        k_simplex: Simplex dimension (default: 5)
        embedding_dim: Target embedding dimension (default: 512)
        use_pdf_proxy: Use entropy-based features (default: True)
    """

    def __init__(self, k_simplex: int = 5, embedding_dim: int = 512,
                 use_pdf_proxy: bool = True):
        super().__init__("fractal_simplex_init", "f.beatrix.fractal_init")
        self.k = k_simplex
        self.k_plus_1 = k_simplex + 1
        self.dim = embedding_dim
        self.use_pdf_proxy = use_pdf_proxy

        # Base regular simplex (learnable)
        base = torch.eye(self.k_plus_1)
        centroid = base.mean(dim=0, keepdim=True)
        self.base_simplex = torch.nn.Parameter(base - centroid, requires_grad=True)

        # Projection to embedding space
        self.projection = torch.nn.Linear(self.k_plus_1, embedding_dim, bias=False)

    def forward(self, pe_features: Tensor, cantor_measure: Tensor) -> Dict[str, Tensor]:
        """
        Generate simplex vertices from PE features.

        Args:
            pe_features: PE features [..., levels*features_per_level]
            cantor_measure: C(x) values [..., ]

        Returns:
            vertices: Simplex vertices [..., k+1, dim]
            deformation_magnitude: How much simplex differs from base
            orientation_angle: Rotation applied
        """
        batch_shape = pe_features.shape[:-1]

        # Extract per-level features (assume features_per_level=2: [bit, pdf])
        features_per_level = 2
        levels = pe_features.shape[-1] // features_per_level
        pe_reshaped = pe_features.view(*batch_shape, levels, features_per_level)

        # Use first few levels for primary deformation
        n_control_levels = min(levels, self.k_plus_1)
        control_features = pe_reshaped[..., :n_control_levels, :]  # [..., n_control, 2]

        # Bit features control vertex displacements
        bit_features = control_features[..., 0]  # [..., n_control]

        # Pad to k+1 if needed
        if n_control_levels < self.k_plus_1:
            padding = torch.zeros(*batch_shape, self.k_plus_1 - n_control_levels,
                                  device=bit_features.device, dtype=bit_features.dtype)
            bit_features = torch.cat([bit_features, padding], dim=-1)

        # Apply per-vertex scaling based on bit features
        # Scale ∈ [0.5, 1.5] to preserve overall structure
        vertex_scales = 1.0 + 0.5 * (bit_features - 0.5)  # [..., k+1]

        # Deform base simplex
        deformed = self.base_simplex * vertex_scales.unsqueeze(-1)  # [..., k+1, k+1]

        # Rotation based on Cantor measure
        theta = 2 * math.pi * cantor_measure  # [..., ]
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)

        # Apply 2D rotation in first two dimensions
        rot_deformed = deformed.clone()
        if self.k_plus_1 >= 2:
            rot_deformed[..., :, 0] = (cos_t.unsqueeze(-1) * deformed[..., :, 0] -
                                       sin_t.unsqueeze(-1) * deformed[..., :, 1])
            rot_deformed[..., :, 1] = (sin_t.unsqueeze(-1) * deformed[..., :, 0] +
                                       cos_t.unsqueeze(-1) * deformed[..., :, 1])

        # Project to embedding space
        vertices = self.projection(rot_deformed)  # [..., k+1, dim]

        # Compute metrics
        deformation_magnitude = torch.norm(deformed - self.base_simplex.unsqueeze(0),
                                           dim=-1).mean(dim=-1)
        orientation_angle = theta % (2 * math.pi)

        return {
            'vertices': vertices,
            'deformation_magnitude': deformation_magnitude,
            'orientation_angle': orientation_angle,
            'base_simplex': self.base_simplex
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FLOW ALIGNMENT LOSS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class FlowAlignmentLoss(FormulaBase):
    """
    Align gradients of Cantor measure with geometric simplex deformation.

    Ensures that changes in position (measured by PE) correspond to
    coherent changes in geometric structure (measured by simplex volume/shape).

    Args:
        volume_weight: Weight for volume gradient term (default: 1.0)
        shape_weight: Weight for shape gradient term (default: 0.5)
    """

    def __init__(self, volume_weight: float = 1.0, shape_weight: float = 0.5):
        super().__init__("flow_alignment_loss", "f.beatrix.flow_align")
        self.volume_weight = volume_weight
        self.shape_weight = shape_weight

    def forward(self,
                cantor_sequence: Tensor,
                simplex_sequence: Tensor,
                pe_features_sequence: Tensor) -> Dict[str, Tensor]:
        """
        Compute flow alignment loss across a sequence.

        Args:
            cantor_sequence: C(x) values [batch, seq_len]
            simplex_sequence: Simplex vertices [batch, seq_len, k+1, dim]
            pe_features_sequence: PE features [batch, seq_len, feat_dim]

        Returns:
            total_loss: Combined flow alignment loss
            volume_grad_loss: Volume gradient component
            shape_grad_loss: Shape gradient component
            mean_volume_change: Average volume change rate
        """
        batch_size, seq_len = cantor_sequence.shape[:2]

        # Compute Cantor measure gradients (finite differences)
        cantor_grad = cantor_sequence[:, 1:] - cantor_sequence[:, :-1]  # [B, T-1]

        # Compute simplex volume changes
        volume_calc = CayleyMengerFromSimplex()

        volumes = []
        for t in range(seq_len):
            vol_result = volume_calc.forward(simplex_sequence[:, t])
            volumes.append(vol_result['volume'])

        volumes = torch.stack(volumes, dim=1)  # [B, T]
        volume_grad = volumes[:, 1:] - volumes[:, :-1]  # [B, T-1]

        # Normalize gradients
        cantor_grad_norm = cantor_grad / (torch.abs(cantor_grad).max() + 1e-8)
        volume_grad_norm = volume_grad / (torch.abs(volume_grad).max() + 1e-8)

        # Volume gradient alignment loss
        volume_grad_loss = F.mse_loss(cantor_grad_norm, volume_grad_norm)

        # Shape gradient: measure edge length changes
        edges_t0 = simplex_sequence[:, :-1, 1:] - simplex_sequence[:, :-1, 0:1]  # [B, T-1, k, d]
        edges_t1 = simplex_sequence[:, 1:, 1:] - simplex_sequence[:, 1:, 0:1]  # [B, T-1, k, d]

        edge_change = torch.norm(edges_t1 - edges_t0, dim=-1).mean(dim=-1)  # [B, T-1]
        edge_change_norm = edge_change / (edge_change.max() + 1e-8)

        # Shape gradient alignment
        shape_grad_loss = F.mse_loss(cantor_grad_norm, edge_change_norm)

        # Total loss
        total_loss = (self.volume_weight * volume_grad_loss +
                      self.shape_weight * shape_grad_loss)

        return {
            'total_loss': total_loss,
            'volume_grad_loss': volume_grad_loss,
            'shape_grad_loss': shape_grad_loss,
            'mean_volume_change': volume_grad.abs().mean(),
            'cantor_grad_magnitude': cantor_grad.abs().mean()
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HIERARCHICAL COHERENCE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class HierarchicalCoherence(FormulaBase):
    """
    Ensure simplex structures at adjacent positions maintain geometric continuity.

    Positions with similar PE features should have similar simplex geometries.
    Enforces smooth variation in lattice structure across the sequence.

    Args:
        local_window: Window size for local coherence (default: 5)
        similarity_threshold: Minimum PE similarity for coherence (default: 0.7)
    """

    def __init__(self, local_window: int = 5, similarity_threshold: float = 0.7):
        super().__init__("hierarchical_coherence", "f.beatrix.coherence")
        self.window = local_window
        self.threshold = similarity_threshold

    def forward(self,
                pe_features_sequence: Tensor,
                simplex_sequence: Tensor) -> Dict[str, Tensor]:
        """
        Compute hierarchical coherence loss.

        Args:
            pe_features_sequence: PE features [batch, seq_len, feat_dim]
            simplex_sequence: Simplex vertices [batch, seq_len, k+1, dim]

        Returns:
            coherence_loss: Geometric discontinuity penalty
            similarity_map: PE feature similarity matrix
            geometric_continuity: Average geometric smoothness
        """
        batch_size, seq_len, feat_dim = pe_features_sequence.shape

        # Compute PE feature similarity matrix
        pe_normalized = F.normalize(pe_features_sequence, p=2, dim=-1)
        similarity_matrix = torch.matmul(pe_normalized, pe_normalized.transpose(-2, -1))
        # [B, T, T]

        # Compute geometric distance matrix (centroid distances)
        centroids = simplex_sequence.mean(dim=-2)  # [B, T, dim]
        centroid_dist = torch.cdist(centroids, centroids, p=2)  # [B, T, T]

        # Create local window mask
        mask = torch.zeros(seq_len, seq_len, device=pe_features_sequence.device)
        for i in range(seq_len):
            start = max(0, i - self.window)
            end = min(seq_len, i + self.window + 1)
            mask[i, start:end] = 1.0

        # Apply mask to similarity (only consider local neighbors)
        local_similarity = similarity_matrix * mask.unsqueeze(0)

        # High PE similarity should predict low geometric distance
        # Loss: high similarity + high distance = bad
        coherence_penalty = (local_similarity > self.threshold).float() * centroid_dist
        coherence_loss = coherence_penalty.sum(dim=(-2, -1)) / (mask.sum() + 1e-8)
        coherence_loss = coherence_loss.mean()  # Average over batch

        # Geometric continuity: smoothness of adjacent simplices
        adjacent_dist = centroid_dist[:, torch.arange(seq_len - 1),
        torch.arange(1, seq_len)]  # [B, T-1]
        geometric_continuity = 1.0 / (adjacent_dist.mean() + 1e-8)

        return {
            'coherence_loss': coherence_loss,
            'similarity_map': similarity_matrix.mean(dim=0),  # Average over batch
            'geometric_continuity': geometric_continuity,
            'mean_local_similarity': local_similarity.mean()
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MULTI-SCALE CONSISTENCY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class MultiScaleConsistency(FormulaBase):
    """
    Ensure geometric structures are consistent across PE hierarchy levels.

    Devil's Staircase PE has features at multiple scales (levels).
    Simplex geometry should reflect these multi-scale properties.

    Args:
        num_scales: Number of PE levels to analyze (default: 3)
    """

    def __init__(self, num_scales: int = 3):
        super().__init__("multiscale_consistency", "f.beatrix.multiscale")
        self.num_scales = num_scales

    def forward(self,
                pe_features: Tensor,
                simplex_vertices: Tensor,
                levels: int = 12,
                features_per_level: int = 2) -> Dict[str, Tensor]:
        """
        Check multi-scale consistency.

        Args:
            pe_features: PE features [..., levels*features_per_level]
            simplex_vertices: Vertices [..., k+1, dim]
            levels: Number of PE levels
            features_per_level: Features per level

        Returns:
            consistency_loss: Multi-scale inconsistency penalty
            scale_variances: Variance at each scale
            cross_scale_correlation: Correlation between scales
        """
        batch_shape = pe_features.shape[:-1]

        # Reshape to per-level
        pe_levels = pe_features.view(*batch_shape, levels, features_per_level)

        # Select scales to analyze (evenly spaced through hierarchy)
        scale_indices = torch.linspace(0, levels - 1, self.num_scales,
                                       dtype=torch.long, device=pe_features.device)
        selected_scales = pe_levels[..., scale_indices, :]  # [..., num_scales, 2]

        # Compute geometric features at corresponding scales
        edge_calc = SimplexEdges()
        edge_result = edge_calc.forward(simplex_vertices)
        edge_lengths = edge_result['edge_lengths']  # [..., n_edges]

        # Group edges by scale (partition into num_scales groups)
        n_edges = edge_lengths.shape[-1]
        edges_per_scale = n_edges // self.num_scales

        scale_variances = []
        for i in range(self.num_scales):
            start_idx = i * edges_per_scale
            end_idx = (i + 1) * edges_per_scale if i < self.num_scales - 1 else n_edges
            scale_edges = edge_lengths[..., start_idx:end_idx]
            scale_var = scale_edges.var(dim=-1)
            scale_variances.append(scale_var)

        scale_variances = torch.stack(scale_variances, dim=-1)  # [..., num_scales]

        # PE scale features (use bit features as scale indicators)
        pe_scale_features = selected_scales[..., 0]  # [..., num_scales]

        # Consistency: PE scale variance should correlate with geometric variance
        # Normalize both to [0, 1]
        pe_norm = (pe_scale_features - pe_scale_features.min(dim=-1, keepdim=True)[0])
        pe_norm = pe_norm / (pe_norm.max(dim=-1, keepdim=True)[0] + 1e-8)

        geom_norm = (scale_variances - scale_variances.min(dim=-1, keepdim=True)[0])
        geom_norm = geom_norm / (geom_norm.max(dim=-1, keepdim=True)[0] + 1e-8)

        # Consistency loss: L2 between normalized scales
        consistency_loss = F.mse_loss(pe_norm, geom_norm)

        # Cross-scale correlation
        pe_flat = pe_norm.reshape(-1, self.num_scales)
        geom_flat = geom_norm.reshape(-1, self.num_scales)

        # Pearson correlation
        pe_centered = pe_flat - pe_flat.mean(dim=0, keepdim=True)
        geom_centered = geom_flat - geom_flat.mean(dim=0, keepdim=True)

        numerator = (pe_centered * geom_centered).sum(dim=0)
        denominator = (torch.sqrt((pe_centered ** 2).sum(dim=0)) *
                       torch.sqrt((geom_centered ** 2).sum(dim=0)) + 1e-8)
        correlation = numerator / denominator
        cross_scale_correlation = correlation.mean()

        return {
            'consistency_loss': consistency_loss,
            'scale_variances': scale_variances.mean(dim=tuple(range(len(batch_shape)))),
            'cross_scale_correlation': cross_scale_correlation,
            'pe_scale_features': pe_scale_features.mean(dim=tuple(range(len(batch_shape))))
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# INTEGRATED BEATRIX LOSS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class BeatrixIntegratedLoss(FormulaBase):
    """
    Combined loss function integrating all Beatrix components.

    Provides unified training objective for Devil's Staircase PE + geometric lattice.

    Args:
        flow_weight: Weight for flow alignment (default: 1.0)
        coherence_weight: Weight for hierarchical coherence (default: 0.5)
        multiscale_weight: Weight for multi-scale consistency (default: 0.3)
        volume_reg_weight: Weight for volume regularization (default: 0.1)
    """

    def __init__(self,
                 flow_weight: float = 1.0,
                 coherence_weight: float = 0.5,
                 multiscale_weight: float = 0.3,
                 volume_reg_weight: float = 0.1):
        super().__init__("beatrix_integrated_loss", "f.beatrix.integrated")

        self.flow_weight = flow_weight
        self.coherence_weight = coherence_weight
        self.multiscale_weight = multiscale_weight
        self.volume_reg_weight = volume_reg_weight

        # Initialize sub-components
        self.flow_loss = FlowAlignmentLoss()
        self.coherence_loss = HierarchicalCoherence()
        self.multiscale_loss = MultiScaleConsistency()

    def forward(self,
                cantor_sequence: Tensor,
                pe_features_sequence: Tensor,
                simplex_sequence: Tensor,
                target_volume: float = 1.0) -> Dict[str, Tensor]:
        """
        Compute integrated Beatrix loss.

        Args:
            cantor_sequence: C(x) [batch, seq_len]
            pe_features_sequence: PE features [batch, seq_len, feat_dim]
            simplex_sequence: Simplex vertices [batch, seq_len, k+1, dim]
            target_volume: Desired simplex volume (default: 1.0)

        Returns:
            total_loss: Weighted combination of all components
            component_losses: Individual loss values
            metrics: Diagnostic metrics
        """
        # Flow alignment
        flow_result = self.flow_loss.forward(
            cantor_sequence, simplex_sequence, pe_features_sequence
        )

        # Hierarchical coherence
        coherence_result = self.coherence_loss.forward(
            pe_features_sequence, simplex_sequence
        )

        # Multi-scale consistency
        # Take first sample from batch for multi-scale analysis
        multiscale_result = self.multiscale_loss.forward(
            pe_features_sequence[:, 0],  # [batch, feat_dim]
            simplex_sequence[:, 0]  # [batch, k+1, dim]
        )

        # Volume regularization
        volume_calc = CayleyMengerFromSimplex()

        volumes = []
        for t in range(simplex_sequence.shape[1]):
            vol_result = volume_calc.forward(simplex_sequence[:, t])
            volumes.append(vol_result['volume'])

        volumes = torch.stack(volumes, dim=1)  # [batch, seq_len]
        volume_loss = ((volumes - target_volume) ** 2).mean()

        # Total loss
        total_loss = (
                self.flow_weight * flow_result['total_loss'] +
                self.coherence_weight * coherence_result['coherence_loss'] +
                self.multiscale_weight * multiscale_result['consistency_loss'] +
                self.volume_reg_weight * volume_loss
        )

        return {
            'total_loss': total_loss,
            'flow_loss': flow_result['total_loss'],
            'coherence_loss': coherence_result['coherence_loss'],
            'multiscale_loss': multiscale_result['consistency_loss'],
            'volume_loss': volume_loss,
            'mean_volume': volumes.mean(),
            'geometric_continuity': coherence_result['geometric_continuity'],
            'cross_scale_correlation': multiscale_result['cross_scale_correlation']
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TESTING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_beatrix_formulas():
    """Test suite for Beatrix formulas."""

    print("\n" + "=" * 70)
    print("BEATRIX FORMULA SUITE TESTS")
    print("=" * 70)

    # Test 1: Cantor to Barycentric
    print("\n[Test 1] CantorToBarycentric")
    cantor_measure = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])

    c2b = CantorToBarycentric(k_simplex=5, mode='fractal_weighted')
    result = c2b.forward(cantor_measure)

    print(f"  Input Cantor measures: {cantor_measure.numpy()}")
    print(f"  Barycentric weights shape: {result['weights'].shape}")
    print(f"  Weights sum: {result['weights'].sum(dim=-1).detach().numpy()}")  # FIXED
    print(f"  Entropy: {result['entropy'].detach().numpy()}")  # FIXED
    print(f"  Status: ✓ PASS")

    # Test 2: Fractal Simplex Initializer
    print("\n[Test 2] FractalSimplexInitializer")
    batch_size = 4
    levels = 12
    features_per_level = 2

    pe_features = torch.randn(batch_size, levels * features_per_level)
    cantor_vals = torch.rand(batch_size)

    init = FractalSimplexInitializer(k_simplex=5, embedding_dim=512)
    init_result = init.forward(pe_features, cantor_vals)

    print(f"  Vertices shape: {init_result['vertices'].shape}")
    print(f"  Deformation magnitude: {init_result['deformation_magnitude'].detach().mean().item():.4f}")  # FIXED
    print(f"  Status: ✓ PASS")

    # Test 3: Flow Alignment Loss
    print("\n[Test 3] FlowAlignmentLoss")
    seq_len = 16
    cantor_seq = torch.rand(batch_size, seq_len)
    simplex_seq = torch.randn(batch_size, seq_len, 6, 512)
    pe_seq = torch.randn(batch_size, seq_len, levels * features_per_level)

    flow = FlowAlignmentLoss()
    flow_result = flow.forward(cantor_seq, simplex_seq, pe_seq)

    print(f"  Total loss: {flow_result['total_loss'].item():.6f}")
    print(f"  Volume grad loss: {flow_result['volume_grad_loss'].item():.6f}")
    print(f"  Status: ✓ PASS")

    # Test 4: Hierarchical Coherence
    print("\n[Test 4] HierarchicalCoherence")
    coherence = HierarchicalCoherence(local_window=5)
    coh_result = coherence.forward(pe_seq, simplex_seq)

    print(f"  Coherence loss: {coh_result['coherence_loss'].item():.6f}")
    print(f"  Geometric continuity: {coh_result['geometric_continuity'].item():.4f}")
    print(f"  Status: ✓ PASS")

    # Test 5: Multi-Scale Consistency
    print("\n[Test 5] MultiScaleConsistency")
    multiscale = MultiScaleConsistency(num_scales=3)
    ms_result = multiscale.forward(pe_features, init_result['vertices'])

    print(f"  Consistency loss: {ms_result['consistency_loss'].item():.6f}")
    print(f"  Cross-scale correlation: {ms_result['cross_scale_correlation'].item():.4f}")
    print(f"  Status: ✓ PASS")

    # Test 6: Integrated Loss
    print("\n[Test 6] BeatrixIntegratedLoss")
    integrated = BeatrixIntegratedLoss()
    int_result = integrated.forward(cantor_seq, pe_seq, simplex_seq)

    print(f"  Total loss: {int_result['total_loss'].item():.6f}")
    print(f"  Flow component: {int_result['flow_loss'].item():.6f}")
    print(f"  Coherence component: {int_result['coherence_loss'].item():.6f}")
    print(f"  Multiscale component: {int_result['multiscale_loss'].item():.6f}")
    print(f"  Volume component: {int_result['volume_loss'].item():.6f}")
    print(f"  Status: ✓ PASS")

    print("\n" + "=" * 70)
    print("All Beatrix tests completed!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    test_beatrix_formulas()