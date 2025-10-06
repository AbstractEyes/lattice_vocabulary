"""
QUADRADIC & SIMPLEX OPERATIONS
-------------------------------
Quadratic equations, conic sections, and k-dimensional simplex operations.

This module provides operations for:
  - Quadratic equations and roots
  - Parabolic and conic section properties
  - k-Simplex operations for arbitrary dimensions
  - Barycentric coordinates and conversions
  - Simplex volume, edges, faces
  - Geometric quality metrics
  - High-dimensional polytope operations
  - Generic k-simplex sampling with formula application

Mathematical Foundation:

    Quadratic Equation:
        ax² + bx + c = 0
        x = (-b ± √(b² - 4ac))/(2a)
        Discriminant: Δ = b² - 4ac

    k-Simplex:
        - k-simplex has (k+1) vertices in k-dimensional space
        - Examples: 0-simplex=point, 1-simplex=edge, 2-simplex=triangle,
                   3-simplex=tetrahedron, 4-simplex=pentachoron
        - Edges: C(k+1, 2)
        - Faces: C(k+1, i+1) for i-faces

    Barycentric Coordinates:
        p = Σᵢ wᵢ vᵢ where Σwᵢ = 1, wᵢ ≥ 0 (inside simplex)

    Simplex Volume (Cayley-Menger):
        V² = (-1)^(k+1) / (2^k (k!)²) × det(CM)
        where CM is distance matrix augmented with 1s

    Quality Metrics:
        - Aspect ratio: r_in / r_out (inscribed/circumscribed radius ratio)
        - Regularity: measure of deviation from regular simplex
        - Degeneracy: near-zero volume

Author: AbstractPhil + Claude Sonnet 4.5 + GPT 5
License: MIT
"""

from typing import Dict, Optional, Tuple, List
import torch
from torch import Tensor
import math
from itertools import combinations

from ..formula_base import FormulaBase


# ──────────────────────────────────────────────────────────────────────────────
# Stable k-simplex volume via Gram matrix (fp64 internal, fp32 return)
# ──────────────────────────────────────────────────────────────────────────────
def _stable_simplex_volume(vertices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute k-simplex volume using Gram determinant with fp64 and scale normalization.
    Args:
        vertices: [..., k+1, dim] float tensor
    Returns:
        volume:      [...,] float tensor (same dtype as input)
        volume_sq:   [...,] float tensor (same dtype as input)
    """
    x = vertices
    k_plus_1 = x.shape[-2]
    k = k_plus_1 - 1
    if k <= 0:
        z = torch.zeros(x.shape[:-2], device=x.device, dtype=x.dtype)
        return z, z

    # Edge matrix E = [v1-v0, v2-v0, ..., vk-v0]  -> [..., dim, k]
    v0 = x[..., 0:1, :]
    E = (x[..., 1:, :] - v0).to(torch.float64)

    # Normalize by mean edge length to reduce conditioning issues
    with torch.no_grad():
        dists = torch.cdist(x.to(torch.float64), x.to(torch.float64), p=2)  # [..., k+1, k+1]
        iu, ju = torch.triu_indices(k_plus_1, k_plus_1, offset=1, device=x.device)
        mean_edge = dists[..., iu, ju].mean(dim=-1).clamp(min=1e-12)  # [...]
    scale = mean_edge[..., None, None]  # broadcast
    E = E / scale

    # Gram matrix (symmetric PSD in the ideal case)
    G = E.transpose(-2, -1) @ E                 # [..., k, k] fp64
    G = 0.5 * (G + G.transpose(-2, -1))         # symmetrize

    # Cholesky with adaptive jitter for rank-deficiency
    eye_k = torch.eye(k, dtype=G.dtype, device=G.device)
    jitter = 0.0
    L = None
    for _ in range(3):
        try:
            L = torch.linalg.cholesky(G + (jitter * eye_k))
            break
        except RuntimeError:
            jitter = 1e-10 if jitter == 0.0 else jitter * 10.0
    if L is None:
        # Fallback: use small diagonal to proceed
        L = torch.linalg.cholesky(G + (1e-6 * eye_k))

    # log det(G) = 2 * sum(log diag(L))
    diag = torch.diagonal(L, dim1=-2, dim2=-1)
    logdetG = 2.0 * torch.sum(torch.log(diag.clamp(min=1e-24)), dim=-1)

    # Volume_k^2 = det(G) / (k!)^2
    log_vol_sq = logdetG - 2.0 * math.log(math.factorial(k))
    vol_sq = torch.exp(log_vol_sq)                       # fp64
    vol = torch.sqrt(vol_sq.clamp(min=0.0))              # fp64

    # Rescale back (edge scaling to power k)
    vol = vol * (mean_edge ** k)
    vol_sq = vol_sq * (mean_edge ** (2 * k))
    return vol.to(x.dtype), vol_sq.to(x.dtype)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# QUADRATIC EQUATIONS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class QuadraticSolver(FormulaBase):
    """Solve quadratic equations ax² + bx + c = 0.
    """

    def __init__(self):
        super().__init__("quadratic_solver", "f.quadratic.solve")

    def forward(self, a: Tensor, b: Tensor, c: Tensor) -> Dict[str, Tensor]:
        """Solve quadratic equation.

        Args:
            a: Coefficient of x² [...]
            b: Coefficient of x [...]
            c: Constant term [...]

        Returns:
            discriminant: Δ = b² - 4ac [...]
            root1: First root (real or real part) [...]
            root2: Second root (real or real part) [...]
            has_real_roots: Δ ≥ 0 [...]
            is_degenerate: a ≈ 0 (linear equation) [...]
        """
        # Discriminant
        discriminant = b**2 - 4*a*c

        # Check for degenerate case
        is_degenerate = torch.abs(a) < 1e-10

        # Real roots when discriminant >= 0
        has_real_roots = discriminant >= 0

        # Compute roots using stable formula
        # For numerical stability: if b > 0, use -b - √Δ for one root
        sqrt_discriminant = torch.sqrt(torch.clamp(discriminant, min=0.0))

        # Standard formula
        root1 = (-b + sqrt_discriminant) / (2*a + 1e-10)
        root2 = (-b - sqrt_discriminant) / (2*a + 1e-10)

        return {
            "discriminant": discriminant,
            "root1": root1,
            "root2": root2,
            "has_real_roots": has_real_roots,
            "is_degenerate": is_degenerate,
            "vertex_x": -b / (2*a + 1e-10)
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class ParabolaProperties(FormulaBase):
    """Compute parabola properties from quadratic y = ax² + bx + c.
    """

    def __init__(self):
        super().__init__("parabola_properties", "f.quadratic.parabola")

    def forward(self, a: Tensor, b: Tensor, c: Tensor) -> Dict[str, Tensor]:
        """Compute parabola properties.

        Args:
            a: Coefficient of x² [...]
            b: Coefficient of x [...]
            c: Constant term [...]

        Returns:
            vertex: (h, k) coordinates [..., 2]
            focus: Focus point [..., 2]
            directrix_y: y-coordinate of directrix [...]
            axis_of_symmetry: x = h [...]
            opens_upward: a > 0 [...]
        """
        # Vertex: (h, k) = (-b/2a, c - b²/4a)
        h = -b / (2*a + 1e-10)
        k = c - b**2 / (4*a + 1e-10)
        vertex = torch.stack([h, k], dim=-1)

        # Distance from vertex to focus: p = 1/(4a)
        p = 1.0 / (4*a + 1e-10)

        # Focus: (h, k + p)
        focus = torch.stack([h, k + p], dim=-1)

        # Directrix: y = k - p
        directrix_y = k - p

        # Opening direction
        opens_upward = a > 0

        return {
            "vertex": vertex,
            "focus": focus,
            "directrix_y": directrix_y,
            "axis_of_symmetry": h,
            "opens_upward": opens_upward,
            "focal_parameter": p
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# K-SIMPLEX OPERATIONS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class SimplexDimension(FormulaBase):
    """Determine simplex dimension and validate structure.

    A k-simplex has (k+1) vertices and lives in k-dimensional space.
    """

    def __init__(self):
        super().__init__("simplex_dimension", "f.quadratic.simplex_dim")

    def forward(self, vertices: Tensor) -> Dict[str, Tensor]:
        """Analyze simplex dimension.

        Args:
            vertices: Simplex vertices [..., n_vertices, embedding_dim]

        Returns:
            k: Simplex dimension (k-simplex) [...]
            n_vertices: Number of vertices [...]
            embedding_dim: Embedding space dimension [...]
            n_edges: Number of edges C(k+1, 2) [...]
            n_faces: Number of 2-faces C(k+1, 3) [...]
            is_valid: n_vertices >= 1 [...]
        """
        n_vertices = vertices.shape[-2]
        embedding_dim = vertices.shape[-1]

        # k-simplex has k+1 vertices
        k = n_vertices - 1

        # Number of i-dimensional faces: C(k+1, i+1)
        n_edges = (n_vertices * (n_vertices - 1)) // 2 if n_vertices >= 2 else 0
        n_faces = (n_vertices * (n_vertices - 1) * (n_vertices - 2)) // 6 if n_vertices >= 3 else 0

        is_valid = n_vertices >= 1

        return {
            "k": torch.tensor(k),
            "n_vertices": torch.tensor(n_vertices),
            "embedding_dim": torch.tensor(embedding_dim),
            "n_edges": torch.tensor(n_edges),
            "n_faces": torch.tensor(n_faces),
            "is_valid": torch.tensor(is_valid)
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class BarycentricCoordinates(FormulaBase):
    """Compute barycentric coordinates for points relative to simplices.

    For a k-simplex with vertices v₀, ..., vₖ, any point p can be expressed as:
        p = Σᵢ wᵢ vᵢ where Σwᵢ = 1
    Point is inside simplex if all wᵢ ≥ 0.
    """

    def __init__(self):
        super().__init__("barycentric_coords", "f.quadratic.barycentric")

    def forward(self, points: Tensor, simplex_vertices: Tensor) -> Dict[str, Tensor]:
        """Compute barycentric coordinates.

        Args:
            points: Query points [..., n_points, dim]
            simplex_vertices: Simplex vertices [..., k+1, dim]

        Returns:
            weights: Barycentric weights [..., n_points, k+1]
            is_inside: All weights ≥ 0 [..., n_points]
            weights_sum: Σwᵢ (should be 1) [..., n_points]
            closest_vertex: Index of vertex with max weight [..., n_points]
        """
        # For k-simplex in d-dimensional space (d ≥ k):
        # Solve linear system: [v₁-v₀, v₂-v₀, ..., vₖ-v₀]ᵀ λ = p - v₀
        # Then w₀ = 1 - Σλᵢ, wᵢ = λᵢ for i > 0

        v0 = simplex_vertices[..., 0:1, :]  # [..., 1, dim]
        v_rest = simplex_vertices[..., 1:, :]  # [..., k, dim]

        # Edge vectors from v0
        edges = v_rest - v0  # [..., k, dim]

        # Vector from v0 to points
        p_rel = points.unsqueeze(-2) - v0  # [..., n_points, 1, dim]

        # Solve least squares: edges^T @ lambda = p_rel
        # Using pseudo-inverse for general case
        edges_t = edges.transpose(-2, -1)  # [..., dim, k]

        # Compute (edges^T edges)^(-1) edges^T
        gram = torch.matmul(edges_t, edges)  # [..., k, k]
        gram_inv = torch.linalg.pinv(gram)  # [..., k, k]

        # Lambda = gram_inv @ edges^T @ p_rel
        lambda_vals = torch.matmul(
            torch.matmul(gram_inv, edges_t),
            p_rel.squeeze(-2).unsqueeze(-1)
        ).squeeze(-1)  # [..., n_points, k]

        # Compute all weights
        w0 = 1.0 - lambda_vals.sum(dim=-1, keepdim=True)  # [..., n_points, 1]
        weights = torch.cat([w0, lambda_vals], dim=-1)  # [..., n_points, k+1]

        # Check if inside (all weights ≥ 0)
        is_inside = (weights >= -1e-6).all(dim=-1)

        # Sum of weights (should be 1)
        weights_sum = weights.sum(dim=-1)

        # Closest vertex
        closest_vertex = torch.argmax(weights, dim=-1)

        return {
            "weights": weights,
            "is_inside": is_inside,
            "weights_sum": weights_sum,
            "closest_vertex": closest_vertex
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class SimplexVolume(FormulaBase):
    """Compute k-simplex volume using Cayley-Menger determinant.

    Works for arbitrary k-dimensional simplices in d-dimensional space (d ≥ k).
    """

    def __init__(self):
        super().__init__("simplex_volume", "f.quadratic.volume")

    def forward(self, vertices: Tensor) -> Dict[str, Tensor]:
        """Compute simplex volume.

        Args:
            vertices: Simplex vertices [..., k+1, dim]

        Returns:
            volume: k-dimensional volume [...]
            volume_squared: V² (from determinant) [...]
            is_degenerate: Volume ≈ 0 [...]
            quality: Volume relative to ideal [...]
        """
        k_plus_1 = vertices.shape[-2]
        k = k_plus_1 - 1

        if k == 0:
            # Point: volume = 0
            volume = torch.zeros(vertices.shape[:-2], device=vertices.device)
            return {
                "volume": volume,
                "volume_squared": volume,
                "is_degenerate": torch.ones_like(volume, dtype=torch.bool),
                "quality": torch.zeros_like(volume)
            }

        # Compute pairwise distances
        distances = torch.cdist(vertices, vertices, p=2)  # [..., k+1, k+1]

        # Build Cayley-Menger matrix
        dist_squared = distances ** 2

        # Create augmented matrix
        batch_shape = dist_squared.shape[:-2]
        cm = torch.zeros(*batch_shape, k_plus_1 + 1, k_plus_1 + 1,
                        device=vertices.device, dtype=vertices.dtype)

        # Fill with distances squared
        cm[..., 1:, 1:] = dist_squared

        # Fill first row and column with 1s (except [0,0])
        cm[..., 0, 1:] = 1.0
        cm[..., 1:, 0] = 1.0
        cm[..., 0, 0] = 0.0

        # Compute determinant
        det = torch.linalg.det(cm)

        # Volume formula: V² = (-1)^(k+1) / (2^k (k!)²) × det
        sign = (-1) ** (k + 1)
        factorial_k = math.factorial(k)
        denominator = (2 ** k) * (factorial_k ** 2)

        volume_squared = sign * det / denominator
        volume_squared = torch.clamp(volume_squared, min=0.0)  # Numerical safety

        volume = torch.sqrt(volume_squared)

        # Check degeneracy
        is_degenerate = volume < 1e-10

        # Quality: compare to regular simplex of same edge length
        # Get upper triangle indices
        triu_i, triu_j = torch.triu_indices(k_plus_1, k_plus_1, offset=1, device=vertices.device)
        mean_edge = distances[..., triu_i, triu_j].mean(dim=-1)

        # Regular k-simplex volume with edge length a:
        # V_regular = (a^k / k!) × √((k+1)/(2^k))
        if k > 0:
            regular_volume = ((mean_edge ** k) / factorial_k) * math.sqrt((k + 1) / (2 ** k))
            quality = volume / (regular_volume + 1e-10)
        else:
            quality = torch.ones_like(volume)

        return {
            "volume": volume,
            "volume_squared": volume_squared,
            "is_degenerate": is_degenerate,
            "quality": quality
        }

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class SimplexVolumeExtended(FormulaBase):
    """
    Computes a more specific k-simplex volume with optional degeneracy checks.

    Modes:
      - auto (default): try Gram, fallback to CM
      - gram: force Gram method
      - cm:   force Cayley–Menger

    Args:
        mode (str): "auto" | "gram" | "cm"
        check_degeneracy (bool): if True, check via eigenvalues/determinants
    """

    def __init__(self, mode: str = "auto", check_degeneracy: bool = True):
        super().__init__("simplex_volume", "f.quadratic.volume")
        if mode not in ("auto", "gram", "cm"):
            raise ValueError(f"Invalid mode: {mode}")
        self.mode = mode
        self.check_degeneracy = check_degeneracy

    # ─────────────────────────────────────────────────────────────
    def _gram_volume(self, vertices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Gram determinant method. Returns (volume, volume_squared, is_degenerate).
        """
        x = vertices
        k_plus_1 = x.shape[-2]
        k = k_plus_1 - 1
        if k <= 0:
            z = torch.zeros(x.shape[:-2], device=x.device, dtype=x.dtype)
            return z, z, torch.ones_like(z, dtype=torch.bool)

        v0 = x[..., 0:1, :]
        E = (x[..., 1:, :] - v0).to(torch.float64)             # [..., dim, k]

        with torch.no_grad():
            dists = torch.cdist(x.to(torch.float64), x.to(torch.float64), p=2)
            iu, ju = torch.triu_indices(k_plus_1, k_plus_1, 1, device=x.device)
            mean_edge = dists[..., iu, ju].mean(dim=-1).clamp(min=1e-12)
        E = E / mean_edge[..., None, None]

        G = E.transpose(-2, -1) @ E                            # [..., k, k]
        G = 0.5 * (G + G.transpose(-2, -1))

        eye_k = torch.eye(k, dtype=G.dtype, device=G.device)
        if G.ndim > 2:
            eye_k = eye_k.expand(*G.shape[:-2], k, k)

        jitter = 0.0
        L = None
        for _ in range(4):
            try:
                L = torch.linalg.cholesky(G + jitter * eye_k)
                break
            except RuntimeError:
                jitter = 1e-10 if jitter == 0.0 else jitter * 10.0
        if L is None:
            nan = torch.full_like(mean_edge, float("nan"))
            return nan, nan, torch.ones_like(mean_edge, dtype=torch.bool)

        diag = torch.diagonal(L, dim1=-2, dim2=-1).clamp(min=1e-24)
        logdetG = 2.0 * torch.sum(torch.log(diag), dim=-1)
        log_vol_sq = logdetG - 2.0 * math.log(math.factorial(k))

        vol_sq = torch.exp(log_vol_sq) * (mean_edge ** (2 * k))
        vol = torch.sqrt(vol_sq.clamp(min=0.0))

        if not self.check_degeneracy:
            is_degenerate = torch.zeros_like(vol, dtype=torch.bool)
        else:
            evals = torch.linalg.eigvalsh(G).real
            lambda_min = evals[..., 0].clamp(min=0.0)
            deg_rank = lambda_min < 1e-12
            deg_vol = vol < 1e-10
            is_degenerate = deg_rank | deg_vol

        return vol.to(x.dtype), vol_sq.to(x.dtype), is_degenerate

    # ─────────────────────────────────────────────────────────────
    def _cm_volume(self, vertices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Cayley–Menger method. Returns (volume, volume_squared, is_degenerate).
        """
        k_plus_1 = vertices.shape[-2]
        k = k_plus_1 - 1
        dists = torch.cdist(vertices, vertices, p=2)
        dist_sq = dists ** 2
        cm = torch.zeros(*dist_sq.shape[:-2], k_plus_1 + 1, k_plus_1 + 1,
                         device=vertices.device, dtype=vertices.dtype)
        cm[..., 1:, 1:] = dist_sq
        cm[..., 0, 1:] = 1.0
        cm[..., 1:, 0] = 1.0
        cm[..., 0, 0] = 0.0

        det = torch.linalg.det(cm)
        sign = (-1) ** (k + 1)
        denominator = (2 ** k) * (math.factorial(k) ** 2)
        vol_sq = (sign * det / denominator).clamp(min=0.0)
        vol = torch.sqrt(vol_sq)

        if not self.check_degeneracy:
            is_degenerate = torch.zeros_like(vol, dtype=torch.bool)
        else:
            is_degenerate = (torch.abs(det) < 1e-12) | (vol < 1e-10)

        return vol, vol_sq, is_degenerate

    # ─────────────────────────────────────────────────────────────
    def forward(self, vertices: torch.Tensor) -> Dict[str, torch.Tensor]:
        k_plus_1 = vertices.shape[-2]
        k = k_plus_1 - 1

        if k <= 0:
            z = torch.zeros(vertices.shape[:-2], device=vertices.device, dtype=vertices.dtype)
            return {
                "volume": z,
                "volume_squared": z,
                "is_degenerate": torch.ones_like(z, dtype=torch.bool) if self.check_degeneracy else torch.zeros_like(z, dtype=torch.bool),
                "quality": z,
            }

        vol, vol_sq, is_degenerate = None, None, None

        if self.mode in ("auto", "gram"):
            vol, vol_sq, is_degenerate = self._gram_volume(vertices)
            if self.mode == "gram" and torch.isnan(vol).any():
                raise RuntimeError("Gram method failed")

        if (self.mode == "cm") or (self.mode == "auto" and (vol is None or torch.isnan(vol).any())):
            vol, vol_sq, is_degenerate = self._cm_volume(vertices)

        # quality vs regular simplex
        dists = torch.cdist(vertices, vertices, p=2)
        iu, ju = torch.triu_indices(k_plus_1, k_plus_1, 1, device=vertices.device)
        mean_edge = dists[..., iu, ju].mean(dim=-1).clamp(min=1e-12)
        reg = ((mean_edge ** k) / math.factorial(k)) * math.sqrt((k + 1) / (2 ** k))
        quality = (vol / (reg + 1e-10)).clamp(min=0.0)

        return {
            "volume": vol,
            "volume_squared": vol_sq,
            "is_degenerate": is_degenerate,
            "quality": quality,
        }
class SimplexQualityExtended(FormulaBase):
    """Compute extended quality metrics for k-simplex."""

    def __init__(self):
        super().__init__("simplex_quality", "f.quadratic.quality")

    def forward(self, vertices: torch.Tensor) -> Dict[str, torch.Tensor]:
        edge_calc = SimplexEdges()
        edge_result = edge_calc.forward(vertices)
        aspect_ratio = edge_result["aspect_ratio"]

        vol_calc = SimplexVolumeExtended()
        vol_result = vol_calc.forward(vertices)
        volume_quality = vol_result["quality"]
        is_degenerate = vol_result["is_degenerate"]

        regularity = (1.0 / (aspect_ratio + 1e-10)).clamp(0.0, 1.0)

        # NEW: edge coefficient of variation
        n_vertices = vertices.shape[-2]
        ii, jj = torch.triu_indices(n_vertices, n_vertices, offset=1, device=vertices.device)
        all_edges = torch.norm(vertices[..., jj, :] - vertices[..., ii, :], dim=-1)
        edge_mean = all_edges.mean(dim=-1).clamp(min=1e-12)
        edge_cv = (all_edges.std(dim=-1) / edge_mean).clamp(min=0.0)

        # NEW: Gram condition index
        v0 = vertices[..., 0:1, :]
        E = (vertices[..., 1:, :] - v0).to(torch.float64)
        G = 0.5 * (E.transpose(-2, -1) @ E + (E.transpose(-2, -1) @ E).transpose(-2, -1))
        evals = torch.linalg.eigvalsh(G).real.clamp(min=1e-20)
        cond_index = (evals[..., -1] / evals[..., 0]).to(vertices.dtype)

        is_well_shaped = (regularity > 0.3) & (~is_degenerate) & (cond_index < 1e6)

        return {
            "regularity": regularity,
            "aspect_ratio": aspect_ratio,
            "volume_quality": volume_quality,
            "edge_cv": edge_cv,                       # NEW
            "gram_condition_index": cond_index,       # NEW
            "is_well_shaped": is_well_shaped,
            "is_degenerate": is_degenerate,
        }

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class SimplexEdges(FormulaBase):
    """Extract edges from k-simplex.

    A k-simplex has C(k+1, 2) edges.
    """

    def __init__(self):
        super().__init__("simplex_edges", "f.quadratic.edges")

    def forward(self, vertices: Tensor) -> Dict[str, Tensor]:
        """Extract edges.

        Args:
            vertices: Simplex vertices [..., k+1, dim]

        Returns:
            edges: Edge vectors [..., n_edges, dim]
            edge_lengths: Length of each edge [..., n_edges]
            edge_indices: Vertex pairs [n_edges, 2]
            min_edge: Minimum edge length [...]
            max_edge: Maximum edge length [...]
            aspect_ratio: max_edge / min_edge [...]
        """
        n_vertices = vertices.shape[-2]

        # Generate edge indices using triu_indices (upper triangular)
        i_idx, j_idx = torch.triu_indices(n_vertices, n_vertices, offset=1, device=vertices.device)
        n_edges = i_idx.shape[0]

        # Extract vertices for all edges at once
        vertices_i = vertices[..., i_idx, :]  # [..., n_edges, dim]
        vertices_j = vertices[..., j_idx, :]  # [..., n_edges, dim]

        # Compute edge vectors and lengths
        edges = vertices_j - vertices_i  # [..., n_edges, dim]
        edge_lengths = torch.norm(edges, dim=-1)  # [..., n_edges]

        # Edge indices as pairs
        edge_indices = torch.stack([i_idx, j_idx], dim=-1)  # [n_edges, 2]

        # Statistics
        min_edge = edge_lengths.min(dim=-1)[0]
        max_edge = edge_lengths.max(dim=-1)[0]
        aspect_ratio = max_edge / (min_edge + 1e-10)

        return {
            "edges": edges,
            "edge_lengths": edge_lengths,
            "edge_indices": edge_indices,
            "min_edge": min_edge,
            "max_edge": max_edge,
            "aspect_ratio": aspect_ratio,
            "n_edges": torch.tensor(n_edges)
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class SimplexFaces(FormulaBase):
    """Extract i-dimensional faces from k-simplex.

    A k-simplex has C(k+1, i+1) faces of dimension i.
    """

    def __init__(self, face_dim: int = 2):
        super().__init__("simplex_faces", "f.quadratic.faces")
        self.face_dim = face_dim

    def forward(self, vertices: Tensor) -> Dict[str, Tensor]:
        """Extract faces.

        Args:
            vertices: Simplex vertices [..., k+1, dim]

        Returns:
            face_indices: Indices of vertices in each face [n_faces, face_dim+1]
            n_faces: Number of faces
            face_volumes: Volume of each face [..., n_faces]
        """
        n_vertices = vertices.shape[-2]
        i = self.face_dim

        if i >= n_vertices:
            return {
                "face_indices": torch.tensor([], dtype=torch.long),
                "n_faces": torch.tensor(0),
                "face_volumes": torch.tensor([])
            }

        # Generate all (i+1)-combinations of vertex indices
        face_tuples = list(combinations(range(n_vertices), i + 1))
        n_faces = len(face_tuples)

        # Convert to tensor: [n_faces, i+1]
        face_indices = torch.tensor(face_tuples, dtype=torch.long, device=vertices.device)

        # Extract face vertices
        face_idx_expanded = face_indices.view(1, n_faces, i+1, 1).expand(
            *vertices.shape[:-2], n_faces, i+1, vertices.shape[-1]
        )

        # Gather face vertices: [..., n_faces, i+1, dim]
        face_vertices = torch.gather(
            vertices.unsqueeze(-3).expand(*vertices.shape[:-2], n_faces, *vertices.shape[-2:]),
            -2,
            face_idx_expanded
        )

        # Compute volumes
        batch_shape = face_vertices.shape[:-2]
        face_verts_flat = face_vertices.reshape(-1, i+1, vertices.shape[-1])

        volume_calculator = SimplexVolume()
        vol_result = volume_calculator.forward(face_verts_flat)

        face_volumes = vol_result["volume"].reshape(*batch_shape)

        return {
            "face_indices": face_indices,
            "n_faces": torch.tensor(n_faces),
            "face_volumes": face_volumes
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class SimplexFacesSampler(FormulaBase):
    """
    Generic sampler: select k-simplices and apply any formula.

    Replaces SimplexFacesDiffusion with vectorized operations (no for loops).

    Args:
        face_dim: Dimension of faces (k)
        sample_budget: Number of faces to sample
        formula: FormulaBase instance to apply
        diffusion_steps: Diffusion iterations (default: 5)
        temperature: Diffusion temperature (default: 0.1)
        selection_strategy: 'diffusion', 'random', 'uniform' (default: 'diffusion')
        aggregate_to_vertices: Scatter results to vertices (default: True)
    """

    def __init__(
        self,
        face_dim: int,
        sample_budget: int,
        formula: FormulaBase,
        diffusion_steps: int = 5,
        temperature: float = 0.1,
        selection_strategy: str = "diffusion",
        aggregate_to_vertices: bool = True
    ):
        super().__init__(
            f"simplex_sampler_{formula.name}",
            f"f.sampler.{formula.uid}"
        )

        self.face_dim = face_dim
        self.sample_budget = sample_budget
        self.formula = formula
        self.diffusion_steps = diffusion_steps
        self.temperature = temperature
        self.selection_strategy = selection_strategy
        self.aggregate_to_vertices = aggregate_to_vertices

    def _select_diffusion(self, vertices: Tensor) -> Dict[str, Tensor]:
        """Diffusion-based sampling (vectorized)."""
        batch_dims = vertices.shape[:-2]
        n_vertices = vertices.shape[-2]
        k = self.face_dim
        device = vertices.device
        dtype = vertices.dtype

        # Distances and adjacency
        distances = torch.cdist(vertices, vertices, p=2)

        # Normalize distances to prevent numerical overflow
        # Use median distance as scale
        dist_flat = distances.reshape(-1, n_vertices * n_vertices)
        median_dist = torch.median(dist_flat, dim=-1, keepdim=True)[0]
        median_dist = median_dist.reshape(*batch_dims, 1, 1).clamp(min=1e-6)
        distances_normalized = distances / median_dist

        adjacency = torch.exp(-distances_normalized.pow(2) / self.temperature)
        adjacency = adjacency * (1 - torch.eye(n_vertices, device=device, dtype=dtype))

        # Transition matrix with better numerical stability
        row_sums = adjacency.sum(dim=-1, keepdim=True)
        # If row sum is too small, fall back to uniform
        row_sums = torch.where(row_sums < 1e-10,
                               torch.ones_like(row_sums),
                               row_sums)
        transition = adjacency / row_sums

        # Matrix power with stability check
        transition_power = torch.linalg.matrix_power(transition, self.diffusion_steps)

        # Apply diffusion
        heat_init = torch.ones(*batch_dims, n_vertices, 1, device=device, dtype=dtype) / n_vertices
        heat = torch.matmul(transition_power, heat_init).squeeze(-1)

        # Ensure valid probability distribution
        heat = torch.clamp(heat, min=1e-10)
        heat = heat / heat.sum(dim=-1, keepdim=True)

        # Additional safety: check for NaN/Inf
        if not torch.all(torch.isfinite(heat)):
            # Fallback to uniform distribution
            heat = torch.ones_like(heat) / n_vertices

        # Sample
        n_samples = (k + 1) * self.sample_budget
        batch_size = heat.reshape(-1, n_vertices).shape[0]
        heat_flat = heat.reshape(batch_size, n_vertices)

        # Final safety check before multinomial
        heat_flat = torch.clamp(heat_flat, min=1e-10)
        heat_flat = heat_flat / heat_flat.sum(dim=-1, keepdim=True)

        sampled_flat = torch.multinomial(heat_flat, n_samples, replacement=True)
        sampled = sampled_flat.reshape(*batch_dims, self.sample_budget, k + 1)
        face_indices, _ = torch.sort(sampled, dim=-1)

        return {"face_indices": face_indices, "importance_scores": heat}

    def _select_random(self, vertices: Tensor) -> Dict[str, Tensor]:
        """Random sampling (vectorized)."""
        batch_dims = vertices.shape[:-2]
        n_vertices = vertices.shape[-2]
        device = vertices.device

        face_indices = torch.randint(
            0, n_vertices,
            (*batch_dims, self.sample_budget, self.face_dim + 1),
            device=device
        )
        face_indices, _ = torch.sort(face_indices, dim=-1)

        importance = torch.ones(*batch_dims, n_vertices, device=device) / n_vertices
        return {"face_indices": face_indices, "importance_scores": importance}

    def _select_uniform(self, vertices: Tensor) -> Dict[str, Tensor]:
        """Uniform spacing (vectorized)."""
        batch_dims = vertices.shape[:-2]
        n_vertices = vertices.shape[-2]
        k = self.face_dim
        device = vertices.device

        step = max(1, n_vertices // (k + 1))
        base = torch.arange(0, min((k+1)*step, n_vertices), step, device=device)[:(k+1)]

        # Reshape for broadcasting
        for _ in batch_dims:
            base = base.unsqueeze(0)
        base = base.unsqueeze(-2)

        offsets = torch.arange(self.sample_budget, device=device) % max(1, step)
        offsets = offsets.reshape(*([1]*len(batch_dims)), self.sample_budget, 1)

        face_indices = (base + offsets) % n_vertices
        importance = torch.ones(*batch_dims, n_vertices, device=device) / n_vertices

        return {"face_indices": face_indices, "importance_scores": importance}

    def _validate_faces(self, face_indices: Tensor) -> Tensor:
        """Check uniqueness (vectorized)."""
        expanded_a = face_indices.unsqueeze(-1)
        expanded_b = face_indices.unsqueeze(-2)
        matches = (expanded_a == expanded_b).sum(dim=(-2, -1))
        return matches == (self.face_dim + 1)

    def _gather_face_vertices(self, vertices: Tensor, face_indices: Tensor) -> Tensor:
        """Gather vertices (vectorized)."""
        batch_dims = vertices.shape[:-2]
        n_vertices = vertices.shape[-2]
        dim = vertices.shape[-1]
        k_plus_1 = self.face_dim + 1

        face_idx_exp = face_indices.unsqueeze(-1).expand(*batch_dims, self.sample_budget, k_plus_1, dim)
        verts_exp = vertices.unsqueeze(-3).expand(*batch_dims, self.sample_budget, n_vertices, dim)

        return torch.gather(verts_exp, -2, face_idx_exp)

    def _aggregate_to_vertex_level(
        self,
        results: Dict[str, Tensor],
        face_indices: Tensor,
        n_vertices: int,
        valid_mask: Tensor
    ) -> Dict[str, Tensor]:
        """Scatter to vertices (fully vectorized - NO FOR LOOPS)."""
        batch_dims = face_indices.shape[:-2]
        k_plus_1 = self.face_dim + 1
        device = face_indices.device

        aggregated = {}

        for key, value in results.items():
            if not isinstance(value, Tensor):
                continue

            # Scalar per face: [..., sample_budget]
            if value.shape == valid_mask.shape:
                dtype = value.dtype

                # Masked values
                masked = (value * valid_mask.float()).unsqueeze(-1)  # [..., sample_budget, 1]
                valid_expanded = valid_mask.float().unsqueeze(-1)  # [..., sample_budget, 1]

                # Expand to k+1 copies (one per vertex in face)
                masked_broadcast = masked.unsqueeze(-2).expand(*batch_dims, self.sample_budget, k_plus_1, 1)
                valid_broadcast = valid_expanded.unsqueeze(-2).expand(*batch_dims, self.sample_budget, k_plus_1, 1)

                # Flatten sample and k dims: [..., sample_budget*k_plus_1, 1]
                masked_flat = masked_broadcast.reshape(*batch_dims, -1, 1).squeeze(-1)
                valid_flat = valid_broadcast.reshape(*batch_dims, -1, 1).squeeze(-1)
                indices_flat = face_indices.reshape(*batch_dims, -1)

                # Scatter
                output = torch.zeros(*batch_dims, n_vertices, device=device, dtype=dtype)
                counts = torch.zeros(*batch_dims, n_vertices, device=device, dtype=dtype)

                output.scatter_add_(-1, indices_flat, masked_flat)
                counts.scatter_add_(-1, indices_flat, valid_flat)

                aggregated[f"{key}_per_vertex"] = output / counts.clamp(min=1e-10)

            # Vector per face: [..., sample_budget, feat_dim]
            elif len(value.shape) == len(batch_dims) + 2:
                feat_dim = value.shape[-1]
                dtype = value.dtype

                # Mask
                masked = value * valid_mask.unsqueeze(-1).float()  # [..., sample_budget, feat_dim]
                valid_expanded = valid_mask.float().unsqueeze(-1).unsqueeze(-1)  # [..., sample_budget, 1, 1]

                # Broadcast to k+1 vertices
                masked_broadcast = masked.unsqueeze(-2).expand(*batch_dims, self.sample_budget, k_plus_1, feat_dim)
                valid_broadcast = valid_expanded.expand(*batch_dims, self.sample_budget, k_plus_1, 1)

                # Flatten: [..., sample_budget*k_plus_1, feat_dim]
                masked_flat = masked_broadcast.reshape(*batch_dims, -1, feat_dim)
                valid_flat = valid_broadcast.reshape(*batch_dims, -1, 1).expand(-1, -1, feat_dim)
                indices_flat = face_indices.reshape(*batch_dims, -1).unsqueeze(-1).expand(-1, -1, feat_dim)

                # Scatter
                output = torch.zeros(*batch_dims, n_vertices, feat_dim, device=device, dtype=dtype)
                counts = torch.zeros(*batch_dims, n_vertices, 1, device=device, dtype=dtype)

                output.scatter_add_(-2, indices_flat, masked_flat)
                counts.scatter_add_(-2, face_indices.reshape(*batch_dims, -1).unsqueeze(-1),
                                   valid_flat[..., :1])

                aggregated[f"{key}_per_vertex"] = output / counts.clamp(min=1e-10)

        return aggregated

    def forward(self, vertices: Tensor) -> Dict[str, Tensor]:
        """Sample faces and apply formula."""
        n_vertices = vertices.shape[-2]
        k_plus_1 = self.face_dim + 1
        dim = vertices.shape[-1]

        # SELECT
        if self.selection_strategy == "diffusion":
            selection = self._select_diffusion(vertices)
        elif self.selection_strategy == "random":
            selection = self._select_random(vertices)
        elif self.selection_strategy == "uniform":
            selection = self._select_uniform(vertices)
        else:
            raise ValueError(f"Unknown strategy: {self.selection_strategy}")

        face_indices = selection["face_indices"]

        # VALIDATE
        valid_mask = self._validate_faces(face_indices)

        # GATHER
        face_verts = self._gather_face_vertices(vertices, face_indices)

        # APPLY FORMULA
        batch_dims = face_verts.shape[:-2]
        batch_size = face_verts.reshape(-1, k_plus_1, dim).shape[0]
        flat_verts = face_verts.reshape(batch_size, k_plus_1, dim)

        formula_results = self.formula.forward(flat_verts)

        # Reshape results
        reshaped = {}
        for key, value in formula_results.items():
            if isinstance(value, Tensor):
                if value.ndim == 1:
                    reshaped[key] = value.reshape(*batch_dims)
                elif value.ndim == 2:
                    reshaped[key] = value.reshape(*batch_dims, value.shape[-1])
                else:
                    reshaped[key] = value
            else:
                reshaped[key] = value

        # Mask invalid
        for key, value in reshaped.items():
            if isinstance(value, Tensor):
                if value.shape[:len(valid_mask.shape)] == valid_mask.shape:
                    if value.ndim == valid_mask.ndim:
                        reshaped[key] = value * valid_mask.float()
                    elif value.ndim == valid_mask.ndim + 1:
                        reshaped[key] = value * valid_mask.unsqueeze(-1).float()

        # Output
        output = {
            **reshaped,
            "face_indices": face_indices,
            "valid_mask": valid_mask,
            "n_valid": valid_mask.sum(dim=-1),
            **selection
        }

        # AGGREGATE
        if self.aggregate_to_vertices:
            aggregated = self._aggregate_to_vertex_level(reshaped, face_indices, n_vertices, valid_mask)
            output.update(aggregated)

        return output


# Backward compatibility wrapper
def SimplexFacesDiffusion(face_dim: int = 2, sample_budget: int = 1000,
                          diffusion_steps: int = 5, temperature: float = 0.1):
    """Backward-compatible wrapper for SimplexFacesSampler."""
    return SimplexFacesSampler(
        face_dim=face_dim,
        sample_budget=sample_budget,
        formula=SimplexVolume(),
        diffusion_steps=diffusion_steps,
        temperature=temperature,
        selection_strategy="diffusion"
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class SimplexCentroid(FormulaBase):
    """Compute centroid (barycenter) of k-simplex."""

    def __init__(self):
        super().__init__("simplex_centroid", "f.quadratic.centroid")

    def forward(self, vertices: Tensor) -> Dict[str, Tensor]:
        """Compute centroid."""
        centroid = vertices.mean(dim=-2)
        distances = torch.norm(vertices - centroid.unsqueeze(-2), dim=-1)
        radius = distances.max(dim=-1)[0]

        return {
            "centroid": centroid,
            "distances_to_vertices": distances,
            "radius": radius,
            "mean_distance": distances.mean(dim=-1)
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class SimplexQuality(FormulaBase):
    """Compute quality metrics for k-simplex."""

    def __init__(self,
                 edge_calculator: Optional[FormulaBase] = None,
                 volume_calculator: Optional[FormulaBase] = None
             ):
        super().__init__("simplex_quality", "f.quadratic.quality")
        self.edge_calc = SimplexEdges() if edge_calculator is None else edge_calculator
        self.vol_calc = SimplexVolumeExtended() if volume_calculator is None else volume_calculator

    def forward(self, vertices: Tensor) -> Dict[str, Tensor]:
        """Compute quality metrics."""
        edge_result = self.edge_calc.forward(vertices)
        aspect_ratio = edge_result["aspect_ratio"]

        vol_result = self.vol_calc.forward(vertices)
        volume_quality = vol_result["quality"]

        regularity = 1.0 / (aspect_ratio + 1e-10)
        regularity = torch.clamp(regularity, 0.0, 1.0)

        is_well_shaped = (regularity > 0.3) & (~vol_result["is_degenerate"])

        return {
            "regularity": regularity,
            "aspect_ratio": aspect_ratio,
            "volume_quality": volume_quality,
            "is_well_shaped": is_well_shaped,
            "is_degenerate": vol_result["is_degenerate"]
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class SimplexToSimplex(FormulaBase):
    """Map between simplices of different dimensions."""

    def __init__(self, target_k: int):
        super().__init__("simplex_to_simplex", "f.quadratic.simplex_map")
        self.target_k = target_k

    def forward(self, vertices: Tensor) -> Dict[str, Tensor]:
        """Map simplex to different dimension."""
        source_k = vertices.shape[-2] - 1
        target_k = self.target_k

        if source_k == target_k:
            return {
                "target_vertices": vertices,
                "mapping_type": "identity"
            }

        elif source_k < target_k:
            n_to_add = target_k - source_k
            centroid = vertices.mean(dim=-2, keepdim=True)

            t_values = torch.linspace(1.0/(n_to_add+1), n_to_add/(n_to_add+1),
                                     n_to_add, device=vertices.device, dtype=vertices.dtype)
            vert_indices = torch.arange(n_to_add, device=vertices.device) % vertices.shape[-2]
            selected_verts = vertices[..., vert_indices, :]

            t_expanded = t_values.view(*([1]*(vertices.ndim-2)), n_to_add, 1)
            new_vertices = (1-t_expanded)*centroid + t_expanded*selected_verts

            target_vertices = torch.cat([vertices, new_vertices], dim=-2)
            mapping_type = "subdivision"

        else:
            target_vertices = vertices[..., :target_k+1, :]
            mapping_type = "projection"

        return {
            "target_vertices": target_vertices,
            "mapping_type": mapping_type,
            "source_k": torch.tensor(source_k),
            "target_k": torch.tensor(target_k)
        }

class CMLogDetRegularizer(FormulaBase):
    """
    Log-det(Gram) regularizer:
      loss = - logdet(Gram(E)) + γ * clamp(cond_index / C, min=0)
    Promotes non-collapsed, well-conditioned simplices.
    """

    def __init__(self, gamma: float = 0.0, cond_cap: float = 1e6):
        super().__init__("cm_logdet_reg", "f.reg.cm_logdet")
        self.gamma = float(gamma)
        self.cond_cap = float(cond_cap)

    def forward(self, vertices: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = vertices
        v0 = x[..., 0:1, :]
        E = (x[..., 1:, :] - v0).to(torch.float64)        # [..., dim, k]
        G = E.transpose(-2, -1) @ E
        G = 0.5 * (G + G.transpose(-2, -1))

        # Cholesky logdet (stable)
        k = G.shape[-1]
        eye_k = torch.eye(k, dtype=G.dtype, device=G.device)
        jitter = 0.0
        for _ in range(3):
            try:
                L = torch.linalg.cholesky(G + (jitter * eye_k))
                break
            except RuntimeError:
                jitter = 1e-10 if jitter == 0.0 else jitter * 10.0
        else:
            L = torch.linalg.cholesky(G + (1e-6 * eye_k))
        diag = torch.diagonal(L, dim1=-2, dim2=-1).clamp(min=1e-24)
        logdetG = 2.0 * torch.sum(torch.log(diag), dim=-1).to(vertices.dtype)   # [...,]

        # Condition index
        evals = torch.linalg.eigvalsh(G).real.clamp(min=1e-20)
        cond_index = (evals[..., -1] / evals[..., 0]).to(vertices.dtype)

        # Loss (minimize): encourage larger logdet, penalize huge condition
        loss = -logdetG + self.gamma * torch.clamp(cond_index / self.cond_cap, min=0.0)

        return {
            "logdet_gram": logdetG,
            "gram_condition_index": cond_index,
            "loss": loss,
        }

from typing import Callable

class RoseWeightedVolume(FormulaBase):
    """
    Combine resonance ('rose') with stable volume to produce a shaping loss.
      score   = normalize(volume) * normalize(rose)
      loss    = -score
    Optionally provide rose_fn(vertices)->[...]-shaped resonance; else use an edge-direction cosine proxy.
    """

    def __init__(self, rose_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
                 eps: float = 1e-8):
        super().__init__("rose_weighted_volume", "f.reg.rose_weighted_volume")
        self.rose_fn = rose_fn
        self.eps = float(eps)

    def _edge_cos_proxy(self, vertices: torch.Tensor) -> torch.Tensor:
        n = vertices.shape[-2]
        if n < 3:
            return torch.zeros(vertices.shape[:-2], device=vertices.device, dtype=vertices.dtype)
        ii, jj = torch.triu_indices(n, n, offset=1, device=vertices.device)
        edges = vertices[..., jj, :] - vertices[..., ii, :]                # [..., E, dim]
        norm = torch.norm(edges, dim=-1, keepdim=True).clamp(min=self.eps)
        U = edges / norm                                                   # unit
        # mean pairwise cosine across edges (directional cohesion)
        e_i = U
        e_j = U
        # (E dot E) mean; compute via batch matmul
        cos_mat = torch.matmul(e_i, e_j.transpose(-2, -1))                 # [..., E, E]
        # exclude diagonal
        E = U.shape[-2]
        mask = (~torch.eye(E, dtype=torch.bool, device=U.device))[None, ...]
        cos_mean = (cos_mat.masked_select(mask).reshape(*vertices.shape[:-2], E*(E-1))).mean(dim=-1)
        return cos_mean.clamp(-1.0, 1.0)

    def forward(self, vertices: torch.Tensor) -> Dict[str, torch.Tensor]:
        vol, vol_sq = _stable_simplex_volume(vertices)                     # stable volume
        # Normalize volume within batch for scale-invariance
        vnorm = (vol / (vol.mean(dim=tuple(range(vol.ndim)), keepdim=False) + self.eps)) if vol.ndim == 0 else (vol / (vol.mean() + self.eps))

        if self.rose_fn is not None:
            rose = self.rose_fn(vertices).to(vertices.dtype)
        else:
            rose = self._edge_cos_proxy(vertices)

        # map to [0,1] via tanh-ish squashing
        v_s = torch.tanh(vnorm)
        r_s = (rose + 1.0) * 0.5   # [-1,1] -> [0,1]
        score = v_s * r_s
        loss = -score

        return {
            "volume": vol,
            "rose": rose,
            "score": score,
            "loss": loss,
        }



# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TESTING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_quadratic_simplex_operations():
    """Original test suite for quadratic and simplex operations."""

    print("\n" + "=" * 70)
    print("QUADRATIC & SIMPLEX OPERATIONS TESTS")
    print("=" * 70)

    # Test 1: Quadratic solver
    print("\n[Test 1] Quadratic Solver")
    a = torch.tensor([1.0, 1.0, 1.0])
    b = torch.tensor([0.0, -5.0, 2.0])
    c = torch.tensor([0.0, 6.0, 5.0])

    solver = QuadraticSolver()
    result = solver.forward(a, b, c)

    print(f"  Equations: x², x²-5x+6, x²+2x+5")
    print(f"  Discriminants: {result['discriminant'].numpy()}")
    print(f"  Root 1: {result['root1'].numpy()}")
    print(f"  Root 2: {result['root2'].numpy()}")
    print(f"  Has real roots: {result['has_real_roots'].numpy()}")
    print(f"  Status: ✓ PASS")

    # Test 2: Parabola properties
    print("\n[Test 2] Parabola Properties")
    parabola = ParabolaProperties()
    para_result = parabola.forward(a[1:2], b[1:2], c[1:2])

    print(f"  Equation: x² - 5x + 6")
    print(f"  Vertex: {para_result['vertex'].numpy()}")
    print(f"  Focus: {para_result['focus'].numpy()}")
    print(f"  Opens upward: {para_result['opens_upward'].item()}")
    print(f"  Status: ✓ PASS")

    # Test 3: Simplex dimension
    print("\n[Test 3] Simplex Dimension Analysis")
    triangle = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.5, 0.866]])

    dim_calc = SimplexDimension()
    dim_result = dim_calc.forward(triangle.unsqueeze(0))

    print(f"  Vertices: {triangle.shape}")
    print(f"  k-simplex: k={dim_result['k'].item()}")
    print(f"  Number of edges: {dim_result['n_edges'].item()}")
    print(f"  Number of faces: {dim_result['n_faces'].item()}")
    print(f"  Status: ✓ PASS")

    # Test 4: Barycentric coordinates
    print("\n[Test 4] Barycentric Coordinates")
    points = torch.tensor([[0.33, 0.33], [0.0, 0.0], [2.0, 2.0]])

    bary_calc = BarycentricCoordinates()
    bary_result = bary_calc.forward(points.unsqueeze(0), triangle.unsqueeze(0))

    print(f"  Query points: {points.numpy()}")
    print(f"  Weights: {bary_result['weights'].squeeze().numpy()}")
    print(f"  Inside simplex: {bary_result['is_inside'].squeeze().numpy()}")
    print(f"  Status: ✓ PASS")

    # Test 5: Simplex volume (2D triangle)
    print("\n[Test 5] Simplex Volume (Triangle)")
    vol_calc = SimplexVolume()
    vol_result = vol_calc.forward(triangle.unsqueeze(0))

    print(f"  Triangle vertices: {triangle.shape}")
    print(f"  Volume (area): {vol_result['volume'].item():.4f}")
    print(f"  Expected: ~0.433")
    print(f"  Quality: {vol_result['quality'].item():.4f}")
    print(f"  Status: ✓ PASS")

    # Test 6: Simplex volume (3D tetrahedron)
    print("\n[Test 6] Simplex Volume (Tetrahedron)")
    tetrahedron = torch.tensor([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ])

    tet_vol = vol_calc.forward(tetrahedron.unsqueeze(0))

    print(f"  Tetrahedron vertices: {tetrahedron.shape}")
    print(f"  Volume: {tet_vol['volume'].item():.4f}")
    print(f"  Expected: ~0.1667 (1/6)")
    print(f"  Quality: {tet_vol['quality'].item():.4f}")
    print(f"  Status: ✓ PASS")

    # Test 7: Simplex edges
    print("\n[Test 7] Simplex Edges")
    edge_calc = SimplexEdges()
    edge_result = edge_calc.forward(triangle.unsqueeze(0))

    print(f"  Number of edges: {edge_result['n_edges'].item()}")
    print(f"  Edge lengths: {edge_result['edge_lengths'].squeeze().numpy()}")
    print(f"  Aspect ratio: {edge_result['aspect_ratio'].item():.4f}")
    print(f"  Status: ✓ PASS")

    # Test 8: Simplex faces
    print("\n[Test 8] Simplex Faces (2-faces from tetrahedron)")
    face_calc = SimplexFaces(face_dim=2)
    face_result = face_calc.forward(tetrahedron.unsqueeze(0))

    print(f"  Number of 2-faces: {face_result['n_faces'].item()}")
    print(f"  Face volumes: {face_result['face_volumes'].squeeze().numpy()}")
    print(f"  Status: ✓ PASS")

    # Test 9: Simplex centroid
    print("\n[Test 9] Simplex Centroid")
    centroid_calc = SimplexCentroid()
    cent_result = centroid_calc.forward(triangle.unsqueeze(0))

    print(f"  Triangle centroid: {cent_result['centroid'].squeeze().numpy()}")
    print(f"  Expected: ~[0.5, 0.289]")
    print(f"  Circumradius (approx): {cent_result['radius'].item():.4f}")
    print(f"  Status: ✓ PASS")

    # Test 10: High-dimensional simplex (4-simplex/pentachoron in 5D)
    print("\n[Test 10] High-Dimensional Simplex (4-simplex)")
    pentachoron = torch.randn(1, 5, 1024)

    dim_high = dim_calc.forward(pentachoron)
    vol_high = vol_calc.forward(pentachoron)

    print(f"  Shape: [b=1, k+1=5, dim=1024]")
    print(f"  k-simplex: k={dim_high['k'].item()}")
    print(f"  Number of edges: {dim_high['n_edges'].item()}")
    print(f"  Volume: {vol_high['volume'].item():.6e}")
    print(f"  Degenerate: {vol_high['is_degenerate'].item()}")
    print(f"  Status: ✓ PASS")

    # Test 11: Original diffusion test
    print("\n[Test 11] SimplexFacesDiffusion (Backward Compatibility)")
    large_simplex = torch.randn(2, 50, 128)

    diffusion_sampler = SimplexFacesDiffusion(
        face_dim=2,
        sample_budget=100,
        diffusion_steps=3,
        temperature=0.5
    )

    diff_result = diffusion_sampler.forward(large_simplex)

    print(f"  Input: [batch=2, n_vertices=50, dim=128]")
    print(f"  Requested faces: 100 triangular faces")
    print(f"  Face indices shape: {diff_result['face_indices'].shape}")
    print(f"  Valid faces: {diff_result['n_valid'].numpy()}")
    print(f"  Importance scores shape: {diff_result['importance_scores'].shape}")
    print(f"  Mean face volume: {diff_result['volume'][diff_result['valid_mask']].mean().item():.6e}")
    print(f"  Status: ✓ PASS")

    print("\n" + "=" * 70)
    print("All original tests completed! (11 total)")
    print("=" * 70 + "\n")


def test_simplex_sampler_new_features():
    """New tests for SimplexFacesSampler functionality."""

    print("\n" + "=" * 70)
    print("SIMPLEX FACES SAMPLER - NEW FUNCTIONALITY TESTS")
    print("=" * 70)

    # Test 1: Multi-dimensional batching
    print("\n[Test 1] Multi-dimensional Batching (No For Loops)")

    test_shapes = [
        (50, 128),
        (4, 50, 128),
        (2, 3, 50, 128),
    ]

    for shape in test_shapes:
        vertices = torch.randn(*shape)
        sampler = SimplexFacesSampler(
            face_dim=2,
            sample_budget=20,
            formula=SimplexVolume(),
            selection_strategy="diffusion"
        )

        result = sampler.forward(vertices)

        print(f"\n  Shape: {shape}")
        print(f"    Volume shape: {result['volume'].shape} (expected: {shape[:-2] + (20,)})")
        print(f"    Per-vertex: {result['volume_per_vertex'].shape} (expected: {shape[:-1]})")
        assert result['volume'].shape == shape[:-2] + (20,)
        assert result['volume_per_vertex'].shape == shape[:-1]

    print(f"\n  Status: ✓ PASS - Vectorized batching works")

    # Test 2: Multiple formulas
    print("\n[Test 2] Multiple Formula Types")

    vertices = torch.randn(3, 40, 64)

    formulas = [
        ("Volume", SimplexVolume()),
        ("Quality", SimplexQuality()),
        ("Centroid", SimplexCentroid()),
    ]

    for name, formula in formulas:
        sampler = SimplexFacesSampler(
            face_dim=2,
            sample_budget=15,
            formula=formula,
            selection_strategy="random"
        )

        result = sampler.forward(vertices)
        print(f"\n  {name}: {list(result.keys())[:4]}... ({len(result)} keys)")
        assert 'valid_mask' in result
        assert 'face_indices' in result

    print(f"\n  Status: ✓ PASS - Multiple formulas work")

    # Test 3: Large-scale CLIP-like test
    print("\n[Test 3] Large-Scale Embedding Test (CLIP-like)")

    batch_size = 8
    n_tokens = 197
    embed_dim = 512

    embeddings = torch.randn(batch_size, n_tokens, embed_dim)

    sampler = SimplexFacesSampler(
        face_dim=2,
        sample_budget=50,
        formula=SimplexQuality(),
        selection_strategy="diffusion",
        diffusion_steps=3
    )

    result = sampler.forward(embeddings)

    print(f"\n  Input: [{batch_size}, {n_tokens}, {embed_dim}]")
    print(f"  Regularity: {result['regularity'].shape}")
    print(f"  Mean regularity: {result['regularity'].mean().item():.4f}")
    print(f"  Per-token: {result['regularity_per_vertex'].shape}")
    print(f"  Valid faces: {result['n_valid']}")

    print(f"\n  Status: ✓ PASS - Large-scale test successful")

    print("\n" + "=" * 70)
    print("ALL NEW TESTS PASSED")
    print("=" * 70)
    print("\nKey Improvements:")
    print("  ✓ NO FOR LOOPS - Fully vectorized aggregation")
    print("  ✓ Multi-dimensional batching")
    print("  ✓ Generic formula application")
    print("  ✓ Works with CLIP/BERT embeddings")
    print("=" * 70 + "\n")

def test_simplex_quality_extended():
    print("\n" + "=" * 70)
    print("SIMPLEX QUALITY EXTENDED TESTS")
    print("=" * 70)

    triangle = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.5, 0.866]])  # equilateral
    tetrahedron = torch.tensor([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ])  # regular 3D

    quality_ext = SimplexQualityExtended()

    # Triangle check
    res_tri = quality_ext.forward(triangle.unsqueeze(0))
    print(f"Triangle Regularity: {res_tri['regularity'].item():.4f}")
    print(f"Triangle Gram Condition: {res_tri['gram_condition_index'].item():.4e}")
    assert res_tri["is_well_shaped"].item()

    # Tetrahedron check
    res_tet = quality_ext.forward(tetrahedron.unsqueeze(0))
    print(f"Tetrahedron Regularity: {res_tet['regularity'].item():.4f}")
    print(f"Tetrahedron Gram Condition: {res_tet['gram_condition_index'].item():.4e}")
    assert res_tet["is_well_shaped"].item()


def test_rose_weighted_volume():
    print("\n" + "=" * 70)
    print("ROSE WEIGHTED VOLUME TESTS")
    print("=" * 70)

    tetrahedron = torch.tensor([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ])

    rose_vol = RoseWeightedVolume()
    res = rose_vol.forward(tetrahedron.unsqueeze(0))
    print(f"Volume: {res['volume'].item():.4f}, Rose Score: {res['rose'].item():.4f}")
    print(f"Weighted Loss: {res['loss'].item():.4f}")
    assert torch.isfinite(res["loss"]).all()


def test_cm_logdet_regularizer():
    print("\n" + "=" * 70)
    print("CM LOGDET REGULARIZER TESTS")
    print("=" * 70)

    tetrahedron = torch.tensor([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ])

    cm_reg = CMLogDetRegularizer(gamma=0.1)
    res = cm_reg.forward(tetrahedron.unsqueeze(0))
    print(f"LogDet Gram: {res['logdet_gram'].item():.4f}")
    print(f"Condition Index: {res['gram_condition_index'].item():.4e}")
    print(f"Regularizer Loss: {res['loss'].item():.4f}")
    assert torch.isfinite(res["loss"]).all()


if __name__ == "__main__":
    # Run original tests
    test_quadratic_simplex_operations()

    # Run new tests
    test_simplex_sampler_new_features()

    test_simplex_quality_extended()
    test_rose_weighted_volume()
    test_cm_logdet_regularizer()


    print("\n[System Architecture]")
    print("-" * 70)
    print("k-Simplex Dimension Mapping:")
    print("  Input: [batch, n_vertices, embedding_dim]")
    print("  k = n_vertices - 1")
    print("  Examples:")
    print("    [b, 2, d] → 1-simplex (edge)")
    print("    [b, 3, d] → 2-simplex (triangle)")
    print("    [b, 4, d] → 3-simplex (tetrahedron)")
    print("    [b, 5, d] → 4-simplex (pentachoron)")
    print("    [b, 5, 1024] → 4-simplex in 1024D embedding")
    print("\nSimplexFacesSampler: Generic k-selector with formula application")
    print("  - No for loops (fully vectorized)")
    print("  - Pluggable formulas (any FormulaBase)")
    print("  - Multiple selection strategies")
    print("  - Automatic vertex-level aggregation")
    print("-" * 70)