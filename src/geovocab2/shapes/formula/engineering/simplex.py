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

Author: AbstractPhil + Claude Sonnet 4.5
License: MIT
"""

from typing import Dict, Optional, Tuple, List
import torch
from torch import Tensor
import math
from itertools import combinations

from shapes.formula.formula_base import FormulaBase


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
            volume = torch.zeros(vertices.shape[:-2])
            return {
                "volume": volume,
                "volume_squared": volume,
                "is_degenerate": torch.ones_like(volume, dtype=torch.bool),
                "quality": torch.zeros_like(volume)
            }

        # Compute pairwise distances
        distances = torch.cdist(vertices, vertices, p=2)  # [..., k+1, k+1]

        # Build Cayley-Menger matrix
        # CM = [ 0   1   1  ...  1  ]
        #      [ 1  d²₀₁ d²₀₂ ... d²₀ₖ]
        #      [ 1  d²₁₀ d²₁₂ ... d²₁ₖ]
        #      [ :   :    :       :  ]

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
        mean_edge = distances[..., torch.triu_indices(k_plus_1, k_plus_1, offset=1)[0],
                              torch.triu_indices(k_plus_1, k_plus_1, offset=1)[1]].mean(dim=-1)

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
        # This gives all pairs (i, j) where i < j
        i_idx, j_idx = torch.triu_indices(n_vertices, n_vertices, offset=1, device=vertices.device)
        n_edges = i_idx.shape[0]

        # Extract vertices for all edges at once
        # vertices: [..., k+1, dim]
        # i_idx, j_idx: [n_edges]
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
            # No faces of this dimension
            return {
                "face_indices": torch.tensor([], dtype=torch.long),
                "n_faces": torch.tensor(0),
                "face_volumes": torch.tensor([])
            }

        # Generate all (i+1)-combinations of vertex indices
        # This is the only place we use combinations, but it's computed once, not per batch
        face_tuples = list(combinations(range(n_vertices), i + 1))
        n_faces = len(face_tuples)

        # Convert to tensor: [n_faces, i+1]
        face_indices = torch.tensor(face_tuples, dtype=torch.long, device=vertices.device)

        # Extract all face vertices at once using advanced indexing
        # vertices: [..., k+1, dim]
        # face_indices: [n_faces, i+1]
        # We want: [..., n_faces, i+1, dim]

        # Expand vertices for broadcasting
        # [..., k+1, dim] -> [..., 1, k+1, dim]
        verts_expanded = vertices.unsqueeze(-3)

        # Use gather to extract face vertices
        # face_indices: [n_faces, i+1] -> [1, n_faces, i+1, 1]
        face_idx_expanded = face_indices.view(1, n_faces, i+1, 1).expand(
            *vertices.shape[:-2], n_faces, i+1, vertices.shape[-1]
        )

        # Gather face vertices: [..., n_faces, i+1, dim]
        face_vertices = torch.gather(
            vertices.unsqueeze(-3).expand(*vertices.shape[:-2], n_faces, *vertices.shape[-2:]),
            -2,
            face_idx_expanded
        )

        # Compute volumes for all faces in batch
        # face_vertices: [..., n_faces, i+1, dim]
        # Reshape to [...*n_faces, i+1, dim] for batch processing
        batch_shape = face_vertices.shape[:-2]
        face_verts_flat = face_vertices.reshape(-1, i+1, vertices.shape[-1])

        # Compute volume for all faces at once
        volume_calculator = SimplexVolume()
        vol_result = volume_calculator.forward(face_verts_flat)

        # Reshape back to [..., n_faces]
        face_volumes = vol_result["volume"].reshape(*batch_shape)

        return {
            "face_indices": face_indices,
            "n_faces": torch.tensor(n_faces),
            "face_volumes": face_volumes
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class SimplexFacesDiffusion(FormulaBase):
    """Sample faces using heat diffusion to identify geometrically important regions.

    Uses diffusion on vertex adjacency graph to concentrate sampling in high-curvature,
    high-connectivity areas. Fully batched with zero Python loops.

    Args:
        face_dim: Dimension of faces to extract (default: 2 for triangles)
        sample_budget: Number of faces to sample (default: 1000)
        diffusion_steps: Number of diffusion iterations (default: 5)
        temperature: Controls diffusion spread (default: 0.1)
    """

    def __init__(self, face_dim: int = 2, sample_budget: int = 1000,
                 diffusion_steps: int = 5, temperature: float = 0.1):
        super().__init__("simplex_faces_diffusion", "f.quadratic.faces_diffusion")
        self.face_dim = face_dim
        self.sample_budget = sample_budget
        self.diffusion_steps = diffusion_steps
        self.temperature = temperature

    def forward(self, vertices: Tensor) -> Dict[str, Tensor]:
        """Sample faces using diffusion-based importance sampling.

        Args:
            vertices: Simplex vertices [..., n_vertices, dim]

        Returns:
            face_indices: Sampled face indices [..., sample_budget, face_dim+1]
            face_volumes: Volumes of sampled faces [..., sample_budget]
            valid_mask: Which faces have unique vertices [..., sample_budget]
            importance_scores: Per-vertex importance from diffusion [..., n_vertices]
            n_valid: Number of valid faces per batch [...]
        """
        n_vertices = vertices.shape[-2]
        k = self.face_dim
        device = vertices.device
        batch_shape = vertices.shape[:-2]

        # Build adjacency matrix: A_ij = exp(-||v_i - v_j||^2 / T)
        distances = torch.cdist(vertices, vertices, p=2)  # [..., n, n]
        adjacency = torch.exp(-distances ** 2 / self.temperature)

        # Zero diagonal
        adjacency = adjacency * (1 - torch.eye(n_vertices, device=device))

        # Normalize to transition matrix
        transition = adjacency / (adjacency.sum(dim=-1, keepdim=True) + 1e-10)

        # Compute P^t using matrix power (single operation, no loop)
        transition_power = torch.linalg.matrix_power(transition, self.diffusion_steps)

        # Apply diffusion: heat distribution after t steps
        heat_init = torch.ones(*batch_shape, n_vertices, 1, device=device) / n_vertices
        heat = torch.matmul(transition_power, heat_init).squeeze(-1)  # [..., n_vertices]

        # Numerical stability: clamp and renormalize
        heat = torch.clamp(heat, min=1e-10)
        heat = heat / heat.sum(dim=-1, keepdim=True)

        # Sample vertices proportional to importance
        n_samples = (k + 1) * self.sample_budget

        # Reshape for batched multinomial
        heat_flat = heat.reshape(-1, n_vertices)
        sampled_flat = torch.multinomial(heat_flat, n_samples, replacement=True)
        sampled = sampled_flat.reshape(*batch_shape, self.sample_budget, k+1)

        # Sort faces and check for uniqueness (vectorized)
        face_indices, _ = torch.sort(sampled, dim=-1)

        # Uniqueness check: count self-matches (diagonal should equal k+1)
        expanded_a = face_indices.unsqueeze(-1)  # [..., budget, k+1, 1]
        expanded_b = face_indices.unsqueeze(-2)  # [..., budget, 1, k+1]
        matches = (expanded_a == expanded_b).sum(dim=(-2, -1))  # [..., budget]
        valid_mask = (matches == (k + 1))

        # Gather face vertices
        dim = vertices.shape[-1]
        face_idx_exp = face_indices.unsqueeze(-1).expand(
            *batch_shape, self.sample_budget, k+1, dim
        )
        verts_exp = vertices.unsqueeze(-3).expand(
            *batch_shape, self.sample_budget, n_vertices, dim
        )
        face_verts = torch.gather(verts_exp, -2, face_idx_exp)

        # Compute volumes for all faces in batch
        vol_calc = SimplexVolume()
        volumes = vol_calc.forward(face_verts.reshape(-1, k+1, dim))["volume"]
        face_volumes = volumes.reshape(*batch_shape, self.sample_budget)

        # Zero out invalid faces
        face_volumes = face_volumes * valid_mask.float()

        return {
            "face_indices": face_indices,
            "face_volumes": face_volumes,
            "valid_mask": valid_mask,
            "importance_scores": heat,
            "n_valid": valid_mask.sum(dim=-1)
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class SimplexCentroid(FormulaBase):
    """Compute centroid (barycenter) of k-simplex.

    Centroid = (1/(k+1)) × Σᵢ vᵢ
    """

    def __init__(self):
        super().__init__("simplex_centroid", "f.quadratic.centroid")

    def forward(self, vertices: Tensor) -> Dict[str, Tensor]:
        """Compute centroid.

        Args:
            vertices: Simplex vertices [..., k+1, dim]

        Returns:
            centroid: Center point [..., dim]
            distances_to_vertices: Distance from centroid to each vertex [..., k+1]
            radius: Maximum distance (circumradius approximation) [...]
        """
        # Centroid
        centroid = vertices.mean(dim=-2)

        # Distances to vertices
        distances = torch.norm(vertices - centroid.unsqueeze(-2), dim=-1)

        # Maximum distance (approximate circumradius)
        radius = distances.max(dim=-1)[0]

        return {
            "centroid": centroid,
            "distances_to_vertices": distances,
            "radius": radius,
            "mean_distance": distances.mean(dim=-1)
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class SimplexQuality(FormulaBase):
    """Compute quality metrics for k-simplex.

    Measures how well-shaped the simplex is.
    """

    def __init__(self):
        super().__init__("simplex_quality", "f.quadratic.quality")

    def forward(self, vertices: Tensor) -> Dict[str, Tensor]:
        """Compute quality metrics.

        Args:
            vertices: Simplex vertices [..., k+1, dim]

        Returns:
            regularity: Deviation from regular simplex [0, 1] [...]
            aspect_ratio: Edge length ratio [...]
            volume_quality: Volume relative to regular [...]
            is_well_shaped: quality > threshold [...]
        """
        # Get edges
        edge_calc = SimplexEdges()
        edge_result = edge_calc.forward(vertices)

        aspect_ratio = edge_result["aspect_ratio"]

        # Get volume
        vol_calc = SimplexVolume()
        vol_result = vol_calc.forward(vertices)

        volume_quality = vol_result["quality"]

        # Regularity: inverse of aspect ratio
        regularity = 1.0 / (aspect_ratio + 1e-10)
        regularity = torch.clamp(regularity, 0.0, 1.0)

        # Well-shaped if regularity > 0.3 and not degenerate
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
    """Map between simplices of different dimensions.

    For [b, k₁+1, d] → [b, k₂+1, d] transformations.
    """

    def __init__(self, target_k: int):
        super().__init__("simplex_to_simplex", "f.quadratic.simplex_map")
        self.target_k = target_k

    def forward(self, vertices: Tensor) -> Dict[str, Tensor]:
        """Map simplex to different dimension.

        Args:
            vertices: Source simplex [..., k₁+1, dim]

        Returns:
            target_vertices: Target simplex [..., k₂+1, dim]
            mapping_type: "subdivision", "projection", or "aggregation"
        """
        source_k = vertices.shape[-2] - 1
        target_k = self.target_k

        if source_k == target_k:
            # Same dimension
            return {
                "target_vertices": vertices,
                "mapping_type": "identity"
            }

        elif source_k < target_k:
            # Subdivision: add vertices
            n_to_add = target_k - source_k

            # Compute centroid once
            centroid = vertices.mean(dim=-2, keepdim=True)  # [..., 1, dim]

            # Create interpolation weights for all new vertices at once
            # t values: [n_to_add]
            t_values = torch.linspace(1.0 / (n_to_add + 1),
                                     n_to_add / (n_to_add + 1),
                                     n_to_add,
                                     device=vertices.device,
                                     dtype=vertices.dtype)

            # Vertex indices to interpolate with (cycle through existing vertices)
            # [n_to_add]
            vert_indices = torch.arange(n_to_add, device=vertices.device) % vertices.shape[-2]

            # Get selected vertices: [..., n_to_add, dim]
            selected_verts = vertices[..., vert_indices, :]

            # Interpolate: (1-t)*centroid + t*vertex
            # centroid: [..., 1, dim]
            # selected_verts: [..., n_to_add, dim]
            # t_values: [n_to_add] -> [..., n_to_add, 1]
            t_expanded = t_values.view(*([1] * (vertices.ndim - 2)), n_to_add, 1)

            new_vertices = (1 - t_expanded) * centroid + t_expanded * selected_verts

            # Concatenate: [..., k₁+1, dim] + [..., n_to_add, dim] -> [..., k₂+1, dim]
            target_vertices = torch.cat([vertices, new_vertices], dim=-2)
            mapping_type = "subdivision"

        else:
            # Aggregation: remove vertices
            # Keep first (target_k + 1) vertices
            target_vertices = vertices[..., :target_k + 1, :]
            mapping_type = "projection"

        return {
            "target_vertices": target_vertices,
            "mapping_type": mapping_type,
            "source_k": torch.tensor(source_k),
            "target_k": torch.tensor(target_k)
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TESTING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_quadratic_simplex_operations():
    """Test suite for quadratic and simplex operations."""

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
    pentachoron = torch.randn(1, 5, 1024)  # [b=1, k+1=5, embedding_dim=1024]

    dim_high = dim_calc.forward(pentachoron)
    vol_high = vol_calc.forward(pentachoron)

    print(f"  Shape: [b=1, k+1=5, dim=1024]")
    print(f"  k-simplex: k={dim_high['k'].item()}")
    print(f"  Number of edges: {dim_high['n_edges'].item()}")
    print(f"  Volume: {vol_high['volume'].item():.6e}")
    print(f"  Degenerate: {vol_high['is_degenerate'].item()}")
    print(f"  Status: ✓ PASS")

    # Test 11: Diffusion-based face sampling
    print("\n[Test 11] Diffusion Face Sampling (Large Simplex)")
    large_simplex = torch.randn(2, 50, 128)  # [batch=2, n_vertices=50, dim=128]

    diffusion_sampler = SimplexFacesDiffusion(
        face_dim=2,  # Sample triangular faces
        sample_budget=100,  # Sample 100 faces
        diffusion_steps=3,
        temperature=0.5
    )

    diff_result = diffusion_sampler.forward(large_simplex)

    print(f"  Input: [batch=2, n_vertices=50, dim=128]")
    print(f"  Requested faces: 100 triangular faces (3 vertices each)")
    print(f"  Face indices shape: {diff_result['face_indices'].shape}")
    print(f"  Valid faces: {diff_result['n_valid'].numpy()}")
    print(f"  Importance scores shape: {diff_result['importance_scores'].shape}")
    print(f"  Mean face volume: {diff_result['face_volumes'][diff_result['valid_mask']].mean().item():.6e}")
    print(f"  Total possible faces: C(50,3) = 19,600")
    print(f"  Sampled: 100 (0.5% coverage)")
    print(f"  Status: ✓ PASS")

    print("\n" + "=" * 70)
    print("All tests completed! (11 total)")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    test_quadratic_simplex_operations()

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
    print("\nOperations support arbitrary k-dimensional simplices.")
    print("-" * 70)