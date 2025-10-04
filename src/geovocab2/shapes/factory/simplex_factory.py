"""
SimplexFactory
--------------
Factory for generating k-simplices with formula-based validation.

A k-simplex is the convex hull of k+1 affinely independent points.
Examples:
    - 0-simplex: point
    - 1-simplex: line segment (2 points)
    - 2-simplex: triangle (3 points)
    - 3-simplex: tetrahedron (4 points)

This factory generates simplices in d-dimensional embedding space and
validates them using Cayley-Menger determinants from FormulaBase.

License: MIT
"""

import numpy as np
from typing import Optional, Tuple, Union

try:
    from .factory_base import FactoryBase, HAS_TORCH
except ImportError:
    from factory_base import FactoryBase, HAS_TORCH

if HAS_TORCH:
    import torch


class SimplexFactory(FactoryBase):
    """
    Generate k-simplices with configurable embedding dimension.

    Args:
        k: Simplex dimension (k+1 vertices)
        embed_dim: Embedding space dimension (must be >= k)
        method: Generation method ('random', 'regular', 'uniform')
        scale: Scaling factor for vertex coordinates
    """

    def __init__(
        self,
        k: int,
        embed_dim: int,
        method: str = "random",
        scale: float = 1.0,
    ):
        if k < 0:
            raise ValueError(f"k must be non-negative, got {k}")
        if embed_dim < k:
            raise ValueError(f"embed_dim ({embed_dim}) must be >= k ({k})")

        super().__init__(
            name=f"simplex_k{k}_d{embed_dim}",
            uid=f"factory.simplex.k{k}.d{embed_dim}.{method}"
        )
        self.k = k
        self.embed_dim = embed_dim
        self.method = method
        self.scale = scale
        self.num_vertices = k + 1

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # NumPy Backend
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def build_numpy(
        self,
        *,
        dtype=np.float32,
        seed: Optional[int] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Build k-simplex using NumPy.

        Returns:
            Array of shape (k+1, embed_dim) representing simplex vertices
        """
        rng = np.random.default_rng(seed)

        if self.method == "random":
            vertices = self._generate_random(rng, dtype)

        elif self.method == "regular":
            vertices = self._generate_regular(dtype)

        elif self.method == "uniform":
            vertices = self._generate_uniform(rng, dtype)

        else:
            raise ValueError(f"Unknown method: {self.method}")

        return vertices * self.scale

    def _generate_random(self, rng, dtype) -> np.ndarray:
        """
        Generate simplex with random vertices.

        Uses QR decomposition to ensure affine independence.
        """
        # Generate random matrix
        raw = rng.standard_normal((self.num_vertices, self.embed_dim))

        # QR decomposition ensures linear independence
        q, r = np.linalg.qr(raw.T)
        vertices = q.T[:self.num_vertices]

        # Center at origin (affine independence, not just linear)
        vertices = vertices - vertices.mean(axis=0, keepdims=True)

        return vertices.astype(dtype, copy=False)

    def _generate_regular(self, dtype) -> np.ndarray:
        """
        Generate regular simplex (all edges equal length = 1.0).

        Uses standard construction where vertex i has:
        - coordinate i: sqrt((k+1)/k)
        - all other coordinates: -1/k

        This guarantees all pairwise distances equal sqrt(2(k+1)/k).
        Then we normalize to unit edge length.
        """
        if self.k == 0:
            return np.zeros((1, self.embed_dim), dtype=dtype)

        # Need k+1 dimensions minimum for k-simplex
        min_dim = self.k + 1

        # Build in minimal embedding space
        vertices_minimal = np.full((self.num_vertices, min_dim), -1.0 / self.k, dtype=dtype)

        # Set diagonal to sqrt((k+1)/k)
        coef = np.sqrt((self.k + 1.0) / self.k)
        np.fill_diagonal(vertices_minimal, coef)

        # Embed into higher dimensional space if needed
        if self.embed_dim > min_dim:
            vertices = np.zeros((self.num_vertices, self.embed_dim), dtype=dtype)
            vertices[:, :min_dim] = vertices_minimal
        else:
            vertices = vertices_minimal[:, :self.embed_dim]

        # Center at origin
        vertices = vertices - vertices.mean(axis=0, keepdims=True)

        # Normalize to unit edge length
        edge_length = np.linalg.norm(vertices[1] - vertices[0])
        if edge_length > 1e-10:
            vertices = vertices / edge_length

        return vertices

    def _generate_uniform(self, rng, dtype) -> np.ndarray:
        """
        Generate simplex with uniform distribution in hypercube.

        Simple method: sample from unit hypercube, no special structure.
        """
        vertices = rng.uniform(
            -1.0, 1.0,
            size=(self.num_vertices, self.embed_dim)
        )

        # Ensure affine independence via small perturbation
        for i in range(1, self.num_vertices):
            vertices[i] += 0.1 * i * np.eye(self.embed_dim)[i % self.embed_dim]

        return vertices.astype(dtype, copy=False)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PyTorch Backend
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def build_torch(
        self,
        *,
        device: str = "cpu",
        dtype: Optional["torch.dtype"] = None,
        seed: Optional[int] = None,
        **kwargs
    ) -> "torch.Tensor":
        """
        Build k-simplex using PyTorch (direct on-device generation).
        """
        if not HAS_TORCH:
            raise RuntimeError("PyTorch required for build_torch")

        target_dtype = dtype or self._infer_torch_dtype(device)
        dev = torch.device(device)

        # Generate on CPU first (RNG is CPU-only)
        if seed is not None:
            gen = torch.Generator(device="cpu")
            gen.manual_seed(seed)
        else:
            gen = None

        if self.method == "random":
            vertices = self._generate_random_torch(gen, target_dtype)

        elif self.method == "regular":
            vertices = self._generate_regular_torch(target_dtype)

        elif self.method == "uniform":
            vertices = self._generate_uniform_torch(gen, target_dtype)

        else:
            raise ValueError(f"Unknown method: {self.method}")

        return (vertices * self.scale).to(dev)

    def _generate_random_torch(self, gen, dtype) -> "torch.Tensor":
        """Random simplex using QR decomposition."""
        raw = torch.randn(
            (self.num_vertices, self.embed_dim),
            generator=gen,
            dtype=dtype
        )

        q, r = torch.linalg.qr(raw.T)
        vertices = q.T[:self.num_vertices]
        vertices = vertices - vertices.mean(dim=0, keepdim=True)

        return vertices

    def _generate_regular_torch(self, dtype) -> "torch.Tensor":
        """Regular simplex construction (vectorized, matches numpy)."""
        if self.k == 0:
            return torch.zeros((1, self.embed_dim), dtype=dtype)

        min_dim = self.k + 1

        # Create matrix filled with -1/k
        vertices_minimal = torch.full((self.num_vertices, min_dim), -1.0 / self.k, dtype=dtype)

        # Set diagonal to sqrt((k+1)/k) - vectorized
        coef = float(np.sqrt((self.k + 1.0) / self.k))
        diag_size = min(self.num_vertices, min_dim)
        diag_indices = torch.arange(diag_size, dtype=torch.long)
        vertices_minimal[diag_indices, diag_indices] = coef

        # Embed into higher dimensional space if needed
        if self.embed_dim > min_dim:
            vertices = torch.zeros((self.num_vertices, self.embed_dim), dtype=dtype)
            vertices[:, :min_dim] = vertices_minimal
        else:
            vertices = vertices_minimal[:, :self.embed_dim]

        # Center and normalize
        vertices = vertices - vertices.mean(dim=0, keepdim=True)

        edge_length = torch.linalg.norm(vertices[1] - vertices[0])
        if edge_length > 1e-10:
            vertices = vertices / edge_length

        return vertices

    def _generate_uniform_torch(self, gen, dtype) -> "torch.Tensor":
        """Uniform hypercube sampling (vectorized)."""
        vertices = torch.rand(
            (self.num_vertices, self.embed_dim),
            generator=gen,
            dtype=dtype
        ) * 2 - 1  # Scale to [-1, 1]

        # Vectorized perturbation for affine independence
        if self.num_vertices > 1:
            eye = torch.eye(self.embed_dim, dtype=dtype)
            indices = torch.arange(1, self.num_vertices, dtype=torch.long)

            # Create perturbations: [num_vertices-1, embed_dim]
            perturbations = 0.1 * indices.unsqueeze(-1).float() * eye[indices % self.embed_dim]
            vertices[1:] += perturbations

        return vertices

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Validation
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def validate(
        self,
        output: Union[np.ndarray, "torch.Tensor"]
    ) -> Tuple[bool, str]:
        """
        Validate simplex using FormulaBase geometric checks.

        Checks:
            1. Shape is (k+1, embed_dim)
            2. No NaN/Inf values
            3. Non-degenerate (positive volume via Cayley-Menger)
        """
        # Check shape
        expected_shape = (self.num_vertices, self.embed_dim)
        if output.shape != expected_shape:
            return False, f"Expected shape {expected_shape}, got {output.shape}"

        # Check for NaN/Inf
        if isinstance(output, np.ndarray):
            if not np.all(np.isfinite(output)):
                return False, "Contains NaN or Inf"
            # Convert to torch for formula
            verts_torch = torch.from_numpy(output).unsqueeze(0)
        else:
            if not torch.all(torch.isfinite(output)):
                return False, "Contains NaN or Inf"
            verts_torch = output.unsqueeze(0) if output.ndim == 2 else output

        # Use CayleyMengerFromSimplex for validation
        try:
            from geovocab2.shapes.formula.symbolic.cayley_menger import CayleyMengerFromSimplex

            validator = CayleyMengerFromSimplex(eps=1e-10, validate_input=False)
            result = validator.forward(verts_torch)

            # Check for degeneracy
            is_degenerate = result["is_degenerate"].squeeze().item()
            volume = result["volume"].squeeze().item()

            if is_degenerate or volume < 1e-10:
                return False, f"Degenerate simplex (volume={volume:.2e})"

            return True, ""

        except ImportError:
            # Fallback to basic rank check if formulas not available
            if isinstance(output, np.ndarray):
                verts = output
            else:
                verts = output.cpu().numpy()

            translated = verts[1:] - verts[0]
            rank = np.linalg.matrix_rank(translated, tol=1e-6)

            if rank < self.k:
                return False, f"Vertices not affinely independent (rank={rank}, need {self.k})"

            return True, ""

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Metadata
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def info(self):
        base_info = super().info()
        base_info.update({
            "description": f"k-simplex factory (k={self.k}, embed_dim={self.embed_dim})",
            "simplex_dimension": self.k,
            "num_vertices": self.num_vertices,
            "embedding_dimension": self.embed_dim,
            "generation_method": self.method,
            "scale": self.scale,
            "output_shape": (self.num_vertices, self.embed_dim)
        })
        return base_info


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Example Usage
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    print("=" * 70)
    print("SIMPLEX FACTORY DEMONSTRATION")
    print("=" * 70)

    # Example 1: 2-simplex (triangle) in 3D
    print("\n[Example 1] 2-simplex (triangle) in R^3, random method")
    factory_2d = SimplexFactory(k=2, embed_dim=3, method="random", scale=1.0)

    triangle = factory_2d.build(backend="numpy", seed=42, validate=True)
    print(f"  Shape: {triangle.shape}")
    print(f"  Vertices:\n{triangle}")
    print(f"  Info: {factory_2d.info()['description']}")

    # Example 2: 3-simplex (tetrahedron) in 4D
    print("\n[Example 2] 3-simplex (tetrahedron) in R^4, regular method")
    factory_3d = SimplexFactory(k=3, embed_dim=4, method="regular")

    tetrahedron = factory_3d.build(backend="numpy", validate=True)
    print(f"  Shape: {tetrahedron.shape}")

    # Check edge lengths (should all be equal for regular simplex)
    edges = []
    for i in range(4):
        for j in range(i+1, 4):
            edge_len = np.linalg.norm(tetrahedron[i] - tetrahedron[j])
            edges.append(edge_len)

    print(f"  Edge lengths (regular): {[f'{e:.6f}' for e in edges[:3]]}...")
    print(f"  All edges: {[f'{e:.6f}' for e in edges]}")
    print(f"  Edge std dev: {np.std(edges):.8f} (should be ~0)")
    print(f"  Edge mean: {np.mean(edges):.6f}")

    # Example 3: High-dimensional simplex
    print("\n[Example 3] 5-simplex in R^10, uniform method")
    factory_hd = SimplexFactory(k=5, embed_dim=10, method="uniform")

    simplex_hd = factory_hd.build(backend="numpy", seed=123, validate=True)
    print(f"  Shape: {simplex_hd.shape}")
    print(f"  Centroid: {simplex_hd.mean(axis=0)[:4]}... (should be near origin)")

    if HAS_TORCH:
        # Example 4: PyTorch backend with CUDA
        print("\n[Example 4] 2-simplex on PyTorch (CPU)")
        triangle_torch = factory_2d.build(
            backend="torch",
            device="cpu",
            seed=42,
            validate=True
        )
        print(f"  Type: {type(triangle_torch)}")
        print(f"  Shape: {triangle_torch.shape}")
        print(f"  Device: {triangle_torch.device}")
        print(f"  Dtype: {triangle_torch.dtype}")

        if torch.cuda.is_available():
            print("\n[Example 5] 3-simplex on CUDA")
            tetra_cuda = factory_3d.build(
                backend="torch",
                device="cuda:0",
                validate=True
            )
            print(f"  Device: {tetra_cuda.device}")
            print(f"  Dtype: {tetra_cuda.dtype}")

    # Example 6: Validation failure case
    print("\n[Example 6] Validation test")
    factory_test = SimplexFactory(k=2, embed_dim=3, method="random")

    # Create degenerate simplex (collinear points)
    bad_simplex = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0]  # Collinear!
    ], dtype=np.float32)

    is_valid, msg = factory_test.validate(bad_simplex)
    print(f"  Degenerate simplex valid? {is_valid}")
    print(f"  Message: {msg}")

    print("\n" + "=" * 70)
    print("SimplexFactory ready for:")
    print("  - Integration with FactoryDataset")
    print("  - Formula-based processing (Cayley-Menger, capacity)")
    print("  - Pipeline stages for geometric validation")
    print("=" * 70)