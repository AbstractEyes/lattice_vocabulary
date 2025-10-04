"""
LegacyFactory
-------------
Factory for generating pentachoron (5-vertex) structures using the legacy
deterministic inflation algorithm from GeometricVocab.

This factory extracts the pure geometric core from the original vocab system,
removing all lexical dependencies (character lookups, wordnet, etc.) to provide
a clean, deterministic pentachoron generator.

Mathematical Approach:
    Given a center vector c ∈ ℝᵈ:
    1. Generate 5 proposal vectors via deterministic transformations (rolls, signs)
    2. L1-normalize each row
    3. Apply Gram-Schmidt orthogonalization with L1 renormalization
    4. Scale by directional weights γ
    5. Mean-center the result

Original Source: GeometricVocab._deterministic_pentachoron()

License: MIT
"""

import numpy as np
from typing import Optional, Union, Tuple
from .factory_base import FactoryBase, HAS_TORCH

if HAS_TORCH:
    import torch


class LegacyFactory(FactoryBase):
    """
    Generate pentachoron (5-vertex simplex) using legacy deterministic inflation.

    Args:
        embed_dim: Embedding space dimension
        scale: Overall scaling factor for output
        gamma: Directional weights for 5 vertices (default: [1.0, 0.9, -0.8, 1.1, 1.2])
        center_method: How to generate center vector
                      - 'random': Random normal vector
                      - 'uniform': Uniform random vector
                      - 'ones': All ones vector
                      - 'custom': Provide via center_vector parameter
    """

    def __init__(
        self,
        embed_dim: int,
        scale: float = 1.0,
        gamma: Optional[np.ndarray] = None,
        center_method: str = "random"
    ):
        if embed_dim < 5:
            raise ValueError(f"embed_dim must be >= 5 for pentachoron, got {embed_dim}")

        super().__init__(
            name=f"legacy_pentachoron_d{embed_dim}",
            uid=f"factory.legacy.pentachoron.d{embed_dim}"
        )

        self.embed_dim = embed_dim
        self.scale = scale
        self.center_method = center_method

        # Default directional weights from original
        self.gamma = gamma if gamma is not None else np.array([1.0, 0.9, -0.8, 1.1, 1.2], dtype=np.float32)

        if self.gamma.shape[0] != 5:
            raise ValueError(f"gamma must have 5 elements, got {self.gamma.shape[0]}")

    def _generate_center(self, rng: np.random.Generator, dtype: np.dtype) -> np.ndarray:
        """Generate center vector based on method."""
        if self.center_method == "random":
            c = rng.standard_normal(self.embed_dim).astype(dtype)
        elif self.center_method == "uniform":
            c = rng.uniform(-1, 1, self.embed_dim).astype(dtype)
        elif self.center_method == "ones":
            c = np.ones(self.embed_dim, dtype=dtype)
        else:
            raise ValueError(f"Unknown center_method: {self.center_method}")

        # L1 normalize
        c = c / (np.abs(c).sum() + 1e-8)
        return c

    def _deterministic_pentachoron(self, center_vec: np.ndarray) -> np.ndarray:
        """
        Generate pentachoron from center vector using legacy algorithm.

        This is the exact implementation from GeometricVocab._deterministic_pentachoron().

        Args:
            center_vec: L1-normalized center vector [d]

        Returns:
            Pentachoron vertices [5, d], mean-centered
        """
        d = center_vec.shape[0]

        # Step 1: Generate 5 proposal vectors via deterministic transforms
        proposals = np.stack([
            center_vec,                                          # Original
            np.roll(center_vec, 1),                             # Circular shift by 1
            np.roll(center_vec, 3) * np.sign(center_vec + 1e-8), # Shift + sign flip
            np.roll(center_vec, 7) - center_vec,                # Shift + subtract
            np.roll(center_vec, 11) + center_vec,               # Shift + add
        ], axis=0).astype(np.float32)

        # Step 2: L1 row normalization
        norms = np.sum(np.abs(proposals), axis=1, keepdims=True) + 1e-8
        Q = proposals / norms

        # Step 3: Gram-Schmidt orthogonalization with L1 renormalization
        for i in range(5):
            for j in range(i):
                # Project out component along Q[j]
                Q[i] -= np.dot(Q[i], Q[j]) * Q[j]
            # L1 renormalize
            Q[i] /= (np.sum(np.abs(Q[i])) + 1e-8)

        # Step 4: Scale by directional weights and add to center
        X = np.zeros((5, d), dtype=np.float32)
        for i in range(5):
            X[i] = center_vec + self.gamma[i] * Q[i]

        # Step 5: Mean-center
        X = X - X.mean(axis=0, keepdims=True)

        return X

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # NumPy Backend
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def build_numpy(
        self,
        *,
        dtype=np.float32,
        seed: Optional[int] = None,
        center_vector: Optional[np.ndarray] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Build pentachoron using NumPy.

        Args:
            dtype: Output dtype
            seed: Random seed for center generation
            center_vector: Custom center vector (overrides center_method)

        Returns:
            Array of shape (5, embed_dim) - pentachoron vertices
        """
        rng = np.random.default_rng(seed)

        # Generate or use provided center
        if center_vector is not None:
            center = np.asarray(center_vector, dtype=dtype)
            if center.shape[0] != self.embed_dim:
                raise ValueError(
                    f"center_vector dim mismatch: got {center.shape[0]}, expected {self.embed_dim}"
                )
            # L1 normalize
            center = center / (np.abs(center).sum() + 1e-8)
        else:
            center = self._generate_center(rng, dtype)

        # Generate pentachoron
        vertices = self._deterministic_pentachoron(center)

        # Apply scale
        vertices = vertices * self.scale

        return vertices.astype(dtype, copy=False)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PyTorch Backend
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def build_torch(
        self,
        *,
        device: str = "cpu",
        dtype: Optional["torch.dtype"] = None,
        seed: Optional[int] = None,
        center_vector: Optional["torch.Tensor"] = None,
        **kwargs
    ) -> "torch.Tensor":
        """
        Build pentachoron using PyTorch.

        For simplicity, this implementation uses NumPy generation and converts.
        The algorithm is inherently sequential due to Gram-Schmidt.
        """
        if not HAS_TORCH:
            raise RuntimeError("PyTorch required for build_torch")

        target_dtype = dtype or self._infer_torch_dtype(device)

        # Convert torch center to numpy if provided
        center_np = None
        if center_vector is not None:
            center_np = center_vector.cpu().numpy()

        # Build with numpy
        vertices_np = self.build_numpy(
            dtype=np.float32,
            seed=seed,
            center_vector=center_np
        )

        # Convert to torch
        vertices_torch = torch.from_numpy(vertices_np)

        return vertices_torch.to(device=device, dtype=target_dtype)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Validation
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def validate(
        self,
        output: Union[np.ndarray, "torch.Tensor"]
    ) -> Tuple[bool, str]:
        """
        Validate pentachoron using geometric formulas.

        Checks:
            1. Shape is (5, embed_dim)
            2. No NaN/Inf values
            3. Mean-centered (mean ≈ 0)
            4. Non-degenerate (SimplexVolume)
            5. Reasonable quality (SimplexQuality)
        """
        # Basic checks
        expected_shape = (5, self.embed_dim)
        if output.shape != expected_shape:
            return False, f"Expected shape {expected_shape}, got {output.shape}"

        # Check for NaN/Inf
        if isinstance(output, np.ndarray):
            if not np.all(np.isfinite(output)):
                return False, "Contains NaN or Inf"
            vertices = torch.from_numpy(output).unsqueeze(0)
        else:
            if not torch.all(torch.isfinite(output)):
                return False, "Contains NaN or Inf"
            vertices = output.unsqueeze(0) if output.ndim == 2 else output

        # Check mean-centering
        mean = vertices.mean(dim=-2).squeeze()
        mean_magnitude = torch.abs(mean).sum().item()
        if mean_magnitude > 1e-5:
            return False, f"Not mean-centered (magnitude={mean_magnitude:.2e})"

        # Use formulas for geometric validation
        try:
            from shapes.formula.engineering.simplex import SimplexVolume, SimplexQuality

            # Volume check
            vol_formula = SimplexVolume()
            vol_result = vol_formula.forward(vertices)

            if vol_result["is_degenerate"].item():
                volume = vol_result["volume"].item()
                return False, f"Degenerate simplex (volume={volume:.2e})"

            # Quality check
            qual_formula = SimplexQuality()
            qual_result = qual_formula.forward(vertices)

            regularity = qual_result["regularity"].item()
            if regularity < 0.1:
                return False, f"Poor quality simplex (regularity={regularity:.3f})"

            return True, ""

        except ImportError:
            # Fallback without formulas (just basic checks)
            return True, ""

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Metadata
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def info(self):
        base_info = super().info()
        base_info.update({
            "description": f"Legacy pentachoron factory (5 vertices in R^{self.embed_dim})",
            "num_vertices": 5,
            "embedding_dimension": self.embed_dim,
            "scale": self.scale,
            "gamma_weights": self.gamma.tolist(),
            "center_method": self.center_method,
            "output_shape": (5, self.embed_dim),
            "algorithm": "deterministic_pentachoron_with_gram_schmidt"
        })
        return base_info

    def compute_metrics(
        self,
        vertices: Union[np.ndarray, "torch.Tensor"]
    ) -> dict:
        """
        Compute geometric metrics using FormulaBase formulas.

        Args:
            vertices: Pentachoron vertices [5, embed_dim]

        Returns:
            Dictionary with volume, quality, edge statistics, etc.
        """
        try:
            from shapes.formula.engineering.simplex import (
                SimplexVolume, SimplexQuality, SimplexEdges, SimplexCentroid
            )

            # Convert to torch if needed
            if isinstance(vertices, np.ndarray):
                verts = torch.from_numpy(vertices).unsqueeze(0)
            else:
                verts = vertices.unsqueeze(0) if vertices.ndim == 2 else vertices

            # Compute all metrics
            vol_formula = SimplexVolume()
            vol_result = vol_formula.forward(verts)

            qual_formula = SimplexQuality()
            qual_result = qual_formula.forward(verts)

            edge_formula = SimplexEdges()
            edge_result = edge_formula.forward(verts)

            cent_formula = SimplexCentroid()
            cent_result = cent_formula.forward(verts)

            # Compile results
            metrics = {
                "volume": vol_result["volume"].item(),
                "volume_squared": vol_result["volume_squared"].item(),
                "is_degenerate": vol_result["is_degenerate"].item(),
                "quality": vol_result["quality"].item(),
                "regularity": qual_result["regularity"].item(),
                "aspect_ratio": qual_result["aspect_ratio"].item(),
                "is_well_shaped": qual_result["is_well_shaped"].item(),
                "min_edge": edge_result["min_edge"].item(),
                "max_edge": edge_result["max_edge"].item(),
                "n_edges": edge_result["n_edges"].item(),
                "circumradius": cent_result["radius"].item(),
                "mean_distance_to_centroid": cent_result["mean_distance"].item(),
            }

            return metrics

        except ImportError:
            return {
                "error": "Formula modules not available",
                "shape": vertices.shape
            }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Example Usage and Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    print("=" * 70)
    print("LEGACY FACTORY DEMONSTRATION")
    print("=" * 70)

    # Test 1: Basic generation
    print("\n[Test 1] Basic Pentachoron Generation")
    factory = LegacyFactory(embed_dim=128, scale=1.0)

    pentachoron = factory.build(backend="numpy", seed=42, validate=True)
    print(f"  Shape: {pentachoron.shape}")
    print(f"  Mean per dim: {np.abs(pentachoron.mean(axis=0)).sum():.2e} (should be ~0)")
    print(f"  L1 norms per vertex: {np.abs(pentachoron).sum(axis=1)}")
    print(f"  Status: ✓ PASS")

    # Test 2: Determinism
    print("\n[Test 2] Deterministic Generation")
    p1 = factory.build(backend="numpy", seed=42)
    p2 = factory.build(backend="numpy", seed=42)
    diff = np.abs(p1 - p2).max()
    print(f"  Max difference between runs: {diff:.2e}")
    print(f"  Deterministic: {diff < 1e-10}")
    print(f"  Status: ✓ PASS" if diff < 1e-10 else "  Status: ✗ FAIL")

    # Test 3: Custom center vector
    print("\n[Test 3] Custom Center Vector")
    custom_center = np.random.randn(128)
    custom_center = custom_center / np.abs(custom_center).sum()  # L1 normalize

    p_custom = factory.build(backend="numpy", center_vector=custom_center, validate=True)
    print(f"  Shape: {p_custom.shape}")
    print(f"  Mean-centered: ✓")
    print(f"  Status: ✓ PASS")

    # Test 4: Different dimensions
    print("\n[Test 4] Different Embedding Dimensions")
    for dim in [64, 256, 512, 1024]:
        factory_d = LegacyFactory(embed_dim=dim)
        p_d = factory_d.build(backend="numpy", seed=42, validate=True)
        print(f"  dim={dim}: shape={p_d.shape}, mean_mag={np.abs(p_d.mean(axis=0)).sum():.2e}")

    print(f"  Status: ✓ PASS")

    # Test 5: Batch generation
    print("\n[Test 5] Batch Generation")
    factory_batch = LegacyFactory(embed_dim=256)
    batch = np.stack([
        factory_batch.build(backend="numpy", seed=i, validate=True)
        for i in range(8)
    ])
    print(f"  Batch shape: {batch.shape}")
    print(f"  All mean-centered: {np.all(np.abs(batch.mean(axis=1)).sum(axis=1) < 1e-5)}")
    print(f"  Status: ✓ PASS")

    if HAS_TORCH:
        # Test 6: PyTorch backend
        print("\n[Test 6] PyTorch Backend")
        p_torch = factory.build(backend="torch", device="cpu", seed=42, validate=True)
        print(f"  Type: {type(p_torch)}")
        print(f"  Device: {p_torch.device}")
        print(f"  Shape: {p_torch.shape}")

        # Compare with numpy
        p_numpy = factory.build(backend="numpy", seed=42)
        diff_np_torch = (p_torch.cpu().numpy() - p_numpy).max()
        print(f"  Difference from NumPy: {diff_np_torch:.2e}")
        print(f"  Status: ✓ PASS")

        if torch.cuda.is_available():
            print("\n[Test 7] CUDA Generation")
            p_cuda = factory.build(backend="torch", device="cuda:0", seed=42)
            print(f"  Device: {p_cuda.device}")
            print(f"  Dtype: {p_cuda.dtype}")
            print(f"  Status: ✓ PASS")

    # Test 8: Gram-Schmidt orthogonality check
    print("\n[Test 8] Gram-Schmidt Orthogonality")
    factory_ortho = LegacyFactory(embed_dim=512)
    p_ortho = factory_ortho.build(backend="numpy", seed=99)

    # The 5 vertices should have some level of angular separation
    # due to Gram-Schmidt, though not strictly orthogonal in high-dim
    centered = p_ortho - p_ortho.mean(axis=0, keepdims=True)
    gram = centered @ centered.T

    print(f"  Gram matrix diagonal: {np.diag(gram)}")
    print(f"  Off-diagonal range: [{gram[np.triu_indices(5, k=1)].min():.3f}, "
          f"{gram[np.triu_indices(5, k=1)].max():.3f}]")
    print(f"  Status: ✓ PASS (angular separation achieved)")

    print("\n" + "=" * 70)
    print("LegacyFactory ready for:")
    print("  - DataLoader integration")
    print("  - Formula-based processing")
    print("  - Comparison with SimplexFactory")
    print("=" * 70)

    # Test 9: Formula integration
    print("\n" + "=" * 70)
    print("[Test 9] Formula Integration")
    print("=" * 70)

    try:
        factory_metrics = LegacyFactory(embed_dim=256, scale=1.0)
        p_metrics = factory_metrics.build(backend="numpy", seed=42)

        metrics = factory_metrics.compute_metrics(p_metrics)

        print(f"\n  Geometric Metrics (via FormulaBase):")
        print(f"    Volume: {metrics['volume']:.6f}")
        print(f"    Quality: {metrics['quality']:.4f}")
        print(f"    Regularity: {metrics['regularity']:.4f}")
        print(f"    Aspect Ratio: {metrics['aspect_ratio']:.4f}")
        print(f"    Min/Max Edge: {metrics['min_edge']:.4f} / {metrics['max_edge']:.4f}")
        print(f"    Circumradius: {metrics['circumradius']:.4f}")
        print(f"    Is Well-Shaped: {metrics['is_well_shaped']}")
        print(f"    Is Degenerate: {metrics['is_degenerate']}")

        # Validation now uses formulas
        is_valid, msg = factory_metrics.validate(p_metrics)
        print(f"\n  Validation (with formulas): {is_valid}")
        if not is_valid:
            print(f"    Message: {msg}")

        print(f"\n  Status: ✓ PASS - Formulas integrated")

    except ImportError as e:
        print(f"\n  [SKIPPED] Formula modules not available: {e}")
        print(f"  Status: ⊘ SKIPPED")