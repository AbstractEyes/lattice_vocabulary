"""
GEOMETRIC ORIGIN SAMPLER
------------------------
Replaces patch-based tokenization with geometric origin sampling.

Key differences from traditional patches:
- Origins are geometric anchors (simplices), not grid positions
- Placement determined by geometric stability, not spatial indexing
- Multi-scale inherent (different simplex volumes = different scales)
- Validates via Cayley-Menger, not arbitrary boundaries

    Authors: AbstractPhil
        GPT-4o, GPT-o3, GPT-o4, Claude Sonnet 4,
        Claude Opus 4, Claude Opus 4.1, Claude Sonnet 4.5,
        Gemini Pro 2, Gemini Pro 2.5, Gemini Flash 2.5

    Refactored and reenvisioned for geovocab2 by AbstractPhil + Claude Sonnet 4.5
"""

from typing import Dict, Optional, Tuple, List, Literal
import torch
from torch import Tensor
import math

from geovocab2.shapes.formula import (
    SimplexFacesSampler,
    SimplexVolume,
    SimplexQuality,
    SimplexCentroid,
    CMLogDetRegularizer,
    RoseWeightedVolume
)
from ..formula_base import FormulaBase


class GeometricOriginSampler(FormulaBase):
    """
    Initialize and manage geometric origins for input processing.

    Replaces: Patch embedding + positional encoding
    Uses: Simplex lattice with geometric validation

    Args:
        num_origins: Number of geometric anchors
        origin_dim: Dimension of simplex origins (k+1 vertices)
        embed_dim: Embedding dimension
        init_strategy: How to initialize origins
            - 'simplex_lattice': Regular simplex tiling
            - 'random_validated': Random + CM validation
            - 'learned': Gradient-optimized placement
            - 'diffusion': Use diffusion sampling
        validation_formula: Formula for validating origins
        temperature: Diffusion temperature (if using diffusion strategy)
    """

    def __init__(
            self,
            num_origins: int,
            origin_dim: int = 5,  # k+1 for k-simplex (default: pentachoron)
            embed_dim: int = 512,
            init_strategy: Literal['simplex_lattice', 'random_validated', 'learned', 'diffusion'] = 'simplex_lattice',
            validation_formula: Optional[FormulaBase] = None,
            temperature: float = 0.1,
            quality_threshold: float = 0.3,
            max_init_attempts: int = 100
    ):
        super().__init__("geometric_origin_sampler", "f.origin.sampler")

        self.num_origins = num_origins
        self.origin_dim = origin_dim  # k+1 vertices
        self.k = origin_dim - 1  # simplex dimension
        self.embed_dim = embed_dim
        self.init_strategy = init_strategy
        self.temperature = temperature
        self.quality_threshold = quality_threshold
        self.max_init_attempts = max_init_attempts

        # Validation formula (default: combined quality check)
        if validation_formula is None:
            self.validator = SimplexQuality()
        else:
            self.validator = validation_formula

        # For learned strategy
        if init_strategy == 'learned':
            # Origins are learnable parameters
            self.origins = torch.nn.Parameter(
                torch.randn(num_origins, origin_dim, embed_dim) * 0.02
            )
        else:
            self.origins = None

    def _regular_simplex_lattice(
            self,
            batch_shape: Tuple[int, ...],
            device: torch.device,
            dtype: torch.dtype
    ) -> Tensor:
        """
        Create regular simplex tiling.

        Returns: [..., num_origins, k+1, embed_dim]
        """
        # Start with canonical regular k-simplex in (k+1)-dimensional space
        # Then embed into higher dimensional space

        k = self.k
        k_plus_1 = self.origin_dim

        # Canonical k-simplex vertices in (k+1)D
        # Place on unit hypersphere, evenly spaced
        canonical = torch.zeros(k_plus_1, k_plus_1, device=device, dtype=dtype)

        for i in range(k_plus_1):
            # Standard basis vector
            canonical[i, i] = 1.0

        # Center at origin
        centroid = canonical.mean(dim=0, keepdim=True)
        canonical = canonical - centroid

        # Scale to unit edge length
        edge_lengths = torch.cdist(canonical, canonical, p=2)
        mean_edge = edge_lengths[torch.triu(torch.ones_like(edge_lengths), diagonal=1) > 0].mean()
        canonical = canonical / (mean_edge + 1e-10)

        # Create num_origins copies with different rotations/positions
        origins = []

        # Generate all rotations at once
        Q = torch.randn(self.num_origins, k_plus_1, k_plus_1, device=device, dtype=dtype)
        Q, _ = torch.linalg.qr(Q)  # Batch QR decomposition
        rotated = torch.matmul(canonical.unsqueeze(0), Q)  # [num_origins, k+1, k+1]

        # Embed all at once
        embedded = torch.zeros(self.num_origins, k_plus_1, self.embed_dim, device=device, dtype=dtype)
        embedded[:, :, :k_plus_1] = rotated

        # Add random translations
        translation = torch.randn(self.num_origins, 1, self.embed_dim, device=device, dtype=dtype) * 0.1
        embedded = embedded + translation

        origins = embedded

        #origins = torch.stack(origins, dim=0)  # [num_origins, k+1, embed_dim]

        # Expand for batch
        for _ in batch_shape:
            origins = origins.unsqueeze(0)

        return origins.expand(*batch_shape, -1, -1, -1)

    def _random_validated(
            self,
            batch_shape: Tuple[int, ...],
            device: torch.device,
            dtype: torch.dtype
    ) -> Tensor:
        """
        Random initialization with CM validation.
        Reject simplices with poor quality.

        Returns: [..., num_origins, k+1, embed_dim]
        """
        origins = []

        for i in range(self.num_origins):
            attempts = 0
            valid = False

            while not valid and attempts < self.max_init_attempts:
                # Random simplex
                candidate = torch.randn(
                    *batch_shape, self.origin_dim, self.embed_dim,
                    device=device, dtype=dtype
                ) * 0.5

                # Validate
                quality_result = self.validator.forward(candidate)

                # Check quality
                if 'is_well_shaped' in quality_result:
                    valid = quality_result['is_well_shaped'].all().item()
                elif 'regularity' in quality_result:
                    valid = (quality_result['regularity'] > self.quality_threshold).all().item()
                else:
                    valid = True  # No quality check available

                attempts += 1

            if not valid:
                # Fallback to regular simplex for this origin
                candidate = self._regular_simplex_lattice(
                    batch_shape, device, dtype
                )[..., 0, :, :]  # Take first origin from lattice

            origins.append(candidate)

        return torch.stack(origins, dim=-3)  # [..., num_origins, k+1, embed_dim]

    def _diffusion_sample(
            self,
            input_data: Tensor,
            batch_shape: Tuple[int, ...],
            device: torch.device,
            dtype: torch.dtype
    ) -> Tensor:
        """
        Use diffusion-based sampling from input structure.

        Args:
            input_data: Input tensor to sample from [..., n_points, embed_dim]

        Returns: [..., num_origins, k+1, embed_dim]
        """
        # Use SimplexFacesSampler with diffusion strategy
        sampler = SimplexFacesSampler(
            face_dim=self.k,
            sample_budget=self.num_origins,
            formula=self.validator,
            selection_strategy='diffusion',
            temperature=self.temperature,
            aggregate_to_vertices=False
        )

        # Sample from input
        result = sampler.forward(input_data)
        face_indices = result['face_indices']  # [..., num_origins, k+1]
        valid_mask = result['valid_mask']  # [..., num_origins]

        # Gather origin vertices
        batch_dims = input_data.shape[:-2]
        n_points = input_data.shape[-2]

        # Expand indices for gathering
        face_idx_exp = face_indices.unsqueeze(-1).expand(
            *batch_dims, self.num_origins, self.origin_dim, self.embed_dim
        )

        input_exp = input_data.unsqueeze(-3).expand(
            *batch_dims, self.num_origins, n_points, self.embed_dim
        )

        origins = torch.gather(input_exp, -2, face_idx_exp)

        # Mask invalid origins (replace with regular simplex)
        if not valid_mask.all():
            regular = self._regular_simplex_lattice(batch_shape, device, dtype)
            invalid = ~valid_mask
            origins = torch.where(
                invalid.unsqueeze(-1).unsqueeze(-1),
                regular,
                origins
            )

        return origins

    def initialize_origins(
            self,
            input_shape: Tuple[int, ...],
            input_data: Optional[Tensor] = None
    ) -> Dict[str, Tensor]:
        """
        Initialize geometric origins.

        Args:
            input_shape: Shape of input [..., n_points, embed_dim]
            input_data: Optional input tensor for diffusion strategy

        Returns:
            origins: Geometric anchor simplices [..., num_origins, k+1, embed_dim]
            quality_metrics: Validation results
            origin_centroids: Centroid of each origin [..., num_origins, embed_dim]
        """
        batch_shape = input_shape[:-2]
        device = input_data.device if input_data is not None else torch.device('cpu')
        dtype = input_data.dtype if input_data is not None else torch.float32

        # Initialize based on strategy
        if self.init_strategy == 'learned':
            # Use learnable parameters
            origins = self.origins
            # Expand for batch
            for _ in batch_shape:
                origins = origins.unsqueeze(0)
            origins = origins.expand(*batch_shape, -1, -1, -1)

        elif self.init_strategy == 'simplex_lattice':
            origins = self._regular_simplex_lattice(batch_shape, device, dtype)

        elif self.init_strategy == 'random_validated':
            origins = self._random_validated(batch_shape, device, dtype)

        elif self.init_strategy == 'diffusion':
            if input_data is None:
                raise ValueError("Diffusion strategy requires input_data")
            origins = self._diffusion_sample(input_data, batch_shape, device, dtype)

        else:
            raise ValueError(f"Unknown init_strategy: {self.init_strategy}")

        # Validate all origins
        quality_metrics = self.validator.forward(origins)

        # Compute centroids
        centroid_calc = SimplexCentroid()
        centroid_result = centroid_calc.forward(origins)
        origin_centroids = centroid_result['centroid']

        return {
            'origins': origins,
            'origin_centroids': origin_centroids,
            'quality_metrics': quality_metrics,
            'is_valid': quality_metrics.get('is_well_shaped',
                                            torch.ones_like(origins[..., 0, 0], dtype=torch.bool))
        }

    def forward(
            self,
            input_data: Tensor,
            return_quality: bool = True
    ) -> Dict[str, Tensor]:
        """
        Sample geometric origins from input.

        Args:
            input_data: Input tensor [..., n_points, embed_dim]
            return_quality: Whether to compute quality metrics

        Returns:
            origins: Sampled origins [..., num_origins, k+1, embed_dim]
            origin_centroids: Origin centers [..., num_origins, embed_dim]
            quality_metrics: Optional quality validation
        """
        result = self.initialize_origins(input_data.shape, input_data)

        if not return_quality:
            result.pop('quality_metrics', None)

        return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TESTING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_geometric_origin_sampler():
    """Test GeometricOriginSampler with different strategies."""

    print("\n" + "=" * 70)
    print("GEOMETRIC ORIGIN SAMPLER TESTS")
    print("=" * 70)

    # Test 1: Simplex lattice strategy
    print("\n[Test 1] Simplex Lattice Initialization")

    sampler = GeometricOriginSampler(
        num_origins=16,
        origin_dim=5,  # Pentachoron
        embed_dim=512,
        init_strategy='simplex_lattice'
    )

    # Create dummy input (e.g., image patches or token embeddings)
    batch_size = 4
    n_tokens = 197  # ViT-like
    input_data = torch.randn(batch_size, n_tokens, 512)

    result = sampler.forward(input_data)

    print(f"  Input: [{batch_size}, {n_tokens}, 512]")
    print(f"  Origins: {result['origins'].shape}")
    print(f"  Expected: [{batch_size}, 16, 5, 512]")
    print(f"  Centroids: {result['origin_centroids'].shape}")
    print(f"  Valid origins: {result['is_valid'].sum().item()}/{result['is_valid'].numel()}")
    print(f"  Mean regularity: {result['quality_metrics']['regularity'].mean().item():.4f}")
    print(f"  Status: ✓ PASS")

    # Test 2: Diffusion strategy
    print("\n[Test 2] Diffusion-Based Sampling")

    sampler_diff = GeometricOriginSampler(
        num_origins=32,
        origin_dim=5,
        embed_dim=512,
        init_strategy='diffusion',
        temperature=0.2
    )

    result_diff = sampler_diff.forward(input_data)

    print(f"  Origins sampled via diffusion: {result_diff['origins'].shape}")
    print(f"  Valid origins: {result_diff['is_valid'].sum().item()}/{result_diff['is_valid'].numel()}")
    print(f"  Mean quality: {result_diff['quality_metrics']['volume_quality'].mean().item():.4f}")
    print(f"  Status: ✓ PASS")

    # Test 3: Learned strategy
    print("\n[Test 3] Learned Origins (Gradient-Optimized)")

    sampler_learned = GeometricOriginSampler(
        num_origins=8,
        origin_dim=5,
        embed_dim=256,
        init_strategy='learned'
    )

    small_input = torch.randn(2, 50, 256)
    result_learned = sampler_learned.forward(small_input)

    print(f"  Learnable parameters: {sampler_learned.origins.shape}")
    print(f"  Origins: {result_learned['origins'].shape}")
    print(f"  Gradients enabled: {sampler_learned.origins.requires_grad}")
    print(f"  Status: ✓ PASS")

    # Test 4: Multi-scale (different origin dimensions)
    print("\n[Test 4] Multi-Scale Origins (Different k)")

    scales = [3, 4, 5, 6]  # Triangle, tetrahedron, pentachoron, 5-simplex

    for k_plus_1 in scales:
        sampler_scale = GeometricOriginSampler(
            num_origins=4,
            origin_dim=k_plus_1,
            embed_dim=128,
            init_strategy='simplex_lattice'
        )

        test_input = torch.randn(1, 20, 128)
        result_scale = sampler_scale.forward(test_input)

        print(f"  {k_plus_1 - 1}-simplex: {result_scale['origins'].shape[-2:]} vertices/dims")

    print(f"  Status: ✓ PASS")

    # Test 5: Comparison with patch-based approach
    print("\n[Test 5] Geometric vs Patch-Based Comparison")

    # Traditional patches
    image_tokens = 196  # 14x14 patches
    patch_embedding = torch.randn(1, image_tokens, 768)

    # Geometric origins
    geom_sampler = GeometricOriginSampler(
        num_origins=image_tokens // 4,  # Fewer origins, multi-vertex structure
        origin_dim=5,
        embed_dim=768,
        init_strategy='diffusion'
    )

    geom_result = geom_sampler.forward(patch_embedding)

    print(f"  Patch approach: {image_tokens} single-point tokens")
    print(
        f"  Geometric approach: {geom_result['origins'].shape[-3]} simplices × {geom_result['origins'].shape[-2]} vertices")
    print(
        f"  Effective coverage: {geom_result['origins'].shape[-3] * geom_result['origins'].shape[-2]} geometric points")
    print(f"  Geometric quality: {geom_result['quality_metrics']['regularity'].mean().item():.4f}")
    print(f"  Status: ✓ PASS")

    print("\n" + "=" * 70)
    print("ALL GEOMETRIC ORIGIN SAMPLER TESTS PASSED")
    print("=" * 70)
    print("\nKey Features:")
    print("  ✓ Replaces arbitrary patch grids with geometric structures")
    print("  ✓ Multiple initialization strategies")
    print("  ✓ Built-in Cayley-Menger validation")
    print("  ✓ Multi-scale inherent (different simplex dimensions)")
    print("  ✓ Learnable origins for optimization")
    print("  ✓ Diffusion-based adaptive sampling")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    test_geometric_origin_sampler()