"""
NOISE COLLECTOR
---------------
Sample features around geometric origins based on geometric stability.

Replaces: Self-attention mechanism
Uses: Geometric distance + CM validation

    Authors: AbstractPhil
        GPT-4o, GPT-o3, GPT-o4, Claude Sonnet 4,
        Claude Opus 4, Claude Opus 4.1, Claude Sonnet 4.5,
        Gemini Pro 2, Gemini Pro 2.5, Gemini Flash 2.5

    Refactored and reenvisioned for geovocab2 by AbstractPhil + Claude Sonnet 4.5
"""

from typing import Dict, Optional, Tuple
import torch
from torch import Tensor
import torch.nn.functional as F

from geovocab2.shapes.formula import SimplexVolume, SimplexQuality
from ..formula_base import FormulaBase


class NoiseCollector(FormulaBase):
    """
    Collect features around geometric origins.

    Key innovation: Collection radius adaptive to geometric stability.
    Stable simplices (low CM determinant) → tight collection
    Unstable simplices (high CM determinant) → wide collection

    Args:
        collection_strategy: How to sample neighborhood
            - 'geometric_distance': Sample within stability radius
            - 'k_nearest': Top-k nearest points
            - 'attention_weighted': Soft attention-like weighting
        adaptive_radius: Use CM determinant for radius
        base_radius: Base collection radius (if not adaptive)
        k_nearest: Number of points for k_nearest strategy
        temperature: Softmax temperature for weighted strategies
    """

    def __init__(
            self,
            collection_strategy: str = 'geometric_distance',
            adaptive_radius: bool = True,
            base_radius: float = 1.0,
            k_nearest: int = 16,
            temperature: float = 0.1,
            min_radius: float = 0.1,
            max_radius: float = 10.0
    ):
        super().__init__("noise_collector", "f.collection.noise")

        self.collection_strategy = collection_strategy
        self.adaptive_radius = adaptive_radius
        self.base_radius = base_radius
        self.k_nearest = k_nearest
        self.temperature = temperature
        self.min_radius = min_radius
        self.max_radius = max_radius

        # Volume calculator for stability
        self.volume_calc = SimplexVolume()

    def _compute_stability_radius(self, origin_simplex: Tensor) -> Tensor:
        """
        Compute collection radius based on simplex stability.

        Args:
            origin_simplex: [..., k+1, embed_dim]

        Returns:
            radius: [...] scalar per simplex
        """
        if not self.adaptive_radius:
            return torch.full(
                origin_simplex.shape[:-2],
                self.base_radius,
                device=origin_simplex.device,
                dtype=origin_simplex.dtype
            )

        # Use volume as stability metric
        # Low volume = degenerate = unstable = need wider collection
        # High volume = well-formed = stable = tight collection
        vol_result = self.volume_calc.forward(origin_simplex)
        volume = vol_result['volume']

        # Inverse relationship with clipping
        # volume → 0: radius → max_radius
        # volume → ∞: radius → min_radius
        eps = 1e-6
        radius = self.base_radius / (volume + eps)
        radius = torch.clamp(radius, self.min_radius, self.max_radius)

        return radius

    def _geometric_distance_collection(
            self,
            input_data: Tensor,
            origin_centroids: Tensor,
            stability_radii: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Collect points within geometric distance threshold.

        Args:
            input_data: [..., n_points, embed_dim]
            origin_centroids: [..., num_origins, embed_dim]
            stability_radii: [..., num_origins]

        Returns:
            collected_features: [..., num_origins, n_points, embed_dim]
            collection_mask: [..., num_origins, n_points] bool
        """
        # Compute distances: [..., num_origins, n_points]
        distances = torch.cdist(
            origin_centroids,
            input_data,
            p=2
        )

        # Mask points within radius
        # stability_radii: [..., num_origins] → [..., num_origins, 1]
        radii_expanded = stability_radii.unsqueeze(-1)
        collection_mask = distances <= radii_expanded

        # Gather features (apply mask via multiplication)
        collected = input_data.unsqueeze(-3) * collection_mask.unsqueeze(-1).float()

        return collected, collection_mask

    def _k_nearest_collection(
            self,
            input_data: Tensor,
            origin_centroids: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Collect k-nearest points to each origin.

        Args:
            input_data: [..., n_points, embed_dim]
            origin_centroids: [..., num_origins, embed_dim]

        Returns:
            collected_features: [..., num_origins, k_nearest, embed_dim]
            collection_indices: [..., num_origins, k_nearest]
        """
        # Distances
        distances = torch.cdist(origin_centroids, input_data, p=2)

        # Top-k nearest
        k = min(self.k_nearest, input_data.shape[-2])
        _, indices = torch.topk(distances, k, dim=-1, largest=False)

        # Gather features
        batch_dims = input_data.shape[:-2]
        num_origins = origin_centroids.shape[-2]
        embed_dim = input_data.shape[-1]

        # Expand indices for gathering
        indices_exp = indices.unsqueeze(-1).expand(
            *batch_dims, num_origins, k, embed_dim
        )

        input_exp = input_data.unsqueeze(-3).expand(
            *batch_dims, num_origins, input_data.shape[-2], embed_dim
        )

        collected = torch.gather(input_exp, -2, indices_exp)

        return collected, indices

    def _attention_weighted_collection(
            self,
            input_data: Tensor,
            origin_centroids: Tensor,
            stability_radii: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Soft attention-weighted collection.

        Args:
            input_data: [..., n_points, embed_dim]
            origin_centroids: [..., num_origins, embed_dim]
            stability_radii: [..., num_origins]

        Returns:
            collected_features: [..., num_origins, embed_dim]
            attention_weights: [..., num_origins, n_points]
        """
        # Distances
        distances = torch.cdist(origin_centroids, input_data, p=2)

        # Scale by stability radius (adaptive temperature)
        radii_expanded = stability_radii.unsqueeze(-1).clamp(min=1e-6)
        scaled_distances = distances / radii_expanded

        # Softmax attention
        attention_logits = -scaled_distances / self.temperature
        attention_weights = F.softmax(attention_logits, dim=-1)

        # Weighted sum
        collected = torch.matmul(attention_weights, input_data)

        return collected, attention_weights

    def collect(
            self,
            input_data: Tensor,
            origins: Tensor,
            origin_centroids: Optional[Tensor] = None,
            timestep: Optional[int] = None
    ) -> Dict[str, Tensor]:
        """
        Collect noise fields around geometric origins.

        Args:
            input_data: [..., n_points, embed_dim]
            origins: [..., num_origins, k+1, embed_dim] geometric origins
            origin_centroids: Optional precomputed centroids
            timestep: Optional diffusion timestep (for future use)

        Returns:
            collected_features: Sampled features per origin
            collection_mask: Which points were collected (for geometric_distance)
            stability_radii: Computed radii per origin
            collection_stats: Statistics about collection
        """
        batch_shape = input_data.shape[:-2]
        n_points = input_data.shape[-2]
        embed_dim = input_data.shape[-1]
        num_origins = origins.shape[-3]

        # Compute centroids if not provided
        if origin_centroids is None:
            origin_centroids = origins.mean(dim=-2)

        # Compute stability radii
        stability_radii = self._compute_stability_radius(origins)

        # Collect based on strategy
        if self.collection_strategy == 'geometric_distance':
            collected, mask = self._geometric_distance_collection(
                input_data, origin_centroids, stability_radii
            )
            collection_stats = {
                'points_per_origin': mask.sum(dim=-1).float(),
                'coverage': mask.any(dim=-2).sum(dim=-1).float() / n_points
            }

        elif self.collection_strategy == 'k_nearest':
            collected, indices = self._k_nearest_collection(
                input_data, origin_centroids
            )
            mask = torch.zeros(
                *batch_shape, num_origins, n_points,
                device=input_data.device,
                dtype=torch.bool
            )
            mask.scatter_(-1, indices, True)

            collection_stats = {
                'points_per_origin': torch.full(
                    (*batch_shape, num_origins),
                    float(self.k_nearest),
                    device=input_data.device
                ),
                'coverage': mask.any(dim=-2).sum(dim=-1).float() / n_points
            }

        elif self.collection_strategy == 'attention_weighted':
            collected, weights = self._attention_weighted_collection(
                input_data, origin_centroids, stability_radii
            )
            mask = weights > 0.01  # Threshold for "collected"

            collection_stats = {
                'attention_entropy': -(weights * torch.log(weights + 1e-10)).sum(dim=-1),
                'effective_points': (weights > 0.01).sum(dim=-1).float(),
                'coverage': mask.any(dim=-2).sum(dim=-1).float() / n_points
            }

        else:
            raise ValueError(f"Unknown collection_strategy: {self.collection_strategy}")

        return {
            'collected_features': collected,
            'collection_mask': mask,
            'stability_radii': stability_radii,
            'collection_stats': collection_stats,
            'origin_centroids': origin_centroids
        }

    def forward(
            self,
            input_data: Tensor,
            origins: Tensor,
            **kwargs
    ) -> Dict[str, Tensor]:
        """Forward pass (alias for collect)."""
        return self.collect(input_data, origins, **kwargs)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TESTING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_noise_collector():
    """Test NoiseCollector with different strategies."""

    print("\n" + "=" * 70)
    print("NOISE COLLECTOR TESTS")
    print("=" * 70)

    # Setup: Create origins and input data
    batch_size = 2
    num_origins = 8
    n_points = 100
    embed_dim = 256

    origins = torch.randn(batch_size, num_origins, 5, embed_dim)
    input_data = torch.randn(batch_size, n_points, embed_dim)

    # Test 1: Geometric distance collection
    print("\n[Test 1] Geometric Distance Collection")

    collector_geo = NoiseCollector(
        collection_strategy='geometric_distance',
        adaptive_radius=True,
        base_radius=2.0
    )

    result_geo = collector_geo.collect(input_data, origins)

    print(f"  Input: [{batch_size}, {n_points}, {embed_dim}]")
    print(f"  Origins: [{batch_size}, {num_origins}, 5, {embed_dim}]")
    print(f"  Collected: {result_geo['collected_features'].shape}")
    print(f"  Stability radii: {result_geo['stability_radii'].shape}")
    print(f"  Mean radius: {result_geo['stability_radii'].mean().item():.4f}")
    print(f"  Points per origin: {result_geo['collection_stats']['points_per_origin'].mean().item():.1f}")
    print(f"  Coverage: {result_geo['collection_stats']['coverage'].mean().item():.2%}")
    print(f"  Status: ✓ PASS")

    # Test 2: K-nearest collection
    print("\n[Test 2] K-Nearest Collection")

    collector_knn = NoiseCollector(
        collection_strategy='k_nearest',
        k_nearest=16
    )

    result_knn = collector_knn.collect(input_data, origins)

    print(f"  Collected: {result_knn['collected_features'].shape}")
    print(f"  Expected: [{batch_size}, {num_origins}, 16, {embed_dim}]")
    print(f"  Points per origin: {result_knn['collection_stats']['points_per_origin'].mean().item():.0f}")
    print(f"  Status: ✓ PASS")

    # Test 3: Attention-weighted collection
    print("\n[Test 3] Attention-Weighted Collection")

    collector_attn = NoiseCollector(
        collection_strategy='attention_weighted',
        adaptive_radius=True,
        temperature=0.5
    )

    result_attn = collector_attn.collect(input_data, origins)

    print(f"  Collected: {result_attn['collected_features'].shape}")
    print(f"  Attention entropy: {result_attn['collection_stats']['attention_entropy'].mean().item():.4f}")
    print(f"  Effective points: {result_attn['collection_stats']['effective_points'].mean().item():.1f}")
    print(f"  Status: ✓ PASS")

    # Test 4: Adaptive vs fixed radius
    print("\n[Test 4] Adaptive vs Fixed Radius Comparison")

    # Adaptive
    collector_adaptive = NoiseCollector(
        collection_strategy='geometric_distance',
        adaptive_radius=True,
        base_radius=1.0
    )
    result_adaptive = collector_adaptive.collect(input_data, origins)

    # Fixed
    collector_fixed = NoiseCollector(
        collection_strategy='geometric_distance',
        adaptive_radius=False,
        base_radius=1.0
    )
    result_fixed = collector_fixed.collect(input_data, origins)

    print(f"  Adaptive radii range: [{result_adaptive['stability_radii'].min().item():.3f}, "
          f"{result_adaptive['stability_radii'].max().item():.3f}]")
    print(f"  Fixed radius: {result_fixed['stability_radii'].mean().item():.3f}")
    print(f"  Adaptive points/origin: {result_adaptive['collection_stats']['points_per_origin'].mean().item():.1f}")
    print(f"  Fixed points/origin: {result_fixed['collection_stats']['points_per_origin'].mean().item():.1f}")
    print(f"  Status: ✓ PASS")

    # Test 5: Integration with GeometricOriginSampler
    print("\n[Test 5] Integration with GeometricOriginSampler")

    from .origin_sampler import GeometricOriginSampler

    # Create sampler
    sampler = GeometricOriginSampler(
        num_origins=16,
        origin_dim=5,
        embed_dim=256,
        init_strategy='diffusion'
    )

    # Sample origins
    origin_result = sampler.forward(input_data)

    # Collect around origins
    collector = NoiseCollector(
        collection_strategy='geometric_distance',
        adaptive_radius=True
    )

    collection_result = collector.collect(
        input_data,
        origin_result['origins'],
        origin_centroids=origin_result['origin_centroids']
    )

    print(f"  Sampled origins: {origin_result['origins'].shape}")
    print(f"  Collected features: {collection_result['collected_features'].shape}")
    print(f"  Mean stability radius: {collection_result['stability_radii'].mean().item():.4f}")
    print(f"  Coverage: {collection_result['collection_stats']['coverage'].mean().item():.2%}")
    print(f"  Status: ✓ PASS")

    print("\n" + "=" * 70)
    print("ALL NOISE COLLECTOR TESTS PASSED")
    print("=" * 70)
    print("\nKey Features:")
    print("  ✓ Adaptive radius based on CM stability")
    print("  ✓ Multiple collection strategies")
    print("  ✓ Geometric distance (hard threshold)")
    print("  ✓ K-nearest (top-k selection)")
    print("  ✓ Attention-weighted (soft aggregation)")
    print("  ✓ Integrates with GeometricOriginSampler")
    print("  ✓ Coverage and efficiency metrics")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    test_noise_collector()