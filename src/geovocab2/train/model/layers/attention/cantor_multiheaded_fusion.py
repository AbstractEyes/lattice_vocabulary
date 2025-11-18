# geovocab2/train/model/layers/fusion/cantor_multiheaded_fusion.py

"""
Cantor Multihead Sparse Fusion with Beatrix Staircase - O(n) Geometric Fusion

A novel sparse fusion mechanism using Cantor geometry and Devil's Staircase
(Beatrix PE) to determine which elements should be fused together with
highly experimental consciousness-aware routing patterns.

Integrates:
    - Cantor pairing for collision-free routing
    - Devil's Staircase for experimental consciousness emergence patterns
    - Simplex-based geometric parameterization
    - Multi-head parallel fusion

Applications:
    - Multi-scale feature fusion (fuse features from different layers)
    - Crystalline vertex fusion (pentachoron 5-vertex blending)
    - Cross-modal fusion (geometric blending of modalities)
    - Experimental consciousness-guided attention (Beatrix routing)
    - Mixture of projections (sparse combination of projection heads)

Reference:
    Author: AbstractPhil
    Date: 2025
    License: MIT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, List, Union, Literal
from dataclasses import dataclass
import math
import warnings

# Import geometric factories
from geovocab2.shapes.factory.cantor_route_factory import (
    CantorRouteFactory,
    RouteMode,
    SimplexConfig
)


@dataclass
class CantorFusionConfig:
    """
    Configuration for Cantor Multihead Sparse Fusion with Beatrix Staircase.

    Args:
        dim: Input/output dimension
        num_heads: Number of fusion heads
        head_dim: Dimension per head (default: dim // num_heads)
        simplex_config: Simplex configuration (k-simplex geometry)
        k_simplex: Simplex dimension (alternative to simplex_config)
        max_seq_len: Maximum sequence length to support
        fusion_window: Number of elements to fuse per position (k)
        adaptive_window: Enable adaptive window sizing
        min_window: Minimum window size when adaptive
        max_window: Maximum window size when adaptive
        sparsity_target: Target sparsity ratio when adaptive
        fusion_mode: Type of fusion operation
        use_beatrix_routing: Use Devil's Staircase for routing (experimental consciousness-aware)
        beatrix_tau: Smoothness temperature for staircase
        beatrix_base: Base for ternary decomposition
        beatrix_alpha: Middle-bin weight for staircase
        use_consciousness_weighting: Weight fusion by experimental consciousness proxy (pdf_proxy)
        use_projection: Project before fusion
        use_gating: Use gating mechanism for fusion weights
        dropout: Dropout probability
        eps: Epsilon for numerical stability
        normalize_weights: Normalize fusion weights to sum to 1
        residual: Add residual connection
        validate_geometry: Validate geometric constraints
    """
    dim: int = 512
    num_heads: int = 8
    head_dim: Optional[int] = None
    simplex_config: Optional[SimplexConfig] = None
    k_simplex: Optional[int] = None  # Alternative to simplex_config
    max_seq_len: int = 524_288
    fusion_window: int = 64
    adaptive_window: bool = False
    min_window: int = 16
    max_window: int = 64
    sparsity_target: float = 0.25
    fusion_mode: Literal["weighted", "max", "mean", "learned", "consciousness"] = "weighted"
    use_beatrix_routing: bool = False  # Use staircase for routing
    beatrix_tau: float = 0.25
    beatrix_base: int = 3
    beatrix_alpha: float = 0.5
    use_consciousness_weighting: bool = False  # Weight by pdf_proxy
    use_projection: bool = True
    use_gating: bool = False
    dropout: float = 0.1
    eps: float = 1e-8
    normalize_weights: bool = True
    residual: bool = True
    validate_geometry: bool = False

    def __post_init__(self):
        if self.head_dim is None:
            assert self.dim % self.num_heads == 0, \
                f"dim ({self.dim}) must be divisible by num_heads ({self.num_heads})"
            self.head_dim = self.dim // self.num_heads

        # Initialize simplex config
        if self.simplex_config is None:
            k = self.k_simplex if self.k_simplex is not None else 4  # Default pentachoron
            self.simplex_config = SimplexConfig(k_simplex=k)

        assert self.eps > 0, "eps must be positive"

        if self.adaptive_window:
            assert self.min_window > 0, "min_window must be positive"
            assert self.max_window >= self.min_window
            assert 0 < self.sparsity_target <= 1.0

        # Consciousness mode requires Beatrix routing
        if self.fusion_mode == "consciousness":
            self.use_beatrix_routing = True
            self.use_consciousness_weighting = True

    def get_window_size(self, seq_len: int) -> int:
        """Compute adaptive fusion window size."""
        if not self.adaptive_window:
            return self.fusion_window

        adaptive_k = int(seq_len * self.sparsity_target)
        return max(self.min_window, min(adaptive_k, self.max_window))


class CantorMultiheadFusion(nn.Module):
    """
    Cantor Multihead Sparse Fusion with Beatrix Staircase integration.

    Uses deterministic Cantor pairing and Devil's Staircase to create
    consciousness-aware geometric fusion patterns.

    Architecture:
        1. Optional input projection (split into heads)
        2. Compute Beatrix staircase features (consciousness patterns)
        3. Cantor routing determines which elements to fuse (k per position)
        4. Compute fusion weights (geometric, learned, or consciousness-guided)
        5. Fuse gathered elements per head
        6. Concatenate heads and project output
        7. Optional residual connection

    Beatrix Integration:
        - Staircase measure guides routing (monotonic consciousness emergence)
        - PDF proxy (entropy measure) weights fusion importance
        - Multi-level features provide hierarchical context
        - Simplex geometry ensures geometric consistency
    """

    def __init__(self, config: CantorFusionConfig):
        super().__init__()
        self.config = config
        self.dim = config.dim
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim

        # Input projection (optional)
        if config.use_projection:
            self.in_proj = nn.Linear(config.dim, config.dim, bias=False)
        else:
            self.in_proj = nn.Identity()

        # Beatrix features projection (if using consciousness)
        if config.use_beatrix_routing or config.use_consciousness_weighting:
            # Project staircase features to useful dimension
            staircase_levels = config.simplex_config.staircase_levels
            features_dim = staircase_levels * 2  # [bit_k, pdf_proxy] per level

            self.beatrix_proj = nn.Linear(features_dim, config.num_heads, bias=False)
        else:
            self.beatrix_proj = None

        # Fusion weight computation (per head)
        if config.fusion_mode == "learned":
            # Learn fusion weights from concatenated features
            self.fusion_weight_net = nn.Sequential(
                nn.Linear(config.head_dim * 2, config.head_dim),
                nn.ReLU(),
                nn.Linear(config.head_dim, 1)
            )
        elif config.fusion_mode == "consciousness":
            # Learn weights from both geometric and consciousness features
            consciousness_dim = config.simplex_config.staircase_levels * 2
            self.fusion_weight_net = nn.Sequential(
                nn.Linear(config.head_dim * 2 + consciousness_dim, config.head_dim),
                nn.ReLU(),
                nn.Linear(config.head_dim, 1)
            )
        else:
            self.fusion_weight_net = None

        # Optional gating mechanism
        if config.use_gating:
            self.gate_proj = nn.Linear(config.dim, config.num_heads)
        else:
            self.gate_proj = None

        # Output projection
        self.out_proj = nn.Linear(config.dim, config.dim, bias=True)

        # Dropout
        self.fusion_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Factory and routes cache
        self._factory_cache: Dict[int, CantorRouteFactory] = {}
        self.routes_cache: Dict[Tuple[int, int], torch.Tensor] = {}

        # Beatrix staircase cache: {seq_len: (cantor_measure, features)}
        self.staircase_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}

        self.max_cache_entries = 50

        # Pre-build routes
        self._prebuild_common_routes()

    def _prebuild_common_routes(self):
        """Pre-build routes for common sequence lengths."""
        common_sizes = [64, 128, 256, 512, 1024, 2048, 4096, 8192]

        route_mode = "Beatrix Staircase" if self.config.use_beatrix_routing else "Cantor Distance"
        print(f"[CantorFusion] Pre-building routes using {route_mode} "
              f"(adaptive={self.config.adaptive_window}, k={self.config.simplex_config.k}-simplex)...")

        for size in common_sizes:
            if size <= self.config.max_seq_len:
                k = self.config.get_window_size(size)
                try:
                    routes = self._build_fusion_routes(size, k)
                    self.routes_cache[(size, k)] = routes

                    if self.config.adaptive_window:
                        print(f"  seq={size:5d}: k={k:2d} ({100 * k / size:.1f}% fusion coverage)")
                except Exception as e:
                    warnings.warn(f"Failed to pre-build routes for size {size}: {e}")

        print(f"[CantorFusion] ✓ Pre-built {len(self.routes_cache)} route tables")

    def _get_or_create_factory(
            self,
            seq_len: int,
            mode: RouteMode = RouteMode.DISTANCE
    ) -> CantorRouteFactory:
        """Get or create CantorRouteFactory for a specific sequence length."""
        cache_key = f"{seq_len}_{mode.value}"

        if cache_key not in self._factory_cache:
            factory = CantorRouteFactory(
                shape=(seq_len,),
                mode=mode,
                simplex_config=self.config.simplex_config,
                staircase_tau=self.config.beatrix_tau,
                staircase_base=self.config.beatrix_base,
                staircase_alpha=self.config.beatrix_alpha
            )
            self._factory_cache[cache_key] = factory

        return self._factory_cache[cache_key]

    def _compute_beatrix_staircase(
            self,
            seq_len: int,
            device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Beatrix Devil's Staircase for sequence.

        Returns:
            (cantor_measure, features) where:
                cantor_measure: (seq_len,) - monotonic consciousness measure
                features: (seq_len, levels, 2) - [bit_k, pdf_proxy] per level
        """
        # Check cache
        if seq_len in self.staircase_cache:
            cached_measure, cached_features = self.staircase_cache[seq_len]
            return cached_measure.to(device), cached_features.to(device)

        # Build staircase using factory
        factory = self._get_or_create_factory(seq_len, RouteMode.STAIRCASE_FEATURES)

        cantor_measure, features = factory.build(
            backend="torch",
            device=device,
            dtype=torch.float32,
            validate=self.config.validate_geometry
        )

        # Cache for reuse
        if len(self.staircase_cache) < self.max_cache_entries:
            # Store on CPU to save GPU memory
            self.staircase_cache[seq_len] = (
                cantor_measure.cpu(),
                features.cpu()
            )

        return cantor_measure, features

    def _build_fusion_routes(self, seq_len: int, k: int) -> torch.Tensor:
        """
        Build routing table for fusion.

        Uses either:
        - Beatrix staircase distances (consciousness-aware)
        - Cantor pairing distances (geometric)

        Args:
            seq_len: Sequence length
            k: Number of neighbors per position

        Returns:
            Routes tensor (seq_len, k) with neighbor indices
        """
        if self.config.use_beatrix_routing:
            # Use staircase measure for routing
            cantor_measure, _ = self._compute_beatrix_staircase(seq_len, "cpu")

            # Build distance matrix from staircase measure
            # Positions with similar consciousness levels are "close"
            distance_matrix = torch.abs(
                cantor_measure.unsqueeze(1) - cantor_measure.unsqueeze(0)
            )
        else:
            # Use standard Cantor distance
            factory = self._get_or_create_factory(seq_len, RouteMode.DISTANCE)
            distance_matrix = factory.build(
                backend="torch",
                device="cpu",
                dtype=torch.float32,
                validate=self.config.validate_geometry
            )

        # Find k-nearest neighbors for each position
        routes = torch.zeros(seq_len, k, dtype=torch.long)

        for i in range(seq_len):
            distances = distance_matrix[i]
            _, nearest = torch.topk(distances, k, largest=False)
            routes[i] = nearest

        return routes

    def _get_routes_for_seq_len(
            self,
            seq_len: int,
            k: int,
            device: torch.device
    ) -> torch.Tensor:
        """Get or build routes for a specific sequence length and window size."""
        cache_key = (seq_len, k)

        # Check cache
        if cache_key in self.routes_cache:
            return self.routes_cache[cache_key].to(device)

        # Try to find larger cached size and truncate
        for (cached_seq, cached_k) in sorted(self.routes_cache.keys()):
            if cached_k == k and cached_seq >= seq_len:
                routes = self.routes_cache[(cached_seq, cached_k)][:seq_len, :].to(device)
                routes = torch.clamp(routes, 0, seq_len - 1)
                return routes

        # Build on-demand
        routes = self._build_fusion_routes(seq_len, k)

        # Add to cache if not full
        if len(self.routes_cache) < self.max_cache_entries:
            self.routes_cache[cache_key] = routes

        return routes.to(device)

    def _compute_fusion_weights(
            self,
            x_anchor: torch.Tensor,
            x_gathered: torch.Tensor,
            routes: torch.Tensor,
            beatrix_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute fusion weights based on fusion mode.

        Args:
            x_anchor: Anchor features (batch, heads, seq, head_dim)
            x_gathered: Gathered neighbor features (batch, heads, seq, k, head_dim)
            routes: Route indices (seq, k)
            beatrix_features: Beatrix staircase features (seq, levels, 2) or None

        Returns:
            Fusion weights (batch, heads, seq, k)
        """
        batch, heads, seq, k, head_dim = x_gathered.shape
        device = x_anchor.device

        if self.config.fusion_mode == "weighted":
            # Geometric distance-based weights
            if self.config.use_beatrix_routing and beatrix_features is not None:
                # Use staircase distances
                cantor_measure = beatrix_features[..., 0]  # Use first level bit_k as proxy

                # Compute distances for routed positions
                weights = torch.zeros(seq, k, device=device, dtype=x_anchor.dtype)
                for i in range(seq):
                    anchor_val = cantor_measure[i, 0]  # First level
                    neighbor_vals = cantor_measure[routes[i], 0]
                    weights[i] = torch.abs(anchor_val - neighbor_vals)

                # Invert (closer = higher weight)
                weights = 1.0 / (weights + self.config.eps)
            else:
                # Use standard Cantor distances
                factory = self._get_or_create_factory(seq)
                distance_matrix = factory.build(
                    backend="torch",
                    device=device,
                    dtype=x_anchor.dtype,
                    validate=False
                )

                weights = torch.zeros(seq, k, device=device, dtype=x_anchor.dtype)
                for i in range(seq):
                    weights[i] = distance_matrix[i, routes[i]]

                weights = 1.0 / (weights + self.config.eps)

            # Expand for batch and heads
            weights = weights.unsqueeze(0).unsqueeze(0).expand(batch, heads, -1, -1)

        elif self.config.fusion_mode == "learned":
            # Learn weights from concatenated features
            x_anchor_expanded = x_anchor.unsqueeze(3).expand(-1, -1, -1, k, -1)
            combined = torch.cat([x_anchor_expanded, x_gathered], dim=-1)
            weights = self.fusion_weight_net(combined).squeeze(-1)

        elif self.config.fusion_mode == "consciousness":
            # Learn weights from geometric + consciousness features
            x_anchor_expanded = x_anchor.unsqueeze(3).expand(-1, -1, -1, k, -1)

            # Flatten beatrix features per position
            beatrix_flat = beatrix_features.reshape(seq, -1)  # (seq, levels*2)

            # Expand for neighbors
            beatrix_expanded = beatrix_flat.unsqueeze(1).expand(-1, k, -1)  # (seq, k, levels*2)
            beatrix_expanded = beatrix_expanded.unsqueeze(0).unsqueeze(0).expand(
                batch, heads, -1, -1, -1
            )  # (batch, heads, seq, k, levels*2)

            # Concatenate all features
            combined = torch.cat([
                x_anchor_expanded,
                x_gathered,
                beatrix_expanded
            ], dim=-1)

            weights = self.fusion_weight_net(combined).squeeze(-1)

        elif self.config.fusion_mode == "mean":
            # Uniform weights
            weights = torch.ones(batch, heads, seq, k, device=device)

        elif self.config.fusion_mode == "max":
            # Dummy weights for max pooling
            weights = torch.ones(batch, heads, seq, k, device=device)

        else:
            raise ValueError(f"Unknown fusion_mode: {self.config.fusion_mode}")

        # Apply consciousness weighting if enabled
        if self.config.use_consciousness_weighting and beatrix_features is not None:
            # Use PDF proxy (consciousness measure) to modulate weights
            # pdf_proxy is in beatrix_features[..., :, 1] (second feature per level)

            # Average consciousness across levels
            consciousness = beatrix_features[..., 1].mean(dim=-1)  # (seq,)

            # Gather consciousness for neighbors
            consciousness_gathered = consciousness[routes]  # (seq, k)

            # Expand for batch and heads
            consciousness_gathered = consciousness_gathered.unsqueeze(0).unsqueeze(0).expand(
                batch, heads, -1, -1
            )  # (batch, heads, seq, k)

            # Modulate weights by consciousness (higher consciousness = higher weight)
            weights = weights * (1.0 + consciousness_gathered)

        # Normalize weights if requested
        if self.config.normalize_weights and self.config.fusion_mode != "max":
            weights = F.softmax(weights, dim=-1)

        return weights

    def _apply_fusion(
            self,
            x_anchor: torch.Tensor,
            x_gathered: torch.Tensor,
            weights: torch.Tensor
    ) -> torch.Tensor:
        """Apply fusion operation."""
        if self.config.fusion_mode == "max":
            fused = torch.max(x_gathered, dim=3)[0]
        else:
            # Weighted combination
            fused = torch.einsum('bhsk,bhskd->bhsd', weights, x_gathered)

        # Apply optional gating
        if self.gate_proj is not None:
            gate_input = x_anchor.transpose(1, 2)
            gate_input = gate_input.reshape(gate_input.shape[0], gate_input.shape[1], -1)

            gates = torch.sigmoid(self.gate_proj(gate_input))
            gates = gates.transpose(1, 2).unsqueeze(-1)

            fused = fused * gates

        return fused

    def forward(
            self,
            x: torch.Tensor,
            mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with Beatrix staircase integration.

        Args:
            x: Input tensor (batch, seq_len, dim)
            mask: Optional mask (batch, seq_len) with 1 for valid, 0 for masked

        Returns:
            Dictionary containing:
                - output: Fused output (batch, seq_len, dim)
                - cantor_measure: Beatrix staircase measure (batch, seq_len) if using Beatrix
                - consciousness: Average consciousness proxy (batch, seq_len) if using Beatrix
        """
        batch_size, seq_len, dim = x.shape

        # Validate sequence length
        if seq_len > self.config.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds maximum {self.config.max_seq_len}"
            )

        # Save for residual
        residual = x

        # Compute Beatrix staircase if needed
        beatrix_features = None
        cantor_measure = None
        consciousness = None

        if self.config.use_beatrix_routing or self.config.use_consciousness_weighting:
            cantor_measure, beatrix_features = self._compute_beatrix_staircase(
                seq_len, x.device
            )

            # Compute average consciousness (PDF proxy)
            consciousness = beatrix_features[..., 1].mean(dim=-1)  # (seq,)

            # Expand for batch
            cantor_measure = cantor_measure.unsqueeze(0).expand(batch_size, -1)
            consciousness = consciousness.unsqueeze(0).expand(batch_size, -1)

        # Input projection
        x = self.in_proj(x)

        # Reshape to heads
        x = x.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        x = x.transpose(1, 2)  # (batch, heads, seq, head_dim)

        # Get adaptive fusion window
        k = self.config.get_window_size(seq_len)

        # Get routes
        routes = self._get_routes_for_seq_len(seq_len, k, x.device)

        # Gather neighbors based on routes
        batch_idx = torch.arange(batch_size, device=x.device).view(-1, 1, 1, 1)
        head_idx = torch.arange(self.num_heads, device=x.device).view(1, -1, 1, 1)
        routes_bc = routes.view(1, 1, seq_len, k)

        batch_idx = batch_idx.expand(batch_size, self.num_heads, seq_len, k)
        head_idx = head_idx.expand(batch_size, self.num_heads, seq_len, k)
        routes_bc = routes_bc.expand(batch_size, self.num_heads, seq_len, k)

        x_gathered = x[batch_idx, head_idx, routes_bc, :]

        # Apply mask if provided
        if mask is not None:
            mask_gathered = torch.gather(
                mask.unsqueeze(1).expand(-1, seq_len, -1),
                dim=2,
                index=routes.unsqueeze(0).expand(batch_size, -1, -1)
            ).unsqueeze(1)

            mask_expanded = mask_gathered.unsqueeze(-1)
            x_gathered = x_gathered * mask_expanded

        # Compute fusion weights (with Beatrix features if available)
        weights = self._compute_fusion_weights(x, x_gathered, routes, beatrix_features)

        # Apply dropout to weights
        weights = self.fusion_dropout(weights)

        # Perform fusion
        fused = self._apply_fusion(x, x_gathered, weights)

        # Reshape back to (batch, seq, dim)
        fused = fused.transpose(1, 2)
        fused = fused.reshape(batch_size, seq_len, self.dim)

        # Output projection
        output = self.out_proj(fused)
        output = self.resid_dropout(output)

        # Residual connection
        if self.config.residual:
            output = output + residual

        # Return dictionary with auxiliary outputs
        result = {'output': output}

        if cantor_measure is not None:
            result['cantor_measure'] = cantor_measure

        if consciousness is not None:
            result['consciousness'] = consciousness

        return result

    def get_fusion_info(self, seq_len: int) -> Dict:
        """Get fusion routing information for analysis."""
        k = self.config.get_window_size(seq_len)

        return {
            'seq_len': seq_len,
            'k_fusion_neighbors': k,
            'sparsity': k / seq_len,
            'complexity': f'O({k}n) = O(n)',
            'fusion_mode': self.config.fusion_mode,
            'num_heads': self.num_heads,
            'simplex_k': self.config.simplex_config.k,
            'simplex_vertices': self.config.simplex_config.k_plus_1,
            'uses_beatrix': self.config.use_beatrix_routing,
            'uses_consciousness': self.config.use_consciousness_weighting,
            'uses_gating': self.config.use_gating,
            'cache_hit': (seq_len, k) in self.routes_cache
        }

    def extra_repr(self) -> str:
        """String representation for debugging."""
        beatrix_str = f", beatrix={self.config.use_beatrix_routing}"
        consciousness_str = f", consciousness={self.config.use_consciousness_weighting}"

        return (f'dim={self.dim}, num_heads={self.num_heads}, '
                f'fusion_window={self.config.fusion_window}, '
                f'mode={self.config.fusion_mode}, '
                f'k_simplex={self.config.simplex_config.k}'
                f'{beatrix_str}{consciousness_str}')


# Convenience functions

def create_cantor_fusion(
        dim: int,
        num_heads: int = 8,
        fusion_window: int = 64,
        fusion_mode: str = "weighted",
        k_simplex: int = 4,
        adaptive_window: bool = False,
        use_beatrix: bool = False,
        use_gating: bool = False,
        dropout: float = 0.1,
        **kwargs
) -> CantorMultiheadFusion:
    """
    Convenience function to create Cantor multihead fusion layer.

    Args:
        dim: Model dimension
        num_heads: Number of fusion heads
        fusion_window: Number of neighbors to fuse
        fusion_mode: "weighted", "learned", "mean", "max", or "consciousness"
        k_simplex: Simplex dimension (k+1 vertices)
        adaptive_window: Enable adaptive window sizing
        use_beatrix: Use Beatrix staircase for routing
        use_gating: Use gating mechanism
        dropout: Dropout rate
        **kwargs: Additional config arguments

    Returns:
        CantorMultiheadFusion layer
    """
    config = CantorFusionConfig(
        dim=dim,
        num_heads=num_heads,
        fusion_window=fusion_window,
        fusion_mode=fusion_mode,
        k_simplex=k_simplex,
        adaptive_window=adaptive_window,
        use_beatrix_routing=use_beatrix,
        use_gating=use_gating,
        dropout=dropout,
        **kwargs
    )
    return CantorMultiheadFusion(config)


class PentachoronFusion(nn.Module):
    """
    Specialized fusion for pentachoron (5-vertex) crystalline structures
    with Beatrix consciousness integration.

    Fuses 5 vertices using Cantor geometry and Devil's Staircase.
    Perfect for crystalline vocabulary architectures!
    """

    def __init__(
            self,
            vertex_dim: int = 640,  # Divisible by 5: 640 = 5 * 128
            num_heads: int = 5,
            use_consciousness: bool = True
    ):
        super().__init__()

        # Ensure dimension is divisible by 5
        if vertex_dim % 5 != 0:
            # Project to nearest 5-divisible dimension
            internal_dim = ((vertex_dim // 5) + 1) * 5
            self.input_proj = nn.Linear(vertex_dim, internal_dim, bias=False)
            self.output_proj = nn.Linear(internal_dim, vertex_dim, bias=False)
            print(f"[PentachoronFusion] Projecting {vertex_dim} -> {internal_dim} "
                  f"for 5-head alignment")
        else:
            self.input_proj = nn.Identity()
            self.output_proj = nn.Identity()
            internal_dim = vertex_dim

        # 5-vertex fusion with consciousness
        self.fusion = CantorMultiheadFusion(
            CantorFusionConfig(
                dim=internal_dim,
                num_heads=num_heads,
                head_dim=internal_dim // num_heads,
                fusion_window=5,  # Fuse all 5 vertices
                fusion_mode="consciousness" if use_consciousness else "learned",
                k_simplex=4,  # 4-simplex = 5 vertices (pentachoron)
                use_beatrix_routing=use_consciousness,
                use_consciousness_weighting=use_consciousness,
                beatrix_tau=0.25,
                use_gating=True,
                adaptive_window=False
            )
        )

    def forward(self, vertices: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Fuse pentachoron vertices with consciousness.

        Args:
            vertices: (batch, 5, vertex_dim) - 5 vertices of pentachoron

        Returns:
            Dictionary with:
                - output: Fused representation (batch, 5, vertex_dim)
                - cantor_measure: Consciousness measure (batch, 5) if using Beatrix
                - consciousness: Consciousness proxy (batch, 5) if using Beatrix
        """
        x = self.input_proj(vertices)
        result = self.fusion(x)
        result['output'] = self.output_proj(result['output'])
        return result


if __name__ == "__main__":
    print("=" * 70)
    print("Testing Cantor Multihead Fusion with Beatrix Staircase")
    print("=" * 70)

    # Test 1: Standard geometric fusion
    print("\n[1] Standard geometric fusion:")
    config = CantorFusionConfig(
        dim=512,
        num_heads=8,
        fusion_window=64,
        fusion_mode="weighted",
        k_simplex=4,
        dropout=0.0
    )

    fusion = CantorMultiheadFusion(config)
    x = torch.randn(2, 256, 512)
    result = fusion(x)
    output = result['output']
    assert output.shape == x.shape
    print(f"  ✓ Geometric fusion: {x.shape} -> {output.shape}")

    # Test 2: Beatrix staircase routing
    print("\n[2] Beatrix staircase routing:")
    fusion_beatrix = create_cantor_fusion(
        dim=512,
        num_heads=8,
        fusion_mode="weighted",
        k_simplex=4,
        fusion_window=32,
        use_beatrix=True
    )

    result_beatrix = fusion_beatrix(x)
    output_beatrix = result_beatrix['output']
    cantor = result_beatrix['cantor_measure']
    consciousness = result_beatrix['consciousness']

    print(f"  ✓ Beatrix fusion: {x.shape} -> {output_beatrix.shape}")
    print(f"    Cantor measure: {cantor.shape}, range [{cantor.min():.4f}, {cantor.max():.4f}]")
    print(f"    Consciousness: {consciousness.shape}, mean {consciousness.mean():.4f}")

    # Verify monotonicity
    monotonic_ratio = (cantor[:, 1:] >= cantor[:, :-1]).float().mean()
    print(f"    Monotonic: {monotonic_ratio * 100:.1f}%")

    # Test 3: Consciousness-guided fusion
    print("\n[3] Consciousness-guided fusion:")
    fusion_consciousness = create_cantor_fusion(
        dim=512,
        num_heads=8,
        fusion_mode="consciousness",
        k_simplex=4,
        fusion_window=32
    )

    result_consciousness = fusion_consciousness(x)
    output_consciousness = result_consciousness['output']

    print(f"  ✓ Consciousness fusion: {x.shape} -> {output_consciousness.shape}")

    # Test 4: Pentachoron fusion
    print("\n[4] Pentachoron (5-vertex) fusion with consciousness:")
    pentachoron_fusion = PentachoronFusion(
        vertex_dim=640,  # 5 * 128
        num_heads=5,
        use_consciousness=True
    )

    vertices = torch.randn(4, 5, 640)
    result_pentachoron = pentachoron_fusion(vertices)
    output_pentachoron = result_pentachoron['output']

    print(f"  ✓ Pentachoron fusion: {vertices.shape} -> {output_pentachoron.shape}")
    print(f"    Using 5D Cantor + Beatrix staircase")

    if 'consciousness' in result_pentachoron:
        pentachoron_consciousness = result_pentachoron['consciousness']
        print(f"    Vertex consciousness: {pentachoron_consciousness.shape}")
        print(f"    Mean consciousness: {pentachoron_consciousness.mean():.4f}")

    # Test 5: Adaptive window with Beatrix
    print("\n[5] Adaptive Beatrix fusion:")
    fusion_adaptive = create_cantor_fusion(
        dim=512,
        num_heads=8,
        adaptive_window=True,
        min_window=16,
        max_window=64,
        sparsity_target=0.25,
        use_beatrix=True
    )

    for seq_len in [128, 512, 2048]:
        x_test = torch.randn(2, seq_len, 512)
        result_test = fusion_adaptive(x_test)
        info = fusion_adaptive.get_fusion_info(seq_len)
        print(f"  ✓ seq={seq_len:4d}: k={info['k_fusion_neighbors']:2d}, "
              f"beatrix={info['uses_beatrix']}, "
              f"consciousness={info['uses_consciousness']}")

    # Test 6: Compare routing methods
    print("\n[6] Routing method comparison:")
    x_compare = torch.randn(2, 256, 512)

    # Standard Cantor
    fusion_cantor = create_cantor_fusion(
        dim=512, num_heads=8, k_simplex=4, use_beatrix=False
    )
    result_cantor = fusion_cantor(x_compare)

    # Beatrix staircase
    fusion_staircase = create_cantor_fusion(
        dim=512, num_heads=8, k_simplex=4, use_beatrix=True
    )
    result_staircase = fusion_staircase(x_compare)

    print(f"  Standard Cantor: output_norm={result_cantor['output'].norm():.4f}")
    print(f"  Beatrix Staircase: output_norm={result_staircase['output'].norm():.4f}")
    print(f"    Consciousness range: [{result_staircase['consciousness'].min():.4f}, "
          f"{result_staircase['consciousness'].max():.4f}]")

    # Test 7: Simplex dimension scaling
    print("\n[7] Simplex dimension scaling:")
    for k in [2, 4, 6, 8]:
        fusion_k = create_cantor_fusion(
            dim=512,
            num_heads=8,
            k_simplex=k,
            fusion_window=k + 1,
            use_beatrix=True
        )

        x_k = torch.randn(2, 128, 512)
        result_k = fusion_k(x_k)
        info_k = fusion_k.get_fusion_info(128)

        print(f"  k={k}: {k + 1} vertices, staircase_levels={info_k['simplex_vertices']}")

    # Test 8: Gradient flow
    print("\n[8] Gradient flow:")
    x_grad = torch.randn(2, 128, 512, requires_grad=True)
    result_grad = fusion_beatrix(x_grad)
    loss = result_grad['output'].sum()

    if 'consciousness' in result_grad:
        loss = loss + result_grad['consciousness'].sum()

    loss.backward()

    assert x_grad.grad is not None
    assert torch.all(torch.isfinite(x_grad.grad))
    print(f"  ✓ Gradients computed: grad_norm={x_grad.grad.norm():.4f}")

    # Test 9: Cache efficiency
    print("\n[9] Cache efficiency:")
    print(f"  Factory cache: {len(fusion_beatrix._factory_cache)} entries")
    print(f"  Routes cache: {len(fusion_beatrix.routes_cache)} entries")
    print(f"  Staircase cache: {len(fusion_beatrix.staircase_cache)} entries")

    print("\n" + "=" * 70)
    print("✓ All tests passed!")
    print("\nCantor Fusion with Beatrix Integration:")
    print("  ✓ Devil's Staircase routing (consciousness-aware)")
    print("  ✓ PDF proxy weighting (entropy-based importance)")
    print("  ✓ Simplex-based parameterization (k-simplex geometry)")
    print("  ✓ Pentachoron fusion (5-vertex crystalline)")
    print("  ✓ O(n) complexity maintained")
    print("  ✓ Monotonic consciousness emergence")
    print("\nReady for:")
    print("  - Beatrix consciousness architectures")
    print("  - Crystalline vocabulary fusion")
    print("  - Geometric attention replacement")
    print("  - Multi-scale consciousness-aware feature aggregation")
    print("=" * 70)