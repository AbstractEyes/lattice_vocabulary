"""
Liminal Staircase - Collective V2 with Organized Fusion + Geometric Initialization
========================================================

COMPLETE INTEGRATED VERSION with:
- Corrected sparse attention indexing
- Truly parameter-efficient shared embeddings
- Organized fusion controller (Alpha, Beta, Gamma, Layer Weights)
- ScaleFusionConfig for easy configuration
- Per-scale hidden dimensions
- Per-scale loss tracking
- Full gradient flow
- GEOMETRIC PENTACHORON INITIALIZATION using SimplexFactory

Author: AbstractPhil + Claude Sonnet 4.5
Date: 2025-11-17 (Geometric Initialization)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import Dict, List, Tuple, Optional, Literal
from dataclasses import dataclass
import math
import time

# Import SimplexFactory for geometric initialization
from geovocab2.shapes.factory.simplex_factory import SimplexFactory


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class CantorAttentionConfig:
    """Configuration for Cantor Global Attention."""
    dim: int = 512
    num_heads: int = 8
    head_dim: Optional[int] = None
    local_window: int = 64
    adaptive_window: bool = True
    min_window: int = 16
    max_window: int = 128
    sparsity_target: float = 0.15
    dropout: float = 0.1
    causal: bool = False
    qkv_bias: bool = True
    out_bias: bool = True

    def __post_init__(self):
        if self.head_dim is None:
            assert self.dim % self.num_heads == 0
            self.head_dim = self.dim // self.num_heads


@dataclass
class ScaleFusionConfig:
    """Configuration for organized scale fusion."""

    # Scale architecture
    scales: List[int] = None
    scale_hidden_dims: Dict[int, int] = None

    # Fusion strategy
    fusion_strategy: str = "learned_weighted"  # For future expansion

    # Alpha: Cantor neighbor bleed-over
    alpha_init: float = 0.1
    alpha_learnable: bool = True
    alpha_per_layer: bool = False
    alpha_per_scale: bool = True
    alpha_min: float = 0.0
    alpha_max: float = 0.5

    # Beta: Geometric vs projection balance
    beta_init: float = 0.5
    beta_learnable: bool = True
    beta_per_scale: bool = True
    beta_min: float = 0.0
    beta_max: float = 1.0

    # Gamma: Scale importance (implemented via learn_scale_weights)
    gamma_init: float = 0.5
    gamma_learnable: bool = True
    gamma_per_scale: bool = True

    # Layer weights
    learn_layer_weights: bool = True
    learn_scale_weights: bool = True

    # Diagnostics
    track_scale_losses: bool = True

    def __post_init__(self):
        if self.scales is None:
            self.scales = [128, 256, 512]
        if self.scale_hidden_dims is None:
            self.scale_hidden_dims = {s: s * 2 for s in self.scales}


@dataclass
class LiminalStaircaseConfig:
    """Configuration for Liminal Staircase with Organized Fusion."""

    # Core architecture
    num_opinion_anchors: int = 225
    pentachoron_dim: int = 512
    cantor_depth: int = 8

    # Encoder configuration
    siglip_hidden_dim: int = 1664
    siglip_num_layers: int = 24
    clip_hidden_dim: int = 768
    clip_num_layers: int = 12
    clip_skip: int = 2

    # Vocabulary
    vocab_size: int = 49408
    max_seq_len: int = 77
    num_heads: int = 8
    dropout: float = 0.1

    # Optimizations
    use_gradient_checkpointing: bool = False
    share_scale_embeddings: bool = True

    # Geometry parameters
    geometry_volume_scale: float = 10.0
    geometry_volume_weight: float = 0.4
    geometry_edge_weight: float = 0.3
    geometry_spread_weight: float = 0.3
    geometry_epsilon: float = 1e-6

    # Geometric initialization
    geometric_init_method: Literal["regular", "random", "uniform", "hybrid"] = "hybrid"
    geometric_init_scale: float = 1.0
    geometric_init_validate: bool = False
    geometric_init_normalize: bool = True
    geometric_init_seed: Optional[int] = 42

    # Layer selection
    siglip_layer_indices: Optional[List[int]] = None
    clip_layer_indices: Optional[List[int]] = None

    # Scale fusion configuration (NEW)
    scale_fusion: ScaleFusionConfig = None

    # Legacy compatibility (kept for backward compatibility)
    scales: List[int] = None
    scale_hidden_dims: Optional[Dict[int, int]] = None
    learn_layer_weights: bool = True
    learn_scale_weights: bool = True
    alpha_init: float = 0.1
    alpha_learnable: bool = True
    alpha_per_layer: bool = False
    alpha_per_scale: bool = True
    alpha_min: float = 0.0
    alpha_max: float = 0.5
    beta_init: float = 0.5
    beta_learnable: bool = True
    beta_per_scale: bool = True
    beta_min: float = 0.0
    beta_max: float = 1.0
    record_scale_losses: bool = True

    def __post_init__(self):
        # Initialize scale_fusion if not provided
        if self.scale_fusion is None:
            # Create from legacy parameters
            self.scale_fusion = ScaleFusionConfig(
                scales=self.scales if self.scales is not None else [128, 256, 512],
                scale_hidden_dims=self.scale_hidden_dims,
                alpha_init=self.alpha_init,
                alpha_learnable=self.alpha_learnable,
                alpha_per_layer=self.alpha_per_layer,
                alpha_per_scale=self.alpha_per_scale,
                alpha_min=self.alpha_min,
                alpha_max=self.alpha_max,
                beta_init=self.beta_init,
                beta_learnable=self.beta_learnable,
                beta_per_scale=self.beta_per_scale,
                beta_min=self.beta_min,
                beta_max=self.beta_max,
                learn_layer_weights=self.learn_layer_weights,
                learn_scale_weights=self.learn_scale_weights,
                track_scale_losses=self.record_scale_losses
            )

        # Sync legacy params with scale_fusion
        self.scales = self.scale_fusion.scales
        self.scale_hidden_dims = self.scale_fusion.scale_hidden_dims

        # Default layer selection
        if self.siglip_layer_indices is None:
            start = max(0, self.siglip_num_layers - 12)
            self.siglip_layer_indices = list(range(start, self.siglip_num_layers))

        if self.clip_layer_indices is None:
            active_clip_layers = self.clip_num_layers - self.clip_skip
            self.clip_layer_indices = list(range(active_clip_layers))

        # Validate geometry weights
        total_weight = self.geometry_volume_weight + self.geometry_edge_weight + self.geometry_spread_weight
        assert abs(total_weight - 1.0) < 1e-5, f"Geometry weights must sum to 1.0, got {total_weight}"


# ============================================================================
# FUSION CONTROLLER COMPONENTS
# ============================================================================

class AlphaController(nn.Module):
    """Controls gradient bleed-over between adjacent Cantor slices."""

    def __init__(
        self,
        num_layers: int,
        num_scales: int,
        config: ScaleFusionConfig
    ):
        super().__init__()

        self.num_layers = num_layers
        self.num_scales = num_scales
        self.config = config

        if config.alpha_per_layer and config.alpha_per_scale:
            shape = (num_layers, num_scales)
        elif config.alpha_per_layer:
            shape = (num_layers,)
        elif config.alpha_per_scale:
            shape = (num_scales,)
        else:
            shape = (1,)

        alpha_init = torch.full(shape, config.alpha_init)

        if config.alpha_learnable:
            self.alpha_raw = nn.Parameter(alpha_init)
        else:
            self.register_buffer('alpha_raw', alpha_init)

    def forward(
        self,
        layer_idx: Optional[int] = None,
        scale_idx: Optional[int] = None
    ) -> torch.Tensor:
        """Get alpha value for layer/scale."""
        alpha = torch.sigmoid(self.alpha_raw)
        alpha = self.config.alpha_min + alpha * (self.config.alpha_max - self.config.alpha_min)

        if self.config.alpha_per_layer and layer_idx is not None:
            if self.config.alpha_per_scale and scale_idx is not None:
                return alpha[layer_idx, scale_idx]
            return alpha[layer_idx]
        elif self.config.alpha_per_scale and scale_idx is not None:
            return alpha[scale_idx]

        return alpha.squeeze()


class BetaController(nn.Module):
    """Controls balance between internal projections and geometric structure."""

    def __init__(self, num_scales: int, config: ScaleFusionConfig):
        super().__init__()

        self.num_scales = num_scales
        self.config = config

        shape = (num_scales,) if config.beta_per_scale else (1,)
        beta_init = torch.full(shape, config.beta_init)

        if config.beta_learnable:
            self.beta_raw = nn.Parameter(beta_init)
        else:
            self.register_buffer('beta_raw', beta_init)

    def forward(self, scale_idx: Optional[int] = None) -> torch.Tensor:
        """Get beta value for scale."""
        beta = torch.sigmoid(self.beta_raw)
        beta = self.config.beta_min + beta * (self.config.beta_max - self.config.beta_min)

        if self.config.beta_per_scale and scale_idx is not None:
            return beta[scale_idx]

        return beta.squeeze()


class GammaController(nn.Module):
    """Controls per-scale importance (via scale weights)."""

    def __init__(self, num_scales: int, config: ScaleFusionConfig):
        super().__init__()

        self.num_scales = num_scales
        self.config = config

        gamma_init = torch.ones(num_scales) / num_scales

        if config.gamma_learnable:
            self.gamma_raw = nn.Parameter(gamma_init)
        else:
            self.register_buffer('gamma_raw', gamma_init)

    def forward(self) -> torch.Tensor:
        """Get normalized gamma weights."""
        return F.softmax(self.gamma_raw, dim=0)


class LayerWeightController(nn.Module):
    """Controls per-layer opinion weights during fusion."""

    def __init__(self, num_layers: int, learnable: bool = True):
        super().__init__()

        self.num_layers = num_layers
        self.learnable = learnable

        weights_init = torch.ones(num_layers) / num_layers

        if learnable:
            self.layer_weights_raw = nn.Parameter(weights_init)
        else:
            self.register_buffer('layer_weights_raw', weights_init)

    def forward(self) -> torch.Tensor:
        """Get normalized layer weights."""
        return F.softmax(self.layer_weights_raw, dim=0)


class OrganizedFusionController(nn.Module):
    """
    Organized hierarchical fusion control system.

    Controls:
    - Alpha: Cantor neighbor bleed-over
    - Beta: Geometric vs projection balance
    - Gamma: Scale importance weights
    - Layer weights: Expert opinion importance
    """

    def __init__(
        self,
        num_layers: int,
        config: ScaleFusionConfig,
        learn_layer_weights: bool = True
    ):
        super().__init__()

        self.config = config
        self.num_layers = num_layers
        self.num_scales = len(config.scales)

        print(f"\n{'='*60}")
        print("ORGANIZED FUSION CONTROLLER")
        print(f"{'='*60}")
        print(f"Layers: {num_layers}")
        print(f"Scales: {config.scales}")

        # Alpha controller
        self.alpha = AlphaController(num_layers, self.num_scales, config)
        print(f"✓ Alpha (bleed-over): init={config.alpha_init}, learnable={config.alpha_learnable}")

        # Beta controller
        self.beta = BetaController(self.num_scales, config)
        print(f"✓ Beta (geometric blend): init={config.beta_init}, learnable={config.beta_learnable}")

        # Gamma controller (scale weights)
        self.gamma = GammaController(self.num_scales, config)
        print(f"✓ Gamma (scale weights): learnable={config.gamma_learnable}")

        # Layer weights
        if learn_layer_weights:
            self.layer_weights = LayerWeightController(num_layers, learnable=True)
            print(f"✓ Layer weights: {num_layers} layers (learnable)")
        else:
            self.layer_weights = None

        # Loss tracking
        if config.track_scale_losses:
            self.register_buffer('scale_losses', torch.zeros(self.num_scales))
            self.register_buffer('scale_loss_counts', torch.zeros(self.num_scales))
            self.register_buffer('scale_beta_losses', torch.zeros(self.num_scales))
            print(f"✓ Scale loss tracking enabled")

        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"✓ Controller params: {total_params} ({trainable_params} trainable)")
        print(f"{'='*60}\n")

    def get_layer_weights(self) -> torch.Tensor:
        """Get normalized layer opinion weights."""
        if self.layer_weights is not None:
            return self.layer_weights()

        device = self.alpha.alpha_raw.device
        return torch.ones(self.num_layers, device=device) / self.num_layers

    def get_scale_weights(self) -> torch.Tensor:
        """Get normalized scale fusion weights (Gamma)."""
        return self.gamma()

    def get_alpha(self, layer_idx: int, scale_idx: int) -> torch.Tensor:
        """Get alpha for specific layer/scale."""
        return self.alpha(layer_idx, scale_idx)

    def get_beta(self, scale_idx: int) -> torch.Tensor:
        """Get beta for specific scale."""
        return self.beta(scale_idx)

    def record_scale_loss(self, scale_idx: int, loss: float, beta_loss: float = 0.0):
        """Record loss for a specific scale."""
        if self.config.track_scale_losses:
            self.scale_losses[scale_idx] += loss
            self.scale_loss_counts[scale_idx] += 1
            self.scale_beta_losses[scale_idx] += beta_loss

    def get_diagnostics(self) -> Dict:
        """Get comprehensive fusion diagnostics."""
        diagnostics = {
            'layer_weights': self.get_layer_weights().detach().cpu().tolist(),
            'scale_weights': self.get_scale_weights().detach().cpu().tolist(),
        }

        # Alpha values per scale
        alpha_values = []
        for scale_idx in range(self.num_scales):
            if self.config.alpha_per_scale:
                alpha = self.alpha(scale_idx=scale_idx).item()
            else:
                alpha = self.alpha().item()
            alpha_values.append(alpha)
        diagnostics['alpha_per_scale'] = alpha_values

        # Beta values per scale
        beta_values = []
        for scale_idx in range(self.num_scales):
            beta = self.beta(scale_idx).item()
            beta_values.append(beta)
        diagnostics['beta_per_scale'] = beta_values

        # Loss statistics
        if self.config.track_scale_losses:
            scale_stats = {}
            with torch.no_grad():
                for i, scale in enumerate(self.config.scales):
                    if self.scale_loss_counts[i] > 0:
                        avg_loss = (self.scale_losses[i] / self.scale_loss_counts[i]).item()
                        avg_beta_loss = (self.scale_beta_losses[i] / self.scale_loss_counts[i]).item()
                        scale_stats[scale] = {
                            'avg_loss': avg_loss,
                            'avg_beta_loss': avg_beta_loss,
                            'count': self.scale_loss_counts[i].item()
                        }
            diagnostics['scale_statistics'] = scale_stats

        return diagnostics

    def reset_losses(self):
        """Reset loss tracking."""
        if self.config.track_scale_losses:
            self.scale_losses.zero_()
            self.scale_loss_counts.zero_()
            self.scale_beta_losses.zero_()


# Alias for backward compatibility
HierarchicalFusionController = OrganizedFusionController


# ============================================================================
# CANTOR ATTENTION (with Alpha) - CACHING OPTIMIZATIONS
# ============================================================================

from functools import lru_cache
import threading

class GlobalCantorRouteCache:
    """Thread-safe global cache for Cantor attention routes across all instances."""

    def __init__(self):
        self._cache = {}
        self._lock = threading.Lock()

    def get_key(self, seq_len: int, k: int, device: str) -> str:
        """Generate cache key."""
        return f"{seq_len}_{k}_{device}"

    def get(self, seq_len: int, k: int, device: torch.device) -> Optional[torch.Tensor]:
        """Thread-safe get from cache."""
        key = self.get_key(seq_len, k, str(device))
        with self._lock:
            return self._cache.get(key)

    def put(self, seq_len: int, k: int, device: torch.device, routes: torch.Tensor):
        """Thread-safe put to cache."""
        key = self.get_key(seq_len, k, str(device))
        with self._lock:
            self._cache[key] = routes

    def clear(self):
        """Clear entire cache."""
        with self._lock:
            self._cache.clear()

# Global singleton cache
_GLOBAL_ROUTE_CACHE = GlobalCantorRouteCache()


@lru_cache(maxsize=256)
def compute_cantor_coords_cached(seq_len: int, depth: int = 8) -> torch.Tensor:
    """LRU-cached Cantor coordinate computation."""
    positions = torch.arange(seq_len, dtype=torch.long)
    x = positions.float() / max(1, seq_len - 1)
    x = torch.clamp(x, 1e-6, 1.0 - 1e-6)

    cantor_val = torch.zeros_like(x)
    factor = 0.5

    for _ in range(depth):
        x_scaled = x * 3.0
        digit = x_scaled.long()
        x_frac = x_scaled - digit.float()

        middle_bit = (digit == 2).float()
        cantor_val = cantor_val + middle_bit * factor

        x = x_frac
        factor *= 0.5

    return torch.clamp(cantor_val, 0.0, 1.0)


class CantorAttention(nn.Module):
    """O(n) attention with aggressive route caching and pre-warming."""

    def __init__(
        self,
        config: CantorAttentionConfig,
        max_seq_len: int = 512,
        k: int = 64
    ):
        super().__init__()
        self.config = config
        self.dim = config.dim
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.k = k
        self.max_seq_len = max_seq_len

        self.qkv = nn.Linear(config.dim, 3 * config.dim, bias=config.qkv_bias)
        self.out_proj = nn.Linear(config.dim, config.dim, bias=config.out_bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Pre-warm cache for common sequence lengths
        self._prewarm_cache()

    def _prewarm_cache(self):
        """Pre-compute and cache routes for common lengths."""
        common_lengths = [64, 77, 128, 196, 256, 384, 512, 1024, 2048]

        for seq_len in common_lengths:
            if seq_len <= self.max_seq_len:
                # Build routes on CPU first
                routes_cpu = self._build_positional_routes_cached(seq_len, self.k)
                # Store in global cache for CPU
                _GLOBAL_ROUTE_CACHE.put(seq_len, self.k, torch.device('cpu'), routes_cpu)

    @staticmethod
    def _build_positional_routes_cached(seq_len: int, k: int) -> torch.Tensor:
        """Cached route building using global LRU-cached coords."""
        coords = compute_cantor_coords_cached(seq_len, depth=8)

        distances = torch.abs(
            coords.unsqueeze(1) - coords.unsqueeze(0)
        )

        _, routes = torch.topk(distances, k, dim=1, largest=False)
        return routes.to(torch.int32)

    def _get_routes_for_seq_len(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Get cached routes with automatic device handling."""
        # Try device-specific cache first
        routes = _GLOBAL_ROUTE_CACHE.get(seq_len, self.k, device)
        if routes is not None:
            return routes.long()

        # Try CPU cache and transfer
        routes_cpu = _GLOBAL_ROUTE_CACHE.get(seq_len, self.k, torch.device('cpu'))
        if routes_cpu is not None:
            routes_device = routes_cpu.to(device, non_blocking=True)
            _GLOBAL_ROUTE_CACHE.put(seq_len, self.k, device, routes_device)
            return routes_device.long()

        # Build new routes (rare case)
        routes = self._build_positional_routes_cached(seq_len, self.k).to(device)
        _GLOBAL_ROUTE_CACHE.put(seq_len, self.k, device, routes)
        return routes.long()

    def _sparse_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        routes: torch.Tensor,
        alpha: float = 0.0
    ) -> torch.Tensor:
        """Optimized sparse attention with minimal overhead."""
        batch_size, num_heads, seq_len, head_dim = q.shape
        k_neighbors = routes.shape[-1]
        device = q.device

        # Expand routes efficiently
        if routes.dim() == 2:
            routes_expanded = routes.unsqueeze(0).expand(batch_size, -1, -1)
        else:
            routes_expanded = routes

        # Create broadcast indices once
        batch_idx = torch.arange(batch_size, device=device).view(-1, 1, 1, 1)
        head_idx = torch.arange(num_heads, device=device).view(1, -1, 1, 1)

        batch_idx = batch_idx.expand(batch_size, num_heads, seq_len, k_neighbors)
        head_idx = head_idx.expand(batch_size, num_heads, seq_len, k_neighbors)
        routes_bc = routes_expanded.unsqueeze(1).expand(batch_size, num_heads, seq_len, k_neighbors)

        # Gather k and v
        k_gathered = k[batch_idx, head_idx, routes_bc, :]
        v_gathered = v[batch_idx, head_idx, routes_bc, :]

        # Alpha bleed-over (only if needed)
        if alpha > 0.0:
            adj_left = torch.clamp(routes_expanded - 1, 0, seq_len - 1)
            adj_right = torch.clamp(routes_expanded + 1, 0, seq_len - 1)

            adj_left_bc = adj_left.unsqueeze(1).expand(batch_size, num_heads, seq_len, k_neighbors)
            adj_right_bc = adj_right.unsqueeze(1).expand(batch_size, num_heads, seq_len, k_neighbors)

            k_left = k[batch_idx, head_idx, adj_left_bc, :]
            k_right = k[batch_idx, head_idx, adj_right_bc, :]
            v_left = v[batch_idx, head_idx, adj_left_bc, :]
            v_right = v[batch_idx, head_idx, adj_right_bc, :]

            k_gathered = k_gathered.mul_(1 - alpha).add_(k_left.add_(k_right).mul_(alpha * 0.5))
            v_gathered = v_gathered.mul_(1 - alpha).add_(v_left.add_(v_right).mul_(alpha * 0.5))

        # Attention computation
        scores = torch.einsum('bhqd,bhqkd->bhqk', q, k_gathered).mul_(self.scale)

        if self.config.causal:
            position_idx = torch.arange(seq_len, device=device).unsqueeze(1)
            causal_mask = routes_expanded > position_idx.unsqueeze(0)
            causal_mask = causal_mask.unsqueeze(1).expand(batch_size, num_heads, -1, -1)
            scores = scores.masked_fill(causal_mask, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        output = torch.einsum('bhqk,bhqkd->bhqd', attn_weights, v_gathered)
        return output

    def forward(
        self,
        x: torch.Tensor,
        anchor_ids: Optional[torch.Tensor] = None,
        alpha: float = 0.0
    ) -> torch.Tensor:
        """Forward with cached routes."""
        batch_size, seq_len, _ = x.shape

        # QKV projection
        qkv = self.qkv(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Get pre-cached routes (no computation!)
        routes = self._get_routes_for_seq_len(seq_len, x.device)

        # Sparse attention
        attn_output = self._sparse_attention(q, k, v, routes, alpha=alpha)

        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.dim)
        output = self.out_proj(attn_output)
        output = self.resid_dropout(output)

        return output


# ============================================================================
# GEOMETRIC POSITIONAL FINGERPRINTING
# ============================================================================

class GeometricPositionalFingerprinter(nn.Module):
    """Fully vectorized geometric computations."""

    def __init__(
        self,
        cantor_depth: int = 8,
        volume_scale: float = 10.0,
        volume_weight: float = 0.4,
        edge_weight: float = 0.3,
        spread_weight: float = 0.3,
        epsilon: float = 1e-6
    ):
        super().__init__()
        self.cantor_depth = cantor_depth
        self.volume_scale = volume_scale
        self.volume_weight = volume_weight
        self.edge_weight = edge_weight
        self.spread_weight = spread_weight
        self.epsilon = epsilon

    def compute_cayley_menger_volume_batched(
        self,
        vertices: torch.Tensor
    ) -> torch.Tensor:
        """Batched Cayley-Menger volume computation."""
        batch_size = vertices.shape[0]
        device = vertices.device
        dtype = vertices.dtype

        diff = vertices.unsqueeze(2) - vertices.unsqueeze(1)
        dist_sq = (diff ** 2).sum(dim=-1)

        M = torch.zeros(batch_size, 6, 6, device=device, dtype=dtype)
        M[:, 0, 1:] = 1.0
        M[:, 1:, 0] = 1.0
        M[:, 1:, 1:] = dist_sq

        det = torch.linalg.det(M)
        volume_sq = (-det / 9216.0).clamp(min=0.0)
        return volume_sq.sqrt()

    def compute_edge_statistics_batched(
        self,
        vertices: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Batched edge statistics."""
        diff = vertices.unsqueeze(2) - vertices.unsqueeze(1)
        dist_sq = (diff ** 2).sum(dim=-1)

        triu_mask = torch.triu(torch.ones(5, 5, device=vertices.device), diagonal=1).bool()
        edge_lengths_sq = dist_sq[:, triu_mask]
        edge_lengths = edge_lengths_sq.sqrt()

        return edge_lengths.mean(dim=1), edge_lengths.std(dim=1)

    def compute_vertex_spread_batched(self, vertices: torch.Tensor) -> torch.Tensor:
        """Batched vertex spread computation."""
        centroid = vertices.mean(dim=1, keepdim=True)
        distances = torch.norm(vertices - centroid, dim=-1)
        return distances.std(dim=1)

    def geometry_to_cantor_position_batched(
        self,
        vertices: torch.Tensor,
        anchor_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Vectorized geometry to Cantor position conversion."""
        volume = self.compute_cayley_menger_volume_batched(vertices)
        mean_edge, std_edge = self.compute_edge_statistics_batched(vertices)
        spread = self.compute_vertex_spread_batched(vertices)

        volume_norm = torch.sigmoid(volume * self.volume_scale)
        edge_ratio = torch.sigmoid(std_edge / (mean_edge + self.epsilon))
        spread_norm = torch.sigmoid(spread)

        seed = (
            volume_norm * self.volume_weight +
            edge_ratio * self.edge_weight +
            spread_norm * self.spread_weight
        )

        if anchor_ids is not None:
            id_contribution = ((anchor_ids * 2654435761) % 1000000) / 1000000.0
            seed = 0.1 * seed + 0.9 * id_contribution.to(seed.device)

        seed = seed.clamp(self.epsilon, 1.0 - self.epsilon)

        x = seed
        cantor_val = torch.zeros_like(x)
        factor = 0.5

        for _ in range(self.cantor_depth):
            x_scaled = x * 3.0
            digit = x_scaled.long()
            x_frac = x_scaled - digit.float()

            middle_bit = (digit == 2).float()
            cantor_val = cantor_val + middle_bit * factor

            x = x_frac
            factor *= 0.5

        return cantor_val.clamp(0.0, 1.0)

    def compute_vocabulary_positions(
        self,
        pentachora: torch.Tensor,
        batch_size: int = 512
    ) -> torch.Tensor:
        """Fully vectorized vocabulary position computation."""
        vocab_size = pentachora.shape[0]
        device = pentachora.device
        positions = torch.zeros(vocab_size, device=device)

        print(f"Computing {vocab_size} positional fingerprints (vectorized, batch={batch_size})...")
        start_time = time.time()

        num_batches = (vocab_size + batch_size - 1) // batch_size

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, vocab_size)

            batch_pentachora = pentachora[start_idx:end_idx]
            batch_anchor_ids = torch.arange(start_idx, end_idx, device=device)
            batch_positions = self.geometry_to_cantor_position_batched(
                batch_pentachora,
                anchor_ids=batch_anchor_ids
            )
            positions[start_idx:end_idx] = batch_positions

            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == num_batches:
                elapsed = time.time() - start_time
                rate = end_idx / elapsed
                print(f"  Batch {batch_idx + 1}/{num_batches} ({end_idx}/{vocab_size}) - {rate:.0f} tokens/sec")

        elapsed = time.time() - start_time
        print(f"✓ Completed in {elapsed:.2f}s ({vocab_size/elapsed:.0f} tokens/sec)")

        return positions


# ============================================================================
# MULTI-SCALE EXPERT COMPANION
# ============================================================================

class MultiScaleExpertCompanion(nn.Module):
    """Expert companion with per-scale hidden dimensions."""

    def __init__(
        self,
        layer_name: str,
        layer_idx: int,
        input_dim: int,
        pentachoron_dim: int,
        scales: List[int],
        scale_hidden_dims: Dict[int, int],
        num_heads: int,
        dropout: float,
        shared_pentachora: torch.Tensor,
        fusion_controller: OrganizedFusionController,
        max_seq_len: int = 512,
        use_checkpoint: bool = False
    ):
        super().__init__()

        self.layer_name = layer_name
        self.layer_idx = layer_idx
        self.input_dim = input_dim
        self.pentachoron_dim = pentachoron_dim
        self.scales = scales
        self.use_checkpoint = use_checkpoint
        self.fusion_controller = fusion_controller

        self.register_buffer('shared_pentachora', shared_pentachora)

        pentachora_centroids = shared_pentachora.mean(dim=1)
        self.register_buffer('pentachora_centroids', F.normalize(pentachora_centroids, dim=-1))

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, pentachoron_dim),
            nn.LayerNorm(pentachoron_dim),
            nn.GELU()
        )

        cantor_config = CantorAttentionConfig(
            dim=pentachoron_dim,
            num_heads=num_heads,
            adaptive_window=True,
            sparsity_target=0.15,
            dropout=dropout
        )

        k_neighbors = int(225 * 0.15)
        k_neighbors = max(16, min(k_neighbors, 64))

        self.cantor_attention = CantorAttention(
            cantor_config,
            max_seq_len=max_seq_len,
            k=k_neighbors
        )

        self.scale_projectors = nn.ModuleDict()
        for i, scale in enumerate(scales):
            hidden_dim = scale_hidden_dims.get(scale, scale * 2)

            self.scale_projectors[str(scale)] = nn.Sequential(
                nn.Linear(pentachoron_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, scale)
            )

        self._init_weights()

    def _init_weights(self):
        """Efficient weight initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def match_to_opinion_anchors(self, features: torch.Tensor) -> torch.Tensor:
        """Match tokens to nearest pentachoron anchors."""
        similarities = torch.matmul(features, self.pentachora_centroids.T)
        return similarities.argmax(dim=-1)

    def _forward_impl(
        self,
        sequence_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Implementation for checkpointing."""
        z = self.input_proj(sequence_features)
        z = F.normalize(z, dim=-1)

        anchor_ids = self.match_to_opinion_anchors(z)

        alpha_values = []
        for scale_idx in range(len(self.scales)):
            alpha = self.fusion_controller.get_alpha(self.layer_idx, scale_idx)
            alpha_values.append(alpha)
        alpha_mean = sum(alpha_values) / len(alpha_values) if alpha_values else 0.0

        z_attended = self.cantor_attention(z, anchor_ids, alpha=alpha_mean)

        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).float()
            z_pooled = (z_attended * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        else:
            z_pooled = z_attended.mean(dim=1)

        scale_opinions = {}
        for scale in self.scales:
            opinion = self.scale_projectors[str(scale)](z_pooled)
            scale_opinions[scale] = opinion

        return {
            'scale_opinions': scale_opinions,
            'pooled_features': z_pooled
        }

    def forward(
        self,
        sequence_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward with optional gradient checkpointing."""
        if self.training and self.use_checkpoint:
            return checkpoint(
                self._forward_impl,
                sequence_features,
                attention_mask,
                use_reentrant=False
            )
        return self._forward_impl(sequence_features, attention_mask)


# ============================================================================
# SHALLOW FUSION WITH CACHED EMBEDDINGS
# ============================================================================

class ShallowTokenPredictionFusion(nn.Module):
    """Shallow fusion with pre-cached position embeddings."""

    def __init__(
        self,
        num_experts: int,
        scales: List[int],
        vocab_size: int,
        fusion_controller: OrganizedFusionController,
        max_seq_len: int = 77,
        num_heads: int = 8,
        dropout: float = 0.1,
        share_embeddings: bool = True
    ):
        super().__init__()

        self.num_experts = num_experts
        self.scales = scales
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.share_embeddings = share_embeddings
        self.fusion_controller = fusion_controller

        if share_embeddings:
            base_dim = max(scales)
            self.shared_position_embeds = nn.Parameter(torch.randn(max_seq_len, base_dim))
            nn.init.normal_(self.shared_position_embeds, std=0.02)
            print(f"\n  💾 Shared embeddings: {max_seq_len * base_dim:,} params")

            # PRE-CACHE position embedding slices for each scale
            self._cached_pos_embeds = {}
            print(f"  🔥 Pre-caching position embeddings for {len(scales)} scales...")
            for scale in scales:
                self._cached_pos_embeds[scale] = None  # Will be populated in forward
        else:
            self._cached_pos_embeds = {}
            for scale in scales:
                position_embeds = nn.Parameter(torch.randn(max_seq_len, scale))
                nn.init.normal_(position_embeds, std=0.02)
                self.register_parameter(f'position_embeds_{scale}', position_embeds)
                self._cached_pos_embeds[scale] = None

        self.scale_output_modules = nn.ModuleDict()

        print("\nInitializing optimized multi-level Cantor output scaffolding...")
        for scale in scales:
            print(f"  Scale {scale}:")

            cantor_config = CantorAttentionConfig(
                dim=scale,
                num_heads=num_heads,
                adaptive_window=False,
                local_window=32,
                dropout=dropout
            )

            k_neighbors = min(32, max_seq_len // 2)
            cantor_attn = CantorAttention(
                cantor_config,
                max_seq_len=max_seq_len,
                k=k_neighbors
            )
            self.scale_output_modules[f'cantor_{scale}'] = cantor_attn
            print(f"    ✓ Cantor attention: k={k_neighbors} neighbors")

            vocab_head = nn.Sequential(
                nn.LayerNorm(scale),
                nn.Linear(scale, vocab_size)
            )
            self.scale_output_modules[f'vocab_{scale}'] = vocab_head

        self._init_weights()

    def _init_weights(self):
        """Efficient initialization."""
        def init_module(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.apply(init_module)

    def _get_position_embeds(self, scale: int) -> torch.Tensor:
        """Get cached position embeddings (no slicing in forward!)."""
        # Use cached slice if available
        if self._cached_pos_embeds[scale] is not None:
            return self._cached_pos_embeds[scale]

        # Create and cache the slice
        if self.share_embeddings:
            # Slice and cache (gradient flows through shared_position_embeds)
            pos_embeds = self.shared_position_embeds[:, :scale]
        else:
            pos_embeds = getattr(self, f'position_embeds_{scale}')

        # Cache for future use
        self._cached_pos_embeds[scale] = pos_embeds
        return pos_embeds

    def forward(
        self,
        expert_opinions: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """Multi-level fusion with zero slicing overhead."""
        batch_size = list(expert_opinions[0]['scale_opinions'].values())[0].shape[0]

        # Get layer weights once
        layer_weights = self.fusion_controller.get_layer_weights()
        weights_view = layer_weights.view(-1, 1, 1)

        scale_token_logits = {}
        scale_feature_pairs = {}

        for scale_idx, scale in enumerate(self.scales):
            # Stack expert opinions
            expert_ops = torch.stack([
                exp['scale_opinions'][scale]
                for exp in expert_opinions
            ], dim=0)

            # Weighted fusion
            collective_opinion = (expert_ops * weights_view).sum(dim=0)

            # Get PRE-CACHED position embeddings (zero overhead!)
            position_embeds = self._get_position_embeds(scale)
            token_features = collective_opinion.unsqueeze(1) + position_embeds.unsqueeze(0)

            # Get beta for this scale
            beta = self.fusion_controller.get_beta(scale_idx)

            # Geometric processing with CACHED routes
            cantor_attn = self.scale_output_modules[f'cantor_{scale}']
            geometric_features = cantor_attn(token_features)

            # Blend
            attended_features = token_features.mul(1 - beta).add_(geometric_features.mul(beta))

            # Store for beta loss computation in trainer
            if self.training:
                scale_feature_pairs[scale] = {
                    'token_features': token_features,
                    'geometric_features': geometric_features,
                    'beta': beta
                }

            # Vocabulary projection
            vocab_head = self.scale_output_modules[f'vocab_{scale}']
            token_logits = vocab_head(attended_features)

            scale_token_logits[scale] = token_logits

        # Final scale fusion
        scale_weights = self.fusion_controller.get_scale_weights()
        logits_list = [scale_token_logits[scale] for scale in self.scales]
        logits_stacked = torch.stack(logits_list, dim=0)
        weights_expanded = scale_weights.view(-1, 1, 1, 1)
        final_token_logits = (logits_stacked * weights_expanded).sum(dim=0)

        return {
            'token_logits': final_token_logits,
            'scale_token_logits': scale_token_logits,
            'scale_feature_pairs': scale_feature_pairs if self.training else {}
        }


# ============================================================================
# LIMINAL STAIRCASE WITH GEOMETRIC INITIALIZATION
# ============================================================================

class LiminalStaircase(nn.Module):
    """Liminal Staircase with organized fusion controller and geometric pentachora."""

    def __init__(self, config: LiminalStaircaseConfig):
        super().__init__()

        self.config = config

        print("=" * 80)
        print("LIMINAL STAIRCASE - With Geometric Pentachora + Organized Fusion")
        print("=" * 80)
        print(f"Opinion anchors: {config.num_opinion_anchors}")
        print(f"SigLIP layers: {len(config.siglip_layer_indices)} (of {config.siglip_num_layers})")
        print(f"CLIP layers: {len(config.clip_layer_indices)} (of {config.clip_num_layers - config.clip_skip})")
        print(f"Scales: {config.scale_fusion.scales}")
        print(f"Scale hidden dims: {config.scale_fusion.scale_hidden_dims}")

        # Initialize fusion controller FIRST
        num_layers = len(config.siglip_layer_indices) + len(config.clip_layer_indices)
        self.fusion_controller = OrganizedFusionController(
            num_layers=num_layers,
            config=config.scale_fusion,
            learn_layer_weights=config.scale_fusion.learn_layer_weights
        )

        # Initialize opinion anchors using GEOMETRIC METHOD
        print("\n🔷 Initializing GEOMETRIC opinion anchors...")
        self.opinion_anchors = self._init_opinion_anchors_geometric()
        print(f"✓ {config.num_opinion_anchors} geometric pentachora created")

        # Compute positional fingerprints
        print("\nComputing positional fingerprints...")
        self.fingerprinter = GeometricPositionalFingerprinter(
            cantor_depth=config.cantor_depth,
            volume_scale=config.geometry_volume_scale,
            volume_weight=config.geometry_volume_weight,
            edge_weight=config.geometry_edge_weight,
            spread_weight=config.geometry_spread_weight,
            epsilon=config.geometry_epsilon
        )
        self.anchor_positions = self.fingerprinter.compute_vocabulary_positions(
            self.opinion_anchors,
            batch_size=512
        )
        print(f"✓ Positions: [{self.anchor_positions.min():.4f}, {self.anchor_positions.max():.4f}]")

        # SigLIP Vision experts
        print("\nCreating SigLIP Vision experts...")
        self.siglip_experts = nn.ModuleDict()
        for expert_idx, layer_idx in enumerate(config.siglip_layer_indices):
            expert = MultiScaleExpertCompanion(
                layer_name=f'siglip_layer_{layer_idx}',
                layer_idx=expert_idx,
                input_dim=config.siglip_hidden_dim,
                pentachoron_dim=config.pentachoron_dim,
                scales=config.scale_fusion.scales,
                scale_hidden_dims=config.scale_fusion.scale_hidden_dims,
                num_heads=config.num_heads,
                dropout=config.dropout,
                shared_pentachora=self.opinion_anchors,
                fusion_controller=self.fusion_controller,
                max_seq_len=512,
                use_checkpoint=config.use_gradient_checkpointing
            )
            self.siglip_experts[f'siglip_layer_{layer_idx}'] = expert

            if (expert_idx + 1) % 6 == 0 or (expert_idx + 1) == len(config.siglip_layer_indices):
                print(f"  ✓ Created {expert_idx + 1}/{len(config.siglip_layer_indices)} experts")

        # CLIP Text experts
        print(f"\nCreating CLIP Text experts...")
        self.clip_experts = nn.ModuleDict()
        num_siglip_experts = len(config.siglip_layer_indices)
        for expert_idx, layer_idx in enumerate(config.clip_layer_indices):
            expert = MultiScaleExpertCompanion(
                layer_name=f'clip_layer_{layer_idx}',
                layer_idx=num_siglip_experts + expert_idx,
                input_dim=config.clip_hidden_dim,
                pentachoron_dim=config.pentachoron_dim,
                scales=config.scale_fusion.scales,
                scale_hidden_dims=config.scale_fusion.scale_hidden_dims,
                num_heads=config.num_heads,
                dropout=config.dropout,
                shared_pentachora=self.opinion_anchors,
                fusion_controller=self.fusion_controller,
                max_seq_len=512,
                use_checkpoint=config.use_gradient_checkpointing
            )
            self.clip_experts[f'clip_layer_{layer_idx}'] = expert
        print(f"  ✓ {len(self.clip_experts)} CLIP text experts")

        # Shallow fusion
        print("\nCreating optimized shallow token prediction fusion...")
        total_experts = len(self.siglip_experts) + len(self.clip_experts)
        self.fusion = ShallowTokenPredictionFusion(
            num_experts=total_experts,
            scales=config.scale_fusion.scales,
            vocab_size=config.vocab_size,
            fusion_controller=self.fusion_controller,
            max_seq_len=config.max_seq_len,
            dropout=config.dropout,
            share_embeddings=config.share_scale_embeddings
        )
        print(f"✓ Fusion: {total_experts} experts → {config.max_seq_len} tokens")

        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"\n{'='*80}")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"{'='*80}\n")

    def _init_opinion_anchors_geometric(self) -> nn.Parameter:
        """Initialize opinion anchors using SimplexFactory for proper 4-simplices."""

        method = self.config.geometric_init_method

        print(f"  Method: {method}")
        print(f"  Dimension: {self.config.pentachoron_dim}")
        print(f"  Scale: {self.config.geometric_init_scale}")
        print(f"  Normalize: {self.config.geometric_init_normalize}")
        print(f"  Validate: {self.config.geometric_init_validate}")

        start_time = time.time()

        if method == "hybrid":
            # Hybrid: mix of regular, random, and uniform
            num_regular = self.config.num_opinion_anchors // 3
            num_random = self.config.num_opinion_anchors // 2
            num_uniform = self.config.num_opinion_anchors - num_regular - num_random

            pentachora_parts = []
            offset = 0

            # Regular
            for method_name, count in [("regular", num_regular), ("random", num_random), ("uniform", num_uniform)]:
                if count > 0:
                    factory = SimplexFactory(
                        k=4,
                        embed_dim=self.config.pentachoron_dim,
                        method=method_name,
                        scale=self.config.geometric_init_scale,
                        seed=self.config.geometric_init_seed
                    )

                    part_pentachora = []
                    for i in range(count):
                        anchor_seed = (self.config.geometric_init_seed + offset + i) if self.config.geometric_init_seed is not None else None
                        pentachoron = factory.build(
                            backend="torch",
                            device="cpu",
                            seed=anchor_seed,
                            validate=self.config.geometric_init_validate
                        )
                        part_pentachora.append(pentachoron)

                    pentachora_parts.append(torch.stack(part_pentachora, dim=0))
                    offset += count
                    print(f"    ✓ Generated {count} {method_name} pentachora")

            pentachora = torch.cat(pentachora_parts, dim=0)

        else:
            # Single method
            factory = SimplexFactory(
                k=4,
                embed_dim=self.config.pentachoron_dim,
                method=method,
                scale=self.config.geometric_init_scale,
                seed=self.config.geometric_init_seed
            )

            pentachora_list = []
            valid_count = 0

            for i in range(self.config.num_opinion_anchors):
                anchor_seed = (self.config.geometric_init_seed + i) if self.config.geometric_init_seed is not None else None

                pentachoron = factory.build(
                    backend="torch",
                    device="cpu",
                    seed=anchor_seed,
                    validate=self.config.geometric_init_validate
                )

                if self.config.geometric_init_validate:
                    is_valid, msg = factory.validate(pentachoron)
                    if is_valid:
                        valid_count += 1
                    else:
                        print(f"    ⚠️  Anchor {i}: {msg}")

                pentachora_list.append(pentachoron)

                if (i + 1) % 50 == 0:
                    elapsed = time.time() - start_time
                    rate = (i + 1) / elapsed
                    print(f"    Generated {i + 1}/{self.config.num_opinion_anchors} ({rate:.0f} anchors/sec)")

            pentachora = torch.stack(pentachora_list, dim=0)

            if self.config.geometric_init_validate:
                print(f"    Valid: {valid_count}/{self.config.num_opinion_anchors} ({100*valid_count/self.config.num_opinion_anchors:.1f}%)")

        # Normalize if requested
        if self.config.geometric_init_normalize:
            pentachora = F.normalize(pentachora, dim=-1)

        elapsed = time.time() - start_time

        # Statistics
        centroid_norms = torch.norm(pentachora.mean(dim=1), dim=-1)
        edge_lengths = torch.norm(pentachora[:, 1] - pentachora[:, 0], dim=-1)

        print(f"    Time: {elapsed:.2f}s")
        print(f"    Shape: {pentachora.shape}")
        print(f"    Range: [{pentachora.min():.4f}, {pentachora.max():.4f}]")
        print(f"    Centroid norms: mean={centroid_norms.mean():.4f}, std={centroid_norms.std():.4f}")
        print(f"    Edge lengths: mean={edge_lengths.mean():.4f}, std={edge_lengths.std():.4f}")

        return nn.Parameter(pentachora, requires_grad=True)

    def forward(
        self,
        siglip_features: Dict[str, torch.Tensor],
        clip_features: Optional[Dict[str, torch.Tensor]] = None,
        siglip_masks: Optional[Dict[str, torch.Tensor]] = None,
        clip_masks: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """Predict tokens from vision and optional text."""
        expert_opinions = []

        for layer_name, features in siglip_features.items():
            if layer_name in self.siglip_experts:
                mask = siglip_masks.get(layer_name) if siglip_masks else None
                opinion = self.siglip_experts[layer_name](features, mask)
                expert_opinions.append(opinion)

        if clip_features is not None:
            for layer_name, features in clip_features.items():
                if layer_name in self.clip_experts:
                    mask = clip_masks.get(layer_name) if clip_masks else None
                    opinion = self.clip_experts[layer_name](features, mask)
                    expert_opinions.append(opinion)

        output = self.fusion(expert_opinions)

        # Add fusion diagnostics
        output['fusion_diagnostics'] = self.fusion_controller.get_diagnostics()

        return output

    def get_info(self) -> Dict:
        """Get model info."""
        return {
            'num_opinion_anchors': self.config.num_opinion_anchors,
            'siglip_experts': len(self.siglip_experts),
            'siglip_layer_indices': self.config.siglip_layer_indices,
            'clip_experts': len(self.clip_experts),
            'clip_layer_indices': self.config.clip_layer_indices,
            'total_experts': len(self.siglip_experts) + len(self.clip_experts),
            'vocab_size': self.config.vocab_size,
            'max_seq_len': self.config.max_seq_len,
            'scales': self.config.scale_fusion.scales,
            'scale_hidden_dims': self.config.scale_fusion.scale_hidden_dims,
            'geometric_init_method': self.config.geometric_init_method,
            'fusion_controller': {
                'alpha_learnable': self.config.scale_fusion.alpha_learnable,
                'beta_learnable': self.config.scale_fusion.beta_learnable,
                'learn_layer_weights': self.config.scale_fusion.learn_layer_weights,
                'learn_scale_weights': self.config.scale_fusion.learn_scale_weights,
            },
            'total_parameters': sum(p.numel() for p in self.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("LIMINAL STAIRCASE - GEOMETRIC INITIALIZATION TEST")
    print("=" * 80 + "\n")

    # Test with ScaleFusionConfig
    fusion_config = ScaleFusionConfig(
        scales=[128, 256, 512],
        scale_hidden_dims={128: 256, 256: 512, 512: 1024},
        alpha_learnable=True,
        beta_learnable=True,
        gamma_learnable=True
    )

    config = LiminalStaircaseConfig(
        num_opinion_anchors=225,
        pentachoron_dim=512,
        siglip_num_layers=24,
        clip_num_layers=12,
        clip_skip=2,
        vocab_size=49408,
        max_seq_len=77,
        siglip_layer_indices=list(range(12, 24)),
        scale_fusion=fusion_config,
        geometric_init_method="hybrid",  # Test hybrid initialization
        geometric_init_validate=True,    # Validate pentachora
        geometric_init_seed=42
    )

    model = LiminalStaircase(config)

    batch_size = 2
    siglip_features = {
        f'siglip_layer_{i}': torch.randn(batch_size, 256, 1664)
        for i in range(12, 24)
    }
    clip_features = {
        f'clip_layer_{i}': torch.randn(batch_size, 77, 768)
        for i in range(10)
    }

    print(f"\n[Forward pass test]")
    with torch.no_grad():
        output = model(siglip_features, clip_features)

    print(f"✓ Token logits shape: {output['token_logits'].shape}")
    print(f"✓ Fusion diagnostics keys: {list(output['fusion_diagnostics'].keys())}")

    info = model.get_info()
    print(f"\n[Model info]")
    print(f"  Total experts: {info['total_experts']}")
    print(f"  Parameters: {info['total_parameters']:,}")
    print(f"  Geometric init: {info['geometric_init_method']}")

    print("\n" + "=" * 80)
    print("✅ TEST COMPLETE - Geometric pentachora + organized fusion ready!")
    print("=" * 80 + "\n")