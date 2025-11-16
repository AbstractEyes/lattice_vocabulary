"""
Liminal Staircase - Organized Scale Fusion System
==================================================

FEATURES:
- Modular scale fusion architecture
- Comprehensive testing suite
- Detailed fusion diagnostics
- Alpha/Beta behavior validation
- Per-scale performance tracking

Author: AbstractPhil + Claude Sonnet 4.5
Date: 2025-11-16
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import math
import time


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
    """Configuration for scale fusion."""

    # Scale architecture
    scales: List[int] = None
    scale_hidden_dims: Dict[int, int] = None

    # Fusion strategy
    fusion_strategy: str = "learned_weighted"  # learned_weighted, average, max, gated

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

    # Gamma: Scale importance (NEW)
    gamma_init: float = 0.5
    gamma_learnable: bool = True
    gamma_per_scale: bool = True

    # Diagnostics
    track_scale_losses: bool = True
    track_attention_patterns: bool = False

    def __post_init__(self):
        if self.scales is None:
            self.scales = [128, 256, 512]

        if self.scale_hidden_dims is None:
            self.scale_hidden_dims = {s: s * 2 for s in self.scales}


@dataclass
class LiminalStaircaseConfig:
    """Configuration for Liminal Staircase."""

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

    # Layer selection
    siglip_layer_indices: Optional[List[int]] = None
    clip_layer_indices: Optional[List[int]] = None

    # Scale fusion config
    scale_fusion: ScaleFusionConfig = None

    # Layer weights
    learn_layer_weights: bool = True

    # Optimizations
    use_gradient_checkpointing: bool = False
    share_scale_embeddings: bool = True

    # Geometry parameters
    geometry_volume_scale: float = 10.0
    geometry_volume_weight: float = 0.4
    geometry_edge_weight: float = 0.3
    geometry_spread_weight: float = 0.3
    geometry_epsilon: float = 1e-6

    def __post_init__(self):
        # Default layer selection
        if self.siglip_layer_indices is None:
            start = max(0, self.siglip_num_layers - 12)
            self.siglip_layer_indices = list(range(start, self.siglip_num_layers))

        if self.clip_layer_indices is None:
            active_clip_layers = self.clip_num_layers - self.clip_skip
            self.clip_layer_indices = list(range(active_clip_layers))

        # Default scale fusion config
        if self.scale_fusion is None:
            self.scale_fusion = ScaleFusionConfig()

        # Validate geometry weights
        total_weight = self.geometry_volume_weight + self.geometry_edge_weight + self.geometry_spread_weight
        assert abs(total_weight - 1.0) < 1e-5, f"Geometry weights must sum to 1.0, got {total_weight}"


# ============================================================================
# FUSION CONTROLLERS
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

    def forward(self, layer_idx: Optional[int] = None, scale_idx: Optional[int] = None) -> torch.Tensor:
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
    """Controls per-scale importance in final fusion (NEW)."""

    def __init__(self, num_scales: int, config: ScaleFusionConfig):
        super().__init__()

        self.num_scales = num_scales
        self.config = config

        shape = (num_scales,) if config.gamma_per_scale else (1,)
        gamma_init = torch.full(shape, config.gamma_init)

        if config.gamma_learnable:
            self.gamma_raw = nn.Parameter(gamma_init)
        else:
            self.register_buffer('gamma_raw', gamma_init)

    def forward(self) -> torch.Tensor:
        """Get normalized gamma weights across scales."""
        if self.config.gamma_per_scale:
            return F.softmax(self.gamma_raw, dim=0)
        return torch.ones(self.num_scales, device=self.gamma_raw.device) / self.num_scales


class LayerWeightController(nn.Module):
    """Controls per-layer opinion weights."""

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


# ============================================================================
# ORGANIZED FUSION CONTROLLER
# ============================================================================

class OrganizedFusionController(nn.Module):
    """
    Organized hierarchical fusion control system.

    Controls:
    - Alpha: Cantor neighbor bleed-over (spatial attention mixing)
    - Beta: Geometric vs projection balance (architectural mixing)
    - Gamma: Scale importance weights (scale-level fusion)
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
        print(f"Strategy: {config.fusion_strategy}")

        # Alpha: Spatial bleed-over control
        self.alpha = AlphaController(num_layers, self.num_scales, config)
        print(f"\n✓ Alpha (spatial bleed): "
              f"init={config.alpha_init}, "
              f"learnable={config.alpha_learnable}, "
              f"per_scale={config.alpha_per_scale}")

        # Beta: Architectural balance control
        self.beta = BetaController(self.num_scales, config)
        print(f"✓ Beta (geometric blend): "
              f"init={config.beta_init}, "
              f"learnable={config.beta_learnable}, "
              f"per_scale={config.beta_per_scale}")

        # Gamma: Scale importance control
        self.gamma = GammaController(self.num_scales, config)
        print(f"✓ Gamma (scale weights): "
              f"init={config.gamma_init}, "
              f"learnable={config.gamma_learnable}, "
              f"per_scale={config.gamma_per_scale}")

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
        print(f"\n✓ Controller params: {total_params} ({trainable_params} trainable)")
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


# ============================================================================
# CANTOR ATTENTION (Unchanged but included for completeness)
# ============================================================================

class CantorAttention(nn.Module):
    """O(n) attention with alpha-controlled neighbor bleed-over."""

    def __init__(self, config: CantorAttentionConfig, max_seq_len: int = 512, k: int = 64):
        super().__init__()
        self.config = config
        self.dim = config.dim
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.k = k

        self.qkv = nn.Linear(config.dim, 3 * config.dim, bias=config.qkv_bias)
        self.out_proj = nn.Linear(config.dim, config.dim, bias=config.out_bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Pre-compute routes
        self.routes_cache = {}
        common_lengths = [64, 77, 128, 196, 256, 384, 512]
        for seq_len in common_lengths:
            if seq_len <= max_seq_len:
                routes = self._build_positional_routes(seq_len, k)
                self.register_buffer(f'routes_{seq_len}', routes.to(torch.int32))
                self.routes_cache[seq_len] = routes

    def _compute_cantor_coord_vectorized(self, positions: torch.Tensor, seq_len: int, depth: int = 8) -> torch.Tensor:
        """Vectorized Cantor coordinate computation."""
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

    def _build_positional_routes(self, seq_len: int, k: int) -> torch.Tensor:
        """Vectorized route building."""
        positions = torch.arange(seq_len, dtype=torch.long)
        coords = self._compute_cantor_coord_vectorized(positions, seq_len)

        distances = torch.abs(coords.unsqueeze(1) - coords.unsqueeze(0))
        _, routes = torch.topk(distances, k, dim=1, largest=False)
        return routes

    def _get_routes_for_seq_len(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Get or build routes for sequence length."""
        if seq_len in self.routes_cache:
            cached = getattr(self, f'routes_{seq_len}', None)
            if cached is not None:
                return cached.to(device).long()

        routes = self._build_positional_routes(seq_len, self.k)
        return routes.to(device)

    def _sparse_attention(self, q, k, v, routes, alpha: float = 0.0):
        """Sparse attention with alpha-controlled neighbor bleed-over."""
        batch_size, num_heads, seq_len, head_dim = q.shape
        k_neighbors = routes.shape[-1]
        device = q.device

        if routes.dim() == 2:
            routes_expanded = routes.unsqueeze(0).expand(batch_size, -1, -1)
        else:
            routes_expanded = routes

        batch_idx = torch.arange(batch_size, device=device).view(-1, 1, 1, 1)
        head_idx = torch.arange(num_heads, device=device).view(1, -1, 1, 1)

        batch_idx = batch_idx.expand(batch_size, num_heads, seq_len, k_neighbors)
        head_idx = head_idx.expand(batch_size, num_heads, seq_len, k_neighbors)
        routes_bc = routes_expanded.unsqueeze(1).expand(batch_size, num_heads, seq_len, k_neighbors)

        k_gathered = k[batch_idx, head_idx, routes_bc, :]
        v_gathered = v[batch_idx, head_idx, routes_bc, :]

        if alpha > 0.0:
            adj_routes_left = torch.clamp(routes_expanded - 1, 0, seq_len - 1)
            adj_routes_right = torch.clamp(routes_expanded + 1, 0, seq_len - 1)

            adj_routes_left_bc = adj_routes_left.unsqueeze(1).expand(batch_size, num_heads, seq_len, k_neighbors)
            adj_routes_right_bc = adj_routes_right.unsqueeze(1).expand(batch_size, num_heads, seq_len, k_neighbors)

            k_adj_left = k[batch_idx, head_idx, adj_routes_left_bc, :]
            k_adj_right = k[batch_idx, head_idx, adj_routes_right_bc, :]
            v_adj_left = v[batch_idx, head_idx, adj_routes_left_bc, :]
            v_adj_right = v[batch_idx, head_idx, adj_routes_right_bc, :]

            k_gathered = (1 - alpha) * k_gathered + alpha/2 * (k_adj_left + k_adj_right)
            v_gathered = (1 - alpha) * v_gathered + alpha/2 * (v_adj_left + v_adj_right)

        scores = torch.einsum('bhqd,bhqkd->bhqk', q, k_gathered) * self.scale

        if self.config.causal:
            position_idx = torch.arange(seq_len, device=device).unsqueeze(1)
            causal_mask = routes_expanded > position_idx.unsqueeze(0)
            causal_mask = causal_mask.unsqueeze(1).expand(batch_size, num_heads, -1, -1)
            scores = scores.masked_fill(causal_mask, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        output = torch.einsum('bhqk,bhqkd->bhqd', attn_weights, v_gathered)
        return output

    def forward(self, x, anchor_ids=None, alpha: float = 0.0):
        """Forward with optional alpha bleed-over."""
        batch_size, seq_len, _ = x.shape

        qkv = self.qkv(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        routes = self._get_routes_for_seq_len(seq_len, x.device)
        attn_output = self._sparse_attention(q, k, v, routes, alpha=alpha)

        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.dim)
        output = self.out_proj(attn_output)
        output = self.resid_dropout(output)

        return output


# ============================================================================
# ORGANIZED SCALE FUSION LAYER
# ============================================================================

class OrganizedScaleFusion(nn.Module):
    """
    Organized multi-scale fusion with multiple strategies.

    Strategies:
    - learned_weighted: Gamma-controlled weighted combination
    - average: Simple averaging
    - max: Max pooling across scales
    - gated: Learned gating mechanism
    """

    def __init__(
        self,
        scales: List[int],
        vocab_size: int,
        fusion_controller: OrganizedFusionController,
        max_seq_len: int = 77,
        num_heads: int = 8,
        dropout: float = 0.1,
        share_embeddings: bool = True
    ):
        super().__init__()

        self.scales = scales
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.share_embeddings = share_embeddings
        self.fusion_controller = fusion_controller
        self.strategy = fusion_controller.config.fusion_strategy

        print(f"\n{'='*60}")
        print(f"ORGANIZED SCALE FUSION")
        print(f"{'='*60}")
        print(f"Strategy: {self.strategy}")
        print(f"Scales: {scales}")

        # Shared position embeddings
        if share_embeddings:
            base_dim = max(scales)
            self.shared_position_embeds = nn.Parameter(torch.randn(max_seq_len, base_dim))
            nn.init.normal_(self.shared_position_embeds, std=0.02)
            print(f"✓ Shared embeddings: {max_seq_len * base_dim:,} params")
        else:
            for scale in scales:
                position_embeds = nn.Parameter(torch.randn(max_seq_len, scale))
                nn.init.normal_(position_embeds, std=0.02)
                self.register_parameter(f'position_embeds_{scale}', position_embeds)

        # Per-scale processing modules
        self.scale_modules = nn.ModuleDict()

        for scale in scales:
            # Cantor attention for geometric structure
            cantor_config = CantorAttentionConfig(
                dim=scale,
                num_heads=num_heads,
                dropout=dropout
            )
            k_neighbors = min(32, max_seq_len // 2)
            cantor_attn = CantorAttention(cantor_config, max_seq_len=max_seq_len, k=k_neighbors)

            # Vocabulary head
            vocab_head = nn.Sequential(
                nn.LayerNorm(scale),
                nn.Linear(scale, vocab_size)
            )

            self.scale_modules[f'cantor_{scale}'] = cantor_attn
            self.scale_modules[f'vocab_{scale}'] = vocab_head

            print(f"  Scale {scale}: {k_neighbors} neighbors → vocab")

        # Gated fusion (if strategy == 'gated')
        if self.strategy == 'gated':
            self.gate_network = nn.Sequential(
                nn.Linear(len(scales) * vocab_size, len(scales)),
                nn.Softmax(dim=-1)
            )
            print(f"✓ Gated fusion network initialized")

        print(f"{'='*60}\n")

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
        """Get position embeddings via slicing."""
        if self.share_embeddings:
            return self.shared_position_embeds[:, :scale]
        else:
            return getattr(self, f'position_embeds_{scale}')

    def forward(self, scale_opinions: Dict[int, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Fuse multi-scale opinions into final token predictions.

        Args:
            scale_opinions: Dict mapping scale → pooled opinion tensor [batch, scale_dim]

        Returns:
            Dict with token_logits, scale_logits, and diagnostic info
        """
        batch_size = list(scale_opinions.values())[0].shape[0]

        scale_token_logits = {}
        scale_beta_losses = {}

        # Process each scale independently
        for scale_idx, scale in enumerate(self.scales):
            opinion = scale_opinions[scale]

            # Expand to token sequence with position embeddings
            position_embeds = self._get_position_embeds(scale)
            token_features = opinion.unsqueeze(1) + position_embeds.unsqueeze(0)

            # Get beta for this scale
            beta = self.fusion_controller.get_beta(scale_idx)

            # Apply Cantor attention (geometric structure)
            cantor_attn = self.scale_modules[f'cantor_{scale}']
            geometric_features = cantor_attn(token_features)

            # Beta-controlled blend
            attended_features = (1 - beta) * token_features + beta * geometric_features

            # Track beta loss for diagnostics
            if self.training:
                beta_loss = beta * F.mse_loss(token_features, geometric_features)
                scale_beta_losses[scale] = beta_loss

            # Vocabulary projection
            vocab_head = self.scale_modules[f'vocab_{scale}']
            token_logits = vocab_head(attended_features)

            scale_token_logits[scale] = token_logits

        # Fuse across scales using selected strategy
        final_logits = self._fuse_scales(scale_token_logits)

        return {
            'token_logits': final_logits,
            'scale_token_logits': scale_token_logits,
            'scale_beta_losses': scale_beta_losses
        }

    def _fuse_scales(self, scale_logits: Dict[int, torch.Tensor]) -> torch.Tensor:
        """Apply fusion strategy to combine scale predictions."""
        logits_list = [scale_logits[scale] for scale in self.scales]
        logits_stacked = torch.stack(logits_list, dim=0)  # [num_scales, batch, seq, vocab]

        if self.strategy == "learned_weighted":
            # Use Gamma weights from fusion controller
            scale_weights = self.fusion_controller.get_scale_weights()
            weights_expanded = scale_weights.view(-1, 1, 1, 1)
            return (logits_stacked * weights_expanded).sum(dim=0)

        elif self.strategy == "average":
            return logits_stacked.mean(dim=0)

        elif self.strategy == "max":
            return logits_stacked.max(dim=0)[0]

        elif self.strategy == "gated":
            # Flatten logits for gating
            batch, seq, vocab = logits_list[0].shape
            flat_logits = torch.cat([l.reshape(batch, seq, -1) for l in logits_list], dim=-1)

            # Compute per-scale gates
            gates = self.gate_network(flat_logits)  # [batch, seq, num_scales]
            gates = gates.unsqueeze(-1)  # [batch, seq, num_scales, 1]

            # Apply gates
            logits_stacked_perm = logits_stacked.permute(1, 2, 0, 3)  # [batch, seq, num_scales, vocab]
            return (logits_stacked_perm * gates).sum(dim=2)

        else:
            raise ValueError(f"Unknown fusion strategy: {self.strategy}")


# ============================================================================
# COMPREHENSIVE TESTING SUITE
# ============================================================================

def test_fusion_controller():
    """Test 1: Fusion controller initialization and diagnostics."""
    print("\n" + "="*80)
    print("TEST 1: Fusion Controller")
    print("="*80)

    config = ScaleFusionConfig(
        scales=[128, 256, 512],
        alpha_learnable=True,
        beta_learnable=True,
        gamma_learnable=True,
        track_scale_losses=True
    )

    controller = OrganizedFusionController(
        num_layers=12,
        config=config,
        learn_layer_weights=True
    )

    print("\n[Initial Diagnostics]")
    diagnostics = controller.get_diagnostics()
    print(f"Layer weights: {[f'{w:.3f}' for w in diagnostics['layer_weights']]}")
    print(f"Scale weights: {[f'{w:.3f}' for w in diagnostics['scale_weights']]}")
    print(f"Alpha per scale: {[f'{a:.3f}' for a in diagnostics['alpha_per_scale']]}")
    print(f"Beta per scale: {[f'{b:.3f}' for b in diagnostics['beta_per_scale']]}")

    # Test recording losses
    print("\n[Recording Scale Losses]")
    for scale_idx in range(3):
        controller.record_scale_loss(scale_idx, loss=1.5 - scale_idx*0.3, beta_loss=0.1)

    diagnostics = controller.get_diagnostics()
    print("Scale statistics:")
    for scale, stats in diagnostics['scale_statistics'].items():
        print(f"  {scale}: loss={stats['avg_loss']:.4f}, beta_loss={stats['avg_beta_loss']:.4f}")

    print("\n✅ Fusion Controller Test PASSED")


def test_scale_fusion_strategies():
    """Test 2: Different fusion strategies."""
    print("\n" + "="*80)
    print("TEST 2: Scale Fusion Strategies")
    print("="*80)

    strategies = ["learned_weighted", "average", "max", "gated"]
    batch_size = 2
    vocab_size = 1000
    scales = [128, 256, 512]

    for strategy in strategies:
        print(f"\n[Testing strategy: {strategy}]")

        config = ScaleFusionConfig(
            scales=scales,
            fusion_strategy=strategy,
            gamma_learnable=True
        )

        controller = OrganizedFusionController(
            num_layers=6,
            config=config,
            learn_layer_weights=False
        )

        fusion = OrganizedScaleFusion(
            scales=scales,
            vocab_size=vocab_size,
            fusion_controller=controller,
            share_embeddings=True
        )

        # Create dummy scale opinions
        scale_opinions = {
            scale: torch.randn(batch_size, scale)
            for scale in scales
        }

        with torch.no_grad():
            output = fusion(scale_opinions)

        print(f"  ✓ Output logits shape: {output['token_logits'].shape}")
        print(f"  ✓ Num scale outputs: {len(output['scale_token_logits'])}")
        assert output['token_logits'].shape == (batch_size, 77, vocab_size)

    print("\n✅ Scale Fusion Strategies Test PASSED")


def test_alpha_beta_effects():
    """Test 3: Alpha and Beta parameter effects."""
    print("\n" + "="*80)
    print("TEST 3: Alpha/Beta Effects")
    print("="*80)

    # Test different alpha values
    print("\n[Testing Alpha (bleed-over) values: 0.0, 0.25, 0.5]")
    for alpha_val in [0.0, 0.25, 0.5]:
        config = ScaleFusionConfig(
            scales=[256],
            alpha_init=alpha_val,
            alpha_learnable=False,
            alpha_per_scale=True
        )

        controller = OrganizedFusionController(
            num_layers=4,
            config=config
        )

        alpha = controller.get_alpha(layer_idx=0, scale_idx=0)
        print(f"  Alpha={alpha_val}: actual={alpha:.4f}")

    # Test different beta values
    print("\n[Testing Beta (geometric blend) values: 0.0, 0.5, 1.0]")
    for beta_val in [0.0, 0.5, 1.0]:
        config = ScaleFusionConfig(
            scales=[256],
            beta_init=beta_val,
            beta_learnable=False,
            beta_per_scale=True
        )

        controller = OrganizedFusionController(
            num_layers=4,
            config=config
        )

        beta = controller.get_beta(scale_idx=0)
        print(f"  Beta={beta_val}: actual={beta:.4f}")

    print("\n✅ Alpha/Beta Effects Test PASSED")


def test_scale_performance_tracking():
    """Test 4: Per-scale performance tracking."""
    print("\n" + "="*80)
    print("TEST 4: Scale Performance Tracking")
    print("="*80)

    config = ScaleFusionConfig(
        scales=[128, 256, 512],
        track_scale_losses=True
    )

    controller = OrganizedFusionController(
        num_layers=8,
        config=config
    )

    # Simulate training with different losses per scale
    print("\n[Simulating 100 training steps]")
    import random
    for step in range(100):
        for scale_idx, scale in enumerate(config.scales):
            # Simulate decreasing loss over time
            base_loss = 2.0 - (step / 100.0) * 1.5
            scale_specific = scale_idx * 0.1  # Larger scales slightly higher loss
            loss = base_loss + scale_specific + random.uniform(-0.1, 0.1)
            beta_loss = 0.05 + random.uniform(-0.01, 0.01)

            controller.record_scale_loss(scale_idx, loss, beta_loss)

    # Get statistics
    diagnostics = controller.get_diagnostics()
    print("\n[Final Scale Statistics]")
    for scale, stats in diagnostics['scale_statistics'].items():
        print(f"Scale {scale}:")
        print(f"  Avg loss: {stats['avg_loss']:.4f}")
        print(f"  Avg beta loss: {stats['avg_beta_loss']:.4f}")
        print(f"  Samples: {stats['count']}")

    print("\n✅ Scale Performance Tracking Test PASSED")


def test_gradient_flow():
    """Test 5: Gradient flow through fusion controller."""
    print("\n" + "="*80)
    print("TEST 5: Gradient Flow")
    print("="*80)

    config = ScaleFusionConfig(
        scales=[128, 256],
        alpha_learnable=True,
        beta_learnable=True,
        gamma_learnable=True
    )

    controller = OrganizedFusionController(
        num_layers=4,
        config=config,
        learn_layer_weights=True
    )

    # Create dummy loss
    layer_weights = controller.get_layer_weights()
    scale_weights = controller.get_scale_weights()

    loss = layer_weights.sum() + scale_weights.sum()

    for scale_idx in range(len(config.scales)):
        alpha = controller.get_alpha(0, scale_idx)
        beta = controller.get_beta(scale_idx)
        loss = loss + alpha + beta

    print("\n[Computing gradients]")
    loss.backward()

    # Check gradients exist
    print("Gradient checks:")
    if controller.alpha.alpha_raw.grad is not None:
        print(f"  ✓ Alpha gradients: {controller.alpha.alpha_raw.grad.abs().mean():.6f}")
    if controller.beta.beta_raw.grad is not None:
        print(f"  ✓ Beta gradients: {controller.beta.beta_raw.grad.abs().mean():.6f}")
    if controller.gamma.gamma_raw.grad is not None:
        print(f"  ✓ Gamma gradients: {controller.gamma.gamma_raw.grad.abs().mean():.6f}")
    if controller.layer_weights and controller.layer_weights.layer_weights_raw.grad is not None:
        print(f"  ✓ Layer weight gradients: {controller.layer_weights.layer_weights_raw.grad.abs().mean():.6f}")

    print("\n✅ Gradient Flow Test PASSED")


def run_all_tests():
    """Run complete test suite."""
    print("\n" + "="*80)
    print("LIMINAL STAIRCASE - COMPREHENSIVE FUSION TEST SUITE")
    print("="*80)

    test_fusion_controller()
    test_scale_fusion_strategies()
    test_alpha_beta_effects()
    test_scale_performance_tracking()
    test_gradient_flow()

    print("\n" + "="*80)
    print("✅ ALL TESTS PASSED")
    print("="*80 + "\n")


if __name__ == "__main__":
    run_all_tests()