"""
ViT-Beans: Multi-Mode Vision Transformer with Cantor Expert Attention
======================================================================

Enhanced with DIRECT TOKEN CONTROL for dense mode.

Supports 3 modes:
- 'dense': Dense masked Cantor attention (FAST - recommended)
  * NOW with explicit token/patch control via manual masks
- 'sparse': Original sparse Cantor attention (SLOW - for experiments)
- 'standard': Standard multi-head attention (BASELINE)

Author: AbstractPhil + Claude Sonnet 4.5
License: Apache 2.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Literal, Union
from dataclasses import dataclass
import math


# ============================================================================
# FINGERPRINTING (shared across modes)
# ============================================================================

class MultiDimensionalCantorFingerprinter:
    """Generate multi-dimensional Cantor fingerprints for token routing."""

    def __init__(self, dimensions: List[int] = [2, 3, 4, 5], depth: int = 8):
        self.dimensions = dimensions
        self.depth = depth

    def _simple_hash_fingerprint(self, num_patches: int, device: torch.device) -> torch.Tensor:
        """Simple deterministic linear spacing."""
        indices = torch.arange(num_patches, dtype=torch.float32, device=device)
        fingerprints = indices / max(num_patches - 1, 1)
        epsilon = 1e-6
        fingerprints = fingerprints.clamp(0.0 + epsilon, 1.0 - epsilon)
        return fingerprints

    def compute_fingerprints(
        self,
        num_patches: int,
        device: torch.device = torch.device('cpu')
    ) -> Dict[int, torch.Tensor]:
        """Compute fingerprints - use simple method for small patch counts."""
        fingerprints = {}

        if num_patches <= 256:
            for dim in self.dimensions:
                base_fp = self._simple_hash_fingerprint(num_patches, device)
                torch.manual_seed(42 + dim)
                noise = torch.randn(num_patches, device=device) * 0.001
                fp = (base_fp + noise).clamp(0.0, 1.0)
                fp = (fp - fp.min()) / (fp.max() - fp.min() + 1e-10)
                fingerprints[dim] = fp

        return fingerprints


# ============================================================================
# MODE 1: DENSE MASKED CANTOR ATTENTION (FAST) - WITH MANUAL CONTROL
# ============================================================================

@dataclass
class DenseCantorExpertConfig:
    """Configuration for dense masked expert."""
    expert_id: int
    num_experts: int
    full_feature_dim: int
    expert_dim: int
    num_heads: int
    dropout: float = 0.1
    alpha_init: float = 1.0


class DenseCantorExpert(nn.Module):
    """Dense expert - processes ALL patches with masking.

    NEW: Supports explicit token masks for manual control.
    """

    def __init__(self, config: DenseCantorExpertConfig):
        super().__init__()

        self.expert_id = config.expert_id
        self.num_experts = config.num_experts
        self.expert_dim = config.expert_dim

        # Fingerprint range (for automatic mode)
        self.fp_min = config.expert_id / config.num_experts
        self.fp_max = (config.expert_id + 1) / config.num_experts
        self.is_last_expert = (config.expert_id == config.num_experts - 1)

        # Feature slice
        slice_size = config.full_feature_dim // config.num_experts
        self.slice_start = config.expert_id * slice_size
        self.slice_end = self.slice_start + slice_size
        self.slice_size = slice_size

        # Alpha gating
        self.alpha = nn.Parameter(torch.tensor(config.alpha_init))
        self.alpha_gate = nn.Sequential(
            nn.Linear(slice_size, slice_size // 4),
            nn.GELU(),
            nn.Linear(slice_size // 4, 1),
            nn.Sigmoid()
        )

        # QKV
        self.q_proj = nn.Linear(slice_size, config.expert_dim, bias=False)
        self.k_proj = nn.Linear(slice_size, config.expert_dim, bias=False)
        self.v_proj = nn.Linear(slice_size, config.expert_dim, bias=False)

        self.dropout = nn.Dropout(config.dropout)
        self._init_weights()

    def _init_weights(self):
        for module in [self.q_proj, self.k_proj, self.v_proj]:
            nn.init.xavier_uniform_(module.weight, gain=1.0)

    def forward(
        self,
        tokens: torch.Tensor,
        fingerprints: Optional[torch.Tensor] = None,
        explicit_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Process ALL patches, mask indicates expert's region.

        Args:
            tokens: [B, N, D] input tokens
            fingerprints: [N] fingerprint values (for automatic routing)
            explicit_mask: [B, N] or [N] boolean mask (for manual control)
                          If provided, OVERRIDES fingerprint-based routing

        Returns:
            Dict with Q, K, V, mask, and statistics
        """
        batch_size, num_patches, _ = tokens.shape

        # Compute mask - EXPLICIT MASK TAKES PRIORITY
        if explicit_mask is not None:
            # Manual control mode
            if explicit_mask.dim() == 1:
                # [N] -> [B, N]
                mask = explicit_mask.unsqueeze(0).expand(batch_size, -1)
            else:
                # Already [B, N]
                mask = explicit_mask
        elif fingerprints is not None:
            # Automatic fingerprint mode
            if self.is_last_expert:
                mask = (fingerprints >= self.fp_min) & (fingerprints <= self.fp_max)
            else:
                mask = (fingerprints >= self.fp_min) & (fingerprints < self.fp_max)
            mask = mask.unsqueeze(0).expand(batch_size, -1)
        else:
            # No routing - process everything (fallback)
            mask = torch.ones(batch_size, num_patches, dtype=torch.bool, device=tokens.device)

        # Extract feature slice for ALL patches
        my_features = tokens[..., self.slice_start:self.slice_end]

        # Alpha gating
        alpha_gate = self.alpha_gate(my_features)
        alpha_weight = torch.sigmoid(self.alpha)
        my_features = my_features * (alpha_gate * alpha_weight + (1 - alpha_weight))

        # QKV for ALL patches
        Q = self.q_proj(my_features)
        K = self.k_proj(my_features)
        V = self.v_proj(my_features)

        return {
            'Q': Q,
            'K': K,
            'V': V,
            'mask': mask,
            'num_patches_processed': mask.sum().item()
        }


class DenseCantorGlobalAttention(nn.Module):
    """Fully vectorized dense attention."""

    def __init__(self, num_experts: int, expert_dim: int, num_heads: int,
                 temperature: float, dropout: float):
        super().__init__()

        self.num_experts = num_experts
        self.expert_dim = expert_dim
        self.num_heads = num_heads
        self.head_dim = expert_dim // num_heads

        self.temperature = nn.Parameter(torch.tensor(temperature))
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        expert_outputs: List[Dict[str, torch.Tensor]],
        num_patches: int
    ) -> torch.Tensor:
        """Vectorized attention with masking."""
        device = expert_outputs[0]['Q'].device
        batch_size = expert_outputs[0]['Q'].shape[0]

        # Stack [E, B, N, D]
        Q_stack = torch.stack([out['Q'] for out in expert_outputs], dim=0)
        K_stack = torch.stack([out['K'] for out in expert_outputs], dim=0)
        V_stack = torch.stack([out['V'] for out in expert_outputs], dim=0)
        masks = torch.stack([out['mask'] for out in expert_outputs], dim=0)

        E, B, N, D = Q_stack.shape
        H = self.num_heads
        head_dim = D // H

        # Multi-head reshape [E, B, H, N, head_dim]
        Q_heads = Q_stack.view(E, B, N, H, head_dim).transpose(2, 3)
        K_heads = K_stack.view(E, B, N, H, head_dim).transpose(2, 3)
        V_heads = V_stack.view(E, B, N, H, head_dim).transpose(2, 3)

        # Attention scores [E, B, H, N, N]
        scores = torch.matmul(Q_heads, K_heads.transpose(-2, -1))
        scores = scores / (math.sqrt(head_dim) * self.temperature.abs())

        # Mask attention to expert regions
        attn_mask = masks.view(E, B, 1, N, 1).float()
        scores = scores + (1.0 - attn_mask.transpose(-2, -1)) * -1e9

        # Softmax and apply
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, V_heads)

        # Reshape back [E, B, N, D]
        out = out.transpose(2, 3).contiguous().view(E, B, N, D)

        # Weight by masks and average
        expert_weights = attn_mask.squeeze(2).squeeze(-1)  # [E, B, N]
        weighted_out = out * expert_weights.unsqueeze(-1)
        fused = weighted_out.sum(dim=0)
        norm_factor = expert_weights.sum(dim=0).unsqueeze(-1).clamp(min=1.0)
        fused = fused / norm_factor

        return fused


# ============================================================================
# MODE 2: SPARSE CANTOR ATTENTION (SLOW - original)
# ============================================================================

class SparseCantorMoELayer(nn.Module):
    """Placeholder for sparse mode - use original implementation."""
    def __init__(self, config):
        super().__init__()
        raise NotImplementedError(
            "Sparse mode requires the original CantorMoELayer. "
            "Copy it from your original vit_beans_moe.py if needed."
        )


# ============================================================================
# UNIFIED MoE LAYER WITH MODES AND MANUAL CONTROL
# ============================================================================

@dataclass
class CantorMoEConfig:
    """Unified configuration for all MoE modes."""
    mode: Literal['dense', 'sparse', 'standard'] = 'dense'
    num_experts: int = 8
    full_feature_dim: int = 512
    expert_dim: int = 128
    num_heads: int = 8
    dropout: float = 0.1
    alpha_init: float = 1.0
    temperature: float = 0.5


class MultiModeCantorMoELayer(nn.Module):
    """
    Multi-mode Cantor MoE layer with MANUAL TOKEN CONTROL.

    Modes:
    - 'dense': Fast dense masked attention (recommended)
      * Supports explicit token masks via expert_masks parameter
    - 'sparse': Original sparse attention (slow)
    - 'standard': Standard multi-head attention (baseline)
    """

    def __init__(self, config: CantorMoEConfig):
        super().__init__()

        self.mode = config.mode
        self.num_experts = config.num_experts
        self.full_feature_dim = config.full_feature_dim

        if config.mode == 'dense':
            # Dense masked experts
            self.experts = nn.ModuleList([
                DenseCantorExpert(DenseCantorExpertConfig(
                    expert_id=i,
                    num_experts=config.num_experts,
                    full_feature_dim=config.full_feature_dim,
                    expert_dim=config.expert_dim,
                    num_heads=config.num_heads,
                    dropout=config.dropout,
                    alpha_init=config.alpha_init
                ))
                for i in range(config.num_experts)
            ])

            self.attention = DenseCantorGlobalAttention(
                num_experts=config.num_experts,
                expert_dim=config.expert_dim,
                num_heads=config.num_heads,
                temperature=config.temperature,
                dropout=config.dropout
            )

            self.fusion_proj = nn.Linear(config.expert_dim, config.full_feature_dim)

        elif config.mode == 'sparse':
            raise NotImplementedError("Use original sparse implementation if needed")

        elif config.mode == 'standard':
            # Standard multi-head attention (no experts)
            self.attention = nn.MultiheadAttention(
                config.full_feature_dim,
                config.num_heads,
                dropout=config.dropout,
                batch_first=True
            )
            self.experts = None
            self.fusion_proj = None

        else:
            raise ValueError(f"Unknown mode: {config.mode}")

        self.norm = nn.LayerNorm(config.full_feature_dim)

    def forward(
        self,
        x: torch.Tensor,
        fingerprints: Optional[torch.Tensor] = None,
        expert_masks: Optional[Union[List[torch.Tensor], torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Dict[str, int]]:
        """Forward pass - mode-dependent with manual control support.

        Args:
            x: [B, N, D] input tokens
            fingerprints: [N] fingerprints for automatic routing (dense mode)
            expert_masks: Manual token routing (dense mode only):
                - List[Tensor]: List of [B, N] or [N] masks, one per expert
                - Tensor: [E, B, N] or [E, N] masks for all experts
                If provided, overrides fingerprint-based routing

        Returns:
            output: [B, N, D] transformed tokens
            stats: Dict with expert utilization statistics
        """

        if self.mode == 'standard':
            # Standard attention (no fingerprints needed)
            x_norm = self.norm(x)
            attn_out, _ = self.attention(x_norm, x_norm, x_norm)
            return x + attn_out, {}

        elif self.mode == 'dense':
            # Dense Cantor attention with optional manual control
            batch_size, num_patches, _ = x.shape
            x_norm = self.norm(x)

            # Parse expert masks if provided
            parsed_masks = None
            if expert_masks is not None:
                if isinstance(expert_masks, list):
                    parsed_masks = expert_masks
                else:
                    # Tensor: split into list
                    if expert_masks.dim() == 2:
                        # [E, N] -> List of [N]
                        parsed_masks = [expert_masks[i] for i in range(expert_masks.shape[0])]
                    else:
                        # [E, B, N] -> List of [B, N]
                        parsed_masks = [expert_masks[i] for i in range(expert_masks.shape[0])]

            # Process all experts
            expert_outputs = []
            expert_utilization = {}

            for i, expert in enumerate(self.experts):
                mask = parsed_masks[i] if parsed_masks is not None else None
                output = expert(x_norm, fingerprints, explicit_mask=mask)
                expert_outputs.append(output)
                expert_utilization[f'expert_{expert.expert_id}'] = output['num_patches_processed']

            # Dense attention
            fused = self.attention(expert_outputs, num_patches)

            # Project and residual
            output = self.fusion_proj(fused)
            return x + output, expert_utilization

        else:
            raise NotImplementedError(f"Mode {self.mode} not implemented")


# ============================================================================
# ViT-BEANS WITH MODES AND MANUAL CONTROL
# ============================================================================

@dataclass
class ViTBeansConfig:
    """Complete ViT-Beans configuration with mode selection."""
    # Architecture
    image_size: int = 224
    patch_size: int = 16
    in_channels: int = 3
    num_layers: int = 12
    feature_dim: int = 1024
    num_classes: int = 1000

    # Attention mode
    attention_mode: Literal['dense', 'sparse', 'standard'] = 'dense'

    # MoE settings (ignored if mode='standard')
    num_experts: int = 16
    expert_dim: int = 128
    num_heads: int = 8
    mlp_ratio: float = 4.0

    # Cantor settings (ignored if mode='standard')
    cantor_dimensions: List[int] = None
    cantor_depth: int = 8
    alpha_init: float = 1.0
    temperature: float = 0.5

    # Training
    dropout: float = 0.1

    def __post_init__(self):
        if self.cantor_dimensions is None:
            self.cantor_dimensions = [2, 3, 4, 5]

        if self.attention_mode != 'standard':
            assert self.feature_dim % self.num_experts == 0, \
                "feature_dim must be divisible by num_experts"


class ViTBeans(nn.Module):
    """ViT-Beans with configurable attention modes and MANUAL TOKEN CONTROL.

    Usage Examples:
    ---------------

    # 1. AUTOMATIC ROUTING (default)
    model = ViTBeans(config)
    logits = model(images)  # Uses fingerprint-based routing

    # 2. MANUAL ROUTING - Equal division
    masks = model.create_manual_masks({
        0: list(range(0, 49)),      # Expert 0: patches 0-48
        1: list(range(49, 98)),     # Expert 1: patches 49-97
        2: list(range(98, 147)),    # Expert 2: patches 98-146
        3: list(range(147, 196))    # Expert 3: patches 147-195
    })
    logits = model(images, expert_masks=masks)

    # 3. MANUAL ROUTING - Spatial regions
    masks = model.create_spatial_masks(
        grid_size=14,  # For 224x224 with patch_size=16
        regions={
            0: [(0, 6), (0, 6)],     # Top-left quadrant
            1: [(8, 14), (0, 6)],    # Top-right quadrant
            2: [(0, 6), (8, 14)],    # Bottom-left quadrant
            3: [(8, 14), (8, 14)]    # Bottom-right quadrant
        }
    )
    logits = model(images, expert_masks=masks)
    """

    def __init__(self, config: ViTBeansConfig):
        super().__init__()

        self.config = config

        # Calculate patches
        assert config.image_size % config.patch_size == 0
        self.num_patches = (config.image_size // config.patch_size) ** 2
        self.grid_size = config.image_size // config.patch_size

        # Patch embedding
        self.patch_embed = nn.Conv2d(
            config.in_channels,
            config.feature_dim,
            kernel_size=config.patch_size,
            stride=config.patch_size
        )

        # Position embedding
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.num_patches + 1, config.feature_dim) * 0.02
        )

        # CLS token
        self.cls_token = nn.Parameter(
            torch.randn(1, 1, config.feature_dim) * 0.02
        )

        # Fingerprinter (only needed for Cantor modes)
        if config.attention_mode != 'standard':
            self.fingerprinter = MultiDimensionalCantorFingerprinter(
                dimensions=config.cantor_dimensions,
                depth=config.cantor_depth
            )
            self.register_buffer('patch_fingerprints', torch.zeros(self.num_patches))
            self._fingerprints_computed = False
        else:
            self.fingerprinter = None
            self.patch_fingerprints = None
            self._fingerprints_computed = True

        # Transformer layers
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'attention': MultiModeCantorMoELayer(CantorMoEConfig(
                    mode=config.attention_mode,
                    num_experts=config.num_experts,
                    full_feature_dim=config.feature_dim,
                    expert_dim=config.expert_dim,
                    num_heads=config.num_heads,
                    dropout=config.dropout,
                    alpha_init=config.alpha_init,
                    temperature=config.temperature
                )),
                'mlp': nn.Sequential(
                    nn.LayerNorm(config.feature_dim),
                    nn.Linear(config.feature_dim, int(config.feature_dim * config.mlp_ratio)),
                    nn.GELU(),
                    nn.Dropout(config.dropout),
                    nn.Linear(int(config.feature_dim * config.mlp_ratio), config.feature_dim),
                    nn.Dropout(config.dropout)
                )
            })
            for _ in range(config.num_layers)
        ])

        # Classification head
        self.norm = nn.LayerNorm(config.feature_dim)
        self.head = nn.Linear(config.feature_dim, config.num_classes)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        nn.init.xavier_uniform_(self.patch_embed.weight)
        if self.patch_embed.bias is not None:
            nn.init.zeros_(self.patch_embed.bias)

        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def _compute_fingerprints(self, device: torch.device):
        """Compute and cache Cantor fingerprints."""
        if self.fingerprinter is not None and not self._fingerprints_computed:
            fingerprints_dict = self.fingerprinter.compute_fingerprints(
                self.num_patches, device
            )
            self.patch_fingerprints = fingerprints_dict[3]
            self._fingerprints_computed = True

    def forward_features(
        self,
        x: torch.Tensor,
        expert_masks: Optional[Union[List[torch.Tensor], torch.Tensor]] = None
    ) -> torch.Tensor:
        """Extract features through ViT-Beans layers.

        Args:
            x: [B, C, H, W] input images
            expert_masks: Optional manual token routing for dense mode
                - Single mask for all layers: [E, B, N] or [E, N]
                - Different masks per layer: List[[E, B, N] or [E, N]]
        """
        batch_size = x.shape[0]
        device = x.device

        # Compute fingerprints if needed
        if self.config.attention_mode != 'standard':
            self._compute_fingerprints(device)

        # Patch embedding
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)

        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Add positional embedding
        x = x + self.pos_embed

        # Handle expert_masks - can be single mask or per-layer masks
        if expert_masks is not None and not isinstance(expert_masks, list):
            # Single mask for all layers
            expert_masks = [expert_masks] * len(self.layers)
        elif expert_masks is not None:
            # List of masks - must match number of layers
            assert len(expert_masks) == len(self.layers), \
                f"Number of expert_masks ({len(expert_masks)}) must match layers ({len(self.layers)})"

        # Process through layers
        for layer_idx, layer in enumerate(self.layers):
            # Get mask for this layer
            layer_mask = expert_masks[layer_idx] if expert_masks is not None else None

            # Attention (skip CLS token for Cantor modes)
            if self.config.attention_mode == 'standard':
                x_attn, _ = layer['attention'](x, None, None)
                x = x_attn
            else:
                x_patches = x[:, 1:]
                x_patches, _ = layer['attention'](
                    x_patches,
                    self.patch_fingerprints,
                    layer_mask
                )
                x = torch.cat([x[:, :1], x_patches], dim=1)

            # MLP
            x = x + layer['mlp'](x)

        return x

    def forward(
        self,
        x: torch.Tensor,
        expert_masks: Optional[Union[List[torch.Tensor], torch.Tensor]] = None
    ) -> torch.Tensor:
        """Full forward pass with optional manual token control.

        Args:
            x: [B, C, H, W] input images
            expert_masks: Optional manual expert routing (dense mode only)
        """
        x = self.forward_features(x, expert_masks)

        # Classification from CLS token
        x = self.norm(x[:, 0])
        logits = self.head(x)

        return logits

    def create_manual_masks(
        self,
        patch_assignments: Dict[int, List[int]],
        batch_size: int = 1
    ) -> torch.Tensor:
        """Helper to create manual expert masks from patch assignments.

        Args:
            patch_assignments: Dict mapping expert_id -> list of patch indices
                              e.g., {0: [0, 1, 2], 1: [3, 4, 5], ...}
            batch_size: Batch size for masks

        Returns:
            masks: [E, B, N] boolean tensor

        Example:
            # Equal division among 4 experts
            masks = model.create_manual_masks({
                0: list(range(0, 49)),
                1: list(range(49, 98)),
                2: list(range(98, 147)),
                3: list(range(147, 196)),
            }, batch_size=8)
            logits = model(images, expert_masks=masks)
        """
        device = next(self.parameters()).device
        masks = torch.zeros(
            self.config.num_experts,
            batch_size,
            self.num_patches,
            dtype=torch.bool,
            device=device
        )

        for expert_id, patch_indices in patch_assignments.items():
            if expert_id < self.config.num_experts:
                masks[expert_id, :, patch_indices] = True

        return masks

    def create_spatial_masks(
        self,
        grid_size: Optional[int] = None,
        regions: Optional[Dict[int, Tuple[Tuple[int, int], Tuple[int, int]]]] = None,
        batch_size: int = 1
    ) -> torch.Tensor:
        """Helper to create spatial region masks for experts.

        Args:
            grid_size: Size of patch grid (e.g., 14 for 224x224 images with 16x16 patches)
                      If None, uses self.grid_size
            regions: Dict mapping expert_id -> ((y_start, y_end), (x_start, x_end))
                    Ranges are inclusive on start, exclusive on end
            batch_size: Batch size for masks

        Returns:
            masks: [E, B, N] boolean tensor

        Example:
            # Divide into quadrants for 4 experts
            masks = model.create_spatial_masks(
                grid_size=14,
                regions={
                    0: ((0, 7), (0, 7)),      # Top-left
                    1: ((0, 7), (7, 14)),     # Top-right
                    2: ((7, 14), (0, 7)),     # Bottom-left
                    3: ((7, 14), (7, 14)),    # Bottom-right
                },
                batch_size=8
            )
            logits = model(images, expert_masks=masks)
        """
        if grid_size is None:
            grid_size = self.grid_size

        device = next(self.parameters()).device
        masks = torch.zeros(
            self.config.num_experts,
            batch_size,
            self.num_patches,
            dtype=torch.bool,
            device=device
        )

        if regions is None:
            return masks

        for expert_id, ((y_start, y_end), (x_start, x_end)) in regions.items():
            if expert_id < self.config.num_experts:
                for y in range(y_start, y_end):
                    for x in range(x_start, x_end):
                        patch_idx = y * grid_size + x
                        if patch_idx < self.num_patches:
                            masks[expert_id, :, patch_idx] = True

        return masks

    def diagnose_expert_coverage(
        self,
        expert_masks: Optional[torch.Tensor] = None
    ) -> Dict:
        """Diagnose expert coverage (only for Cantor modes).

        Args:
            expert_masks: Optional manual masks to diagnose. If None, uses fingerprints.
        """
        if self.config.attention_mode == 'standard':
            return {'mode': 'standard', 'no_experts': True}

        device = next(self.parameters()).device

        coverage = {}
        total_covered = 0

        if expert_masks is not None:
            # Diagnose manual masks
            if expert_masks.dim() == 3:
                # [E, B, N] - use first batch
                masks = expert_masks[:, 0, :]
            else:
                # [E, N]
                masks = expert_masks

            for expert_id in range(self.config.num_experts):
                mask = masks[expert_id]
                num_patches = mask.sum().item()
                coverage[f'expert_{expert_id}'] = {
                    'patches': num_patches,
                    'mode': 'manual'
                }
                total_covered += num_patches

        else:
            # Diagnose fingerprint-based routing
            self._compute_fingerprints(device)
            fingerprints = self.patch_fingerprints

            moe = self.layers[0]['attention']
            for expert in moe.experts:
                if expert.is_last_expert:
                    mask = (fingerprints >= expert.fp_min) & (fingerprints <= expert.fp_max)
                else:
                    mask = (fingerprints >= expert.fp_min) & (fingerprints < expert.fp_max)

                num_patches = mask.sum().item()
                coverage[f'expert_{expert.expert_id}'] = {
                    'patches': num_patches,
                    'range': f'[{expert.fp_min:.4f}, {expert.fp_max:.4f}{")" if not expert.is_last_expert else "]"}',
                    'is_last': expert.is_last_expert,
                    'mode': 'fingerprint'
                }
                total_covered += num_patches

            coverage['max_fingerprint'] = fingerprints.max().item()
            coverage['min_fingerprint'] = fingerprints.min().item()

        coverage['total_patches'] = self.num_patches
        coverage['total_covered'] = total_covered

        return coverage


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("ViT-Beans with Manual Token Control")
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    config = ViTBeansConfig(
        image_size=224,
        patch_size=16,
        num_layers=4,
        feature_dim=512,
        num_experts=8,
        expert_dim=64,
        num_classes=1000,
        attention_mode='dense'
    )

    model = ViTBeans(config).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Patches: {model.num_patches} ({model.grid_size}x{model.grid_size})")

    # Test 1: Automatic routing
    print("\n" + "=" * 80)
    print("Test 1: AUTOMATIC ROUTING")
    print("=" * 80)

    x = torch.randn(2, 3, 224, 224, device=device)
    model.eval()
    with torch.no_grad():
        logits_auto = model(x)
        coverage_auto = model.diagnose_expert_coverage()

    print(f"Output shape: {logits_auto.shape}")
    print("Expert coverage (fingerprint-based):")
    for i in range(config.num_experts):
        info = coverage_auto[f'expert_{i}']
        print(f"  Expert {i}: {info['patches']} patches, range {info['range']}")

    # Test 2: Manual equal division
    print("\n" + "=" * 80)
    print("Test 2: MANUAL EQUAL DIVISION")
    print("=" * 80)

    patches_per_expert = model.num_patches // config.num_experts
    manual_equal = {
        i: list(range(i * patches_per_expert, (i + 1) * patches_per_expert))
        for i in range(config.num_experts)
    }
    # Handle remainder
    manual_equal[config.num_experts - 1].extend(
        list(range(config.num_experts * patches_per_expert, model.num_patches))
    )

    masks_equal = model.create_manual_masks(manual_equal, batch_size=2)
    with torch.no_grad():
        logits_equal = model(x, expert_masks=masks_equal)
        coverage_equal = model.diagnose_expert_coverage(masks_equal)

    print(f"Output shape: {logits_equal.shape}")
    print("Expert coverage (manual equal):")
    for i in range(config.num_experts):
        info = coverage_equal[f'expert_{i}']
        print(f"  Expert {i}: {info['patches']} patches")

    # Test 3: Spatial quadrants
    print("\n" + "=" * 80)
    print("Test 3: SPATIAL REGION CONTROL")
    print("=" * 80)

    # For 8 experts, create 8 spatial regions
    half = model.grid_size // 2
    quarter_x = model.grid_size // 4
    quarter_y = model.grid_size // 4

    spatial_regions = {
        0: ((0, quarter_y), (0, half)),              # Top-left corner
        1: ((0, quarter_y), (half, model.grid_size)), # Top-right corner
        2: ((quarter_y, half), (0, half)),           # Left region
        3: ((quarter_y, half), (half, model.grid_size)), # Right region
        4: ((half, half + quarter_y), (0, half)),    # Lower-left region
        5: ((half, half + quarter_y), (half, model.grid_size)), # Lower-right region
        6: ((half + quarter_y, model.grid_size), (0, half)),    # Bottom-left
        7: ((half + quarter_y, model.grid_size), (half, model.grid_size))  # Bottom-right
    }

    masks_spatial = model.create_spatial_masks(
        regions=spatial_regions,
        batch_size=2
    )

    with torch.no_grad():
        logits_spatial = model(x, expert_masks=masks_spatial)
        coverage_spatial = model.diagnose_expert_coverage(masks_spatial)

    print(f"Output shape: {logits_spatial.shape}")
    print("Expert coverage (spatial):")
    for i in range(config.num_experts):
        info = coverage_spatial[f'expert_{i}']
        print(f"  Expert {i}: {info['patches']} patches")

    print("\n" + "=" * 80)
    print("✅ All manual control modes working!")
    print("=" * 80)

    print("\nUSAGE SUMMARY:")
    print("-" * 80)
    print("1. Automatic:  logits = model(images)")
    print("2. Manual:     logits = model(images, expert_masks=masks)")
    print("3. Equal div:  masks = model.create_manual_masks({0: [0,1,2], ...})")
    print("4. Spatial:    masks = model.create_spatial_masks(regions={...})")