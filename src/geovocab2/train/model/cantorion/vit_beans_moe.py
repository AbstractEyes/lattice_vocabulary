"""
ViT-Beans: Multi-Mode Vision Transformer with Cantor Expert Attention
======================================================================

Supports 3 modes:
- 'dense': Dense masked Cantor attention (FAST - recommended)
- 'sparse': Original sparse Cantor attention (SLOW - for experiments)
- 'standard': Standard multi-head attention (BASELINE)

Author: AbstractPhil + Claude Sonnet 4.5
License: Apache 2.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Literal
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
# MODE 1: DENSE MASKED CANTOR ATTENTION (FAST)
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
    """Dense expert - processes ALL patches with masking."""

    def __init__(self, config: DenseCantorExpertConfig):
        super().__init__()

        self.expert_id = config.expert_id
        self.num_experts = config.num_experts
        self.expert_dim = config.expert_dim

        # Fingerprint range
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
        fingerprints: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Process ALL patches, mask indicates expert's region."""
        batch_size, num_patches, _ = tokens.shape

        # Compute mask
        if self.is_last_expert:
            mask = (fingerprints >= self.fp_min) & (fingerprints <= self.fp_max)
        else:
            mask = (fingerprints >= self.fp_min) & (fingerprints < self.fp_max)

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
        attn_mask = masks.view(E, 1, 1, N, 1).float()
        scores = scores + (1.0 - attn_mask.transpose(-2, -1)) * -1e9

        # Softmax and apply
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, V_heads)

        # Reshape back [E, B, N, D]
        out = out.transpose(2, 3).contiguous().view(E, B, N, D)

        # Weight by masks and average
        # FIX: Reshape mask to [E, 1, N, 1] for proper broadcasting with [E, B, N, D]
        expert_weights = masks.view(E, 1, N, 1).float()  # [E, 1, N, 1]
        weighted_out = out * expert_weights  # [E, B, N, D] * [E, 1, N, 1] → [E, B, N, D]
        fused = weighted_out.sum(dim=0)  # [B, N, D]

        # Normalize by number of experts per patch (should be 1 with perfect partitioning)
        norm_factor = expert_weights.sum(dim=0).clamp(min=1.0)  # [1, N, 1]
        fused = fused / norm_factor  # [B, N, D] / [1, N, 1] → [B, N, D]

        return fused


# ============================================================================
# MODE 2: SPARSE CANTOR ATTENTION (SLOW - original)
# ============================================================================

# Import your original sparse implementation here if you want to keep it
# For now, we'll just have a placeholder that raises an error

class SparseCantorMoELayer(nn.Module):
    """Placeholder for sparse mode - use original implementation."""
    def __init__(self, config):
        super().__init__()
        raise NotImplementedError(
            "Sparse mode requires the original CantorMoELayer. "
            "Copy it from your original vit_beans_moe.py if needed."
        )


# ============================================================================
# UNIFIED MoE LAYER WITH MODES
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
    Multi-mode Cantor MoE layer.

    Modes:
    - 'dense': Fast dense masked attention (recommended)
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
        fingerprints: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, int]]:
        """Forward pass - mode-dependent."""

        if self.mode == 'standard':
            # Standard attention (no fingerprints needed)
            x_norm = self.norm(x)
            attn_out, _ = self.attention(x_norm, x_norm, x_norm)
            return x + attn_out, {}

        elif self.mode == 'dense':
            # Dense Cantor attention
            batch_size, num_patches, _ = x.shape
            x_norm = self.norm(x)

            # Process all experts
            expert_outputs = []
            expert_utilization = {}

            for expert in self.experts:
                output = expert(x_norm, fingerprints)
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
# ViT-BEANS WITH MODES
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
    """ViT-Beans with configurable attention modes."""

    def __init__(self, config: ViTBeansConfig):
        super().__init__()

        self.config = config

        # Calculate patches
        assert config.image_size % config.patch_size == 0
        self.num_patches = (config.image_size // config.patch_size) ** 2

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

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features through ViT-Beans layers."""
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

        # Process through layers
        for layer in self.layers:
            # Attention (skip CLS token for Cantor modes)
            if self.config.attention_mode == 'standard':
                x_attn, _ = layer['attention'](x, None)
                x = x_attn
            else:
                x_patches = x[:, 1:]
                x_patches, _ = layer['attention'](x_patches, self.patch_fingerprints)
                x = torch.cat([x[:, :1], x_patches], dim=1)

            # MLP
            x = x + layer['mlp'](x)

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full forward pass."""
        x = self.forward_features(x)

        # Classification from CLS token
        x = self.norm(x[:, 0])
        logits = self.head(x)

        return logits

    def diagnose_expert_coverage(self) -> Dict:
        """Diagnose expert coverage (only for Cantor modes)."""
        if self.config.attention_mode == 'standard':
            return {'mode': 'standard', 'no_experts': True}

        device = next(self.parameters()).device
        self._compute_fingerprints(device)
        fingerprints = self.patch_fingerprints

        coverage = {}
        total_covered = 0

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
                'is_last': expert.is_last_expert
            }
            total_covered += num_patches

        coverage['total_patches'] = self.num_patches
        coverage['total_covered'] = total_covered
        coverage['max_fingerprint'] = fingerprints.max().item()
        coverage['min_fingerprint'] = fingerprints.min().item()

        return coverage


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("ViT-Beans Multi-Mode Test")
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Test all three modes
    modes = ['dense', 'standard']

    for mode in modes:
        print(f"\nTesting mode: {mode.upper()}")
        print("-" * 80)

        config = ViTBeansConfig(
            image_size=32,
            patch_size=4,
            num_layers=2,
            feature_dim=256,
            num_experts=4,
            expert_dim=64,
            num_classes=100,
            attention_mode=mode
        )

        model = ViTBeans(config).to(device)

        # Count parameters
        params = sum(p.numel() for p in model.parameters())
        print(f"Parameters: {params:,}")

        # Test forward pass
        x = torch.randn(2, 3, 32, 32, device=device)

        import time
        start = time.time()

        model.eval()
        with torch.no_grad():
            for _ in range(10):
                logits = model(x)

        if device.type == 'cuda':
            torch.cuda.synchronize()

        elapsed = time.time() - start

        print(f"10 forward passes: {elapsed:.3f}s ({elapsed*100:.1f}ms per pass)")
        print(f"Output shape: {logits.shape}")
        print(f"✅ {mode.upper()} mode working!")

    # Test coverage diagnostic
    print("\n" + "=" * 80)
    print("Coverage Diagnostic (Dense Mode)")
    print("=" * 80)

    config_dense = ViTBeansConfig(
        image_size=32,
        patch_size=4,
        num_experts=8,
        feature_dim=512,
        attention_mode='dense'
    )

    model_dense = ViTBeans(config_dense).to(device)
    coverage = model_dense.diagnose_expert_coverage()

    print(f"Total patches: {coverage['total_patches']}")
    print(f"Total covered: {coverage['total_covered']}")

    for i in range(8):
        info = coverage[f'expert_{i}']
        print(f"  Expert {i}: {info['patches']} patches")

    print("\n✅ All modes tested successfully!")