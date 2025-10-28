# geovocab2.train.model.layers.conv.py
# ============================================================
# CantorConv2d: Fractal‑structured convolutional layer
# Author: AbstractPhil
# Date: 2025‑10‑28
# License: MIT
# ============================================================

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import math


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass
class CantorConv2dConfig:
    in_channels: int
    out_channels: int
    kernel_size: int | tuple[int, int]
    stride: int | tuple[int, int] = 1
    padding: int | tuple[int, int] | str = 0
    dilation: int | tuple[int, int] = 1
    groups: int = 1
    bias: bool = True
    padding_mode: str = 'zeros'
    depth: int = 8  # Cantor recursion depth
    mask_mode: str = "alpha"  # ['prune', 'scale', 'alpha']
    mask_floor: float = 0.25  # Minimum mask scaled value
    mask_scale: float = 0.5  # Initial scaling factor for cantor mask
    alpha_mode: str = "sigmoid"  # ['sigmoid', 'softplus', 'exp', 'raw']
    alpha_min: float = 0.1  # Minimum alpha value (for constrained modes)
    alpha_max: float = 1.0  # Maximum alpha value (for constrained modes)
    per_output_alpha: bool = False  # Use per-output alpha parameters
    dtype: torch.dtype = torch.float32
    device: str | None = None


# ============================================================
# CANTOR CONV2D LAYER
# ============================================================

class CantorConv2d(nn.Module):
    """
    Conv2d layer whose kernel weights follow Cantor‑set recursion.
    Each weight in the 4D tensor gets masked based on its position
    in the fractal structure.
    """

    def __init__(self, cfg: CantorConv2dConfig):
        super().__init__()
        self.cfg = cfg
        self.in_channels = cfg.in_channels
        self.out_channels = cfg.out_channels

        # Handle kernel_size as int or tuple
        if isinstance(cfg.kernel_size, int):
            self.kernel_size = (cfg.kernel_size, cfg.kernel_size)
        else:
            self.kernel_size = cfg.kernel_size

        if isinstance(cfg.stride, int):
            self.stride = (cfg.stride, cfg.stride)
        else:
            self.stride = cfg.stride

        if isinstance(cfg.padding, int):
            self.padding = (cfg.padding, cfg.padding)
        else:
            self.padding = cfg.padding

        if isinstance(cfg.dilation, int):
            self.dilation = (cfg.dilation, cfg.dilation)
        else:
            self.dilation = cfg.dilation

        self.groups = cfg.groups
        self.padding_mode = cfg.padding_mode
        self.depth = cfg.depth
        self.mask_mode = cfg.mask_mode
        self.mask_floor = cfg.mask_floor
        self.mask_scale = cfg.mask_scale

        # Parameters
        self.weight = nn.Parameter(
            torch.empty(
                cfg.out_channels,
                cfg.in_channels // cfg.groups,
                self.kernel_size[0],
                self.kernel_size[1],
                dtype=cfg.dtype,
                device=cfg.device
            )
        )
        self.bias = nn.Parameter(
            torch.empty(cfg.out_channels, dtype=cfg.dtype, device=cfg.device)
        ) if cfg.bias else None

        # Alpha parameter setup
        if self.mask_mode == "alpha":
            self.alpha_mode = cfg.alpha_mode
            self.alpha_min = cfg.alpha_min
            self.alpha_max = cfg.alpha_max
            self.per_output_alpha = cfg.per_output_alpha

            # Determine initialization value based on mode
            if self.alpha_mode in ["sigmoid", "softplus"]:
                init_val = 0.0  # sigmoid(0) = 0.5, good starting point
            elif self.alpha_mode == "exp":
                init_val = math.log(cfg.mask_scale)
            else:  # raw
                init_val = cfg.mask_scale

            # Create parameter (scalar or per-output-channel)
            if self.per_output_alpha:
                self._alpha_raw = nn.Parameter(
                    torch.full((cfg.out_channels,), init_val, dtype=cfg.dtype, device=cfg.device)
                )
            else:
                self._alpha_raw = nn.Parameter(
                    torch.tensor(init_val, dtype=cfg.dtype, device=cfg.device)
                )
        else:
            self._alpha_raw = None

        # Build Cantor mask
        mask = self._build_cantor_mask(
            cfg.out_channels,
            cfg.in_channels // cfg.groups,
            self.kernel_size[0],
            self.kernel_size[1],
            cfg.depth
        )
        self.register_buffer("mask", mask, persistent=False)

        # Initialize parameters
        self.reset_parameters()

    # --------------------------------------------------------
    def reset_parameters(self):
        """Kaiming uniform initialization for conv layers."""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    # --------------------------------------------------------
    @property
    def alpha(self):
        """
        Compute constrained alpha from raw parameter.
        Returns appropriately shaped tensor for broadcasting.
        """
        if self._alpha_raw is None:
            return None

        if self.alpha_mode == "sigmoid":
            # Map to [alpha_min, alpha_max] using sigmoid
            normalized = torch.sigmoid(self._alpha_raw)
            alpha_val = self.alpha_min + (self.alpha_max - self.alpha_min) * normalized
        elif self.alpha_mode == "softplus":
            # Always positive, offset by alpha_min
            alpha_val = F.softplus(self._alpha_raw) + self.alpha_min
            alpha_val = torch.clamp(alpha_val, max=self.alpha_max)
        elif self.alpha_mode == "exp":
            # Exponential ensures positivity, clamp in log space
            alpha_val = torch.exp(self._alpha_raw.clamp(-3, 2))
        else:  # raw
            alpha_val = torch.clamp(self._alpha_raw, self.alpha_min, self.alpha_max)

        # Reshape for broadcasting if per-output-channel
        if self.per_output_alpha:
            return alpha_val.view(-1, 1, 1, 1)  # [out_channels, 1, 1, 1]
        else:
            return alpha_val

    # --------------------------------------------------------
    def _build_cantor_mask(
            self,
            out_ch: int,
            in_ch: int,
            kh: int,
            kw: int,
            depth: int
    ) -> torch.Tensor:
        """
        Constructs a Cantor mask for Conv2d weights.
        Each weight at position (o, i, h, w) is masked based on
        whether its flattened index survives the Cantor recursion.
        """
        mask = torch.zeros(out_ch, in_ch, kh, kw)
        total_elements = out_ch * in_ch * kh * kw

        for o in range(out_ch):
            for i in range(in_ch):
                for h in range(kh):
                    for w in range(kw):
                        # Flatten 4D index to 1D
                        flat_idx = (((o * in_ch + i) * kh + h) * kw + w)
                        # Normalize to [0, 1]
                        index_val = flat_idx / (total_elements + 1e-9)

                        # Check Cantor validity
                        valid = True
                        x = index_val
                        for _ in range(depth):
                            x *= 3.0
                            digit = int(x)
                            x -= digit
                            if digit == 1:  # middle third removed
                                valid = False
                                break
                        if valid:
                            mask[o, i, h, w] = 1.0
        return mask

    # --------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Cantor‑masked convolution."""
        if self.mask_mode == "scale":
            effective_weight = self.weight * (self.mask_floor + self.mask_scale * self.mask)
        elif self.mask_mode == "alpha":
            effective_weight = self.weight * (self.mask_floor + self.alpha * self.mask)
        else:  # "prune"
            effective_weight = self.weight * self.mask

        return F.conv2d(
            x,
            effective_weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups
        )

    # --------------------------------------------------------
    def get_alpha_stats(self) -> dict[str, float]:
        """Return statistics about current alpha values (useful for logging)."""
        if self._alpha_raw is None:
            return {}

        alpha_val = self.alpha
        if self.per_output_alpha:
            alpha_val = alpha_val.squeeze()

        return {
            "alpha_mean": alpha_val.mean().item(),
            "alpha_std": alpha_val.std().item() if self.per_output_alpha else 0.0,
            "alpha_min": alpha_val.min().item(),
            "alpha_max": alpha_val.max().item(),
            "alpha_raw_mean": self._alpha_raw.mean().item(),
        }


# ============================================================
# ACTIVATION TEST BLOCK
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Testing CantorConv2d with improved alpha handling")
    print("=" * 60)

    # Test 1: Sigmoid-constrained alpha
    print("\n[Test 1] Sigmoid-constrained alpha (single)")
    cfg = CantorConv2dConfig(
        in_channels=3,
        out_channels=64,
        kernel_size=3,
        padding=1,
        depth=6,
        mask_mode="alpha",
        alpha_mode="sigmoid",
        alpha_min=0.1,
        alpha_max=1.0
    )
    layer = CantorConv2d(cfg)

    x = torch.randn(4, 3, 32, 32)
    y = layer(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Weight shape: {layer.weight.shape}")
    print(f"Mask density: {layer.mask.mean().item():.4f}")
    print(f"Alpha value: {layer.alpha.item():.4f}")
    print(f"Alpha stats: {layer.get_alpha_stats()}")

    # Test 2: Per-output-channel alpha
    print("\n[Test 2] Per-output-channel alpha")
    cfg2 = CantorConv2dConfig(
        in_channels=16,
        out_channels=32,
        kernel_size=3,
        padding=1,
        depth=6,
        mask_mode="alpha",
        alpha_mode="sigmoid",
        per_output_alpha=True
    )
    layer2 = CantorConv2d(cfg2)

    x2 = torch.randn(2, 16, 28, 28)
    y2 = layer2(x2)

    print(f"Output shape: {y2.shape}")
    print(f"Alpha shape: {layer2.alpha.shape}")
    print(f"Alpha stats: {layer2.get_alpha_stats()}")

    print("\n" + "=" * 60)
    print("All tests passed!")