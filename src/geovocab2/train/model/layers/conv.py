# geovocab2.train.model.layers.conv.py
# ============================================================
# CantorConv2d: Fractal‑structured convolutional layer
# Author: AbstractPhil
# Assistant: Claude Sonnet (4.5)
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
    mask_scale: float = 0.5  # Scaling factor for cantor mask
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

        if self.mask_mode == "alpha":
            self.alpha = nn.Parameter(
                torch.tensor(cfg.mask_scale, dtype=cfg.dtype, device=cfg.device)
            )
        else:
            self.alpha = None

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


# ============================================================
# ACTIVATION TEST BLOCK
# ============================================================

if __name__ == "__main__":
    cfg = CantorConv2dConfig(
        in_channels=3,
        out_channels=64,
        kernel_size=3,
        padding=1,
        depth=6
    )
    layer = CantorConv2d(cfg)

    x = torch.randn(4, 3, 32, 32)
    y = layer(x)

    print("Input shape:", x.shape)
    print("Output shape:", y.shape)
    print("Weight shape:", layer.weight.shape)
    print("Mask density:", layer.mask.mean().item())
    print("Sample output stats:")
    print(f"  Mean: {y.mean().item():.4f}")
    print(f"  Std: {y.std().item():.4f}")