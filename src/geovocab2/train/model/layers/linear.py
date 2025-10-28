# geovocab2.train.model.layers.linear.py
# ============================================================
# CantorLinear: Fractal‑structured linear transformation layer
# Author: AbstractPhil
# Assistant: Quartermaster Mirel GPT 4o + GPT 5 intervention
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
class CantorLinearConfig:
    in_features: int
    out_features: int
    depth: int = 8               # Cantor recursion depth
    bias: bool = True
    mask_mode: str = "prune"     # ['prune', 'scale']
    dtype: torch.dtype = torch.float32
    device: str | None = None


# ============================================================
# CANTOR UTILITY FUNCTIONS
# ============================================================

def cantor_slice(x: float, bits: int = 8) -> float:
    """Return Cantor‑mapped value in [0,1]."""
    out = 0.0
    factor = 0.5
    for _ in range(bits):
        x *= 3.0
        digit = int(x)
        x -= digit
        if digit == 2:
            out += factor
        elif digit == 1:
            # middle third removed → no contribution
            pass
        factor *= 0.5
    return out


def cantor_mask_for_index(index: int, depth: int) -> list[int]:
    """
    Produce ternary digits for the Cantor mask associated with a neuron index.
    Returns digits ∈ {0,1,2}, length=depth.
    """
    digits = []
    val = cantor_slice(index / (10_000.0 + index + 1e-9), bits=depth)
    x = val
    for _ in range(depth):
        x *= 3.0
        d = int(x)
        x -= d
        digits.append(d)
    return digits


# ============================================================
# CANTOR LINEAR LAYER
# ============================================================

class CantorLinear(nn.Module):
    """
    Linear layer whose connection topology follows Cantor‑set recursion.

    Each output neuron is assigned a unique Cantor step which determines
    which input connections are active (mask=1) or suppressed (mask=0).
    """

    def __init__(self, cfg: CantorLinearConfig):
        super().__init__()
        self.cfg = cfg
        self.in_features = cfg.in_features
        self.out_features = cfg.out_features
        self.depth = cfg.depth
        self.mask_mode = cfg.mask_mode

        # Parameters
        self.weight = nn.Parameter(
            torch.empty(cfg.out_features, cfg.in_features, dtype=cfg.dtype, device=cfg.device)
        )
        self.bias = nn.Parameter(
            torch.empty(cfg.out_features, dtype=cfg.dtype, device=cfg.device)
        ) if cfg.bias else None

        # Build Cantor mask once
        mask = self._build_cantor_mask(cfg.out_features, cfg.in_features, cfg.depth)
        self.register_buffer("mask", mask, persistent=False)

        # Initialize parameters
        self.reset_parameters()

    # --------------------------------------------------------
    def reset_parameters(self):
        bound = 1.0 / math.sqrt(self.in_features)
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    # --------------------------------------------------------
    def _build_cantor_mask(self, out_dim: int, in_dim: int, depth: int) -> torch.Tensor:
        """
        Constructs a true recursive Cantor mask.
        Each (i, j) connection is retained only if
        the full ternary expansion has no '1's.
        """
        mask = torch.zeros(out_dim, in_dim)
        for i in range(out_dim):
            for j in range(in_dim):
                # Normalize combined index to [0, 1]
                index_val = (i * in_dim + j) / (out_dim * in_dim + 1e-9)
                valid = True
                x = index_val
                for _ in range(depth):
                    x *= 3.0
                    digit = int(x)
                    x -= digit
                    if digit == 1:
                        valid = False
                        break
                if valid:
                    mask[i, j] = 1.0
        return mask

    # --------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply Cantor‑modulated linear transformation.
        If mask_mode == 'prune', removed weights are zeroed.
        If mask_mode == 'scale', removed weights are attenuated by 0.5.
        """
        if self.mask_mode == "scale":
            effective_weight = self.weight * (0.5 + 0.5 * self.mask)
        else:
            effective_weight = self.weight * self.mask
        return F.linear(x, effective_weight, self.bias)


# ============================================================
# ACTIVATION TEST BLOCK
# ============================================================

if __name__ == "__main__":
    cfg = CantorLinearConfig(in_features=256, out_features=256, depth=6)
    layer = CantorLinear(cfg)

    x = torch.randn(4, 256)
    y = layer(x)

    print("Input shape:", x.shape)
    print("Output shape:", y.shape)
    print("Mask density:", layer.mask.mean().item())
    print("First output sample:", y[0, :8])
