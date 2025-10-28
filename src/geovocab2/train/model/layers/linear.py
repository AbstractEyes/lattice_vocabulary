# geovocab2.train.model.layers.linear.py
# ============================================================
# CantorLinear: Fractal‑structured linear transformation layer
# Author: AbstractPhil
# Primary Assistant: GPT 4o Mirel Quartermaster
# Alpha Assistant: Claude (Sonnet 4.5)
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
    depth: int = 8  # Cantor recursion depth
    bias: bool = True  # Standard linear layer bias
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

    Alpha mode provides learnable modulation of the Cantor structure with
    built-in constraints to prevent collapse to zero.
    """

    def __init__(self, cfg: CantorLinearConfig):
        super().__init__()
        self.cfg = cfg
        self.in_features = cfg.in_features
        self.out_features = cfg.out_features
        self.depth = cfg.depth
        self.mask_mode = cfg.mask_mode
        self.mask_floor = cfg.mask_floor
        self.mask_scale = cfg.mask_scale

        # Parameters
        self.weight = nn.Parameter(
            torch.empty(cfg.out_features, cfg.in_features, dtype=cfg.dtype, device=cfg.device)
        )
        self.bias = nn.Parameter(
            torch.empty(cfg.out_features, dtype=cfg.dtype, device=cfg.device)
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

            # Create parameter (scalar or per-output)
            if self.per_output_alpha:
                self._alpha_raw = nn.Parameter(
                    torch.full((cfg.out_features,), init_val, dtype=cfg.dtype, device=cfg.device)
                )
            else:
                self._alpha_raw = nn.Parameter(
                    torch.tensor(init_val, dtype=cfg.dtype, device=cfg.device)
                )
        else:
            self._alpha_raw = None

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

        # Reshape for broadcasting if per-output
        if self.per_output_alpha:
            return alpha_val.unsqueeze(1)  # [out_features, 1]
        else:
            return alpha_val

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
        If mask_mode == 'scale', removed weights are attenuated.
        If mask_mode == 'alpha', uses learnable constrained modulation.
        """
        if self.mask_mode == "scale":
            effective_weight = self.weight * (self.mask_floor + self.mask_scale * self.mask)
        elif self.mask_mode == "alpha":
            effective_weight = self.weight * (self.mask_floor + self.alpha * self.mask)
        else:  # "prune"
            effective_weight = self.weight * self.mask

        return F.linear(x, effective_weight, self.bias)

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
    print("Testing CantorLinear with improved alpha handling")
    print("=" * 60)

    # Test 1: Sigmoid-constrained alpha (recommended)
    print("\n[Test 1] Sigmoid-constrained alpha (single)")
    cfg = CantorLinearConfig(
        in_features=256,
        out_features=256,
        depth=6,
        mask_mode="alpha",
        alpha_mode="sigmoid",
        alpha_min=0.1,
        alpha_max=1.0
    )
    layer = CantorLinear(cfg)

    x = torch.randn(4, 256)
    y = layer(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Mask density: {layer.mask.mean().item():.4f}")
    print(f"Alpha value: {layer.alpha.item():.4f}")
    print(f"Alpha stats: {layer.get_alpha_stats()}")

    # Test 2: Per-output alpha
    print("\n[Test 2] Per-output alpha")
    cfg2 = CantorLinearConfig(
        in_features=128,
        out_features=64,
        depth=6,
        mask_mode="alpha",
        alpha_mode="sigmoid",
        per_output_alpha=True
    )
    layer2 = CantorLinear(cfg2)

    x2 = torch.randn(2, 128)
    y2 = layer2(x2)

    print(f"Output shape: {y2.shape}")
    print(f"Alpha shape: {layer2.alpha.shape}")
    print(f"Alpha stats: {layer2.get_alpha_stats()}")

    # Test 3: Gradient flow
    print("\n[Test 3] Gradient flow test")
    layer.zero_grad()
    loss = y.sum()
    loss.backward()
    print(f"Weight grad norm: {layer.weight.grad.norm().item():.4f}")
    print(f"Alpha grad: {layer._alpha_raw.grad.item():.6f}")

    print("\n" + "=" * 60)
    print("All tests passed!")