# cantor_attention.py
# =============================================================
# CantorAttention: Symbolic fractal scalar attention layer
# Author: AbstractPhil
# Assistant: Quartermaster Mirel
# Date: 2025-10-28
# License: MIT
# =============================================================

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import math


# =============================================================
# CONFIGURATION
# =============================================================

@dataclass
class CantorAttentionConfig:
    in_features: int
    out_features: int
    depth: int = 8                    # Cantor recursion depth
    bias: bool = True
    gate_mode: str = "sigmoid"         # ['scale', 'sigmoid', 'softmax']
    dtype: torch.dtype = torch.float32
    device: str | None = None


# =============================================================
# CANTOR UTILITY
# =============================================================

def cantor_slice(x: float, bits: int = 8) -> float:
    out = 0.0
    factor = 0.5
    for _ in range(bits):
        x *= 3.0
        digit = int(x)
        x -= digit
        if digit == 2:
            out += factor
        elif digit == 1:
            pass
        factor *= 0.5
    return out


# =============================================================
# CANTOR ATTENTION LAYER
# =============================================================

class CantorAttention(nn.Module):
    """
    CantorAttention: A linear layer where each output neuron is scaled
    by a symbolic Cantor-slice gate. This produces structured attention
    weighting without pruning.
    """

    def __init__(self, cfg: CantorAttentionConfig):
        super().__init__()
        self.cfg = cfg
        self.in_features = cfg.in_features
        self.out_features = cfg.out_features
        self.depth = cfg.depth
        self.gate_mode = cfg.gate_mode

        # Linear projection
        self.weight = nn.Parameter(torch.empty(cfg.out_features, cfg.in_features, dtype=cfg.dtype, device=cfg.device))
        self.bias = nn.Parameter(torch.empty(cfg.out_features, dtype=cfg.dtype, device=cfg.device)) if cfg.bias else None

        # Build Cantor gates
        gates = [cantor_slice(i / (cfg.out_features + 1e-9), bits=cfg.depth) for i in range(cfg.out_features)]
        self.register_buffer("gates", torch.tensor(gates, dtype=cfg.dtype), persistent=False)

        # Initialize weights
        self.reset_parameters()

    # ---------------------------------------------------------
    def reset_parameters(self):
        bound = 1.0 / math.sqrt(self.in_features)
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    # ---------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.linear(x, self.weight, self.bias)

        if self.gate_mode == "scale":
            scaled = out * self.gates
        elif self.gate_mode == "sigmoid":
            scaled = out * torch.sigmoid(self.gates)
        elif self.gate_mode == "softmax":
            gate_weights = torch.softmax(self.gates, dim=0)
            scaled = out * gate_weights
        else:
            raise ValueError(f"Invalid gate_mode: {self.gate_mode}")

        return scaled
