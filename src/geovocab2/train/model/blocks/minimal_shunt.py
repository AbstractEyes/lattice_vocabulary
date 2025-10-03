"""
    Minimal Shunt Module for Cross-Attention and Dynamic Projections
    ---------------------------------------------------
    Author: AbstractPhil + GPT-3.5 Turbo
    ---------------------------------------------------
    Description:
    This module implements a minimal shunt architecture that utilizes a cross-attention multihead mechanism
    to facilitate information exchange between two input tensors. It includes dynamic projection layers,
    gating mechanisms, and additional modules for enhanced feature extraction.

    Key Features:
    - Cross-Attention Multihead Mechanism
    - Dynamic Projection Layers with optional normalization and dropout
    - Gating Mechanism for controlled feature modulation

    License: MIT

"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class MinimalShunt(nn.Module):
    def __init__(
        self,
        input_dim=384,
        bneck=256,
        heads=8,
        tau_init=0.01,
        use_norm=True,
        use_do=True,
        do_p=0.1,
        proj_depth=2
    ):
        super().__init__()

        def build_projection(input_dim, output_dim):
            layers = []
            last_dim = input_dim
            if use_norm:
                layers.append(nn.LayerNorm(last_dim))
            for i in range(proj_depth):
                next_dim = bneck * (2 if i == 0 and proj_depth > 1 else 1)
                layers.append(nn.Linear(last_dim, next_dim))
                layers.append(nn.GELU())
                if use_do:
                    layers.append(nn.Dropout(do_p))
                last_dim = next_dim
            layers.append(nn.Linear(last_dim, output_dim))
            return nn.Sequential(*layers)

        self.proj_a = build_projection(input_dim, bneck)
        self.proj_b = build_projection(input_dim, bneck)

        self.cross_a2b = nn.MultiheadAttention(bneck, heads, batch_first=True, dropout=do_p)
        self.cross_b2a = nn.MultiheadAttention(bneck, heads, batch_first=True, dropout=do_p)

        self.delta_proj = build_projection(bneck, input_dim)
        self.gate_proj = nn.Sequential(
            nn.LayerNorm(bneck),
            nn.Linear(bneck, bneck),
            nn.GELU(),
            nn.Linear(bneck, 1),
            nn.Tanh(),
            nn.Sigmoid()
        )

        self.tau = nn.Parameter(torch.full((heads, 1, 1), tau_init))

        # Additional dynamic modules
        self.anchor_proj = build_projection(bneck, input_dim)
        self.cre_basis = nn.Parameter(torch.randn(6, input_dim))  # Collapse Residual Emission basis vectors
        self.attractor = nn.Parameter(torch.randn(1, 1, input_dim))  # Collapse Vector Potential attractor

    def forward(self, input_a: torch.Tensor, input_b: torch.Tensor):
        assert input_a.shape[-1] == input_b.shape[-1], "Input tensors must have the same feature dimension"

        a_b = self.proj_a(input_a)
        b_b = self.proj_b(input_b)

        a2b, attn_a2b = self.cross_a2b(a_b, b_b, b_b, need_weights=True, average_attn_weights=False)
        b2a, attn_b2a = self.cross_b2a(b_b, a_b, a_b, need_weights=True, average_attn_weights=False)

        core = (a2b.mean(1, keepdim=True) + b2a.mean(1, keepdim=True)) / 2

        gate = self.gate_proj(core)
        delta = self.delta_proj(core) * gate
        anchor = self.anchor_proj(core)

        # Collapse Residual Emission: project delta against CRE basis
        cre_energy = torch.stack([torch.sum(delta * b, dim=-1) for b in self.cre_basis], dim=-1)

        # Collapse Vector Potential Loss (alignment with attractor)
        attractor_norm = F.normalize(self.attractor, dim=-1)
        delta_norm = F.normalize(delta, dim=-1)
        cvp_alignment = torch.sum(delta_norm * attractor_norm, dim=-1, keepdim=True)

        return {
            "delta": delta,
            "anchor": anchor,
            "gate": gate,
            "attn_a2b": attn_a2b,
            "attn_b2a": attn_b2a,
            "tau": self.tau,
            "cre_energy": cre_energy,
            "cvp_alignment": cvp_alignment
        }