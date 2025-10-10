"""
Beatrix Staircase Positional Encodings
Based on Cantor's Devil's Staircase PE - let alpha float naturally.
Author: AbstractPhil + Claude Sonnet 4.5 + GPT-4o

License MIT
"""
import math

import torch
from torch import nn
import torch.nn.functional as F

class BeatrixStaircasePositionalEncodings(nn.Module):
    """Based on Cantor's Devil's Staircase PE - let alpha float naturally."""

    def __init__(self, levels=20, features_per_level=4, smooth_tau=0.25, base=3):
        super().__init__()
        self.levels = levels
        self.features_per_level = features_per_level
        self.tau = smooth_tau
        self.base = base

        self.alpha = nn.Parameter(torch.tensor(0.1))

        self.base_features = 2
        if features_per_level > 2:
            self.feature_expansion = nn.Linear(self.base_features, features_per_level)
        else:
            self.feature_expansion = None

    def forward(self, positions, seq_len):
        x = positions.float() / max(1, (seq_len - 1))
        x = x.clamp(1e-6, 1.0 - 1e-6)

        feats = []
        Cx = torch.zeros_like(x)

        for k in range(1, self.levels + 1):
            scale = self.base ** k
            y = (x * scale) % self.base

            centers = torch.tensor([0.5, 1.5, 2.5], device=x.device, dtype=x.dtype)
            d2 = (y.unsqueeze(-1) - centers) ** 2
            logits = -d2 / (self.tau + 1e-8)
            p = F.softmax(logits, dim=-1)

            bit_k = p[..., 2] + self.alpha * p[..., 1]
            Cx = Cx + bit_k * (0.5 ** k)

            ent = -(p * p.clamp_min(1e-8).log()).sum(dim=-1)
            pdf_proxy = 1.1 - ent / math.log(3.0)

            base_feat = torch.stack([bit_k, pdf_proxy], dim=-1)

            if self.feature_expansion is not None:
                level_feat = self.feature_expansion(base_feat)
            else:
                level_feat = base_feat

            feats.append(level_feat)

        pe_levels = torch.stack(feats, dim=1)
        return pe_levels, Cx