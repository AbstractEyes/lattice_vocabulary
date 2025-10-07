"""
Devil's Staircase Positional Encoding
--------------------------------------------
An experimental positional encoding based on the Devil's Staircase (Cantor function).
"""

# =========================
# CONFIG
# =========================
CONFIG = {
    "base": 3,  # triadic construction
    "levels": 16,  # triadic depth; ~ log_3(context)
    "features_per_level": 2,  # [measure, pdf_proxy]; can add sin/cos on measure if desired
    "smooth_tau": 0.25,  # temperature for soft trit; lower = sharper steps
    "mode": "soft",  # "soft" or "ste"
    "add_sin_cos": False,  # optionally add sin/cos on measure for compatibility
    "sin_cos_factors": [1, 2, 4],
    "eps": 1e-6,  # numerical
}

# =========================
# IMPLEMENTATION
# =========================
import torch
import torch.nn as nn
import torch.nn.functional as F


def _normalize_positions(pos, seq_len=None):
    """
    Normalize integer or float positions to x in [0, 1].
    If seq_len given, pos in [0, seq_len-1] -> [0,1].
    Otherwise assume pos already in [0,1] and clamp.
    """
    if seq_len is not None:
        x = pos.float() / max(1, (seq_len - 1))
    else:
        x = pos.float().clamp(0.0, 1.0)
    return x


def _soft_trit(y, tau):
    """
    Soft-assign a scalar y in [0, 3) to triadic digit {0,1,2} via relaxed categorical.
    Returns probs p0, p1, p2 that sum to 1.
    We center three bumps at 0.5, 1.5, 2.5 using negative squared distances / tau.
    """
    centers = torch.tensor([0.5, 1.5, 2.5], device=y.device, dtype=y.dtype)
    d2 = (y[..., None] - centers) ** 2
    logits = -d2 / (tau + 1e-8)
    p = F.softmax(logits, dim=-1)  # (..., 3)
    return p  # p[...,0], p[...,1], p[...,2]


def _hard_trit(y):
    """
    Hard digit in {0,1,2} using floor. y in [0,3).
    """
    d = torch.floor(y).clamp(0, 2)
    # return one-hot
    p = F.one_hot(d.long(), num_classes=3).to(y.dtype)
    return p  # (..., 3)


def _cantor_bit_from_trit(p_trit):
    """
    Map tri-digit {0,1,2} to binary digit used by the Cantor function:
        0 -> 0, 1 -> (ambiguous; treat as soft mix), 2 -> 1.
    With soft trits, we take expectation under p_trit:
        bit = p2 * 1 + p1 * alpha + p0 * 0
    We set alpha=0.5 (symmetric), but make it learnable if desired.
    """
    p0, p1, p2 = p_trit[..., 0], p_trit[..., 1], p_trit[..., 2]
    bit = p2 + 0.5 * p1
    return bit  # (...,)


class DevilStaircasePE(nn.Module):
    """
    Devil's Staircase (Cantor) Positional Encoding.
    - Multi-level triadic decomposition with differentiable (soft) or STE (hard) trits.
    - Emits per-level features: [measure_k(x), pdf_proxy_k(x)] and optional sin/cos on the global measure.

    Shapes:
        input: positions tensor of shape (...,) with optional seq_len (int) to normalize
        output: (..., D) where D = levels * features_per_level (+ optional sincos)
    """

    def __init__(self,
                 levels=CONFIG["levels"],
                 features_per_level=CONFIG["features_per_level"],
                 smooth_tau=CONFIG["smooth_tau"],
                 mode=CONFIG["mode"],
                 add_sin_cos=CONFIG["add_sin_cos"],
                 sin_cos_factors=CONFIG["sin_cos_factors"],
                 base=CONFIG["base"],
                 eps=CONFIG["eps"]):
        super().__init__()
        assert base == 3, "Current implementation assumes triadic base=3."
        self.levels = levels
        self.features_per_level = features_per_level
        self.tau = smooth_tau
        self.mode = mode
        self.add_sin_cos = add_sin_cos
        self.sin_cos_factors = sin_cos_factors
        self.base = base
        self.eps = eps

        # optional learnable mixing for the ambiguous trit=1 (advanced use)
        self.alpha = nn.Parameter(torch.tensor(0.5), requires_grad=False)

    def forward(self, positions: torch.Tensor, seq_len: int | None = None):
        """
        positions: (...,) int or float
        seq_len: optional int, for normalization
        returns: (..., D)
        """
        x = _normalize_positions(positions, seq_len=seq_len)  # (...,)
        x = x.clamp(self.eps, 1.0 - self.eps)  # avoid exact endpoints

        # build per-level features
        feats = []
        # also accumulate the global Cantor measure C(x) from bits
        Cx = torch.zeros_like(x)

        for k in range(1, self.levels + 1):
            scale = self.base ** k  # 3^k
            y = (x * scale) % self.base  # y in [0,3)

            if self.mode == "soft":
                p_trit = _soft_trit(y, self.tau)
            elif self.mode == "ste":
                # straight-through estimator: hard forward, soft backward via identity
                y_detached = y.detach()
                p_hard = _hard_trit(y_detached)
                p_soft = _soft_trit(y, self.tau)
                p_trit = p_hard + (p_soft - p_soft.detach())  # STE
            else:
                raise ValueError("mode must be 'soft' or 'ste'.")

            # binary bit for Cantor function at level k
            bit_k = p_trit[..., 2] + self.alpha * p_trit[..., 1]  # 2->1, 1->alpha, 0->0
            Cx = Cx + bit_k * (0.5 ** k)

            # local pdf proxy: magnitude of change near tri-boundaries
            # approximate via entropy of p_trit (low entropy near boundaries after sharpening)
            ent = -(p_trit * (p_trit.clamp_min(1e-8)).log()).sum(dim=-1)
            # invert entropy to suggest "boundary proximity" (normalized)
            pdf_proxy = (1.1 - ent / torch.log(torch.tensor(3.0, device=ent.device)))  # ~[0,1]

            # per-level features
            if self.features_per_level == 1:
                feats.append(bit_k[..., None])
            elif self.features_per_level == 2:
                feats.append(torch.stack([bit_k, pdf_proxy], dim=-1))
            else:
                # extendable: add (bit_k, pdf_proxy, ent, etc.)
                feats.append(torch.stack([bit_k, pdf_proxy], dim=-1))

        F_levels = torch.cat(feats, dim=-1)  # (..., levels * features_per_level)

        # Optional sin/cos band on the global measure C(x)
        if self.add_sin_cos:
            bands = []
            for f in self.sin_cos_factors:
                bands.append(torch.sin(f * 2.0 * torch.pi * Cx))
                bands.append(torch.cos(f * 2.0 * torch.pi * Cx))
            sincos = torch.stack(bands, dim=-1)  # (..., 2*len(factors))
            out = torch.cat([F_levels, sincos], dim=-1)
        else:
            out = F_levels

        return out, Cx  # (features, scalar measure)


# =========================
# ACTIVATION / EXAMPLE
# =========================
if __name__ == "__main__":
    B, L = 2, 16
    pos = torch.arange(L)  # [0..L-1]
    pe = DevilStaircasePE(levels=12, features_per_level=2, smooth_tau=0.2, mode="soft", add_sin_cos=False)
    feats, measure = pe(pos, seq_len=L)
    print("Feats:", feats.shape)  # (L, 24) for levels=12 & fpl=2
    print("Measure range:", measure.min().item(), measure.max().item())
