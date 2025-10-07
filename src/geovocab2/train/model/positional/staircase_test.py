# devils_staircase_offset_solidity.py
# ============================================================
# CONFIG
# ============================================================
CONFIG = {
    # Devil's Staircase PE params
    "pe": {
        "base": 3,
        "levels": 12,
        "features_per_level": 2,
        "smooth_tau": 0.2,
        "mode": "soft",          # "soft" or "ste"
        "add_sin_cos": False,
        "sin_cos_factors": [1, 2, 4],
        "eps": 1e-6,
    },

    # Synthesis / evaluation params
    "window_len": 128,           # size of each sampled window
    "global_horizon": 4096,      # max absolute range for global norm test
    "num_trials": 10000,         # number of random offset trials
    "device": "cpu",             # "cuda" if available
    "dtype": "float32",

    # Metrics / numerical tolerances
    "relative_tol": 1e-6,        # expected near-zero MSE tolerance for Test A
    "pair_dist_sample": 256,     # subsample pairs to limit O(N^2) cost (<= window_len^2)
    "report_every": 1000,        # logging cadence
}

# ============================================================
# IMPLEMENTATION (Devil's Staircase PE)
# ============================================================
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

def _normalize_positions(pos, seq_len=None):
    """
    Normalize integer/float positions to x in [0,1].
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
    Soft-assign y in [0,3) to triadic digit {0,1,2} via relaxed categorical.
    Centers at 0.5, 1.5, 2.5 with negative squared distance / tau.
    """
    centers = torch.tensor([0.5, 1.5, 2.5], device=y.device, dtype=y.dtype)
    d2 = (y[..., None] - centers) ** 2
    logits = -d2 / (tau + 1e-8)
    p = F.softmax(logits, dim=-1)  # (..., 3)
    return p

def _hard_trit(y):
    """
    Hard digit in {0,1,2} using floor. y in [0,3).
    """
    d = torch.floor(y).clamp(0, 2)
    p = F.one_hot(d.long(), num_classes=3).to(y.dtype)
    return p

class DevilStaircasePE(nn.Module):
    """
    Devil's Staircase (Cantor) Positional Encoding.
    Produces per-level features: [bit_k, pdf_proxy_k] and optional sin/cos over global measure C(x).

    forward(inputs):
        positions: (...,) integer or float
        seq_len: optional normalization length (int)
    returns:
        features: (..., D) where D = levels * features_per_level (+ optional 2*len(sin_cos_factors))
        measure:  (...,) global Cantor measure C(x) \in [0,1]
    """
    def __init__(self,
                 levels=12,
                 features_per_level=2,
                 smooth_tau=0.2,
                 mode="soft",
                 add_sin_cos=False,
                 sin_cos_factors=(1, 2, 4),
                 base=3,
                 eps=1e-6):
        super().__init__()
        assert base == 3, "Implementation assumes triadic base=3."
        self.levels = levels
        self.features_per_level = features_per_level
        self.tau = smooth_tau
        self.mode = mode
        self.add_sin_cos = add_sin_cos
        self.sin_cos_factors = list(sin_cos_factors)
        self.base = base
        self.eps = eps

        # ambiguous trit=1 mixing coefficient (fixed 0.5 by default, enable grad if desired)
        self.alpha = nn.Parameter(torch.tensor(0.5), requires_grad=False)

    def forward(self, positions: torch.Tensor, seq_len: int | None = None):
        x = _normalize_positions(positions, seq_len=seq_len).clamp(self.eps, 1.0 - self.eps)
        feats = []
        Cx = torch.zeros_like(x)

        for k in range(1, self.levels + 1):
            scale = self.base ** k  # 3^k
            y = (x * scale) % self.base  # in [0,3)

            if self.mode == "soft":
                p_trit = _soft_trit(y, self.tau)
            elif self.mode == "ste":
                y_det = y.detach()
                p_hard = _hard_trit(y_det)
                p_soft = _soft_trit(y, self.tau)
                p_trit = p_hard + (p_soft - p_soft.detach())
            else:
                raise ValueError("mode must be 'soft' or 'ste'.")

            # bit_k: map tri-digit to binary via (0->0, 2->1, 1->alpha)
            bit_k = p_trit[..., 2] + self.alpha * p_trit[..., 1]
            Cx = Cx + bit_k * (0.5 ** k)

            # pdf proxy via inverted (normalized) entropy
            ent = -(p_trit * p_trit.clamp_min(1e-8).log()).sum(dim=-1)
            pdf_proxy = (1.1 - ent / math.log(3.0))  # ~[0,1]

            if self.features_per_level == 1:
                feats.append(bit_k[..., None])
            else:
                feats.append(torch.stack([bit_k, pdf_proxy], dim=-1))

        F_levels = torch.cat(feats, dim=-1)  # (..., levels * features_per_level)

        if self.add_sin_cos:
            bands = []
            for f in self.sin_cos_factors:
                bands.append(torch.sin(f * 2.0 * math.pi * Cx))
                bands.append(torch.cos(f * 2.0 * math.pi * Cx))
            sincos = torch.stack(bands, dim=-1)
            out = torch.cat([F_levels, sincos], dim=-1)
        else:
            out = F_levels

        return out, Cx

# ============================================================
# OFFSET SOLIDITY TESTS
# ============================================================
@torch.no_grad()
def pairwise_distances(X: torch.Tensor, sample_pairs: int | None = None) -> torch.Tensor:
    """
    Compute condensed pairwise distances for a window of embeddings X: (N, D).
    Returns a 1D tensor of length N*(N-1)/2 or a subsampled vector if sample_pairs is set.
    Uses Euclidean distances by default (stable and interpretable).
    """
    N = X.shape[0]
    if sample_pairs is not None:
        # random sample of index pairs (i<j)
        pairs = []
        for _ in range(sample_pairs):
            i = random.randrange(0, N - 1)
            j = random.randrange(i + 1, N)
            pairs.append((i, j))
        d = []
        for (i, j) in pairs:
            d.append(torch.linalg.vector_norm(X[i] - X[j]))
        return torch.stack(d, dim=0)

    # full condensed vector
    dists = []
    for i in range(N - 1):
        v = torch.linalg.vector_norm(X[i + 1:] - X[i], dim=1)
        dists.append(v)
    return torch.cat(dists, dim=0)

@torch.no_grad()
def relative_solidity_test(pe: DevilStaircasePE,
                           window_len: int,
                           device: str,
                           dtype: torch.dtype) -> float:
    """
    Test A: Relative solidity.
    Use local normalization (seq_len = window_len) for two windows at different offsets.
    Since positions are always 0..window_len-1 under local norm, embeddings should match -> MSE ~ 0.
    """
    pos0 = torch.arange(window_len, device=device).to(dtype)
    feats0, _ = pe(pos0, seq_len=window_len)

    # pick a random offset, but still feed relative positions (0..W-1)
    pos_rel = torch.arange(window_len, device=device).to(dtype)
    feats1, _ = pe(pos_rel, seq_len=window_len)

    mse = F.mse_loss(feats0, feats1).item()
    return mse

@torch.no_grad()
def global_robustness_test(pe: DevilStaircasePE,
                           window_len: int,
                           global_horizon: int,
                           device: str,
                           dtype: torch.dtype,
                           pair_dist_sample: int | None = None) -> tuple[float, float]:
    """
    Test B: Global robustness to offsets under global normalization.
    1) Take baseline window at offset 0: positions [0..W-1], seq_len = global_horizon
    2) Take shifted window at random offset o: positions [o..o+W-1], seq_len = global_horizon
    3) Compare pairwise distance structures between windows (cosine similarity and L1 error)

    Returns:
        (cos_sim, l1_err)
    """
    assert global_horizon >= window_len + 1, "global_horizon must exceed window_len."

    pos_base = torch.arange(window_len, device=device).to(dtype)
    feats_base, _ = pe(pos_base, seq_len=global_horizon)

    max_off = global_horizon - window_len
    o = random.randrange(0, max_off + 1)
    pos_shift = torch.arange(o, o + window_len, device=device).to(dtype)
    feats_shift, _ = pe(pos_shift, seq_len=global_horizon)

    # distance fingerprints
    d0 = pairwise_distances(feats_base, sample_pairs=pair_dist_sample)
    d1 = pairwise_distances(feats_shift, sample_pairs=pair_dist_sample)

    # align shapes (in case of sampling variation, but here they match)
    n = min(d0.shape[0], d1.shape[0])
    d0 = d0[:n]
    d1 = d1[:n]

    # cosine similarity & L1 error between condensed vectors
    cos_sim = F.cosine_similarity(d0, d1, dim=0).item()
    l1_err = torch.mean(torch.abs(d0 - d1)).item()
    return cos_sim, l1_err

def summarize_stats(values: list[float]) -> tuple[float, float, float]:
    t = torch.tensor(values)
    return t.mean().item(), t.min().item(), t.max().item()

# ============================================================
# ACTIVATION / MAIN
# ============================================================
def main(cfg: dict = CONFIG):
    device = cfg["device"]
    dtype = getattr(torch, cfg["dtype"])
    torch.set_grad_enabled(False)

    pe_cfg = cfg["pe"]
    pe = DevilStaircasePE(
        levels=pe_cfg["levels"],
        features_per_level=pe_cfg["features_per_level"],
        smooth_tau=pe_cfg["smooth_tau"],
        mode=pe_cfg["mode"],
        add_sin_cos=pe_cfg["add_sin_cos"],
        sin_cos_factors=pe_cfg["sin_cos_factors"],
        base=pe_cfg["base"],
        eps=pe_cfg["eps"],
    ).to(device=device, dtype=dtype)
    pe.eval()

    W = cfg["window_len"]
    H = cfg["global_horizon"]
    T = cfg["num_trials"]
    pair_sample = cfg["pair_dist_sample"]
    rel_tol = cfg["relative_tol"]

    rel_mses = []
    cos_sims = []
    l1_errs = []

    # Precompute a single baseline for relative solidity (deterministic)
    base_rel_mse = relative_solidity_test(pe, W, device, dtype)
    rel_mses.append(base_rel_mse)

    for t in range(1, T + 1):
        # Relative solidity should be invariant—periodically re-check
        mse = relative_solidity_test(pe, W, device, dtype)
        rel_mses.append(mse)

        # Global robustness under absolute offsets
        cos_sim, l1_err = global_robustness_test(pe, W, H, device, dtype, pair_dist_sample=pair_sample)
        cos_sims.append(cos_sim)
        l1_errs.append(l1_err)

        if t % cfg["report_every"] == 0:
            rel_mean, rel_min, rel_max = summarize_stats(rel_mses)
            cos_mean, cos_min, cos_max = summarize_stats(cos_sims)
            l1_mean, l1_min, l1_max = summarize_stats(l1_errs)
            print(f"[{t:5d}/{T}] Relative MSE: mean={rel_mean:.3e} min={rel_min:.3e} max={rel_max:.3e} "
                  f"| Global pairwise: cos={cos_mean:.4f} [{cos_min:.4f},{cos_max:.4f}] "
                  f"L1={l1_mean:.4e} [{l1_min:.4e},{l1_max:.4e}]")

    # Final summary
    rel_mean, rel_min, rel_max = summarize_stats(rel_mses)
    cos_mean, cos_min, cos_max = summarize_stats(cos_sims)
    l1_mean, l1_min, l1_max = summarize_stats(l1_errs)

    print("\n=== Devil’s Staircase Positional Encoding — Offset Solidity Report ===")
    print(f"- Trials: {T}, Window: {W}, GlobalHorizon: {H}, Levels: {pe.levels}, Mode: {pe.mode}, Tau: {pe.tau}")
    print(f"- Test A (Relative Solidity; local norm SeqLen={W}):")
    print(f"    MSE mean={rel_mean:.3e}  min={rel_min:.3e}  max={rel_max:.3e}  (target ≤ {rel_tol:.1e})")
    print(f"- Test B (Global Robustness; global norm SeqLen={H}):")
    print(f"    Pairwise Cosine mean={cos_mean:.4f}  range=[{cos_min:.4f}, {cos_max:.4f}]")
    print(f"    Pairwise L1 mean={l1_mean:.4e}      range=[{l1_min:.4e}, {l1_max:.4e}]")

if __name__ == "__main__":
    main()
