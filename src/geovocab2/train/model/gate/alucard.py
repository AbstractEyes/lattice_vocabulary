# alucard_formula_block.py
# Author: Phil + Mirel (Quartermaster)
# Purpose: Batched, learnable Alucard-style fold-gate for David blocks.

from dataclasses import dataclass
from typing import Optional, Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


# src/geovocab2/train/config/alucard_formula_config.py

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from geovocab2.train.config.config_base import BaseConfig

@dataclass
class AlucardFormulaConfig(BaseConfig):
    """Configuration for Alucard Formula Block."""
    model_dim: int = 768
    ffn_expand: float = 2.0
    dropout: float = 0.05
    norm: str = "rms"
    stride: int = 4
    fold_mode: str = "mean"        # "mean" | "max" | "conv1d"
    conv_kernel: int = 3
    harmonic_levels: int = 4
    cfg_temperature: float = 0.5
    cfg_clamp: float = 2.0
    shiva_cool: float = 0.15
    shiva_learnable: bool = True
    residual_alpha: float = 1.0
    init_scale: float = 0.5

    # inherited BaseConfig provides:
    # - to_dict(), from_dict()
    # - save_json(path)
    # - load_json(path)
    # - repr(), validate_fields(), freeze()

# ──────────────────────────────────────────────────────────────────────────────
# HELPERS (minimal, no monkey-patching)
# ──────────────────────────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,T,D]
        norm = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return self.weight * x * norm

def select_norm(kind: str, d: int) -> nn.Module:
    if kind == "rms":
        return RMSNorm(d)
    elif kind == "layer":
        return nn.LayerNorm(d)
    else:
        raise ValueError(f"Unknown norm kind: {kind}")

def cosine_cfg_blend(base: torch.Tensor,
                     target: torch.Tensor,
                     cfg: torch.Tensor,
                     clamp: float = 2.0) -> torch.Tensor:
    """
    base, target: [B,T,D], cfg: [B,1,1] or [B,T,1]
    returns: [B,T,D]
    """
    cfg = torch.clamp(cfg, -clamp, clamp)
    # Cosine-signed scaling for stability
    b = F.normalize(base, dim=-1)
    t = F.normalize(target, dim=-1)
    cos = (b * t).sum(dim=-1, keepdim=True)  # [B,T,1]
    scale = cfg * (1.0 + cos) * 0.5  # map cos∈[-1,1] → [0,1]
    return base + scale * (target - base)

def stride_fold(x: torch.Tensor,
                stride: int,
                mode: str = "mean",
                conv1d: Optional[nn.Conv1d] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    x: [B,T,D]  → folded: [B,T,D], mask: [B,T,1]
    Windowed folding without length shrinkage (align to left; partial windows masked).
    """
    B, T, D = x.shape
    # Build index windows
    idx = torch.arange(T, device=x.device)[None, :, None]  # [1,T,1]
    win = torch.arange(stride, device=x.device)[None, None, :]  # [1,1,S]
    gather = idx + win  # [1,T,S]
    gather = torch.clamp(gather, max=T-1)  # repeat last for tail

    x_exp = x[:, :, None, :].expand(B, T, stride, D)             # [B,T,S,D]
    x_g   = torch.gather(x, 1, gather.expand(B, T, stride))      # [B,T,S] (for indices)
    # We need to gather along T for D; use take_along_dim:
    x_take = torch.take_along_dim(x_exp, gather[..., None].expand(B, T, stride, D), dim=1)

    if mode == "mean":
        folded = x_take.mean(dim=2)
    elif mode == "max":
        folded, _ = x_take.max(dim=2)
    elif mode == "conv1d":
        assert conv1d is not None, "conv1d module required for fold_mode='conv1d'"
        # reshape to [B*D, 1, T]
        z = x.transpose(1, 2)  # [B,D,T]
        z = z.reshape(B * D, 1, T)
        z = conv1d(z)  # [B*D,1,T]
        folded = z.reshape(B, D, T).transpose(1, 2)  # [B,T,D]
    else:
        raise ValueError(f"Unknown fold mode: {mode}")

    # Mask marks positions whose window would overflow (true=valid)
    last_full = T - (stride - 1)
    valid = torch.arange(T, device=x.device) < last_full
    mask = valid.view(1, T, 1).expand(B, T, 1).to(x.dtype)

    return folded, mask


# ──────────────────────────────────────────────────────────────────────────────
# MAIN BLOCK
# ──────────────────────────────────────────────────────────────────────────────

class AlucardFormulaBlock(nn.Module):
    """
    Batched Alucard-style fold gate for David.
    - Stride folding over temporal tokens
    - Harmonic gating (multi-frequency positional gates)
    - Cosine-CFG blend toward a learned style projection
    - Shiva 'cool' noise (learnable amplitude)
    - Residual projection with FFN

    Inputs:
        x: [B,T,D]  (David feature stream at insertion depth)
        style: Optional[Tensor] [B,1,D] or [B,T,D] (style/warper guidance)
        cfg_scale: Optional[Tensor] [B] or [B,1]  (per-sample CFG)
        pos: Optional[Tensor] [B,T]  (positions; if None, arange)

    Returns:
        y: [B,T,D]
        aux: Dict[str,Tensor] with gates/masks/diagnostics
    """
    def __init__(self, cfg: Dict = CONFIG):
        super().__init__()
        d = cfg["model_dim"]
        self.cfg = cfg
        self.norm_in = select_norm(cfg["norm"], d)

        # Folding projection pre/post
        self.pre = nn.Linear(d, d, bias=False)
        nn.init.orthogonal_(self.pre.weight, gain=cfg["init_scale"])

        # Optional conv kernel for fold_mode='conv1d'
        if cfg["fold_mode"] == "conv1d":
            k = cfg["conv_kernel"]
            pad = (k - 1) // 2
            self.fold_conv = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=k, padding=pad)
        else:
            self.fold_conv = None

        # Harmonic gate parameters (sin/cos stack → gate)
        H = cfg["harmonic_levels"]
        self.harm_fc = nn.Linear(2 * H, d)
        nn.init.zeros_(self.harm_fc.weight); nn.init.zeros_(self.harm_fc.bias)

        # Style projection head (target embedding direction)
        self.style_proj = nn.Linear(d, d, bias=False)
        nn.init.orthogonal_(self.style_proj.weight, gain=cfg["init_scale"])

        # Shiva noise
        self.shiva_amp = nn.Parameter(torch.tensor(cfg["shiva_cool"])) if cfg["shiva_learnable"] else None

        # Merge + FFN
        hidden = int(d * cfg["ffn_expand"])
        self.merge = nn.Linear(2 * d, d)
        self.ffn = nn.Sequential(
            nn.Linear(d, hidden),
            nn.GELU(),
            nn.Dropout(cfg["dropout"]),
            nn.Linear(hidden, d),
            nn.Dropout(cfg["dropout"]),
        )
        self.norm_out = select_norm(cfg["norm"], d)

        # Residual scale
        self.res_alpha = cfg["residual_alpha"]

    def forward(self,
                x: torch.Tensor,
                style: Optional[torch.Tensor] = None,
                cfg_scale: Optional[torch.Tensor] = None,
                pos: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        x: [B,T,D], style: [B,1,D] or [B,T,D], cfg_scale: [B] or [B,1], pos: [B,T]
        """
        B, T, D = x.shape
        assert D == self.cfg["model_dim"], f"Dim mismatch: got {D}, expected {self.cfg['model_dim']}"

        # Norm-in + pre-proj
        z = self.norm_in(x)
        z = self.pre(z)

        # Stride fold (batched, no shrink)
        folded, mask = stride_fold(
            z, stride=self.cfg["stride"], mode=self.cfg["fold_mode"], conv1d=self.fold_conv
        )
        # Harmonic gates
        if pos is None:
            pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)  # [B,T]
        posf = pos.float() / max(1, (T - 1))
        hs = []
        for k in range(1, self.cfg["harmonic_levels"] + 1):
            hs.append(torch.sin(2.0 * torch.pi * k * posf))
            hs.append(torch.cos(2.0 * torch.pi * k * posf))
        Hst = torch.stack(hs, dim=-1)  # [B,T,2H]
        gate = torch.sigmoid(self.harm_fc(Hst))  # [B,T,D]

        # Apply gate on folded
        gfold = folded * gate

        # Style projection & cosine-CFG
        if style is None:
            style = z.mean(dim=1, keepdim=True)  # fallback: global pooled
        if style.shape[1] == 1:
            style = style.expand(B, T, D)
        target = self.style_proj(style)

        # cfg scaling tensor
        if cfg_scale is None:
            cfg_scale = torch.full((B, 1, 1), self.cfg["cfg_temperature"], device=x.device, dtype=x.dtype)
        elif cfg_scale.ndim == 1:
            cfg_scale = cfg_scale.view(B, 1, 1)
        elif cfg_scale.ndim == 2:
            cfg_scale = cfg_scale.view(B, -1, 1)
        cfg_scale = cfg_scale.to(x.dtype)

        cfg_blend = cosine_cfg_blend(gfold, target, cfg_scale, clamp=self.cfg["cfg_clamp"])

        # Shiva cool (structured noise)
        if (self.cfg["shiva_cool"] > 0.0) or (self.shiva_amp is not None):
            amp = self.shiva_amp.abs() if self.shiva_amp is not None else x.new_tensor(self.cfg["shiva_cool"])
            noise = torch.randn_like(cfg_blend) * amp
            cfg_blend = cfg_blend + noise * mask  # mask tail padding

        # Merge prior with transformed
        merged = torch.cat([x, cfg_blend], dim=-1)  # [B,T,2D]
        y = self.merge(merged)
        y = self.ffn(self.norm_out(y)) + self.res_alpha * x

        aux = {
            "mask": mask,               # [B,T,1]
            "gate_mean": gate.mean(dim=(1, 2)),  # [B]
            "fold_mode": torch.tensor(0 if self.cfg["fold_mode"]=="mean" else (1 if self.cfg["fold_mode"]=="max" else 2),
                                      device=x.device),
            "cfg_scale": cfg_scale.squeeze(-1).mean(dim=1),  # [B] or [B,T]→[B]
            "shiva_amp": (self.shiva_amp.abs() if self.shiva_amp is not None else x.new_tensor(self.cfg["shiva_cool"]))
        }
        return y, aux


# ──────────────────────────────────────────────────────────────────────────────
# ACTIVATION / EXAMPLE (base)
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(0)
    B, T, D = 2, 32, CONFIG["model_dim"]
    x = torch.randn(B, T, D)

    block = AlucardFormulaBlock(CONFIG)
    style = torch.randn(B, 1, D)                # single style vector per sample
    cfg_scale = torch.tensor([0.7, 1.2])        # different per sample

    y, aux = block(x, style=style, cfg_scale=cfg_scale)
    print("y:", y.shape, "gate_mean:", aux["gate_mean"])
