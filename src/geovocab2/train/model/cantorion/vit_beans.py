"""
GeoFractalEncoder-v2
====================

Conforms to new axiom:

    K = Cantor-modulated keys
    Q = Simplex-modulated geometric queries
    V = Flow-matched, simplex-gated values

    Z = F(K, Q, V)  → single wide Cantor latent token

Author: AbstractPhil + Mirel
"""

from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import nn
import torch.nn.functional as F

# ----------------------------------------------------------------------
# Imports from your project (existing components)
# ----------------------------------------------------------------------
from geovocab2.train.model.layers.attention.cantor_global import (
    CantorAttention,
    CantorAttentionConfig
)
from geovocab2.shapes.factory.simplex_factory import SimplexFactory


# ==========================================================================
# CONFIG
# ==========================================================================

@dataclass
class ViTCantorCatBeansConfig:
    simplex_seed: int = 42
    image_size: int = 224
    patch_size: int = 16
    in_channels: int = 3

    dim: int = 512            # base token dim
    cantor_dim: int = 256     # K-space dim
    simplex_dim: int = 256    # Q-space dim
    value_dim: int = 256      # V-space dim
    fusion_dim: int = 2048    # ultra wide Cantor latent

    num_routes: int = 16
    simplex_k: int = 4        # 4-simplex / pentachoron

    # CantorAttention params
    num_heads: int = 8
    cantor_depth: int = 8
    max_seq_len: int = 512_000
    local_window: int = 64
    dropout: float = 0.1

    def make_cantor_cfg(self):
        return CantorAttentionConfig(
            dim=self.dim,
            num_heads=self.num_heads,
            depth=self.cantor_depth,
            max_seq_len=self.max_seq_len,
            local_window=self.local_window,
            dropout=self.dropout,
        )


# ==========================================================================
# PATCH EMBEDDING
# ==========================================================================

class PatchEmbed(nn.Module):
    def __init__(self, cfg: ViTCantorCatBeansConfig):
        super().__init__()
        assert cfg.image_size % cfg.patch_size == 0
        self.num_patches = (cfg.image_size // cfg.patch_size) ** 2
        self.proj = nn.Conv2d(cfg.in_channels, cfg.dim,
                              cfg.patch_size, cfg.patch_size)

    def forward(self, x):
        x = self.proj(x)                 # B, D, H/p, W/p
        return x.flatten(2).transpose(1, 2)  # B, N, D


# ==========================================================================
# 1. CANTOR-MODULATED KEYS (K)
# ==========================================================================

class KeyProjector(nn.Module):
    """
    K = CantorAttention(tokens)
    Followed by route-based Cantor fingerprint association.
    """
    def __init__(self, cfg):
        super().__init__()
        self.cantor_global = CantorAttention(cfg.make_cantor_cfg())
        self.route_logits = nn.Linear(cfg.dim, cfg.num_routes)
        self.to_cantor = nn.Linear(cfg.dim, cfg.cantor_dim, bias=False)

    def forward(self, tokens):
        # Cantor global mixing first
        k = self.cantor_global(tokens)  # [B, N, D]

        # Route assignment
        logits = self.route_logits(k)   # [B, N, R]
        route_w = logits.transpose(1,2)
        route_w = F.softmax(route_w, dim=-1)  # [B,R,N]

        # K_fingerprints
        cantor_tokens = self.to_cantor(k)    # [B,N,cantor_dim]
        K = torch.einsum("brn,bnd->brd", route_w, cantor_tokens)
        return K, route_w, k


# ==========================================================================
# 2. SIMPLEX-MODULATED QUERIES (Q)
# ==========================================================================

class SimplexQueryProjector(nn.Module):
    """
    Q = SimplexFactory projection of K_cantor
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        factory = SimplexFactory(
            k=cfg.simplex_k,
            embed_dim=cfg.simplex_dim,
            method="regular",
            scale=1.0
        )
        verts = factory.build_torch(device="cpu", dtype=torch.float32)  # [V, D]
        self.register_buffer("vertices", verts.unsqueeze(0), persistent=False)
        self.to_simplex = nn.Linear(cfg.cantor_dim, cfg.simplex_dim)

    def forward(self, K):
        # K: [B,R,cantor_dim]
        q = self.to_simplex(K)  # [B,R,simplex_dim]

        B,R,_ = q.shape
        V = self.vertices.shape[1]

        verts = self.vertices.to(q.device, q.dtype)
        verts = verts.unsqueeze(1).expand(B,R,V,-1)

        q_exp = q.unsqueeze(2)
        dots = (q_exp * verts).sum(-1)
        bary = F.softmax(dots, dim=-1)

        Q = torch.einsum("brv,brvd->brd", bary, verts)
        return Q, bary


# ==========================================================================
# 3. FLOW-MATCHED VALUE PROJECTION (V)
# ==========================================================================

class FlowMatchedValue(nn.Module):
    """
    V = flow-matched value projection
      = MLP(flow(tokens)) ⊙ simplex_gate
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.to_value = nn.Sequential(
            nn.Linear(cfg.dim, cfg.value_dim),
            nn.GELU(),
            nn.Linear(cfg.value_dim, cfg.value_dim)
        )
        self.simplex_gate = nn.Linear(cfg.simplex_dim, cfg.value_dim)

    def forward(self, tokens, route_weights, Q):
        # Project to value space
        V_raw = self.to_value(tokens)      # [B,N,value_dim]

        # Route aggregate
        V_route = torch.einsum("brn,bnd->brd", route_weights, V_raw)

        # Simplex gating
        gate = torch.sigmoid(self.simplex_gate(Q))
        V = V_route * (1 + gate)
        return V


# ==========================================================================
# 4. FUSION CORE (ONE WIDE CANTOR TOKEN)
# ==========================================================================

class CantorFusionCore(nn.Module):
    """
    Z = F(K, Q, V)
    Collapses all route experts into one ultra-wide Cantor latent.
    """
    def __init__(self, cfg):
        super().__init__()
        self.in_dim = cfg.cantor_dim + cfg.simplex_dim + cfg.value_dim
        self.fuse = nn.Linear(self.in_dim, cfg.fusion_dim)
        self.act = nn.GELU()
        self.post = nn.Linear(cfg.fusion_dim, cfg.fusion_dim)

    def forward(self, K, Q, V):
        fused = torch.cat([K, Q, V], dim=-1)
        fused = self.act(self.fuse(fused))
        # collapse experts to 1 token
        Z = fused.mean(dim=1)
        Z = self.act(self.post(Z))
        return Z


# ==========================================================================
# TOP-LEVEL ENCODER
# ==========================================================================

class ViTCatBeans(nn.Module):
    """
    Final encoder:
        Z = F(K(Q(tokens)), V(tokens))
    Produces a single wide Cantor latent token.
    """
    def __init__(self, cfg: ViTCantorCatBeansConfig):
        super().__init__()
        self.cfg = cfg

        self.patch = PatchEmbed(cfg)
        self.key_projector = KeyProjector(cfg)
        self.query_projector = SimplexQueryProjector(cfg)
        self.value_projector = FlowMatchedValue(cfg)
        self.fusion = CantorFusionCore(cfg)

    def forward(self, x):
        tokens = self.patch(x)  # [B,N,D]

        K, route_w, k_tokens = self.key_projector(tokens)
        Q, bary = self.query_projector(K)
        V = self.value_projector(k_tokens, route_w, Q)

        Z = self.fusion(K, Q, V)
        return Z

if __name__ == "__main__":
    print("[Cantorian: Cat-Beans] Running sanity tests...")

    cfg = ViTCantorCatBeansConfig(
        image_size=224,
        patch_size=16,
        dim=512,
        cantor_dim=256,
        simplex_dim=256,
        value_dim=256,
        fusion_dim=2048,
        num_routes=16,
        simplex_k=4,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Cat-Beans] Device: {device}")

    model = ViTCatBeans(cfg).to(device)
    model.eval()

    B = 2
    x = torch.randn(B, 3, cfg.image_size, cfg.image_size, device=device)

    with torch.no_grad():
        Z = model(x)

    print(f"[Cat-Beans] Output latent shape: {Z.shape}")
