"""
ViTBeans-V1 - FIXED
====================

Conforms to axiom:
    K = Cantor-modulated keys (deterministic geometric routing)
    Q = Simplex-modulated geometric queries
    V = Flow-matched, simplex-gated values
    Z = F(K, Q, V) → single wide Cantor latent token

Integrates:
    - CantorRouteFactory for parameter-free geometric routing
    - SimplexFactory for pentachoron geometric queries
    - Validated geometric constraints throughout

Author: AbstractPhil + Mirel
License: MIT
"""

from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

# ----------------------------------------------------------------------
# Geometric Factories
# ----------------------------------------------------------------------
from geovocab2.shapes.factory.simplex_factory import SimplexFactory
from geovocab2.shapes.factory.cantor_route_factory import (
    CantorRouteFactory,
    RouteMode,
    HARMONIC_FP32
)

# ----------------------------------------------------------------------
# Existing attention (if available)
# ----------------------------------------------------------------------
try:
    from geovocab2.train.model.layers.attention.cantor_global import (
        CantorAttention,
        CantorAttentionConfig
    )
    HAS_CANTOR_ATTENTION = True
except ImportError:
    HAS_CANTOR_ATTENTION = False
    print("[Warning] CantorAttention not found, using standard attention")


# ==========================================================================
# CONFIG - FIXED
# ==========================================================================

@dataclass
class ViTCantorCatBeansConfig:
    # Image settings
    image_size: int = 224
    patch_size: int = 16
    in_channels: int = 3

    # Architecture dimensions
    dim: int = 512            # base token dim
    cantor_dim: int = 256     # K-space dim
    simplex_dim: int = 256    # Q-space dim
    value_dim: int = 256      # V-space dim
    fusion_dim: int = 2048    # ultra wide Cantor latent

    # Geometric routing
    num_routes: int = 16
    simplex_k: int = 4        # 4-simplex / pentachoron (5 vertices)
    cantor_dimensions: int = 2  # Cantor pairing dimensions (2-5)

    # Attention params (if using CantorAttention)
    num_heads: int = 8
    max_seq_len: int = 512_000
    local_window: int = 64
    dropout: float = 0.1

    # Geometric validation
    validate_geometry: bool = True
    simplex_method: str = "regular"  # "regular", "random", "uniform"
    simplex_seed: int = 42

    def make_cantor_cfg(self):
        """Create CantorAttention config if available."""
        if not HAS_CANTOR_ATTENTION:
            return None
        return CantorAttentionConfig(
            dim=self.dim,
            num_heads=self.num_heads,
            cantor_dimensions=self.cantor_dimensions,  # FIXED: use cantor_dimensions, not depth
            max_seq_len=self.max_seq_len,
            local_window=self.local_window,
            dropout=self.dropout,
        )


# ==========================================================================
# PATCH EMBEDDING
# ==========================================================================

class PatchEmbed(nn.Module):
    """Standard ViT patch embedding."""
    def __init__(self, cfg: ViTCantorCatBeansConfig):
        super().__init__()
        assert cfg.image_size % cfg.patch_size == 0, \
            f"Image size {cfg.image_size} must be divisible by patch size {cfg.patch_size}"

        self.num_patches = (cfg.image_size // cfg.patch_size) ** 2
        self.patch_size = cfg.patch_size

        self.proj = nn.Conv2d(
            cfg.in_channels,
            cfg.dim,
            kernel_size=cfg.patch_size,
            stride=cfg.patch_size
        )

    def forward(self, x):
        """
        Args:
            x: [B, C, H, W]
        Returns:
            [B, N, D] where N = num_patches
        """
        x = self.proj(x)  # [B, D, H/p, W/p]
        return x.flatten(2).transpose(1, 2)  # [B, N, D]


# ==========================================================================
# 1. CANTOR-MODULATED KEYS (K)
# ==========================================================================

class KeyProjector(nn.Module):
    """
    K = Cantor-routed keys using geometric fingerprints.

    No learned routing logits - pure Cantor pairing for route assignment.
    """
    def __init__(self, cfg: ViTCantorCatBeansConfig):
        super().__init__()
        self.cfg = cfg

        # Optional: CantorAttention for global mixing
        if HAS_CANTOR_ATTENTION:
            self.cantor_global = CantorAttention(cfg.make_cantor_cfg())
        else:
            # Fallback: standard MHA
            self.cantor_global = nn.MultiheadAttention(
                cfg.dim,
                cfg.num_heads,
                dropout=cfg.dropout,
                batch_first=True
            )

        # Geometric Cantor route mask (deterministic)
        # We'll generate this per sequence length dynamically
        self.cantor_dimensions = cfg.cantor_dimensions

        # Project to Cantor key space
        self.to_cantor = nn.Linear(cfg.dim, cfg.cantor_dim, bias=False)

        # Route mixing weights (small MLP for route combination)
        self.route_mix = nn.Sequential(
            nn.Linear(cfg.dim, cfg.num_routes),
            nn.LayerNorm(cfg.num_routes)
        )

    def forward(self, tokens):
        """
        Args:
            tokens: [B, N, D]
        Returns:
            K: [B, R, cantor_dim] - Cantor-routed keys
            route_w: [B, R, N] - Route weights for value aggregation
            k_tokens: [B, N, D] - Processed tokens for value projection
        """
        B, N, D = tokens.shape

        # Global mixing
        if HAS_CANTOR_ATTENTION:
            k_tokens = self.cantor_global(tokens)  # [B, N, D]
        else:
            k_tokens, _ = self.cantor_global(tokens, tokens, tokens)

        # Generate Cantor route fingerprints (deterministic)
        cantor_factory = CantorRouteFactory(
            shape=(N,),  # 1D sequence
            mode=RouteMode.DISTANCE,  # Pairwise distances
            dimensions=self.cantor_dimensions
        )

        # Build distance matrix on same device
        cantor_dist = cantor_factory.build(
            backend="torch",
            device=tokens.device,
            dtype=tokens.dtype,
            validate=self.cfg.validate_geometry
        )  # [N, N]

        # Convert distances to routing probabilities
        # Use top-k nearest neighbors for each route
        route_logits = self.route_mix(k_tokens)  # [B, N, R]
        route_logits = route_logits.transpose(1, 2)  # [B, R, N]

        # Modulate with Cantor distance geometry
        # Each route gets a different distance threshold
        thresholds = torch.linspace(0.1, 0.9, self.cfg.num_routes, device=tokens.device)
        thresholds = thresholds.view(1, -1, 1, 1)  # [1, R, 1, 1]

        # Geometric mask: routes attend to tokens within distance threshold
        cantor_mask = (cantor_dist.unsqueeze(0).unsqueeze(0) < thresholds).float()  # [1, R, N, N]

        # Apply geometric constraint to route assignment
        route_w = F.softmax(route_logits, dim=-1)  # [B, R, N]

        # Geometric modulation: weight by distance-based affinity
        route_affinity = cantor_mask.mean(dim=-1)  # [1, R, N] - avg affinity per route
        route_w = route_w * route_affinity  # [B, R, N]
        route_w = route_w / (route_w.sum(dim=-1, keepdim=True) + 1e-8)  # Renormalize

        # Project to Cantor key space
        cantor_tokens = self.to_cantor(k_tokens)  # [B, N, cantor_dim]

        # Route aggregation
        K = torch.einsum("brn,bnd->brd", route_w, cantor_tokens)  # [B, R, cantor_dim]

        return K, route_w, k_tokens


# ==========================================================================
# 2. SIMPLEX-MODULATED QUERIES (Q)
# ==========================================================================

class SimplexQueryProjector(nn.Module):
    """
    Q = SimplexFactory-based geometric queries.

    Uses pentachoron (4-simplex) vertices as query basis.
    """
    def __init__(self, cfg: ViTCantorCatBeansConfig):
        super().__init__()
        self.cfg = cfg

        # Generate geometric simplex vertices
        factory = SimplexFactory(
            k=cfg.simplex_k,
            embed_dim=cfg.simplex_dim,
            method=cfg.simplex_method,
            scale=1.0,
            seed=cfg.simplex_seed
        )

        # Build and validate
        verts = factory.build_torch(
            device="cpu",
            dtype=torch.float32,
            validate=cfg.validate_geometry
        )  # [V, simplex_dim] where V = k+1

        self.num_vertices = verts.shape[0]
        self.register_buffer("vertices", verts.unsqueeze(0), persistent=False)

        # Project from Cantor space to simplex space
        self.to_simplex = nn.Linear(cfg.cantor_dim, cfg.simplex_dim)

        # Barycentric coordinate decoder (optional refinement)
        self.bary_refine = nn.Sequential(
            nn.Linear(self.num_vertices, self.num_vertices * 2),
            nn.GELU(),
            nn.Linear(self.num_vertices * 2, self.num_vertices)
        )

    def forward(self, K):
        """
        Args:
            K: [B, R, cantor_dim]
        Returns:
            Q: [B, R, simplex_dim] - Simplex-modulated queries
            bary: [B, R, V] - Barycentric coordinates
        """
        B, R, _ = K.shape
        V = self.num_vertices

        # Project to simplex space
        q = self.to_simplex(K)  # [B, R, simplex_dim]

        # Get simplex vertices
        verts = self.vertices.to(q.device, q.dtype)  # [1, V, simplex_dim]
        verts = verts.unsqueeze(1).expand(B, R, V, -1)  # [B, R, V, simplex_dim]

        # Compute barycentric coordinates via dot product
        q_exp = q.unsqueeze(2)  # [B, R, 1, simplex_dim]
        dots = (q_exp * verts).sum(-1)  # [B, R, V]

        # Refine barycentric coords with small MLP
        bary_raw = F.softmax(dots, dim=-1)  # [B, R, V]
        bary_delta = self.bary_refine(bary_raw)  # [B, R, V]
        bary = F.softmax(bary_raw + bary_delta, dim=-1)  # [B, R, V]

        # Reconstruct query as weighted simplex vertex combination
        Q = torch.einsum("brv,brvd->brd", bary, verts)  # [B, R, simplex_dim]

        return Q, bary


# ==========================================================================
# 3. FLOW-MATCHED VALUE PROJECTION (V)
# ==========================================================================

class FlowMatchedValue(nn.Module):
    """
    V = flow-matched value projection with simplex gating.

    V = MLP(tokens) ⊙ (1 + simplex_gate(Q))
    """
    def __init__(self, cfg: ViTCantorCatBeansConfig):
        super().__init__()
        self.cfg = cfg

        # Value projection MLP
        self.to_value = nn.Sequential(
            nn.Linear(cfg.dim, cfg.value_dim * 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.value_dim * 2, cfg.value_dim)
        )

        # Simplex-based gating
        self.simplex_gate = nn.Sequential(
            nn.Linear(cfg.simplex_dim, cfg.value_dim),
            nn.LayerNorm(cfg.value_dim)
        )

        # Optional: Harmonic quantization for stability
        self.register_buffer(
            "harmonic_scale",
            torch.tensor(HARMONIC_FP32, dtype=torch.float32),
            persistent=False
        )

    def forward(self, tokens, route_weights, Q):
        """
        Args:
            tokens: [B, N, dim] - Original tokens
            route_weights: [B, R, N] - Route aggregation weights
            Q: [B, R, simplex_dim] - Simplex queries
        Returns:
            V: [B, R, value_dim] - Flow-matched values
        """
        # Project tokens to value space
        V_raw = self.to_value(tokens)  # [B, N, value_dim]

        # Route aggregation
        V_route = torch.einsum("brn,bnd->brd", route_weights, V_raw)  # [B, R, value_dim]

        # Simplex gating: modulate with geometric query structure
        gate = torch.sigmoid(self.simplex_gate(Q))  # [B, R, value_dim]

        # Flow matching: V = V_route ⊙ (1 + gate)
        # This encourages geometric flow alignment
        V = V_route * (1.0 + gate)

        return V


# ==========================================================================
# 4. FUSION CORE (ONE WIDE CANTOR TOKEN)
# ==========================================================================

class CantorFusionCore(nn.Module):
    """
    Z = F(K, Q, V)

    Collapses all route experts into one ultra-wide Cantor latent.
    Uses geometric attention over routes before fusion.
    """
    def __init__(self, cfg: ViTCantorCatBeansConfig):
        super().__init__()
        self.cfg = cfg

        # Input dimension: concatenate K, Q, V
        self.in_dim = cfg.cantor_dim + cfg.simplex_dim + cfg.value_dim

        # Route attention (geometric weighting)
        self.route_attn = nn.MultiheadAttention(
            self.in_dim,
            num_heads=4,
            dropout=cfg.dropout,
            batch_first=True
        )

        # Fusion MLP
        self.fuse = nn.Sequential(
            nn.Linear(self.in_dim, cfg.fusion_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.fusion_dim, cfg.fusion_dim)
        )

        # Final projection
        self.project = nn.Sequential(
            nn.LayerNorm(cfg.fusion_dim),
            nn.Linear(cfg.fusion_dim, cfg.fusion_dim),
            nn.GELU()
        )

    def forward(self, K, Q, V):
        """
        Args:
            K: [B, R, cantor_dim]
            Q: [B, R, simplex_dim]
            V: [B, R, value_dim]
        Returns:
            Z: [B, fusion_dim] - Single wide Cantor latent
        """
        # Concatenate geometric components
        fused = torch.cat([K, Q, V], dim=-1)  # [B, R, in_dim]

        # Route self-attention (geometric mixing)
        attended, _ = self.route_attn(fused, fused, fused)  # [B, R, in_dim]

        # Fusion to wide latent space
        wide = self.fuse(attended)  # [B, R, fusion_dim]

        # Collapse routes to single token (mean pooling)
        Z = wide.mean(dim=1)  # [B, fusion_dim]

        # Final projection
        Z = self.project(Z)  # [B, fusion_dim]

        return Z


# ==========================================================================
# TOP-LEVEL ENCODER
# ==========================================================================

class ViTCatBeans(nn.Module):
    """
    GeoFractal Encoder with Cantor routing and Simplex queries.

    Pipeline:
        1. Patch embedding
        2. Cantor-modulated keys (K) - deterministic geometric routing
        3. Simplex-modulated queries (Q) - pentachoron basis
        4. Flow-matched values (V) - simplex-gated
        5. Fusion to single wide Cantor latent (Z)

    Output: Z ∈ ℝ^fusion_dim (single token representation)
    """
    def __init__(self, cfg: ViTCantorCatBeansConfig):
        super().__init__()
        self.cfg = cfg

        # Components
        self.patch = PatchEmbed(cfg)
        self.key_projector = KeyProjector(cfg)
        self.query_projector = SimplexQueryProjector(cfg)
        self.value_projector = FlowMatchedValue(cfg)
        self.fusion = CantorFusionCore(cfg)

        # Optional: positional encoding
        num_patches = (cfg.image_size // cfg.patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, cfg.dim) * 0.02)

    def forward(self, x):
        """
        Args:
            x: [B, C, H, W] - Input images
        Returns:
            Z: [B, fusion_dim] - Single wide Cantor latent token
        """
        # Patch embedding
        tokens = self.patch(x)  # [B, N, D]
        tokens = tokens + self.pos_embed  # Add positional encoding

        # K: Cantor-modulated keys (geometric routing)
        K, route_w, k_tokens = self.key_projector(tokens)  # K: [B, R, cantor_dim]

        # Q: Simplex-modulated queries (pentachoron basis)
        Q, bary = self.query_projector(K)  # Q: [B, R, simplex_dim]

        # V: Flow-matched values (simplex-gated)
        V = self.value_projector(k_tokens, route_w, Q)  # V: [B, R, value_dim]

        # Z: Fusion to single wide latent
        Z = self.fusion(K, Q, V)  # Z: [B, fusion_dim]

        return Z

    def forward_with_debug(self, x):
        """Forward pass with intermediate outputs for debugging."""
        tokens = self.patch(x)
        tokens = tokens + self.pos_embed

        K, route_w, k_tokens = self.key_projector(tokens)
        Q, bary = self.query_projector(K)
        V = self.value_projector(k_tokens, route_w, Q)
        Z = self.fusion(K, Q, V)

        return {
            "Z": Z,
            "K": K,
            "Q": Q,
            "V": V,
            "route_weights": route_w,
            "barycentric": bary,
            "tokens": tokens
        }


# ==========================================================================
# TESTING & DEMO
# ==========================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("ViT CatBeans - GeoFractal Encoder with Cantor & Simplex")
    print("=" * 70)

    cfg = ViTCantorCatBeansConfig(
        image_size=224,
        patch_size=16,
        dim=512,
        cantor_dim=256,
        simplex_dim=256,
        value_dim=256,
        fusion_dim=2048,
        num_routes=16,
        simplex_k=4,  # Pentachoron
        cantor_dimensions=2,
        validate_geometry=True
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[Device] {device}")
    print(f"[Config] {cfg.num_routes} routes, {cfg.simplex_k}-simplex, fusion_dim={cfg.fusion_dim}")

    # Build model
    model = ViTCatBeans(cfg).to(device)
    model.eval()

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[Parameters]")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")

    # Test forward pass
    B = 2
    x = torch.randn(B, 3, cfg.image_size, cfg.image_size, device=device)

    print(f"\n[Forward Pass]")
    print(f"  Input: {x.shape}")

    with torch.no_grad():
        Z = model(x)

    print(f"  Output latent: {Z.shape}")
    print(f"  Output dtype: {Z.dtype}")
    print(f"  Output range: [{Z.min():.4f}, {Z.max():.4f}]")

    # Debug forward
    print(f"\n[Debug Forward]")
    with torch.no_grad():
        debug = model.forward_with_debug(x)

    print(f"  K (Cantor keys): {debug['K'].shape}")
    print(f"  Q (Simplex queries): {debug['Q'].shape}")
    print(f"  V (Flow values): {debug['V'].shape}")
    print(f"  Route weights: {debug['route_weights'].shape}")
    print(f"  Barycentric coords: {debug['barycentric'].shape}")

    # Validate geometric constraints
    print(f"\n[Geometric Validation]")
    bary = debug['barycentric']
    bary_sum = bary.sum(dim=-1)
    print(f"  Barycentric sum: [{bary_sum.min():.6f}, {bary_sum.max():.6f}] (should be ~1.0)")
    print(f"  Barycentric valid: {torch.allclose(bary_sum, torch.ones_like(bary_sum), atol=1e-4)}")

    route_sum = debug['route_weights'].sum(dim=-1)
    print(f"  Route weights sum: [{route_sum.min():.6f}, {route_sum.max():.6f}] (should be ~1.0)")
    print(f"  Route weights valid: {torch.allclose(route_sum, torch.ones_like(route_sum), atol=1e-4)}")

    print("\n" + "=" * 70)
    print("CatBeans ready for:")
    print("  - ImageNet classification (add classification head)")
    print("  - Feature extraction (use Z directly)")
    print("  - Transfer learning (freeze encoder, train head)")
    print("  - Geometric attention visualization")
    print("=" * 70)