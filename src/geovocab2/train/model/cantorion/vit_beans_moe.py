"""
ViTBeans-V1 - Corrected Per-Expert Architecture
=================================================

Key fix: Experts maintain identity through to classification!
- Fusion returns per-expert features
- Each expert classifies independently with Cantor masks
- Late fusion via weighted expert voting on logits
"""

from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import nn
import torch.nn.functional as F
import math

from geovocab2.shapes.factory.simplex_factory import SimplexFactory
from geovocab2.shapes.factory.cantor_route_factory import (
    CantorRouteFactory,
    RouteMode,
    HARMONIC_FP32
)

try:
    from geovocab2.train.model.layers.attention.cantor_global import (
        CantorAttention,
        CantorAttentionConfig
    )
    HAS_CANTOR_ATTENTION = True
except ImportError:
    HAS_CANTOR_ATTENTION = False


@dataclass
class ViTCantorCatBeansConfig:
    image_size: int = 224
    patch_size: int = 16
    in_channels: int = 3

    dim: int = 512
    cantor_dim: int = 256
    simplex_dim: int = 256
    value_dim: int = 256
    fusion_dim: int = 2048

    num_routes: int = 16
    simplex_k: int = 4
    cantor_dimensions: int = 2

    # Per-expert Cantor dimensions
    expert_cantor_dims: list[int] = None

    # Masking configuration
    use_cantor_masks: bool = True
    mask_alpha: float = 0.5  # Mask strength [0,1]
    mask_floor: float = 0.3  # Minimum weight preservation

    num_heads: int = 8
    max_seq_len: int = 512_000
    local_window: int = 64
    dropout: float = 0.1

    validate_geometry: bool = False
    simplex_method: str = "regular"
    simplex_seed: int = 42

    def __post_init__(self):
        if self.expert_cantor_dims is None:
            # Cycle through 2, 3, 4, 5 dimensions
            self.expert_cantor_dims = [2 + (i % 4) for i in range(self.num_routes)]

    def make_cantor_cfg(self):
        if not HAS_CANTOR_ATTENTION:
            return None
        return CantorAttentionConfig(
            dim=self.dim,
            num_heads=self.num_heads,
            cantor_dimensions=self.cantor_dimensions,
            max_seq_len=self.max_seq_len,
            local_window=self.local_window,
            dropout=self.dropout,
        )


# ==========================================================================
# GLOBAL CANTOR FINGERPRINT CACHE
# ==========================================================================

class CantorFingerprintCache(nn.Module):
    """Global Cantor fingerprint cache using CantorRouteFactory."""
    def __init__(
        self,
        num_experts: int,
        max_seq_len: int,
        expert_cantor_dims: list[int]
    ):
        super().__init__()
        self.num_experts = num_experts
        self.max_seq_len = max_seq_len
        self.expert_dims = expert_cantor_dims

        # Create per-expert factories
        self.factories = [
            CantorRouteFactory(
                shape=(max_seq_len, 1),
                mode=RouteMode.NORMALIZED,
                dimensions=expert_cantor_dims[i],
                harmonic_quantize=True
            )
            for i in range(num_experts)
        ]

        # Pre-compute fingerprints
        fingerprint_cache = self._build_fingerprint_cache()
        self.register_buffer('fingerprint_cache', fingerprint_cache, persistent=False)

        # Expert centers
        expert_centers = torch.linspace(0, 1, num_experts + 1)[:-1] + 0.5 / num_experts
        self.register_buffer('expert_centers', expert_centers, persistent=False)

        # Expert assignments
        expert_assignments = self._assign_expert_regions()
        self.register_buffer('expert_assignments', expert_assignments, persistent=False)

    def _build_fingerprint_cache(self) -> torch.Tensor:
        cache = torch.zeros(self.max_seq_len, self.num_experts, dtype=torch.float32)

        for expert_id in range(self.num_experts):
            fingerprints_2d = self.factories[expert_id].build(
                backend="torch",
                device="cpu",
                dtype=torch.float32,
                validate=False
            )
            cache[:, expert_id] = fingerprints_2d.squeeze(-1)

        return cache

    def _assign_expert_regions(self) -> torch.Tensor:
        assignments = torch.zeros(self.max_seq_len, dtype=torch.long)

        for pos in range(self.max_seq_len):
            fingerprints = self.fingerprint_cache[pos]
            avg_fingerprint = fingerprints.mean()
            distances = torch.abs(self.expert_centers - avg_fingerprint)
            assignments[pos] = distances.argmin()

        return assignments

    def get_expert_routing_weights(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Get soft routing weights [seq_len, num_experts]."""
        fingerprints = self.fingerprint_cache[:seq_len].to(device)
        fingerprints_avg = fingerprints.mean(dim=1, keepdim=True)
        distances = torch.abs(fingerprints_avg - self.expert_centers.view(1, -1).to(device))
        weights = F.softmax(-distances * 10.0, dim=1)
        return weights

    def get_expert_feature_mask(self, expert_id: int, feature_dim: int, device: torch.device) -> torch.Tensor:
        """Generate Cantor mask for expert's feature dimension."""
        mask_factory = CantorRouteFactory(
            shape=(feature_dim, 1),
            mode=RouteMode.ALPHA,
            dimensions=self.expert_dims[expert_id],
            harmonic_quantize=True
        )

        mask = mask_factory.build(
            backend="torch",
            device=device,
            dtype=torch.float32,
            validate=False
        )

        return mask.squeeze(-1)  # [feature_dim]


# ==========================================================================
# PATCH EMBEDDING
# ==========================================================================

class PatchEmbed(nn.Module):
    """Standard patch embedding."""
    def __init__(self, cfg: ViTCantorCatBeansConfig):
        super().__init__()
        assert cfg.image_size % cfg.patch_size == 0

        self.num_patches = (cfg.image_size // cfg.patch_size) ** 2
        self.patch_size = cfg.patch_size

        self.proj = nn.Conv2d(
            cfg.in_channels,
            cfg.dim,
            kernel_size=cfg.patch_size,
            stride=cfg.patch_size
        )

    def forward(self, x):
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


# ==========================================================================
# KEY PROJECTOR
# ==========================================================================

class KeyProjector(nn.Module):
    def __init__(self, cfg: ViTCantorCatBeansConfig, fingerprint_cache: CantorFingerprintCache):
        super().__init__()
        self.cfg = cfg
        self.fingerprint_cache = fingerprint_cache

        # Global attention
        if HAS_CANTOR_ATTENTION:
            self.cantor_global = CantorAttention(cfg.make_cantor_cfg())
        else:
            self.cantor_global = nn.MultiheadAttention(
                cfg.dim, cfg.num_heads, dropout=cfg.dropout, batch_first=True
            )

        self.to_cantor = nn.Linear(cfg.dim, cfg.cantor_dim, bias=False)

    def forward(self, tokens):
        B, N, D = tokens.shape
        device = tokens.device

        # Global mixing
        if HAS_CANTOR_ATTENTION:
            k_tokens = self.cantor_global(tokens)
        else:
            k_tokens, _ = self.cantor_global(tokens, tokens, tokens)

        # Get routing weights
        route_w = self.fingerprint_cache.get_expert_routing_weights(N, device)
        route_w = route_w.unsqueeze(0).expand(B, -1, -1).transpose(1, 2)  # [B, R, N]

        # Project to Cantor space
        cantor_tokens = self.to_cantor(k_tokens)

        # Aggregate to routes
        K = torch.einsum("brn,bnd->brd", route_w, cantor_tokens)

        return K, route_w, k_tokens


# ==========================================================================
# SIMPLEX QUERY PROJECTOR
# ==========================================================================

class SimplexQueryProjector(nn.Module):
    def __init__(self, cfg: ViTCantorCatBeansConfig):
        super().__init__()
        self.cfg = cfg

        factory = SimplexFactory(
            k=cfg.simplex_k,
            embed_dim=cfg.simplex_dim,
            method=cfg.simplex_method,
            scale=1.0,
            seed=cfg.simplex_seed
        )

        verts = factory.build_torch(device="cpu", dtype=torch.float32, validate=cfg.validate_geometry)
        self.num_vertices = verts.shape[0]
        self.register_buffer("vertices", verts.unsqueeze(0), persistent=False)

        self.to_simplex = nn.Linear(cfg.cantor_dim, cfg.simplex_dim)

        self.bary_refine = nn.Sequential(
            nn.Linear(self.num_vertices, self.num_vertices * 2),
            nn.GELU(),
            nn.Linear(self.num_vertices * 2, self.num_vertices)
        )

    def forward(self, K):
        B, R, _ = K.shape
        V = self.num_vertices

        q = self.to_simplex(K)
        verts = self.vertices.to(q.device, q.dtype).unsqueeze(1).expand(B, R, V, -1)

        q_exp = q.unsqueeze(2)
        dots = (q_exp * verts).sum(-1)

        bary_raw = F.softmax(dots, dim=-1)
        bary_delta = self.bary_refine(bary_raw)
        bary = F.softmax(bary_raw + bary_delta, dim=-1)

        Q = torch.einsum("brv,brvd->brd", bary, verts)

        return Q, bary


# ==========================================================================
# FLOW-MATCHED VALUE PROJECTOR
# ==========================================================================

class FlowMatchedValue(nn.Module):
    def __init__(self, cfg: ViTCantorCatBeansConfig):
        super().__init__()
        self.cfg = cfg

        self.value_proj = nn.Sequential(
            nn.Linear(cfg.dim, cfg.value_dim * 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.value_dim * 2, cfg.value_dim)
        )

        self.simplex_gate = nn.Sequential(
            nn.Linear(cfg.simplex_dim, cfg.value_dim),
            nn.LayerNorm(cfg.value_dim)
        )

        self.register_buffer(
            "harmonic_scale",
            torch.tensor(HARMONIC_FP32, dtype=torch.float32),
            persistent=False
        )

    def forward(self, tokens, route_weights, Q):
        V_raw = self.value_proj(tokens)
        V_route = torch.einsum("brn,bnd->brd", route_weights, V_raw)
        gate = torch.sigmoid(self.simplex_gate(Q))
        V = V_route * (1.0 + gate)
        return V


# ==========================================================================
# PER-EXPERT PROCESSOR
# ==========================================================================

class CantorExpert(nn.Module):
    """Expert with Cantor-masked linear layers."""
    def __init__(
        self,
        cfg: ViTCantorCatBeansConfig,
        expert_id: int,
        fingerprint_cache: CantorFingerprintCache
    ):
        super().__init__()
        self.expert_id = expert_id
        self.fingerprint_cache = fingerprint_cache
        self.cfg = cfg

        # Standard linear layers
        self.k_layer = nn.Linear(cfg.cantor_dim, cfg.fusion_dim)
        self.q_layer = nn.Linear(cfg.simplex_dim, cfg.fusion_dim)
        self.v_layer = nn.Linear(cfg.value_dim, cfg.fusion_dim)

        self.flow_net = nn.Sequential(
            nn.Linear(cfg.fusion_dim, cfg.fusion_dim),
            nn.LayerNorm(cfg.fusion_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.fusion_dim, cfg.fusion_dim)
        )

        if cfg.use_cantor_masks:
            self._register_masks()

    def _register_masks(self):
        """Pre-compute Cantor masks."""
        device = next(self.parameters()).device if len(list(self.parameters())) > 0 else "cpu"

        k_mask = self.fingerprint_cache.get_expert_feature_mask(
            self.expert_id, self.cfg.fusion_dim, device
        )
        q_mask = self.fingerprint_cache.get_expert_feature_mask(
            self.expert_id, self.cfg.fusion_dim, device
        )
        v_mask = self.fingerprint_cache.get_expert_feature_mask(
            self.expert_id, self.cfg.fusion_dim, device
        )
        flow_mask = self.fingerprint_cache.get_expert_feature_mask(
            self.expert_id, self.cfg.fusion_dim, device
        )

        self.register_buffer('k_mask', k_mask, persistent=False)
        self.register_buffer('q_mask', q_mask, persistent=False)
        self.register_buffer('v_mask', v_mask, persistent=False)
        self.register_buffer('flow_mask', flow_mask, persistent=False)

    def _apply_mask(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Apply Cantor mask."""
        if not self.cfg.use_cantor_masks:
            return x

        mask = mask.to(x.device)
        scaled_mask = self.cfg.mask_floor + self.cfg.mask_alpha * mask
        return x * scaled_mask

    def forward(self, k, q, v):
        """Flow match with Cantor masks."""
        k_transformed = self._apply_mask(self.k_layer(k), self.k_mask)
        q_transformed = self._apply_mask(self.q_layer(q), self.q_mask)
        v_transformed = self._apply_mask(self.v_layer(v), self.v_mask)

        # Flow matching
        flow_direction = q_transformed * v_transformed
        flow_state = k_transformed + flow_direction

        flow_output = self.flow_net(flow_state)
        expert_output = self._apply_mask(flow_output, self.flow_mask)

        return expert_output


# ==========================================================================
# CANTOR MoE FUSION (CORRECTED)
# ==========================================================================

class CantorMoEFusion(nn.Module):
    """
    MoE fusion that PRESERVES per-expert features.
    ✓ FIXED: Returns both fused Z and per-expert features
    """
    def __init__(self, cfg: ViTCantorCatBeansConfig, fingerprint_cache: CantorFingerprintCache):
        super().__init__()
        self.cfg = cfg
        self.num_experts = cfg.num_routes
        self.fingerprint_cache = fingerprint_cache

        # Create experts
        self.experts = nn.ModuleList([
            CantorExpert(cfg, i, fingerprint_cache)
            for i in range(self.num_experts)
        ])

        # Expert attention
        self.expert_attention = nn.MultiheadAttention(
            cfg.fusion_dim,
            num_heads=4,
            dropout=cfg.dropout,
            batch_first=True
        )

        # Gate network
        self.expert_gate = nn.Sequential(
            nn.Linear(cfg.fusion_dim * self.num_experts, cfg.fusion_dim),
            nn.LayerNorm(cfg.fusion_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.fusion_dim, self.num_experts)
        )

        # Final projection (for auxiliary tasks)
        self.output_proj = nn.Sequential(
            nn.Linear(cfg.fusion_dim, cfg.fusion_dim),
            nn.LayerNorm(cfg.fusion_dim),
            nn.GELU()
        )

    def forward(self, K, Q, V, bary=None, return_expert_features=False):
        """
        Args:
            K, Q, V: Route features
            bary: Barycentric weights
            return_expert_features: If True, return per-expert features

        Returns:
            If return_expert_features=False:
                Z: [B, fusion_dim] - Fused representation
            If return_expert_features=True:
                dict with:
                    - expert_features: [B, num_experts, fusion_dim]
                    - expert_weights: [B, num_experts]
                    - fused_z: [B, fusion_dim]
        """
        B, R, _ = K.shape

        # Process each expert
        expert_outputs = []
        for i in range(self.num_experts):
            k_i = K[:, i, :]
            q_i = Q[:, i, :]
            v_i = V[:, i, :]

            expert_out = self.experts[i](k_i, q_i, v_i)
            expert_outputs.append(expert_out)

        all_experts = torch.stack(expert_outputs, dim=1)  # [B, R, fusion_dim]

        # Geometric attention
        attended_experts, _ = self.expert_attention(all_experts, all_experts, all_experts)

        # Gate network
        flattened = attended_experts.reshape(B, -1)
        gate_logits = self.expert_gate(flattened)

        # Barycentric weighting
        if bary is not None:
            bary_entropy = -(bary * torch.log(bary + 1e-8)).sum(dim=-1)
            gate_logits = gate_logits - bary_entropy

        expert_weights = F.softmax(gate_logits, dim=-1)

        if return_expert_features:
            # Return per-expert features for classification
            Z_fused = torch.einsum('br,brd->bd', expert_weights, attended_experts)
            Z_fused = self.output_proj(Z_fused)

            return {
                'expert_features': attended_experts,  # [B, R, fusion_dim]
                'expert_weights': expert_weights,     # [B, R]
                'fused_z': Z_fused                    # [B, fusion_dim]
            }
        else:
            # Original behavior
            Z = torch.einsum('br,brd->bd', expert_weights, attended_experts)
            Z = self.output_proj(Z)
            return Z


# ==========================================================================
# GEOMETRIC CLASSIFICATION HEAD
# ==========================================================================

class GeometricClassificationHead(nn.Module):
    """
    Geometric classification with TRUE per-expert voting.
    ✓ FIXED: Each expert maintains identity through classification
    """
    def __init__(
        self,
        cfg: ViTCantorCatBeansConfig,
        num_classes: int,
        fingerprint_cache: CantorFingerprintCache,
        use_class_simplex: bool = True
    ):
        super().__init__()
        self.cfg = cfg
        self.num_classes = num_classes
        self.num_experts = cfg.num_routes
        self.fingerprint_cache = fingerprint_cache
        self.use_class_simplex = use_class_simplex

        # Per-expert classifiers
        self.expert_classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(cfg.fusion_dim, cfg.fusion_dim // 2),
                nn.LayerNorm(cfg.fusion_dim // 2),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(cfg.fusion_dim // 2, num_classes)
            )
            for _ in range(self.num_experts)
        ])

        # Optional: Class simplex embeddings
        if use_class_simplex:
            self.class_simplex_embeddings = nn.Parameter(
                torch.randn(num_classes, cfg.simplex_dim) * 0.02
            )
            self.class_simplex_projs = nn.ModuleList([
                nn.Linear(cfg.fusion_dim, cfg.simplex_dim)
                for _ in range(self.num_experts)
            ])

        self.norm = nn.LayerNorm(num_classes)

    def _get_expert_mask(self, expert_id: int, device: torch.device) -> torch.Tensor:
        """Get Cantor mask for expert's class predictions."""
        return self.fingerprint_cache.get_expert_feature_mask(
            expert_id, self.num_classes, device
        )

    def forward(self, expert_features: torch.Tensor, expert_weights: torch.Tensor) -> torch.Tensor:
        """
        Args:
            expert_features: [B, num_experts, fusion_dim]
            expert_weights: [B, num_experts]

        Returns:
            logits: [B, num_classes]
        """
        B, R, D = expert_features.shape
        device = expert_features.device

        expert_logits = []

        # Each expert votes independently
        for i in range(self.num_experts):
            expert_feat = expert_features[:, i, :]

            # Expert classification
            logits = self.expert_classifiers[i](expert_feat)

            # Apply Cantor mask
            if self.cfg.use_cantor_masks:
                mask = self._get_expert_mask(i, device)
                scaled_mask = self.cfg.mask_floor + self.cfg.mask_alpha * mask
                logits = logits * scaled_mask

            # Add simplex affinity
            if self.use_class_simplex:
                expert_simplex = self.class_simplex_projs[i](expert_feat)
                expert_simplex = F.normalize(expert_simplex, dim=-1)

                class_emb = F.normalize(self.class_simplex_embeddings, dim=-1)
                simplex_scores = torch.matmul(expert_simplex, class_emb.T)

                # 70% classifier, 30% simplex
                logits = 0.7 * logits + 0.3 * simplex_scores

            expert_logits.append(logits)

        # Stack and weight
        all_logits = torch.stack(expert_logits, dim=1)  # [B, R, num_classes]
        expert_weights_exp = expert_weights.unsqueeze(-1)
        combined_logits = (all_logits * expert_weights_exp).sum(dim=1)

        combined_logits = self.norm(combined_logits)

        return combined_logits

    def get_expert_predictions(self, expert_features: torch.Tensor, expert_weights: torch.Tensor):
        """Get detailed per-expert predictions."""
        B, R, D = expert_features.shape
        device = expert_features.device

        expert_logits = []

        for i in range(self.num_experts):
            expert_feat = expert_features[:, i, :]
            logits = self.expert_classifiers[i](expert_feat)

            if self.cfg.use_cantor_masks:
                mask = self._get_expert_mask(i, device)
                scaled_mask = self.cfg.mask_floor + self.cfg.mask_alpha * mask
                logits = logits * scaled_mask

            expert_logits.append(logits)

        all_logits = torch.stack(expert_logits, dim=1)

        return {
            "expert_logits": all_logits,
            "expert_weights": expert_weights,
            "expert_features": expert_features
        }


# ==========================================================================
# TOP-LEVEL ENCODER
# ==========================================================================

class ViTCatBeans(nn.Module):
    """GeoFractal Encoder with per-expert paths."""
    def __init__(self, cfg: ViTCantorCatBeansConfig):
        super().__init__()
        self.cfg = cfg

        num_patches = (cfg.image_size // cfg.patch_size) ** 2

        self.fingerprint_cache = CantorFingerprintCache(
            num_experts=cfg.num_routes,
            max_seq_len=num_patches,
            expert_cantor_dims=cfg.expert_cantor_dims
        )

        self.patch = PatchEmbed(cfg)
        self.key_projector = KeyProjector(cfg, self.fingerprint_cache)
        self.query_projector = SimplexQueryProjector(cfg)
        self.value_projector = FlowMatchedValue(cfg)
        self.fusion = CantorMoEFusion(cfg, self.fingerprint_cache)

        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, cfg.dim) * 0.02)

    def forward(self, x, return_expert_features=False):
        """
        Args:
            x: [B, C, H, W]
            return_expert_features: Whether to return per-expert features

        Returns:
            If return_expert_features=False: Z [B, fusion_dim]
            If return_expert_features=True: dict with expert features
        """
        tokens = self.patch(x)
        tokens = tokens + self.pos_embed

        K, route_w, k_tokens = self.key_projector(tokens)
        Q, bary = self.query_projector(K)
        V = self.value_projector(k_tokens, route_w, Q)

        return self.fusion(K, Q, V, bary, return_expert_features=return_expert_features)


if __name__ == "__main__":
    print("=" * 70)
    print("ViT CatBeans - Corrected Per-Expert Architecture")
    print("=" * 70)

    cfg = ViTCantorCatBeansConfig(
        image_size=32,
        patch_size=4,
        dim=384,
        cantor_dim=192,
        simplex_dim=192,
        value_dim=192,
        fusion_dim=1536,
        num_routes=12,
        use_cantor_masks=True,
        mask_alpha=0.5,
        mask_floor=0.3,
        validate_geometry=False
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[Device] {device}")
    print(f"[Experts] {cfg.num_routes}")

    model = ViTCatBeans(cfg).to(device)
    total_params = sum(p.numel() for p in model.parameters())

    print(f"[Parameters] {total_params:,}")

    x = torch.randn(2, 3, cfg.image_size, cfg.image_size, device=device)

    with torch.no_grad():
        # Test both modes
        Z = model(x, return_expert_features=False)
        print(f"\n[Fused Output] {Z.shape}")

        expert_output = model(x, return_expert_features=True)
        print(f"[Expert Features] {expert_output['expert_features'].shape}")
        print(f"[Expert Weights] {expert_output['expert_weights'].shape}")

    print("\n✓ Per-expert architecture ready!")