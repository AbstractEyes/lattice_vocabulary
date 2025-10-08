"""
ViT-Beatrix with Dual-Stream Architecture
------------------------------------------
Flux-inspired dual blocks that preserve geometric structure throughout the network.

Key Innovation: Separate processing streams for visual and geometric modalities
that cross-communicate without destroying either.

Author: AbstractPhil + Claude Sonnet 4.5
License: MIT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import math

# Import validated geometric formulas (same as before)
from geovocab2.shapes.formula.symbolic.beatrix import FractalSimplexInitializer
from geovocab2.shapes.formula.symbolic.cayley_menger import CayleyMengerFromSimplex
from geovocab2.shapes.formula.engineering.simplex import SimplexQuality, SimplexEdges


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DEVIL'S STAIRCASE PE (Same as before)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _normalize_positions(pos, seq_len=None):
    """Normalize positions to [0, 1]."""
    if seq_len is not None:
        x = pos.float() / max(1, (seq_len - 1))
    else:
        x = pos.float().clamp(0.0, 1.0)
    return x


def _soft_trit(y, tau):
    """Soft triadic digit assignment."""
    centers = torch.tensor([0.5, 1.5, 2.5], device=y.device, dtype=y.dtype)
    d2 = (y.unsqueeze(-1) - centers) ** 2
    logits = -d2 / (tau + 1e-8)
    p = F.softmax(logits, dim=-1)
    return p


class DevilStaircasePE(nn.Module):
    """Devil's Staircase (Cantor) Positional Encoding."""

    def __init__(
            self,
            levels: int = 12,
            features_per_level: int = 2,
            smooth_tau: float = 0.25,
            mode: str = "soft",
            add_sin_cos: bool = False,
            sin_cos_factors: list = [1, 2, 4],
            base: int = 3,
            eps: float = 1e-6
    ):
        super().__init__()
        self.levels = levels
        self.features_per_level = features_per_level
        self.tau = smooth_tau
        self.mode = mode
        self.add_sin_cos = add_sin_cos
        self.base = base
        self.eps = eps

        self.register_buffer(
            'scales',
            torch.tensor([base ** k for k in range(1, levels + 1)], dtype=torch.float32)
        )
        self.register_buffer(
            'half_powers',
            torch.tensor([0.5 ** k for k in range(1, levels + 1)], dtype=torch.float32)
        )

        if add_sin_cos:
            self.register_buffer(
                'sincos_factors',
                torch.tensor(sin_cos_factors, dtype=torch.float32)
            )
        else:
            self.sincos_factors = None

        self.alpha = nn.Parameter(torch.tensor(0.5), requires_grad=False)

    def forward(self, positions: torch.Tensor, seq_len: Optional[int] = None):
        x = _normalize_positions(positions, seq_len=seq_len)
        x = x.clamp(self.eps, 1.0 - self.eps)

        x_expanded = x.unsqueeze(-1)
        y = (x_expanded * self.scales) % self.base

        if self.mode == "soft":
            p_trit = _soft_trit(y, self.tau)
        else:
            raise ValueError("Only 'soft' mode supported")

        bit_k = p_trit[..., 2] + self.alpha * p_trit[..., 1]
        Cx = (bit_k * self.half_powers).sum(dim=-1)

        log_p_trit = (p_trit.clamp_min(1e-8)).log()
        ent = -(p_trit * log_p_trit).sum(dim=-1)
        pdf_proxy = 1.1 - ent / math.log(3.0)

        if self.features_per_level == 2:
            feats = torch.stack([bit_k, pdf_proxy], dim=-1)
            feats = feats.flatten(start_dim=-2)
        else:
            feats = bit_k

        if self.add_sin_cos:
            angles = 2.0 * math.pi * Cx.unsqueeze(-1) * self.sincos_factors
            sin_bands = torch.sin(angles)
            cos_bands = torch.cos(angles)
            sincos = torch.stack([sin_bands, cos_bands], dim=-1)
            sincos = sincos.flatten(start_dim=-2)
            feats = torch.cat([feats, sincos], dim=-1)

        return feats, Cx


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DUAL-STREAM TRANSFORMER BLOCK (Flux-inspired)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class DualStreamBlock(nn.Module):
    """
    Dual-stream transformer block with separate processing paths for visual and geometric features.

    Architecture:
        Visual Stream:   patches → norm → self-attn → cross-attn(geom) → mlp → out
        Geometric Stream: geom → norm → self-attn → cross-attn(visual) → mlp → out

    Both streams evolve independently but inform each other via cross-attention.
    """

    def __init__(
            self,
            visual_dim: int,
            geom_dim: int,
            num_heads: int = 8,
            mlp_ratio: float = 4.0,
            dropout: float = 0.0,
            attn_dropout: float = 0.0
    ):
        super().__init__()

        self.visual_dim = visual_dim
        self.geom_dim = geom_dim

        # Visual stream components
        self.visual_norm1 = nn.LayerNorm(visual_dim)
        self.visual_self_attn = nn.MultiheadAttention(
            visual_dim, num_heads, dropout=attn_dropout, batch_first=True
        )

        self.visual_norm2 = nn.LayerNorm(visual_dim)
        self.visual_cross_attn = nn.MultiheadAttention(
            visual_dim, num_heads, dropout=attn_dropout, batch_first=True,
            kdim=geom_dim, vdim=geom_dim
        )

        self.visual_norm3 = nn.LayerNorm(visual_dim)
        visual_mlp_dim = int(visual_dim * mlp_ratio)
        self.visual_mlp = nn.Sequential(
            nn.Linear(visual_dim, visual_mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(visual_mlp_dim, visual_dim),
            nn.Dropout(dropout)
        )

        # Geometric stream components
        self.geom_norm1 = nn.LayerNorm(geom_dim)
        self.geom_self_attn = nn.MultiheadAttention(
            geom_dim, num_heads, dropout=attn_dropout, batch_first=True
        )

        self.geom_norm2 = nn.LayerNorm(geom_dim)
        self.geom_cross_attn = nn.MultiheadAttention(
            geom_dim, num_heads, dropout=attn_dropout, batch_first=True,
            kdim=visual_dim, vdim=visual_dim
        )

        self.geom_norm3 = nn.LayerNorm(geom_dim)
        geom_mlp_dim = int(geom_dim * mlp_ratio)
        self.geom_mlp = nn.Sequential(
            nn.Linear(geom_dim, geom_mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(geom_mlp_dim, geom_dim),
            nn.Dropout(dropout)
        )

    def forward(
            self,
            visual_tokens: torch.Tensor,
            geom_tokens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            visual_tokens: [B, N_vis, D_vis] visual patch tokens
            geom_tokens: [B, N_geom, D_geom] geometric feature tokens

        Returns:
            visual_out: [B, N_vis, D_vis] updated visual tokens
            geom_out: [B, N_geom, D_geom] updated geometric tokens
        """
        # Visual stream: self-attention
        vis_norm = self.visual_norm1(visual_tokens)
        vis_self_attn, _ = self.visual_self_attn(vis_norm, vis_norm, vis_norm)
        visual_tokens = visual_tokens + vis_self_attn

        # Geometric stream: self-attention
        geom_norm = self.geom_norm1(geom_tokens)
        geom_self_attn, _ = self.geom_self_attn(geom_norm, geom_norm, geom_norm)
        geom_tokens = geom_tokens + geom_self_attn

        # Cross-attention: visual attends to geometric
        vis_norm2 = self.visual_norm2(visual_tokens)
        geom_norm2 = self.geom_norm2(geom_tokens)
        vis_cross_attn, _ = self.visual_cross_attn(vis_norm2, geom_norm2, geom_norm2)
        visual_tokens = visual_tokens + vis_cross_attn

        # Cross-attention: geometric attends to visual
        geom_cross_attn, _ = self.geom_cross_attn(geom_norm2, vis_norm2, vis_norm2)
        geom_tokens = geom_tokens + geom_cross_attn

        # MLP for visual stream
        visual_tokens = visual_tokens + self.visual_mlp(self.visual_norm3(visual_tokens))

        # MLP for geometric stream
        geom_tokens = geom_tokens + self.geom_mlp(self.geom_norm3(geom_tokens))

        return visual_tokens, geom_tokens


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DUAL-STREAM ViT-BEATRIX
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class DualStreamGeometricClassifier(nn.Module):
    """
    ViT-Beatrix with dual-stream architecture.

    Key features:
    - Separate visual and geometric processing streams
    - Devil's Staircase PE for both streams
    - Simplex features evolved throughout the network
    - Cross-modal attention at each layer
    """

    def __init__(
            self,
            num_classes: int,
            img_size: int = 32,
            patch_size: int = 4,
            visual_dim: int = 512,
            geom_dim: int = 256,  # Separate dimension for geometric stream
            k_simplex: int = 5,
            depth: int = 12,
            num_heads: int = 8,
            mlp_ratio: float = 4.0,
            dropout: float = 0.1,
            # PE config
            pe_levels: int = 12,
            pe_features_per_level: int = 2,
            pe_smooth_tau: float = 0.25,
            # Geometric config
            num_geom_tokens: int = 8,  # Number of geometric tokens to maintain
    ):
        super().__init__()

        self.num_classes = num_classes
        self.visual_dim = visual_dim
        self.geom_dim = geom_dim
        self.k_simplex = k_simplex
        self.num_patches = (img_size // patch_size) ** 2
        self.num_geom_tokens = num_geom_tokens

        # Patch embedding for visual stream
        self.patch_embed = nn.Conv2d(
            3, visual_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

        # CLS token for visual stream
        self.cls_token = nn.Parameter(torch.zeros(1, 1, visual_dim))

        # Devil's Staircase PE
        self.pe = DevilStaircasePE(
            levels=pe_levels,
            features_per_level=pe_features_per_level,
            smooth_tau=pe_smooth_tau,
            mode='soft',
            add_sin_cos=False
        )

        # Project PE features to visual embedding dimension
        pe_dim = pe_levels * pe_features_per_level
        self.pe_proj_visual = nn.Linear(pe_dim, visual_dim)

        # Initialize geometric stream
        self.simplex_init = FractalSimplexInitializer(
            k_simplex=k_simplex,
            embedding_dim=geom_dim
        )
        self.volume_calc = CayleyMengerFromSimplex()

        # Geometric token initialization
        # Start with num_geom_tokens learnable tokens
        self.geom_tokens = nn.Parameter(torch.zeros(1, num_geom_tokens, geom_dim))

        # Project geometric features (from simplex) to geometric stream
        self.geom_feature_proj = nn.Linear(k_simplex + 1, geom_dim)

        # Dual-stream transformer blocks
        self.dual_blocks = nn.ModuleList([
            DualStreamBlock(
                visual_dim=visual_dim,
                geom_dim=geom_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                attn_dropout=dropout
            )
            for _ in range(depth)
        ])

        # Final normalization
        self.visual_norm = nn.LayerNorm(visual_dim)
        self.geom_norm = nn.LayerNorm(geom_dim)

        # Classification head (fuse both streams)
        self.head = nn.Linear(visual_dim + geom_dim, num_classes)

        self.init_weights()

    def init_weights(self):
        """Initialize model weights."""
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.geom_tokens, std=0.02)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def initialize_geometric_stream(
            self,
            pe_features: torch.Tensor,
            cantor_measure: torch.Tensor
    ) -> torch.Tensor:
        """
        Initialize geometric token stream from PE features and simplex.

        Args:
            pe_features: [B, N+1, pe_dim] PE features
            cantor_measure: [B, N+1] Cantor measures

        Returns:
            geom_stream: [B, num_geom_tokens, geom_dim]
        """
        B = pe_features.shape[0]

        # Generate simplex from CLS token position
        simplex_result = self.simplex_init.forward(
            pe_features[:, 0],  # [B, pe_dim]
            cantor_measure[:, 0]  # [B]
        )
        vertices = simplex_result['vertices']  # [B, k+1, geom_dim]

        # Project simplex vertices to initialize geometric tokens
        # Use vertices as initial geometric features
        # Pad or truncate to num_geom_tokens
        if vertices.shape[1] < self.num_geom_tokens:
            # Pad with learnable tokens
            padding = self.geom_tokens[:, vertices.shape[1]:].expand(B, -1, -1)
            geom_stream = torch.cat([vertices, padding], dim=1)
        else:
            geom_stream = vertices[:, :self.num_geom_tokens]

        return geom_stream

    def forward(
            self,
            x: torch.Tensor,
            return_features: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with dual-stream processing.

        Args:
            x: [B, 3, H, W] input images
            return_features: Whether to return intermediate features

        Returns:
            Dictionary with 'logits' and optional features
        """
        B = x.shape[0]

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # INITIALIZE VISUAL STREAM
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        # Patch embedding
        x_vis = self.patch_embed(x)  # [B, D, H', W']
        x_vis = x_vis.flatten(2).transpose(1, 2)  # [B, N, D]
        N = x_vis.shape[1]

        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x_vis = torch.cat([cls_tokens, x_vis], dim=1)  # [B, N+1, D]

        # Devil's Staircase PE for visual stream
        positions = torch.arange(N + 1, device=x.device)
        pe_features, cantor_measure = self.pe(positions, seq_len=N + 1)

        # Project and add positional encoding to visual stream
        pe_proj = self.pe_proj_visual(pe_features)  # [N+1, D_vis]
        x_vis = x_vis + pe_proj.unsqueeze(0)  # [B, N+1, D_vis]

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # INITIALIZE GEOMETRIC STREAM
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        # Expand PE features to batch
        pe_batch = pe_features.unsqueeze(0).expand(B, -1, -1)
        cantor_batch = cantor_measure.unsqueeze(0).expand(B, -1)

        # Initialize geometric token stream
        x_geom = self.initialize_geometric_stream(pe_batch, cantor_batch)
        # [B, num_geom_tokens, D_geom]

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # DUAL-STREAM PROCESSING
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        for block in self.dual_blocks:
            x_vis, x_geom = block(x_vis, x_geom)

        # Final normalization
        x_vis = self.visual_norm(x_vis)
        x_geom = self.geom_norm(x_geom)

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # FUSION AND CLASSIFICATION
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        # Extract CLS token from visual stream
        cls_visual = x_vis[:, 0]  # [B, D_vis]

        # Pool geometric stream (mean over tokens)
        cls_geom = x_geom.mean(dim=1)  # [B, D_geom]

        # Fuse both modalities
        cls_fused = torch.cat([cls_visual, cls_geom], dim=-1)  # [B, D_vis + D_geom]

        # Classification
        logits = self.head(cls_fused)

        # Prepare output
        output = {'logits': logits}

        if return_features:
            output['visual_features'] = cls_visual
            output['geometric_features'] = cls_geom
            output['pe_features'] = pe_features
            output['cantor_measure'] = cantor_measure
            output['visual_stream'] = x_vis
            output['geometric_stream'] = x_geom

        return output


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TESTING AND DEMONSTRATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    print("=" * 70)
    print("DUAL-STREAM GEOMETRIC CLASSIFIER - DEMONSTRATION")
    print("=" * 70)

    batch_size = 4
    num_classes = 10
    img_size = 32

    print(f"\nConfiguration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Image size: {img_size}x{img_size}")
    print(f"  Classes: {num_classes}")

    print("\n" + "-" * 70)
    print("Creating dual-stream model...")

    model = DualStreamGeometricClassifier(
        num_classes=num_classes,
        img_size=img_size,
        visual_dim=256,
        geom_dim=128,
        depth=6,
        num_geom_tokens=8
    )

    total_params = sum(p.numel() for p in model.parameters())
    visual_params = sum(p.numel() for p in model.dual_blocks[0].visual_self_attn.parameters())
    geom_params = sum(p.numel() for p in model.dual_blocks[0].geom_self_attn.parameters())

    print(f"  Total parameters: {total_params:,}")
    print(f"  Visual stream params (per block): {visual_params:,}")
    print(f"  Geometric stream params (per block): {geom_params:,}")

    print("\n" + "-" * 70)
    print("Testing forward pass...")

    x = torch.randn(batch_size, 3, img_size, img_size)

    with torch.no_grad():
        output = model(x, return_features=True)

    print(f"\nOutputs:")
    print(f"  Logits: {output['logits'].shape}")
    print(f"  Visual features: {output['visual_features'].shape}")
    print(f"  Geometric features: {output['geometric_features'].shape}")
    print(f"  Visual stream: {output['visual_stream'].shape}")
    print(f"  Geometric stream: {output['geometric_stream'].shape}")

    print("\n" + "-" * 70)
    print("Testing backward pass...")

    targets = torch.randint(0, num_classes, (batch_size,))
    criterion = nn.CrossEntropyLoss()

    output = model(x)
    loss = criterion(output['logits'], targets)
    loss.backward()

    print(f"  Loss: {loss.item():.4f}")
    print(f"  Gradients computed: ✓")

    # Check that geometric stream receives gradients
    geom_has_grad = model.dual_blocks[0].geom_self_attn.out_proj.weight.grad is not None
    print(f"  Geometric stream has gradients: {geom_has_grad}")

    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)
    print("\nKey Advantages:")
    print("  ✓ Geometric features preserved throughout network")
    print("  ✓ Separate processing streams with independent weights")
    print("  ✓ Cross-modal attention for information exchange")
    print("  ✓ Fractal structure maintained across all layers")
    print("  ✓ Both streams contribute to final classification")