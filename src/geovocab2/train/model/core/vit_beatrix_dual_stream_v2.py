"""
ViT-Beatrix with Dual-Stream Architecture + Fractal Simplex Integration
-------------------------------------------------------------------------
Factory-validated base structure + dynamic fractal-guided geometric evolution.

Author: AbstractPhil + Claude Sonnet 4.5
License: MIT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import math

# Import validated geometric formulas
from geovocab2.shapes.formula.symbolic.beatrix import (
    FractalSimplexInitializer,
    BeatrixIntegratedLoss,
    CantorToBarycentric,
    FlowAlignmentLoss,
    HierarchicalCoherence,
    MultiScaleConsistency
)
from geovocab2.shapes.formula.symbolic.cayley_menger import CayleyMengerFromSimplex
from geovocab2.shapes.formula.engineering.simplex import SimplexQuality, SimplexEdges
from geovocab2.shapes.factory.simplex_factory import SimplexFactory


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DEVIL'S STAIRCASE PE
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
# DUAL-STREAM TRANSFORMER BLOCK
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class DualStreamBlock(nn.Module):
    """Dual-stream transformer block with separate processing paths."""

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

        # Visual stream
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

        # Geometric stream
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
        # Visual stream: self-attention
        vis_norm = self.visual_norm1(visual_tokens)
        vis_self_attn, _ = self.visual_self_attn(vis_norm, vis_norm, vis_norm)
        visual_tokens = visual_tokens + vis_self_attn

        # Geometric stream: self-attention
        geom_norm = self.geom_norm1(geom_tokens)
        geom_self_attn, _ = self.geom_self_attn(geom_norm, geom_norm, geom_norm)
        geom_tokens = geom_tokens + geom_self_attn

        # Cross-attention: visual → geometric
        vis_norm2 = self.visual_norm2(visual_tokens)
        geom_norm2 = self.geom_norm2(geom_tokens)
        vis_cross_attn, _ = self.visual_cross_attn(vis_norm2, geom_norm2, geom_norm2)
        visual_tokens = visual_tokens + vis_cross_attn

        # Cross-attention: geometric → visual
        geom_cross_attn, _ = self.geom_cross_attn(geom_norm2, vis_norm2, vis_norm2)
        geom_tokens = geom_tokens + geom_cross_attn

        # MLPs
        visual_tokens = visual_tokens + self.visual_mlp(self.visual_norm3(visual_tokens))
        geom_tokens = geom_tokens + self.geom_mlp(self.geom_norm3(geom_tokens))

        return visual_tokens, geom_tokens


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DUAL-STREAM ViT-BEATRIX WITH FRACTAL SIMPLEX
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class DualStreamGeometricClassifier(nn.Module):
    """
    ViT-Beatrix with dual-stream architecture and fractal simplex generation.

    Key features:
    - Factory-validated base simplex structure
    - Dynamic fractal-guided simplex generation per position
    - Separate visual and geometric processing streams
    - Devil's Staircase PE for both streams
    - Beatrix integrated loss for geometric regularization
    """

    def __init__(
            self,
            num_classes: int,
            img_size: int = 32,
            patch_size: int = 4,
            visual_dim: int = 512,
            geom_dim: int = 256,
            k_simplex: int = 4,
            depth: int = 12,
            num_heads: int = 8,
            mlp_ratio: float = 4.0,
            dropout: float = 0.1,
            # PE config
            pe_levels: int = 12,
            pe_features_per_level: int = 2,
            pe_smooth_tau: float = 0.25,
            # SimplexFactory config (for validation)
            simplex_method: str = "regular",
            simplex_scale: float = 1.0,
            simplex_seed: Optional[int] = None,
            # Beatrix loss config
            use_beatrix_loss: bool = True,
            flow_weight: float = 1.0,
            coherence_weight: float = 0.5,
            multiscale_weight: float = 0.3,
            volume_reg_weight: float = 0.1,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.visual_dim = visual_dim
        self.geom_dim = geom_dim
        self.k_simplex = k_simplex
        self.num_patches = (img_size // patch_size) ** 2
        self.use_beatrix_loss = use_beatrix_loss

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # GEOMETRIC INITIALIZATION
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        # SimplexFactory for canonical validation
        self.simplex_factory = SimplexFactory(
            k=k_simplex,
            embed_dim=k_simplex + 1,  # Native simplex dimension
            method=simplex_method,
            scale=simplex_scale,
            seed=simplex_seed
        )

        # Generate and validate canonical simplex
        canonical_simplex = self.simplex_factory.build(
            backend="torch",
            device="cpu",
            dtype=torch.float32,
            validate=True
        )
        self.register_buffer('canonical_simplex', canonical_simplex)

        # FractalSimplexInitializer for dynamic generation
        self.fractal_init = FractalSimplexInitializer(
            k_simplex=k_simplex,
            embedding_dim=geom_dim,
            use_pdf_proxy=True
        )

        # Initialize base simplex from canonical structure
        with torch.no_grad():
            self.fractal_init.base_simplex.data = canonical_simplex.clone()

        # Validation modules
        self.volume_calc = CayleyMengerFromSimplex()
        self.quality_calc = SimplexQuality()
        self.edge_calc = SimplexEdges()

        # Beatrix loss components
        if use_beatrix_loss:
            self.beatrix_loss = BeatrixIntegratedLoss(
                flow_weight=flow_weight,
                coherence_weight=coherence_weight,
                multiscale_weight=multiscale_weight,
                volume_reg_weight=volume_reg_weight
            )
            self.cantor_to_bary = CantorToBarycentric(k_simplex=k_simplex)

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # VISUAL STREAM
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        # Patch embedding
        self.patch_embed = nn.Conv2d(
            3, visual_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, visual_dim))

        # Devil's Staircase PE
        self.pe = DevilStaircasePE(
            levels=pe_levels,
            features_per_level=pe_features_per_level,
            smooth_tau=pe_smooth_tau,
            mode='soft',
            add_sin_cos=False
        )

        # Project PE features to visual dimension
        pe_dim = pe_levels * pe_features_per_level
        self.pe_proj_visual = nn.Linear(pe_dim, visual_dim)

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # DUAL-STREAM TRANSFORMER
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

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

        # Classification head
        self.head = nn.Linear(visual_dim + geom_dim, num_classes)

        self.init_weights()

    def init_weights(self):
        """Initialize model weights."""
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def generate_geometric_stream(
            self,
            pe_features: torch.Tensor,
            cantor_measure: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Generate dynamic geometric token stream using fractal initialization.

        Args:
            pe_features: [N+1, pe_dim] PE features for each position
            cantor_measure: [N+1] Cantor measures

        Returns:
            geom_stream: [B, N+1, k+1, geom_dim] geometric tokens (simplex vertices)
            metrics: Dictionary of geometric validation metrics
        """
        # Generate simplex for each position
        fractal_result = self.fractal_init.forward(pe_features, cantor_measure)
        vertices = fractal_result['vertices']  # [N+1, k+1, geom_dim]

        # Compute geometric metrics
        metrics = {
            'deformation_magnitude': fractal_result['deformation_magnitude'],
            'orientation_angle': fractal_result['orientation_angle']
        }

        # Validate geometric properties
        volume_result = self.volume_calc.forward(vertices)
        quality_result = self.quality_calc.forward(vertices)
        edge_result = self.edge_calc.forward(vertices)

        metrics.update({
            'volume': volume_result['volume'],
            'is_degenerate': volume_result['is_degenerate'],
            'quality': quality_result['quality'],
            'edge_lengths': edge_result['edge_lengths']
        })

        return vertices, metrics

    def forward(
            self,
            x: torch.Tensor,
            return_features: bool = False,
            return_geometric_loss: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with dual-stream processing and dynamic geometric generation.

        Args:
            x: [B, 3, H, W] input images
            return_features: Whether to return intermediate features
            return_geometric_loss: Whether to compute Beatrix geometric loss

        Returns:
            Dictionary with 'logits' and optional features/losses
        """
        B = x.shape[0]
        device = x.device

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # INITIALIZE VISUAL STREAM
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        # Patch embedding
        x_vis = self.patch_embed(x)  # [B, D, H', W']
        x_vis = x_vis.flatten(2).transpose(1, 2)  # [B, N, D]
        N = x_vis.shape[1]

        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x_vis = torch.cat([cls_tokens, x_vis], dim=1)  # [B, N+1, D]

        # Devil's Staircase PE
        positions = torch.arange(N + 1, device=device)
        pe_features, cantor_measure = self.pe(positions, seq_len=N + 1)
        # pe_features: [N+1, pe_dim], cantor_measure: [N+1]

        # Project and add PE to visual stream
        pe_proj = self.pe_proj_visual(pe_features)  # [N+1, D_vis]
        x_vis = x_vis + pe_proj.unsqueeze(0)  # [B, N+1, D_vis]

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # GENERATE GEOMETRIC STREAM (Fractal Simplex)
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        # Generate position-dependent simplices
        vertices_per_pos, geom_metrics = self.generate_geometric_stream(
            pe_features, cantor_measure
        )
        # vertices_per_pos: [N+1, k+1, geom_dim]

        # Expand to batch dimension and flatten to tokens
        # Each position gets its own simplex with k+1 vertices
        x_geom = vertices_per_pos.unsqueeze(0).expand(B, -1, -1, -1)
        # [B, N+1, k+1, geom_dim]

        # Reshape for processing: treat vertices as separate tokens
        # [B, (N+1)*(k+1), geom_dim]
        B, seq_len, k_plus_1, geom_d = x_geom.shape
        x_geom = x_geom.reshape(B, seq_len * k_plus_1, geom_d)

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # DUAL-STREAM PROCESSING
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        for block in self.dual_blocks:
            x_vis, x_geom = block(x_vis, x_geom)

        # Final normalization
        x_vis = self.visual_norm(x_vis)
        x_geom = self.geom_norm(x_geom)

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # FUSION AND CLASSIFICATION
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        # Extract CLS token from visual stream
        cls_visual = x_vis[:, 0]  # [B, D_vis]

        # Pool geometric stream
        cls_geom = x_geom.mean(dim=1)  # [B, D_geom]

        # Fuse both modalities
        cls_fused = torch.cat([cls_visual, cls_geom], dim=-1)

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
            output['simplex_vertices'] = vertices_per_pos
            output['geometric_metrics'] = geom_metrics

        # Compute Beatrix geometric loss if requested
        if return_geometric_loss and self.use_beatrix_loss:
            # Reshape for loss computation
            cantor_batch = cantor_measure.unsqueeze(0).expand(B, -1)  # [B, N+1]
            pe_batch = pe_features.unsqueeze(0).expand(B, -1, -1)  # [B, N+1, pe_dim]
            simplex_batch = vertices_per_pos.unsqueeze(0).expand(B, -1, -1, -1)
            # [B, N+1, k+1, geom_dim]

            beatrix_result = self.beatrix_loss.forward(
                cantor_batch, pe_batch, simplex_batch
            )
            output['beatrix_loss'] = beatrix_result

        return output


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TESTING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    print("=" * 70)
    print("DUAL-STREAM BEATRIX WITH FRACTAL SIMPLEX GENERATION")
    print("=" * 70)

    batch_size = 4
    num_classes = 10
    img_size = 32

    print(f"\nConfiguration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Image size: {img_size}x{img_size}")
    print(f"  Classes: {num_classes}")

    print("\n" + "-" * 70)
    print("Creating model with fractal simplex generation...")

    model = DualStreamGeometricClassifier(
        num_classes=num_classes,
        img_size=img_size,
        visual_dim=256,
        geom_dim=128,
        k_simplex=4,
        depth=6,
        num_heads=8,
        simplex_method="regular",
        use_beatrix_loss=True
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")

    # Validate canonical simplex
    print(f"  Canonical simplex shape: {model.canonical_simplex.shape}")
    is_valid, msg = model.simplex_factory.validate(model.canonical_simplex)
    print(f"  Canonical simplex valid: {is_valid} {msg if msg else '✓'}")

    print("\n" + "-" * 70)
    print("Testing forward pass with geometric generation...")

    x = torch.randn(batch_size, 3, img_size, img_size)

    with torch.no_grad():
        output = model(x, return_features=True, return_geometric_loss=True)

    print(f"\n  Outputs:")
    print(f"    Logits: {output['logits'].shape}")
    print(f"    Visual features: {output['visual_features'].shape}")
    print(f"    Geometric features: {output['geometric_features'].shape}")
    print(f"    Simplex vertices: {output['simplex_vertices'].shape}")

    # Display geometric metrics
    metrics = output['geometric_metrics']
    print(f"\n  Geometric Metrics (per position):")
    print(f"    Mean volume: {metrics['volume'].mean().item():.6f}")
    print(f"    Mean quality: {metrics['quality'].mean().item():.6f}")
    print(f"    Mean deformation: {metrics['deformation_magnitude'].mean().item():.6f}")
    print(f"    Any degenerate: {metrics['is_degenerate'].any().item()}")

    # Display Beatrix loss components
    if 'beatrix_loss' in output:
        bl = output['beatrix_loss']
        print(f"\n  Beatrix Loss Components:")
        print(f"    Total: {bl['total_loss'].item():.6f}")
        print(f"    Flow alignment: {bl['flow_loss'].item():.6f}")
        print(f"    Coherence: {bl['coherence_loss'].item():.6f}")
        print(f"    Multi-scale: {bl['multiscale_loss'].item():.6f}")
        print(f"    Volume reg: {bl['volume_loss'].item():.6f}")

    print("\n" + "-" * 70)
    print("Testing backward pass with combined losses...")

    targets = torch.randint(0, num_classes, (batch_size,))
    criterion = nn.CrossEntropyLoss()

    output = model(x, return_geometric_loss=True)

    # Combined loss: classification + geometric regularization
    cls_loss = criterion(output['logits'], targets)

    if 'beatrix_loss' in output:
        total_loss = cls_loss + 0.1 * output['beatrix_loss']['total_loss']
    else:
        total_loss = cls_loss

    total_loss.backward()

    print(f"  Classification loss: {cls_loss.item():.4f}")
    if 'beatrix_loss' in output:
        print(f"  Geometric loss: {output['beatrix_loss']['total_loss'].item():.4f}")
    print(f"  Total loss: {total_loss.item():.4f}")
    print(f"  Gradients computed: ✓")

    # Check gradient flow
    fractal_has_grad = model.fractal_init.base_simplex.grad is not None
    visual_has_grad = model.cls_token.grad is not None
    print(f"  Fractal base has gradients: {fractal_has_grad}")
    print(f"  Visual tokens have gradients: {visual_has_grad}")

    print("\n" + "=" * 70)
    print("FRACTAL SIMPLEX INTEGRATION COMPLETE")
    print("=" * 70)
    print("\nKey Advantages:")
    print("  ✓ Factory-validated canonical base structure")
    print("  ✓ Dynamic position-dependent simplex generation")
    print("  ✓ PE features directly control geometric deformation")
    print("  ✓ Fractal hierarchy preserved in geometric stream")
    print("  ✓ Beatrix loss ensures geometric coherence")
    print("  ✓ Flow alignment between PE and geometry")
    print("  ✓ Multi-scale consistency enforcement")
    print("  ✓ Both streams fully differentiable")