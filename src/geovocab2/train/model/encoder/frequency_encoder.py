#!/usr/bin/env python3
"""
Pentachoron Frequency Encoder - Decoupled Module
Apache-2.0
Author: AbstractPhil

Multi-channel spectral encoder with frequency-band extraction,
cross-attention fusion, and contrastive loss functions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict


# =====================================================================
# ENCODER COMPONENTS
# =====================================================================

class PentaFreqExtractor(nn.Module):
    """
    Multi-channel spectral extractor:
      - Input: [B, C*H*W], unflatten -> [B, C, H, W]
      - 5 frequency bands -> encode to base_dim each
    """

    def __init__(self, input_dim: int = 784, input_ch: int = 1,
                 base_dim: int = 64, channels: int = 12):
        super().__init__()
        self.input_dim = input_dim
        self.input_ch = int(input_ch)
        side_f = (input_dim / max(1, self.input_ch)) ** 0.5
        side = int(side_f)
        assert side * side * self.input_ch == input_dim, \
            f"input_dim ({input_dim}) != C*H*W with H=W; C={self.input_ch}, side≈{side_f:.3f}"

        self.unflatten = nn.Unflatten(1, (self.input_ch, side, side))
        self.base_dim = base_dim

        # Vertex 0 (ultra-high frequency)
        self.v0_ultrahigh = nn.Sequential(
            nn.Conv2d(self.input_ch, channels, 3, padding=1),
            nn.BatchNorm2d(channels), nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.BatchNorm2d(channels), nn.ReLU(),
            nn.AdaptiveAvgPool2d(7), nn.Flatten()
        )
        self.v0_encode = nn.Linear(channels * 49, base_dim)

        # Vertex 1 (high frequency)
        self.v1_high = nn.Sequential(
            nn.Conv2d(self.input_ch, channels, 3, padding=1),
            nn.BatchNorm2d(channels), nn.Tanh(),
            nn.MaxPool2d(2),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels), nn.Tanh(),
            nn.AdaptiveAvgPool2d(7), nn.Flatten()
        )
        self.v1_encode = nn.Linear(channels * 49, base_dim)

        # Vertex 2 (mid frequency)
        self.v2_mid = nn.Sequential(
            nn.Conv2d(self.input_ch, channels, 5, padding=2, stride=2),
            nn.BatchNorm2d(channels), nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels), nn.GELU(),
            nn.AdaptiveAvgPool2d(7), nn.Flatten()
        )
        self.v2_encode = nn.Linear(channels * 49, base_dim)

        # Vertex 3 (low-mid frequency)
        self.v3_lowmid = nn.Sequential(
            nn.AvgPool2d(2),
            nn.Conv2d(self.input_ch, channels, 7, padding=3),
            nn.BatchNorm2d(channels), nn.SiLU(),
            nn.AdaptiveAvgPool2d(7), nn.Flatten()
        )
        self.v3_encode = nn.Linear(channels * 49, base_dim)

        # Vertex 4 (low frequency)
        self.v4_low = nn.Sequential(
            nn.AvgPool2d(4),
            nn.Conv2d(self.input_ch, channels, 7, padding=3),
            nn.BatchNorm2d(channels), nn.Sigmoid(),
            nn.AdaptiveAvgPool2d(7), nn.Flatten()
        )
        self.v4_encode = nn.Linear(channels * 49, base_dim)

        # Adjacency matrix for pentachoron (complete graph on 5 vertices)
        self.register_buffer("adjacency_matrix", torch.ones(5, 5) - torch.eye(5))
        self._init_edge_kernels(channels)

    @torch.no_grad()
    def _init_edge_kernels(self, channels: int):
        """Initialize first conv layer with edge detection kernels"""
        if channels < 5:
            return
        conv0 = self.v0_ultrahigh[0]
        if not isinstance(conv0, nn.Conv2d):
            return
        if conv0.weight.shape[1] >= 1:
            k = conv0.weight
            # Sobel X, Sobel Y, Laplacian, Diagonal, Vertical
            k[0, 0] = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=k.dtype) / 4
            k[1, 0] = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=k.dtype) / 4
            k[2, 0] = torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=k.dtype) / 4
            k[3, 0] = torch.tensor([[1, 0, 0], [0, -1, 0], [0, 0, 0]], dtype=k.dtype) / 2
            k[4, 0] = torch.tensor([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=k.dtype) / 3

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, C*H*W] flattened input
        Returns:
            vertices: [B, 5, base_dim] - frequency band encodings
            adjacency: [5, 5] - adjacency matrix
        """
        img = self.unflatten(x)
        v0 = self.v0_encode(self.v0_ultrahigh(img))
        v1 = self.v1_encode(self.v1_high(img))
        v2 = self.v2_encode(self.v2_mid(img))
        v3 = self.v3_encode(self.v3_lowmid(img))
        v4 = self.v4_encode(self.v4_low(img))
        vertices = torch.stack([v0, v1, v2, v3, v4], dim=1)  # [B, 5, D]
        return vertices, self.adjacency_matrix


class PentachoronCrossAttention(nn.Module):
    """Cross-attention over pentachoron vertices using adjacency masking"""

    def __init__(self, dim: int, num_heads: int = 14, dropout: float = 0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )

    def _row_to_attn_mask(self, row: torch.Tensor) -> torch.Tensor:
        """Convert adjacency row to attention mask (0 -> -inf)"""
        mask = torch.zeros(1, row.numel(), device=row.device, dtype=torch.float32)
        mask[(row == 0).unsqueeze(0)] = float("-inf")
        return mask

    def forward(self, vertices: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        """
        Args:
            vertices: [B, 5, D]
            adjacency: [5, 5]
        Returns:
            attended: [B, 5, D]
        """
        B, V, D = vertices.shape
        outs = []
        for i in range(V):
            q = vertices[:, i:i + 1, :]  # Query: single vertex
            k = v = vertices  # Key/Value: all vertices
            mask = self._row_to_attn_mask(adjacency[i].to(vertices.device))
            out, _ = self.attn(q, k, v, attn_mask=mask, need_weights=False)
            outs.append(out)
        return torch.cat(outs, dim=1)


class PentachoronOpinionFusion(nn.Module):
    """Fuses pentachoron vertex opinions using soft geometry and cross-attention"""

    def __init__(self, base_dim: int = 64, proj_dim: Optional[int] = None,
                 num_heads: int = 14, p_dropout: float = 0.2):
        super().__init__()
        self.cross = PentachoronCrossAttention(dim=base_dim, num_heads=num_heads)

        self.fusion = nn.Sequential(
            nn.Linear(base_dim * 5, base_dim * 3),
            nn.BatchNorm1d(base_dim * 3), nn.ReLU(), nn.Dropout(p_dropout),
            nn.Linear(base_dim * 3, base_dim * 2),
            nn.BatchNorm1d(base_dim * 2), nn.ReLU(),
            nn.Linear(base_dim * 2, base_dim),
        )

        self.projection = None if proj_dim is None else nn.Linear(base_dim, proj_dim, bias=False)
        self._lambda_raw = nn.Parameter(torch.tensor(0.0))

    @staticmethod
    def _softmax_geometry(vertices: torch.Tensor,
                          adjacency: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute softmax-weighted geometry based on vertex similarities"""
        v_norm = F.normalize(vertices, dim=2, eps=1e-8)
        sims = torch.bmm(v_norm, v_norm.transpose(1, 2))
        edge_strengths = sims * adjacency.to(vertices.dtype).unsqueeze(0)
        weights = F.softmax(edge_strengths.sum(dim=2), dim=1)  # [B, 5]
        weighted = vertices * weights.unsqueeze(2)
        return weighted, weights

    def forward(self, vertices: torch.Tensor, adjacency: torch.Tensor,
                return_diag: bool = False):
        """
        Args:
            vertices: [B, 5, base_dim]
            adjacency: [5, 5]
            return_diag: whether to return diagnostic info
        Returns:
            z: [B, base_dim or proj_dim] - normalized latent
            diag: Optional[Dict] - diagnostics (lambda, softmax_weights)
        """
        soft_out, weights = self._softmax_geometry(vertices, adjacency)
        attn_out = self.cross(vertices, adjacency)

        # Learnable blend between geometry and attention
        lam = torch.sigmoid(self._lambda_raw)
        combined = lam * soft_out + (1.0 - lam) * attn_out

        fused = self.fusion(combined.flatten(1))

        if self.projection is not None:
            fused = self.projection(fused)

        z = F.normalize(fused, dim=1)

        if not return_diag:
            return z, None

        return z, {
            "lambda": lam.detach(),
            "softmax_weights": weights.detach()
        }


class PentaFreqEncoderV2(nn.Module):
    """
    Complete Pentachoron Frequency Encoder V2

    Combines frequency extraction and opinion fusion into a single encoder.
    """

    def __init__(self, input_dim: int = 784, input_ch: int = 1,
                 base_dim: int = 64, proj_dim: Optional[int] = None,
                 num_heads: int = 14, channels: int = 12):
        super().__init__()
        self.extractor = PentaFreqExtractor(
            input_dim=input_dim,
            input_ch=input_ch,
            base_dim=base_dim,
            channels=channels
        )
        self.opinion = PentachoronOpinionFusion(
            base_dim=base_dim,
            proj_dim=proj_dim,
            num_heads=num_heads
        )

    @torch.no_grad()
    def get_frequency_contributions(self, x: torch.Tensor) -> torch.Tensor:
        """Get softmax weights showing contribution of each frequency band"""
        verts, adj = self.extractor(x)
        _, w = self.opinion._softmax_geometry(verts, adj)
        return w

    def forward(self, x: torch.Tensor, return_diag: bool = False):
        """
        Args:
            x: [B, input_dim] - flattened input
            return_diag: whether to return diagnostics
        Returns:
            z: [B, base_dim or proj_dim] - latent encoding
            diag: Optional[Dict] - diagnostics
        """
        verts, adj = self.extractor(x)
        z, diag = self.opinion(verts, adj, return_diag)
        return (z, diag) if return_diag else z


# =====================================================================
# LOSS FUNCTIONS
# =====================================================================

def dual_contrastive_loss(latents: torch.Tensor,
                          targets: torch.Tensor,
                          constellation,
                          temp: float = 0.7) -> torch.Tensor:
    """
    Dual contrastive loss over dispatcher and specialist pentachora.

    Args:
        latents: [B, D] - encoder outputs
        targets: [B] - class labels
        constellation: BatchedPentachoronConstellation instance
        temp: temperature for softmax
    Returns:
        loss: scalar
    """
    B = latents.size(0)
    z = F.normalize(latents, dim=1, eps=1e-8)

    # Normalize pentachora
    disp = F.normalize(constellation.dispatchers, dim=2, eps=1e-8)  # [P, 5, D]
    spec = F.normalize(constellation.specialists, dim=2, eps=1e-8)  # [P, 5, D]

    # Compute similarities
    disp_logits = torch.einsum('bd,pvd->bpv', z, disp) / temp  # [B, P, 5]
    spec_logits = torch.einsum('bd,pvd->bpv', z, spec) / temp  # [B, P, 5]

    # Get target vertex indices
    tvert = constellation.vertex_map[targets]  # [B]
    idx = tvert.view(B, 1, 1).expand(B, disp_logits.size(1), 1)

    # Positive scores
    disp_pos = disp_logits.gather(2, idx).squeeze(2)  # [B, P]
    spec_pos = spec_logits.gather(2, idx).squeeze(2)  # [B, P]

    # Log-sum-exp over vertices
    disp_lse = torch.logsumexp(disp_logits, dim=2)  # [B, P]
    spec_lse = torch.logsumexp(spec_logits, dim=2)  # [B, P]

    # InfoNCE loss
    return (disp_lse - disp_pos).mean() + (spec_lse - spec_pos).mean()


def rose_score_magnitude(x: torch.Tensor,
                         need: torch.Tensor,
                         relation: torch.Tensor,
                         purpose: torch.Tensor,
                         eps: float = 1e-8) -> torch.Tensor:
    """
    Compute ROSE (Relational Opinion Score Estimation) metric.

    Combines cosine similarity with magnitude for relational scoring.
    """
    x_n = F.normalize(x, dim=-1, eps=eps)
    n_n = F.normalize(need, dim=-1, eps=eps)
    r_n = F.normalize(relation, dim=-1, eps=eps)
    p_n = F.normalize(purpose, dim=-1, eps=eps)

    # Average cosine similarity
    r7 = ((x_n * n_n).sum(-1) + (x_n * r_n).sum(-1) + (x_n * p_n).sum(-1)) / 3.0

    # Magnitude component
    r8 = x.norm(dim=-1).clamp_min(eps)

    return r7 * r8


def rose_contrastive_loss(latents: torch.Tensor,
                          targets: torch.Tensor,
                          constellation,
                          temp: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    ROSE-weighted contrastive loss with need/relation/purpose components.

    Args:
        latents: [B, D]
        targets: [B]
        constellation: BatchedPentachoronConstellation instance
        temp: temperature
    Returns:
        loss: scalar
        rose_scores: [B] - detached ROSE scores for diagnostics
    """
    B, D = latents.shape
    tvert = constellation.vertex_map[targets]

    # Compute ROSE components
    need = constellation.specialists[:, tvert, :].mean(dim=0)  # [B, D]
    relation = constellation.dispatchers[:, tvert, :].mean(dim=0)  # [B, D]
    purpose = constellation.specialists.mean(dim=(0, 1)).unsqueeze(0).expand(B, D)

    rose = rose_score_magnitude(latents, need, relation, purpose)
    weights = (1.0 - torch.tanh(rose)).detach()  # Higher loss for low ROSE

    # Standard contrastive loss
    spec = F.normalize(constellation.specialists.mean(dim=0), dim=1, eps=1e-8)
    disp = F.normalize(constellation.dispatchers.mean(dim=0), dim=1, eps=1e-8)
    z = F.normalize(latents, dim=1, eps=1e-8)

    spec_logits = (z @ spec.T) / temp
    disp_logits = (z @ disp.T) / temp

    spec_pos = spec_logits.gather(1, tvert.view(-1, 1)).squeeze(1)
    disp_pos = disp_logits.gather(1, tvert.view(-1, 1)).squeeze(1)

    spec_lse = torch.logsumexp(spec_logits, dim=1)
    disp_lse = torch.logsumexp(disp_logits, dim=1)

    per_sample = 0.5 * ((spec_lse - spec_pos) + (disp_lse - disp_pos))

    return (per_sample * weights).mean(), rose.detach()


class RoseDiagnosticHead(nn.Module):
    """Diagnostic head to predict ROSE scores from latents"""

    def __init__(self, latent_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# =====================================================================
# MAIN - COLAB DEMO
# =====================================================================

def main():
    """
    Demo for Colab: Create encoder, process random batch, show outputs.
    """
    print("=" * 60)
    print("PENTACHORON FREQUENCY ENCODER V2 - COLAB DEMO")
    print("=" * 60)

    # Configuration
    batch_size = 8
    input_dim = 28 * 28  # MNIST-like
    input_ch = 1
    base_dim = 64
    num_heads = 4  # Reduced for demo
    channels = 12

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # Create encoder
    print("\nInitializing encoder...")
    encoder = PentaFreqEncoderV2(
        input_dim=input_dim,
        input_ch=input_ch,
        base_dim=base_dim,
        proj_dim=None,
        num_heads=num_heads,
        channels=channels
    ).to(device)

    param_count = sum(p.numel() for p in encoder.parameters())
    print(f"Total parameters: {param_count:,}")

    # Generate random batch
    print(f"\nGenerating random batch (size={batch_size})...")
    x = torch.randn(batch_size, input_dim).to(device)

    # Forward pass without diagnostics
    print("\n1. Standard forward pass:")
    encoder.eval()
    with torch.no_grad():
        z = encoder(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {z.shape}")
    print(f"   Output norm: {z.norm(dim=1).mean().item():.4f} (should be ~1.0)")

    # Forward pass with diagnostics
    print("\n2. Forward pass with diagnostics:")
    with torch.no_grad():
        z, diag = encoder(x, return_diag=True)
    print(f"   Lambda (geometry ↔ attention): {diag['lambda'].item():.4f}")
    print(f"   Softmax weights shape: {diag['softmax_weights'].shape}")
    print(f"   Frequency contributions (mean across batch):")
    for i, w in enumerate(diag['softmax_weights'].mean(dim=0).cpu().numpy()):
        freq_names = ["Ultra-high", "High", "Mid", "Low-mid", "Low"]
        print(f"      Vertex {i} ({freq_names[i]:10s}): {w:.4f}")

    # Get frequency contributions directly
    print("\n3. Frequency contribution analysis:")
    with torch.no_grad():
        freq_contribs = encoder.get_frequency_contributions(x)
    print(f"   Shape: {freq_contribs.shape}")
    print(f"   Sum per sample: {freq_contribs.sum(dim=1).mean().item():.4f}")

    # Show model structure
    print("\n4. Model structure:")
    print(f"   Extractor: {encoder.extractor.__class__.__name__}")
    print(f"      - 5 frequency vertices, each -> {base_dim}D")
    print(f"   Opinion: {encoder.opinion.__class__.__name__}")
    print(f"      - Cross-attention heads: {num_heads}")
    print(f"      - Fusion: 5×{base_dim} -> {base_dim}")

    # Show loss function usage (requires constellation)
    print("\n5. Loss function notes:")
    print("   - dual_contrastive_loss: requires constellation model")
    print("   - rose_contrastive_loss: requires constellation model")
    print("   - RoseDiagnosticHead: can be used standalone")

    # Create diagnostic head
    diag_head = RoseDiagnosticHead(base_dim).to(device)
    with torch.no_grad():
        rose_pred = diag_head(z)
    print(f"\n   Diagnostic head predictions: {rose_pred.squeeze().shape}")

    print("\n" + "=" * 60)
    print("Demo complete! Encoder is ready for training.")
    print("=" * 60)

    return encoder, diag_head


if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    encoder, diag_head = main()