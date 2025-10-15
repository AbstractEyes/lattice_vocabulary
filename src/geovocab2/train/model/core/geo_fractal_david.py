"""
GeoFractalDavid: Pure Geometric Basin Architecture (Refactored)
================================================================
Compatible with pure geometric basin training (no cross-entropy).
Uses 4 geometric loss components: coherence, separation, discretization, geometry.

Author: AbstractPhil + Claude Sonnet 4.5
License: MIT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple


class CantorStaircase(nn.Module):
    """
    Learnable soft Cantor staircase with alpha-normalized middle weighting.
    Follows Beatrix paradigm: normalize to [0, 1], then apply triadic decomposition.
    """

    def __init__(self, alpha_init: float = 0.5, tau: float = 0.25, base: int = 3):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha_init))
        self.tau = tau
        self.base = base

        # Fixed centers for triadic intervals [left, middle, right]
        self.register_buffer('centers', torch.tensor([0.5, 1.5, 2.5]))

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y: Input features [batch_size, feature_dim]
        Returns:
            cantor_values: Soft Cantor values in [0, 1] [batch_size]
        """
        # Reduce to scalar per sample via mean
        if y.dim() > 1:
            y = y.mean(dim=1, keepdim=True)  # [batch_size, 1]

        # BEATRIX PARADIGM: Normalize to [0, 1] first (like positions)
        # Use sigmoid to map features → [0, 1] (learnable mapping)
        x = torch.sigmoid(y).squeeze(-1)  # [batch_size] in (0, 1)

        # Clamp away from boundaries for numerical stability
        x = x.clamp(1e-6, 1.0 - 1e-6)

        # BEATRIX TRIADIC DECOMPOSITION
        # Map to [0, base) range - this creates the hierarchical structure
        y_triadic = (x * self.base) % self.base  # [batch_size] in [0, 3)

        # Compute distances to centers [0.5, 1.5, 2.5]
        d2 = (y_triadic.unsqueeze(-1) - self.centers) ** 2  # [batch_size, 3]

        # Soft assignment via temperature-scaled softmax
        logits = -d2 / (self.tau + 1e-8)
        p = F.softmax(logits, dim=-1)  # [batch_size, 3]

        # Alpha-normalized soft assignment (Beatrix style)
        # bit_k = p[left] * 0 + p[middle] * alpha + p[right] * 1
        bit_k = p[:, 1] * self.alpha + p[:, 2]

        return bit_k

    def get_interval_distribution(self, y: torch.Tensor) -> Dict[str, float]:
        """Get soft interval distribution for diagnostics."""
        if y.dim() > 1:
            y = y.mean(dim=1, keepdim=True)

        # Same normalization as forward
        x = torch.sigmoid(y).squeeze(-1)
        x = x.clamp(1e-6, 1.0 - 1e-6)
        y_triadic = (x * self.base) % self.base

        d2 = (y_triadic.unsqueeze(-1) - self.centers) ** 2
        logits = -d2 / (self.tau + 1e-8)
        p = F.softmax(logits, dim=-1)

        return {
            'left': p[:, 0].mean().item(),
            'middle': p[:, 1].mean().item(),
            'right': p[:, 2].mean().item()
        }


class GeometricHead(nn.Module):
    """
    Multi-scale geometric classification head.
    Provides components for pure geometric basin training.
    """

    def __init__(
            self,
            feature_dim: int,
            scale_dim: int,
            num_classes: int,
            k: int = 5
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.scale_dim = scale_dim
        self.num_classes = num_classes
        self.k = k

        # Project to scale dimension
        hidden_dim = scale_dim * 2
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, scale_dim)
        )

        # Class prototypes in scale space (for geometry loss)
        self.class_prototypes = nn.Parameter(
            torch.randn(num_classes, scale_dim)
        )

        # Cantor prototypes (scalar per class, for coherence loss)
        self.cantor_prototypes = nn.Parameter(
            torch.rand(num_classes) * 0.5 + 0.25  # Initialize in [0.25, 0.75]
        )

        # Geometric weights for inference (feature, cantor, crystal)
        self.geo_weights = nn.Parameter(torch.zeros(3))

    def forward(
            self,
            features: torch.Tensor,
            cantor_values: torch.Tensor
    ) -> torch.Tensor:
        """
        Inference-time forward pass combining geometric components.

        Args:
            features: Input features [batch_size, feature_dim]
            cantor_values: Cantor values [batch_size]
        Returns:
            logits: Classification logits [batch_size, num_classes]
        """
        batch_size = features.size(0)

        # Project to scale space
        query = self.projection(features)
        query = F.normalize(query, dim=-1)

        # 1. Feature similarity logits
        prototypes_norm = F.normalize(self.class_prototypes, dim=-1)
        feature_logits = query @ prototypes_norm.t()

        # 2. Cantor coherence logits
        cantor_distances = (
            cantor_values.unsqueeze(1) - self.cantor_prototypes.unsqueeze(0)
        ) ** 2
        cantor_logits = -cantor_distances

        # 3. Crystal geometry logits
        distances = torch.cdist(
            query.unsqueeze(1),
            prototypes_norm.unsqueeze(0)
        ).squeeze(1)
        crystal_logits = -distances

        # Geometric basin combination
        weights = F.softmax(self.geo_weights, dim=0)
        logits = (
                weights[0] * feature_logits +
                weights[1] * cantor_logits +
                weights[2] * crystal_logits
        )

        return logits


class GeoFractalDavid(nn.Module):
    """
    GeoFractalDavid: Pure Geometric Basin Architecture

    Multi-scale classifier with geometric structure:
    - Soft Cantor staircase encoding
    - Crystal simplex geometry
    - Learnable geometric basin weights

    Designed for training with pure geometric losses (no cross-entropy).
    Compatible with: coherence, separation, discretization, geometry losses.
    """

    def __init__(
            self,
            feature_dim: int = 512,
            num_classes: int = 1000,
            k: int = 5,
            scales: Optional[List[int]] = None,
            alpha_init: float = 0.5,
            tau: float = 0.25
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.k = k
        self.scales = scales or [256, 384, 512, 768, 1024, 1280]

        # Shared Cantor staircase
        self.cantor_stairs = CantorStaircase(alpha_init=alpha_init, tau=tau)

        # Multi-scale heads
        self.heads = nn.ModuleDict({
            str(scale): GeometricHead(
                feature_dim=feature_dim,
                scale_dim=scale,
                num_classes=num_classes,
                k=k
            ) for scale in self.scales
        })

        # Learnable fusion weights across scales
        self.fusion_weights = nn.Parameter(torch.ones(len(self.scales)))

        self._init_prototypes()

    def _init_prototypes(self):
        """Initialize class prototypes on unit sphere."""
        for head in self.heads.values():
            nn.init.xavier_uniform_(head.class_prototypes)
            head.class_prototypes.data = F.normalize(
                head.class_prototypes.data, dim=-1
            )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Inference forward pass with multi-scale fusion.

        Args:
            features: Input embeddings [batch_size, feature_dim]
        Returns:
            logits: Fused classification logits [batch_size, num_classes]
        """
        batch_size = features.size(0)

        # Normalize input features
        features = F.normalize(features, dim=-1)

        # Compute Cantor values (shared across scales)
        cantor_values = self.cantor_stairs(features)

        # Get logits from each scale
        scale_logits = []
        for scale in self.scales:
            head = self.heads[str(scale)]
            logits = head(features, cantor_values)
            scale_logits.append(logits)

        # Fuse across scales with learnable weights
        scale_logits_tensor = torch.stack(scale_logits, dim=0)
        weights = F.softmax(self.fusion_weights, dim=0).view(-1, 1, 1)
        fused_logits = (weights * scale_logits_tensor).sum(dim=0)

        return fused_logits

    def get_scale_logits(self, features: torch.Tensor) -> Dict[int, torch.Tensor]:
        """Get individual scale logits for analysis."""
        features = F.normalize(features, dim=-1)
        cantor_values = self.cantor_stairs(features)

        scale_logits = {}
        for scale in self.scales:
            head = self.heads[str(scale)]
            scale_logits[scale] = head(features, cantor_values)

        return scale_logits

    def get_cantor_values(self, features: torch.Tensor) -> torch.Tensor:
        """Get Cantor values for features."""
        features = F.normalize(features, dim=-1)
        return self.cantor_stairs(features)

    def get_geometric_weights(self) -> Dict[int, Dict[str, float]]:
        """Get current geometric basin weights per scale."""
        weights = {}
        for scale in self.scales:
            head = self.heads[str(scale)]
            w = F.softmax(head.geo_weights, dim=0).detach().cpu().numpy()
            weights[scale] = {
                'feature': float(w[0]),
                'cantor': float(w[1]),
                'crystal': float(w[2])
            }
        return weights

    def get_cantor_interval_distribution(self, features: torch.Tensor) -> Dict[int, Dict[str, float]]:
        """Get Cantor interval distribution per scale for diagnostics."""
        features = F.normalize(features, dim=-1)
        distributions = {}

        for scale in self.scales:
            head = self.heads[str(scale)]
            proj_features = head.projection(features)
            dist = self.cantor_stairs.get_interval_distribution(proj_features)
            distributions[scale] = dist

        return distributions

    def __repr__(self):
        s = f"GeoFractalDavid (Pure Geometric Basin)\n"
        s += f"  Simplex: k={self.k} ({self.k + 1} vertices)\n"
        s += f"  Scales: {self.scales}\n"
        s += f"  Cantor Alpha: {self.cantor_stairs.alpha.item():.4f}\n"
        s += f"  Parameters: {sum(p.numel() for p in self.parameters()):,}"
        return s


if __name__ == "__main__":
    # Test the model
    model = GeoFractalDavid(
        feature_dim=512,
        num_classes=1000,
        k=5,
        scales=[256, 384, 512, 768, 1024, 1280]
    )

    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass
    batch_size = 16
    features = torch.randn(batch_size, 512)

    logits = model(features)
    print(f"\nInput shape: {features.shape}")
    print(f"Output shape: {logits.shape}")

    # Test scale logits
    scale_logits = model.get_scale_logits(features)
    print(f"\nScale logits:")
    for scale, logits in scale_logits.items():
        print(f"  Scale {scale}: {logits.shape}")

    # Test Cantor values
    cantor_vals = model.get_cantor_values(features)
    print(f"\nCantor values shape: {cantor_vals.shape}")
    print(f"Cantor value range: [{cantor_vals.min():.3f}, {cantor_vals.max():.3f}]")

    # Test geometric weights
    geo_weights = model.get_geometric_weights()
    print(f"\nGeometric weights:")
    for scale, weights in geo_weights.items():
        print(f"  Scale {scale}: feature={weights['feature']:.3f}, "
              f"cantor={weights['cantor']:.3f}, crystal={weights['crystal']:.3f}")

    # Test Cantor interval distribution
    interval_dist = model.get_cantor_interval_distribution(features)
    print(f"\nCantor interval distribution:")
    for scale, dist in interval_dist.items():
        print(f"  Scale {scale}: L={dist['left']:.3f}, M={dist['middle']:.3f}, R={dist['right']:.3f}")