"""
GeoFractalDavid: Pure Geometric Basin Architecture
Designed for training with geometric losses instead of cross-entropy.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional


class CantorStaircase(nn.Module):
    """
    Learnable soft Cantor staircase with alpha-normalized middle weighting.
    Maps features to [0, 1] via soft triadic assignment.
    """

    def __init__(self, alpha_init: float = 0.5, tau: float = 0.25):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha_init))
        self.tau = tau

        # Fixed centers for triadic intervals
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

        # Compute distances to centers
        d2 = (y - self.centers.unsqueeze(0)) ** 2  # [batch_size, 3]

        # Soft assignment via temperature-scaled softmax
        logits = -d2 / (self.tau + 1e-8)
        p = F.softmax(logits, dim=-1)  # [batch_size, 3]

        # Alpha-normalized soft assignment
        # bit_k = p[..., 2] + alpha * p[..., 1]
        bit_k = p[:, 2] + self.alpha * p[:, 1]

        return bit_k.squeeze()


class GeometricHead(nn.Module):
    """
    Multi-scale geometric classification head.
    Combines feature similarity, Cantor coherence, and crystal geometry.
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

        # Class prototypes in scale space (k+1 vertices per class)
        self.class_prototypes = nn.Parameter(
            torch.randn(num_classes, scale_dim)
        )

        # Cantor prototypes (scalar per class)
        self.cantor_prototypes = nn.Parameter(
            torch.rand(num_classes) * 0.5 + 0.25  # Initialize in [0.25, 0.75]
        )

        # Geometric weights (feature, cantor, crystal)
        self.geo_weights = nn.Parameter(torch.zeros(3))

    def forward(
            self,
            features: torch.Tensor,
            cantor_values: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            features: Input features [batch_size, feature_dim]
            cantor_values: Cantor values [batch_size]
        Returns:
            logits: Classification logits [batch_size, num_classes]
        """
        batch_size = features.size(0)

        # Project to scale space
        query = self.projection(features)  # [batch_size, scale_dim]
        query = F.normalize(query, dim=-1)

        # 1. Feature similarity logits
        prototypes_norm = F.normalize(self.class_prototypes, dim=-1)
        feature_logits = query @ prototypes_norm.t()  # [batch_size, num_classes]

        # 2. Cantor coherence logits
        cantor_distances = (
                                   cantor_values.unsqueeze(1) - self.cantor_prototypes.unsqueeze(0)
                           ) ** 2  # [batch_size, num_classes]
        cantor_logits = -cantor_distances  # Lower distance = higher logit

        # 3. Crystal geometry logits (distance-based)
        distances = torch.cdist(
            query.unsqueeze(1),
            prototypes_norm.unsqueeze(0)
        ).squeeze(1)  # [batch_size, num_classes]
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

    Designed for training with geometric losses (no cross-entropy required).
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
        scale_logits_tensor = torch.stack(scale_logits, dim=0)  # [num_scales, batch_size, num_classes]
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

    def __repr__(self):
        s = f"GeoFractalDavid (Pure Geometric)\n"
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