"""
GeoFractalDavid: Pure Geometric Basin Architecture (Refactored)
================================================================
Compatible with pure geometric basin training (no cross-entropy).
Uses 4 geometric loss components: coherence, separation, discretization, geometry.
Uses SimplexFactory for proper k-simplex class prototypes.

Author: AbstractPhil + Claude Sonnet 4.5
License: MIT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple

from geovocab2.shapes.factory import SimplexFactory


class CantorStaircase(nn.Module):
    """
    Learnable soft Cantor staircase with alpha-normalized middle weighting.
    Provides infinite lattice structure that features traverse through.

    Combines:
    - Position-based lattice structure (Beatrix paradigm)
    - Feature-driven modulation (learnable trajectories)
    """

    def __init__(self, feature_dim: int, alpha_init: float = 0.5, tau: float = 0.25, base: int = 3):
        super().__init__()
        self.feature_dim = feature_dim
        self.alpha = nn.Parameter(torch.tensor(alpha_init))
        self.tau = tau
        self.base = base

        # Fixed centers for triadic intervals [left, middle, right]
        self.register_buffer('centers', torch.tensor([0.5, 1.5, 2.5]))

        # Learnable feature modulation (features influence position in lattice)
        self.feature_to_position = nn.Linear(feature_dim, 1)

    def forward(self, y: torch.Tensor, positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            y: Input features [batch_size, feature_dim]
            positions: Optional position indices [batch_size]
        Returns:
            cantor_values: Soft Cantor values in [0, 1] [batch_size]
        """
        batch_size = y.size(0)

        # Get base positions (lattice structure)
        if positions is None:
            positions = torch.arange(batch_size, device=y.device, dtype=torch.float32)

        # Normalize positions to [0, 1] (base lattice)
        x_base = positions / max(batch_size - 1, 1.0)
        x_base = x_base.clamp(1e-6, 1.0 - 1e-6)

        # Feature-driven modulation (trajectories through lattice)
        feature_shift = self.feature_to_position(y).squeeze(-1)  # [batch_size]
        feature_shift = torch.tanh(feature_shift) * 0.3  # Bounded shift ±0.3

        # Combine: base structure + learned trajectory
        x = (x_base + feature_shift).clamp(1e-6, 1.0 - 1e-6)

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

    def get_interval_distribution(self, y: torch.Tensor, positions: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """Get soft interval distribution for diagnostics."""
        batch_size = y.size(0)

        if positions is None:
            positions = torch.arange(batch_size, device=y.device, dtype=torch.float32)

        x_base = positions / max(batch_size - 1, 1.0)
        x_base = x_base.clamp(1e-6, 1.0 - 1e-6)

        feature_shift = self.feature_to_position(y).squeeze(-1)
        feature_shift = torch.tanh(feature_shift) * 0.3

        x = (x_base + feature_shift).clamp(1e-6, 1.0 - 1e-6)
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
    Uses k-simplex structures for class prototypes.
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
        self.num_vertices = k + 1

        # Project to scale dimension
        hidden_dim = scale_dim * 2
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, scale_dim)
        )

        # Class prototypes as k-simplices (k+1 vertices per class)
        # Shape: [num_classes, k+1, scale_dim]
        # Use RANDOM method with proper seed per class for distinguishability
        simplex_factory = SimplexFactory(k=k, embed_dim=scale_dim, method="random")

        # Build simplices in batches to avoid memory explosion
        # Generate on CPU first, then move to GPU
        class_simplices = []
        batch_size = 100  # Process 100 classes at a time

        print(f"    Generating {num_classes} {k}-simplices for scale {scale_dim}...")

        with torch.no_grad():  # No gradients needed during initialization
            for batch_start in range(0, num_classes, batch_size):
                batch_end = min(batch_start + batch_size, num_classes)
                batch_simplices = []

                for class_idx in range(batch_start, batch_end):
                    # Use hash for better seed distribution
                    seed = hash(f"simplex_class_{class_idx}") % (2**32)

                    # Build on CPU first
                    simplex = simplex_factory.build(
                        backend="torch",
                        device="cpu",  # CPU to avoid GPU memory accumulation
                        dtype=torch.float32,
                        seed=seed,
                        validate=False  # Skip validation during init for speed
                    )
                    batch_simplices.append(simplex)

                # Move batch to target device
                device = "cuda" if torch.cuda.is_available() else "cpu"
                batch_tensor = torch.stack(batch_simplices).to(device)
                class_simplices.append(batch_tensor)

                # Clear cache after each batch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                print(f"      {batch_end}/{num_classes} complete", end='\r')

        print(f"      {num_classes}/{num_classes} complete ✓")

        # Concatenate all batches: [num_classes, k+1, scale_dim]
        simplices_tensor = torch.cat(class_simplices, dim=0)
        self.class_prototypes = nn.Parameter(simplices_tensor)

        # Cantor prototypes (scalar per class, for coherence loss)
        # Initialize evenly spaced across [0, 1] like original FractalDavid
        self.cantor_prototypes = nn.Parameter(
            torch.linspace(0.0, 1.0, num_classes, dtype=torch.float32)
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
        # Project to scale space
        query = self.projection(features)
        query = F.normalize(query, dim=-1)

        return self.forward_from_projection(query, cantor_values)

    def forward_from_projection(
            self,
            query: torch.Tensor,
            cantor_values: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward from already-projected features (avoids recomputation).

        Args:
            query: Projected features [batch_size, scale_dim] (already normalized)
            cantor_values: Cantor values [batch_size]
        Returns:
            logits: Classification logits [batch_size, num_classes]
        """
        batch_size = query.size(0)

        # 1. Feature similarity logits (use simplex centroids)
        # Centroid of each class simplex: [num_classes, scale_dim]
        simplex_centroids = self.class_prototypes.mean(dim=1)
        centroids_norm = F.normalize(simplex_centroids, dim=-1)
        feature_logits = query @ centroids_norm.t()

        # 2. Cantor coherence logits
        cantor_distances = (
            cantor_values.unsqueeze(1) - self.cantor_prototypes.unsqueeze(0)
        ) ** 2
        cantor_logits = -cantor_distances

        # 3. Crystal geometry logits (distance to nearest simplex vertex)
        # MEMORY EFFICIENT: Avoid [B, C, k+1, D] broadcast
        # Use: ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a·b

        # Flatten simplices: [num_classes * (k+1), scale_dim]
        simplices_flat = self.class_prototypes.view(-1, self.scale_dim)  # [C*(k+1), D]

        # Compute squared distances efficiently
        query_norm = (query ** 2).sum(dim=1, keepdim=True)  # [B, 1]
        simplices_norm = (simplices_flat ** 2).sum(dim=1, keepdim=True).t()  # [1, C*(k+1)]

        # Dot product: [B, C*(k+1)]
        dot_product = query @ simplices_flat.t()

        # Squared distances: [B, C*(k+1)]
        sq_distances = query_norm + simplices_norm - 2 * dot_product
        sq_distances = sq_distances.clamp(min=0)  # Numerical stability

        # Reshape to [B, C, k+1] and take min over vertices
        sq_distances = sq_distances.view(batch_size, self.num_classes, self.num_vertices)
        min_sq_distances, _ = sq_distances.min(dim=-1)  # [B, C]

        # Crystal logits (negative distance)
        crystal_logits = -torch.sqrt(min_sq_distances + 1e-8)

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

        # Shared Cantor staircase (takes BOTH positions and features)
        self.cantor_stairs = CantorStaircase(
            feature_dim=feature_dim,  # Now needs feature_dim for modulation
            alpha_init=alpha_init,
            tau=tau
        )

        # Multi-scale heads (will generate simplices for each scale)
        print(f"\n[🔷] Initializing {len(self.scales)} scales with k={k} simplices...")
        self.heads = nn.ModuleDict()
        for i, scale in enumerate(self.scales, 1):
            print(f"  Scale {i}/{len(self.scales)}: {scale}")
            self.heads[str(scale)] = GeometricHead(
                feature_dim=feature_dim,
                scale_dim=scale,
                num_classes=num_classes,
                k=k
            )
        print(f"[✅] All scales initialized\n")

        # Learnable fusion weights across scales
        self.fusion_weights = nn.Parameter(torch.ones(len(self.scales)))

        self._init_prototypes()

        # Final memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _init_prototypes(self):
        """Initialize class simplex prototypes - keep them unnormalized for proper centroids."""
        # Do NOT normalize vertices to unit sphere
        # Centroids need to be distinguishable, not collapsed to origin
        pass

    def forward(self, features: torch.Tensor, return_intermediates: bool = False) -> torch.Tensor:
        """
        Inference forward pass with multi-scale fusion.

        Args:
            features: Input embeddings [batch_size, feature_dim]
            return_intermediates: If True, return (logits, cantor_values, proj_features_dict)
        Returns:
            logits: Fused classification logits [batch_size, num_classes]
            OR (logits, cantor_values, proj_features_dict) if return_intermediates=True
        """
        batch_size = features.size(0)

        # Normalize input features
        features = F.normalize(features, dim=-1)

        # Generate positions for lattice structure
        positions = torch.arange(batch_size, device=features.device, dtype=torch.float32)

        # Compute Cantor values (lattice + feature trajectories)
        cantor_values = self.cantor_stairs(features, positions)

        # Get logits from each scale
        scale_logits = []
        proj_features_dict = {} if return_intermediates else None

        for scale in self.scales:
            head = self.heads[str(scale)]

            # Store projected features if needed for loss
            if return_intermediates:
                proj_features = head.projection(features)
                proj_features = F.normalize(proj_features, dim=-1)
                proj_features_dict[scale] = proj_features
                # Use stored proj_features in head forward
                logits = head.forward_from_projection(proj_features, cantor_values)
            else:
                logits = head(features, cantor_values)

            scale_logits.append(logits)

        # Fuse across scales with learnable weights
        scale_logits_tensor = torch.stack(scale_logits, dim=0)
        weights = F.softmax(self.fusion_weights, dim=0).view(-1, 1, 1)
        fused_logits = (weights * scale_logits_tensor).sum(dim=0)

        if return_intermediates:
            return fused_logits, cantor_values, proj_features_dict
        return fused_logits

    def get_scale_logits(self, features: torch.Tensor) -> Dict[int, torch.Tensor]:
        """Get individual scale logits for analysis."""
        features = F.normalize(features, dim=-1)
        batch_size = features.size(0)
        positions = torch.arange(batch_size, device=features.device, dtype=torch.float32)
        cantor_values = self.cantor_stairs(features, positions)

        scale_logits = {}
        for scale in self.scales:
            head = self.heads[str(scale)]
            scale_logits[scale] = head(features, cantor_values)

        return scale_logits

    def get_cantor_values(self, features: torch.Tensor) -> torch.Tensor:
        """Get Cantor values for features."""
        features = F.normalize(features, dim=-1)
        batch_size = features.size(0)
        positions = torch.arange(batch_size, device=features.device, dtype=torch.float32)
        return self.cantor_stairs(features, positions)

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
        batch_size = features.size(0)
        positions = torch.arange(batch_size, device=features.device, dtype=torch.float32)

        # Cantor staircase operates on ORIGINAL features, not projected
        dist = self.cantor_stairs.get_interval_distribution(features, positions)

        # Same distribution applies to all scales (shared Cantor staircase)
        distributions = {scale: dist for scale in self.scales}

        return distributions

    def __repr__(self):
        s = f"GeoFractalDavid (Pure Geometric Basin + SimplexFactory)\n"
        s += f"  Simplex: k={self.k} ({self.k + 1} vertices per class)\n"
        s += f"  Scales: {self.scales}\n"
        s += f"  Cantor Alpha: {self.cantor_stairs.alpha.item():.4f}\n"
        s += f"  Total Classes: {self.num_classes}\n"
        s += f"  Total Simplex Vertices: {self.num_classes * (self.k + 1):,}\n"
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
    print(f"\nCantor interval distribution (Beatrix paradigm):")
    for scale, dist in interval_dist.items():
        print(f"  Scale {scale}: L={dist['left']:.3f}, M={dist['middle']:.3f}, R={dist['right']:.3f}")

    # Check simplex structure
    print(f"\nSimplex structure:")
    for scale in [256, 512, 1024]:
        head = model.heads[str(scale)]
        print(f"  Scale {scale}:")
        print(f"    Class prototypes shape: {head.class_prototypes.shape}")
        print(f"    Expected: [{model.num_classes}, {model.k + 1}, {scale}]")

        # Validate one simplex
        simplex_0 = head.class_prototypes[0]  # First class
        centroid = simplex_0.mean(dim=0)
        print(f"    Class 0 centroid norm: {torch.norm(centroid):.4f}")

        # Check distances between vertices
        distances = []
        for i in range(model.k + 1):
            for j in range(i + 1, model.k + 1):
                d = torch.norm(simplex_0[i] - simplex_0[j])
                distances.append(d.item())
        print(f"    Edge lengths: min={min(distances):.4f}, max={max(distances):.4f}, "
              f"mean={sum(distances)/len(distances):.4f}")