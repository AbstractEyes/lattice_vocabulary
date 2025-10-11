"""
GEOMETRIC CLASSIFICATION LOSS FRAMEWORK
----------------------------------------
Leverages Devil's Staircase PE and Beatrix formulas for classification.

Key Idea:
- Map samples to Cantor measure space
- Use geometric properties (simplex volumes, distances, coherence) as losses
- Guide model to learn representations that respect fractal structure

This enables:
1. Hierarchical class structure (coarse → fine via PE levels)
2. Contrastive learning with geometric consistency
3. Better generalization via geometric constraints
4. OOD detection through geometric violations

These have partial implications for interpretability and robustness with transformer-based architectures.

Author: AbstractPhil + Claude Sonnet 4.5
License: MIT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import math


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# GEOMETRIC CLASSIFICATION LOSS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class GeometricClassificationLoss(nn.Module):
    """
    Classification loss augmented with geometric structure from Devil's Staircase PE.

    Components:
    1. Standard cross-entropy (base classification)
    2. Cantor measure consistency (same class → similar measures)
    3. Simplex volume coherence (within-class compactness)
    4. Geometric margin (between-class separation)
    5. Hierarchical alignment (class relationships via PE levels)

    Args:
        num_classes: Number of classification targets
        ce_weight: Weight for cross-entropy loss (default: 1.0)
        cantor_weight: Weight for Cantor consistency loss (default: 0.3)
        volume_weight: Weight for volume coherence loss (default: 0.2)
        margin_weight: Weight for geometric margin loss (default: 0.1)
        hierarchy_weight: Weight for hierarchical alignment (default: 0.1)
        temperature: Temperature for soft assignments (default: 0.1)
    """

    def __init__(
            self,
            num_classes: int,
            ce_weight: float = 1.0,
            cantor_weight: float = 0.3,
            volume_weight: float = 0.2,
            margin_weight: float = 0.1,
            hierarchy_weight: float = 0.1,
            temperature: float = 0.1,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.ce_weight = ce_weight
        self.cantor_weight = cantor_weight
        self.volume_weight = volume_weight
        self.margin_weight = margin_weight
        self.hierarchy_weight = hierarchy_weight
        self.temperature = temperature

        # Learnable class prototypes in Cantor space
        self.class_cantor_prototypes = nn.Parameter(
            torch.linspace(0.0, 1.0, num_classes),
            requires_grad=True
        )

        # Learnable target volumes for each class
        self.class_target_volumes = nn.Parameter(
            torch.ones(num_classes),
            requires_grad=True
        )

    def cantor_consistency_loss(
            self,
            cantor_measures: torch.Tensor,
            labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Enforce samples from same class to have similar Cantor measures.

        Args:
            cantor_measures: [B] Cantor measures for batch
            labels: [B] class labels

        Returns:
            loss: Cantor consistency loss
        """
        batch_size = cantor_measures.shape[0]

        # Get class prototypes for each sample
        class_prototypes = self.class_cantor_prototypes[labels]  # [B]

        # L2 distance to class prototype
        distances = (cantor_measures - class_prototypes) ** 2

        # Mean distance
        loss = distances.mean()

        return loss

    def volume_coherence_loss(
            self,
            simplex_vertices: torch.Tensor,
            labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Enforce within-class volume consistency.

        Args:
            simplex_vertices: [B, k+1, dim] simplex vertices
            labels: [B] class labels

        Returns:
            loss: Volume coherence loss
        """
        batch_size = simplex_vertices.shape[0]

        # Compute simplex volumes (using determinant approximation)
        # For efficiency, use centroid spread as volume proxy
        centroids = simplex_vertices.mean(dim=1)  # [B, dim]
        spreads = torch.norm(simplex_vertices - centroids.unsqueeze(1), dim=-1).mean(dim=-1)  # [B]

        # Get target volumes for each class
        target_volumes = self.class_target_volumes[labels]  # [B]

        # L2 distance to target volume
        volume_loss = ((spreads - target_volumes) ** 2).mean()

        return volume_loss

    def geometric_margin_loss(
            self,
            simplex_vertices: torch.Tensor,
            labels: torch.Tensor,
            margin: float = 1.0
    ) -> torch.Tensor:
        """
        Enforce geometric separation between different classes.

        Args:
            simplex_vertices: [B, k+1, dim] simplex vertices
            labels: [B] class labels
            margin: Minimum geometric distance between classes

        Returns:
            loss: Geometric margin loss
        """
        batch_size = simplex_vertices.shape[0]

        # Compute centroids
        centroids = simplex_vertices.mean(dim=1)  # [B, dim]

        # Pairwise distances between all samples
        dists = torch.cdist(centroids, centroids, p=2)  # [B, B]

        # Create mask for different-class pairs
        label_match = labels.unsqueeze(0) == labels.unsqueeze(1)  # [B, B]
        different_class = ~label_match  # [B, B]

        # Only consider different-class pairs
        # Apply hinge loss: max(0, margin - distance)
        margin_violations = F.relu(margin - dists)  # [B, B]
        margin_violations = margin_violations * different_class.float()

        # Average over different-class pairs
        num_pairs = different_class.float().sum()
        if num_pairs > 0:
            loss = margin_violations.sum() / num_pairs
        else:
            loss = torch.tensor(0.0, device=simplex_vertices.device)

        return loss

    def hierarchical_alignment_loss(
            self,
            pe_features: torch.Tensor,
            labels: torch.Tensor,
            levels: int = 12,
            features_per_level: int = 2
    ) -> torch.Tensor:
        """
        Enforce hierarchical class structure via PE levels.

        Early levels should capture coarse class groupings,
        later levels capture fine-grained distinctions.

        Args:
            pe_features: [B, levels*features_per_level] PE features
            labels: [B] class labels
            levels: Number of PE levels
            features_per_level: Features per level

        Returns:
            loss: Hierarchical alignment loss
        """
        batch_size = pe_features.shape[0]

        # Reshape to per-level: [B, levels, features_per_level]
        pe_levels = pe_features.view(batch_size, levels, features_per_level)

        # Extract bit features (first feature of each level)
        bits = pe_levels[:, :, 0]  # [B, levels]

        # Compute class similarity matrix based on labels
        # Classes with similar IDs should be similar
        class_dists = torch.abs(labels.unsqueeze(0) - labels.unsqueeze(1)).float()  # [B, B]
        class_sims = torch.exp(-class_dists / self.num_classes)  # [B, B]

        # Compute feature similarity at each level
        losses = []
        for level_idx in range(levels):
            level_feats = bits[:, level_idx:level_idx + 1]  # [B, 1]

            # Pairwise distances at this level
            feat_dists = torch.cdist(level_feats, level_feats, p=2)  # [B, B]

            # Feature similarity (higher for closer features)
            feat_sims = torch.exp(-feat_dists)  # [B, B]

            # Loss: MSE between class similarity and feature similarity
            level_loss = F.mse_loss(feat_sims, class_sims)

            # Weight: early levels get more weight for coarse structure
            level_weight = 1.0 / (level_idx + 1)
            losses.append(level_weight * level_loss)

        total_loss = torch.stack(losses).mean()

        return total_loss

    def forward(
            self,
            logits: torch.Tensor,
            labels: torch.Tensor,
            cantor_measures: torch.Tensor,
            pe_features: torch.Tensor,
            simplex_vertices: torch.Tensor,
            return_components: bool = False
    ) -> torch.Tensor | Dict[str, torch.Tensor]:
        """
        Compute full geometric classification loss.

        Args:
            logits: [B, num_classes] classification logits
            labels: [B] class labels
            cantor_measures: [B] Cantor measures
            pe_features: [B, feat_dim] PE features
            simplex_vertices: [B, k+1, dim] simplex vertices
            return_components: If True, return loss components

        Returns:
            loss: Total loss (or dict of components if return_components=True)
        """
        # 1. Standard cross-entropy
        ce_loss = F.cross_entropy(logits, labels)

        # 2. Cantor consistency
        cantor_loss = self.cantor_consistency_loss(cantor_measures, labels)

        # 3. Volume coherence
        volume_loss = self.volume_coherence_loss(simplex_vertices, labels)

        # 4. Geometric margin
        margin_loss = self.geometric_margin_loss(simplex_vertices, labels)

        # 5. Hierarchical alignment
        hierarchy_loss = self.hierarchical_alignment_loss(pe_features, labels)

        # Total weighted loss
        total_loss = (
                self.ce_weight * ce_loss +
                self.cantor_weight * cantor_loss +
                self.volume_weight * volume_loss +
                self.margin_weight * margin_loss +
                self.hierarchy_weight * hierarchy_loss
        )

        if return_components:
            return {
                'total': total_loss,
                'ce': ce_loss,
                'cantor': cantor_loss,
                'volume': volume_loss,
                'margin': margin_loss,
                'hierarchy': hierarchy_loss
            }

        return total_loss


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONTRASTIVE GEOMETRIC LOSS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ContrastiveGeometricLoss(nn.Module):
    """
    Contrastive learning with geometric structure.

    Similar to SimCLR/MoCo but uses Cantor measure and simplex geometry
    to define positive/negative pairs.

    Args:
        temperature: Temperature for contrastive loss (default: 0.07)
        cantor_threshold: Cantor distance threshold for positives (default: 0.1)
        volume_threshold: Volume similarity threshold for positives (default: 0.2)
    """

    def __init__(
            self,
            temperature: float = 0.07,
            cantor_threshold: float = 0.1,
            volume_threshold: float = 0.2
    ):
        super().__init__()
        self.temperature = temperature
        self.cantor_threshold = cantor_threshold
        self.volume_threshold = volume_threshold

    def forward(
            self,
            embeddings: torch.Tensor,
            cantor_measures: torch.Tensor,
            simplex_vertices: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute contrastive loss with geometric guidance.

        Args:
            embeddings: [B, dim] feature embeddings
            cantor_measures: [B] Cantor measures
            simplex_vertices: [B, k+1, dim] simplex vertices

        Returns:
            loss: Contrastive loss
        """
        batch_size = embeddings.shape[0]
        device = embeddings.device

        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=-1)

        # Compute similarity matrix
        sim_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature  # [B, B]

        # Compute Cantor distances
        cantor_dists = torch.abs(
            cantor_measures.unsqueeze(0) - cantor_measures.unsqueeze(1)
        )  # [B, B]

        # Compute volume similarities
        centroids = simplex_vertices.mean(dim=1)  # [B, dim]
        spreads = torch.norm(simplex_vertices - centroids.unsqueeze(1), dim=-1).mean(dim=-1)  # [B]
        volume_dists = torch.abs(spreads.unsqueeze(0) - spreads.unsqueeze(1))  # [B, B]

        # Define geometric positives:
        # Samples are positive if they have similar Cantor measures AND volumes
        geo_positives = (
                (cantor_dists < self.cantor_threshold) &
                (volume_dists < self.volume_threshold)
        ).float()  # [B, B]

        # Mask out self-comparisons
        mask = torch.eye(batch_size, device=device)
        geo_positives = geo_positives * (1 - mask)

        # Compute contrastive loss
        # For each sample, pull positives closer, push negatives away

        # Numerator: sum of similarities to geometric positives
        exp_sim = torch.exp(sim_matrix)  # [B, B]
        pos_sum = (exp_sim * geo_positives).sum(dim=1)  # [B]

        # Denominator: sum over all non-self samples
        neg_mask = 1 - mask
        all_sum = (exp_sim * neg_mask).sum(dim=1)  # [B]

        # Avoid division by zero
        pos_sum = pos_sum.clamp(min=1e-8)
        all_sum = all_sum.clamp(min=1e-8)

        # Loss: -log(positive_similarity / all_similarity)
        loss = -torch.log(pos_sum / all_sum)

        # Only compute loss for samples that have positives
        has_positives = geo_positives.sum(dim=1) > 0
        if has_positives.sum() > 0:
            loss = loss[has_positives].mean()
        else:
            loss = torch.tensor(0.0, device=device)

        return loss


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# OOD DETECTION LOSS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class GeometricOODLoss(nn.Module):
    """
    Out-of-distribution detection via geometric violations.

    In-distribution samples should have:
    1. Cantor measures within [0, 1] with proper density
    2. Simplex volumes within expected range
    3. Coherent geometric structure

    OOD samples will violate these properties.

    Args:
        cantor_margin: Margin for Cantor density (default: 0.05)
        volume_margin: Margin for volume range (default: 0.1)
    """

    def __init__(
            self,
            cantor_margin: float = 0.05,
            volume_margin: float = 0.1
    ):
        super().__init__()
        self.cantor_margin = cantor_margin
        self.volume_margin = volume_margin

        # Learnable statistics for in-distribution
        self.register_buffer('id_cantor_mean', torch.tensor(0.5))
        self.register_buffer('id_cantor_std', torch.tensor(0.25))
        self.register_buffer('id_volume_mean', torch.tensor(1.0))
        self.register_buffer('id_volume_std', torch.tensor(0.3))

    def update_statistics(
            self,
            cantor_measures: torch.Tensor,
            simplex_vertices: torch.Tensor
    ):
        """Update in-distribution statistics (call during training)."""
        with torch.no_grad():
            self.id_cantor_mean = cantor_measures.mean()
            self.id_cantor_std = cantor_measures.std()

            centroids = simplex_vertices.mean(dim=1)
            spreads = torch.norm(
                simplex_vertices - centroids.unsqueeze(1),
                dim=-1
            ).mean(dim=-1)

            self.id_volume_mean = spreads.mean()
            self.id_volume_std = spreads.std()

    def compute_ood_score(
            self,
            cantor_measures: torch.Tensor,
            simplex_vertices: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute OOD score for each sample.
        Higher score = more likely OOD.

        Args:
            cantor_measures: [B] Cantor measures
            simplex_vertices: [B, k+1, dim] simplex vertices

        Returns:
            ood_scores: [B] OOD scores (higher = more OOD)
        """
        # Cantor measure deviation
        cantor_z_scores = torch.abs(
            (cantor_measures - self.id_cantor_mean) / (self.id_cantor_std + 1e-8)
        )

        # Volume deviation
        centroids = simplex_vertices.mean(dim=1)
        spreads = torch.norm(
            simplex_vertices - centroids.unsqueeze(1),
            dim=-1
        ).mean(dim=-1)

        volume_z_scores = torch.abs(
            (spreads - self.id_volume_mean) / (self.id_volume_std + 1e-8)
        )

        # Combined OOD score
        ood_scores = cantor_z_scores + volume_z_scores

        return ood_scores

    def forward(
            self,
            cantor_measures: torch.Tensor,
            simplex_vertices: torch.Tensor,
            is_ood: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute OOD detection loss.

        Args:
            cantor_measures: [B] Cantor measures
            simplex_vertices: [B, k+1, dim] simplex vertices
            is_ood: [B] binary labels (1=OOD, 0=ID), optional

        Returns:
            loss: OOD detection loss
        """
        ood_scores = self.compute_ood_score(cantor_measures, simplex_vertices)

        if is_ood is not None:
            # Supervised OOD detection
            # Maximize scores for OOD, minimize for ID
            target_scores = is_ood.float() * 10.0  # High target for OOD
            loss = F.mse_loss(ood_scores, target_scores)
        else:
            # Unsupervised: just regularize to be close to in-distribution
            loss = ood_scores.mean()

        return loss


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# EXAMPLE USAGE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def demo_geometric_losses():
    """Demonstrate geometric classification losses."""

    print("=" * 70)
    print("GEOMETRIC CLASSIFICATION LOSS DEMONSTRATION")
    print("=" * 70)

    batch_size = 32
    num_classes = 10
    feat_dim = 512
    k_simplex = 5
    pe_levels = 12
    pe_features_per_level = 2

    # Mock data
    logits = torch.randn(batch_size, num_classes)
    labels = torch.randint(0, num_classes, (batch_size,))
    cantor_measures = torch.rand(batch_size)
    pe_features = torch.randn(batch_size, pe_levels * pe_features_per_level)
    simplex_vertices = torch.randn(batch_size, k_simplex + 1, feat_dim)

    print("\n[1] Geometric Classification Loss")
    geo_loss = GeometricClassificationLoss(num_classes=num_classes)

    loss_components = geo_loss(
        logits, labels, cantor_measures, pe_features, simplex_vertices,
        return_components=True
    )

    print(f"  Total Loss: {loss_components['total'].item():.4f}")
    print(f"  Components:")
    print(f"    Cross-Entropy:      {loss_components['ce'].item():.4f}")
    print(f"    Cantor Consistency: {loss_components['cantor'].item():.4f}")
    print(f"    Volume Coherence:   {loss_components['volume'].item():.4f}")
    print(f"    Geometric Margin:   {loss_components['margin'].item():.4f}")
    print(f"    Hierarchical:       {loss_components['hierarchy'].item():.4f}")

    print("\n[2] Contrastive Geometric Loss")
    embeddings = torch.randn(batch_size, feat_dim)
    contrastive_loss = ContrastiveGeometricLoss()

    loss = contrastive_loss(embeddings, cantor_measures, simplex_vertices)
    print(f"  Contrastive Loss: {loss.item():.4f}")

    print("\n[3] OOD Detection Loss")
    ood_loss = GeometricOODLoss()

    # Update statistics with ID data
    ood_loss.update_statistics(cantor_measures, simplex_vertices)

    # Compute OOD scores
    ood_scores = ood_loss.compute_ood_score(cantor_measures, simplex_vertices)
    print(f"  Mean OOD Score: {ood_scores.mean().item():.4f}")
    print(f"  OOD Score Range: [{ood_scores.min().item():.4f}, {ood_scores.max().item():.4f}]")

    # Compute loss
    is_ood = torch.zeros(batch_size)  # All ID samples
    loss = ood_loss(cantor_measures, simplex_vertices, is_ood)
    print(f"  OOD Detection Loss: {loss.item():.4f}")

    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)
    print("\nKey Advantages:")
    print("  ✓ Hierarchical class structure via PE levels")
    print("  ✓ Geometric consistency within classes")
    print("  ✓ Contrastive learning with fractal guidance")
    print("  ✓ OOD detection via geometric violations")
    print("  ✓ All components differentiable and trainable")


if __name__ == "__main__":
    demo_geometric_losses()