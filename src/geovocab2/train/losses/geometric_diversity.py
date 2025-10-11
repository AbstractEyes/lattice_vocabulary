"""
Batched Geometric Diversity Loss
---------------------------------
Fully vectorized loss to encourage 3D geometric distinctiveness.
Maximizes inter-class separation, minimizes intra-class variance.

Key insight: Geometry should preserve 3D salient forms (faces, animals, vehicles)
and ignore flat/generic objects (furniture, plates).

These have proven fruitless.

Author: AbstractPhil + Claude Sonnet 4.5
License: MIT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class BatchedGeometricDiversityLoss(nn.Module):
    """
    Fully batched geometric diversity loss.
    Encourages geometric stream to learn discriminative 3D structures.

    All operations are vectorized for GPU efficiency.
    """

    def __init__(
            self,
            margin: float = 2.0,
            temperature: float = 0.1,
            intra_weight: float = 1.0,
            inter_weight: float = 2.0,
            volume_weight: float = 0.5,
            complexity_weight: float = 0.1
    ):
        super().__init__()
        self.margin = margin
        self.temperature = temperature
        self.intra_weight = intra_weight
        self.inter_weight = inter_weight
        self.volume_weight = volume_weight
        self.complexity_weight = complexity_weight

    def forward(
            self,
            geometric_stream: torch.Tensor,  # [B, num_tokens, geom_dim]
            simplex_volumes: torch.Tensor,  # [B,]
            labels: torch.Tensor,  # [B,]
            token_diversity: torch.Tensor = None  # [B,] optional
    ) -> Dict[str, torch.Tensor]:
        """
        Compute batched geometric diversity loss.

        Returns dict with:
            - total: combined loss
            - intra_class_loss: compactness within classes
            - inter_class_loss: separation between classes
            - volume_diversity: encourages varied simplex volumes
            - complexity_reward: rewards high volumes (3D salience)
        """
        B = geometric_stream.shape[0]
        device = geometric_stream.device

        # Pool geometric features: [B, num_tokens, geom_dim] -> [B, geom_dim]
        geom_pooled = geometric_stream.mean(dim=1)

        # Normalize for stable distances
        geom_norm = F.normalize(geom_pooled, p=2, dim=1)

        # ====================================================================
        # 1. PAIRWISE DISTANCES (fully vectorized)
        # ====================================================================
        # Compute all pairwise distances: [B, B]
        dists = torch.cdist(geom_norm, geom_norm, p=2)

        # Create label comparison matrix: [B, B]
        # same_class[i,j] = 1 if labels[i] == labels[j], else 0
        same_class = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        diff_class = 1.0 - same_class

        # Mask out diagonal (self-comparisons)
        mask = 1.0 - torch.eye(B, device=device)
        same_class = same_class * mask
        diff_class = diff_class * mask

        # ====================================================================
        # 2. INTRA-CLASS COMPACTNESS (pull same class together)
        # ====================================================================
        # Average distance to same-class neighbors
        same_class_dists = dists * same_class

        # Only compute if there are same-class pairs
        num_same_pairs = same_class.sum()
        if num_same_pairs > 0:
            intra_loss = same_class_dists.sum() / num_same_pairs
        else:
            intra_loss = torch.tensor(0.0, device=device)

        # ====================================================================
        # 3. INTER-CLASS SEPARATION (push different classes apart)
        # ====================================================================
        # Use contrastive margin loss: penalize if distance < margin
        diff_class_dists = dists * diff_class

        # Apply margin: max(0, margin - distance)
        margin_violations = F.relu(self.margin - diff_class_dists) * diff_class

        # Only compute if there are different-class pairs
        num_diff_pairs = diff_class.sum()
        if num_diff_pairs > 0:
            inter_loss = margin_violations.sum() / num_diff_pairs
        else:
            inter_loss = torch.tensor(0.0, device=device)

        # ====================================================================
        # 4. VOLUME DIVERSITY (different classes should have different volumes)
        # ====================================================================
        # Encourage high variance in simplex volumes
        # Higher std = more diverse volumes = better discrimination
        volume_diversity_loss = -simplex_volumes.std()  # Negative = penalize low variance

        # ====================================================================
        # 5. COMPLEXITY REWARD (reward 3D salient objects)
        # ====================================================================
        # Higher volumes = more 3D complex = should be preserved
        # This biases geometry toward faces, animals, vehicles vs flat objects
        complexity_reward = -simplex_volumes.mean()  # Negative = maximize volumes

        # ====================================================================
        # 6. COMBINE LOSSES
        # ====================================================================
        total_loss = (
                self.intra_weight * intra_loss +  # Compact clusters
                self.inter_weight * inter_loss +  # Separated classes
                self.volume_weight * volume_diversity_loss +  # Diverse volumes
                self.complexity_weight * complexity_reward  # Reward complexity
        )

        return {
            'total': total_loss,
            'intra_class_loss': intra_loss,
            'inter_class_loss': inter_loss,
            'volume_diversity_loss': volume_diversity_loss,
            'complexity_reward': complexity_reward,
            'mean_intra_dist': same_class_dists.sum() / num_same_pairs if num_same_pairs > 0 else torch.tensor(0.0),
            'mean_inter_dist': diff_class_dists.sum() / num_diff_pairs if num_diff_pairs > 0 else torch.tensor(0.0)
        }


class PrototypicalGeometricLoss(nn.Module):
    """
    Alternative: Prototypical Networks style loss.
    Maintains running class prototypes (EMA) and pulls samples toward their prototype.

    More stable for streaming training with partial class coverage per batch.
    """

    def __init__(
            self,
            num_classes: int,
            geom_dim: int,
            momentum: float = 0.9,
            temperature: float = 0.1,
            margin: float = 2.0
    ):
        super().__init__()
        self.num_classes = num_classes
        self.geom_dim = geom_dim
        self.momentum = momentum
        self.temperature = temperature
        self.margin = margin

        # Running class prototypes: [num_classes, geom_dim]
        self.register_buffer('prototypes', torch.randn(num_classes, geom_dim))
        self.register_buffer('prototype_counts', torch.zeros(num_classes))

        # Normalize prototypes
        self.prototypes = F.normalize(self.prototypes, p=2, dim=1)

    def forward(
            self,
            geometric_stream: torch.Tensor,  # [B, num_tokens, geom_dim]
            simplex_volumes: torch.Tensor,  # [B,]
            labels: torch.Tensor  # [B,]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute prototypical geometric loss with EMA prototypes.
        """
        B = geometric_stream.shape[0]
        device = geometric_stream.device

        # Pool and normalize geometric features
        geom_pooled = geometric_stream.mean(dim=1)
        geom_norm = F.normalize(geom_pooled, p=2, dim=1)

        # ====================================================================
        # 1. UPDATE PROTOTYPES (EMA)
        # ====================================================================
        if self.training:
            with torch.no_grad():
                for c in labels.unique():
                    mask = labels == c
                    class_features = geom_norm[mask].mean(dim=0)

                    # EMA update
                    self.prototypes[c] = (
                            self.momentum * self.prototypes[c] +
                            (1 - self.momentum) * class_features
                    )
                    self.prototypes[c] = F.normalize(self.prototypes[c], p=2, dim=0)
                    self.prototype_counts[c] += mask.sum()

        # ====================================================================
        # 2. COMPUTE DISTANCES TO PROTOTYPES
        # ====================================================================
        # Distance to all prototypes: [B, num_classes]
        proto_dists = torch.cdist(geom_norm, self.prototypes, p=2)

        # Distance to own-class prototype
        own_proto_dists = proto_dists[torch.arange(B), labels]

        # ====================================================================
        # 3. INTRA-CLASS LOSS (pull toward own prototype)
        # ====================================================================
        intra_loss = own_proto_dists.mean()

        # ====================================================================
        # 4. INTER-CLASS LOSS (push away from other prototypes)
        # ====================================================================
        # Create mask for other-class prototypes
        other_class_mask = torch.ones_like(proto_dists)
        other_class_mask[torch.arange(B), labels] = 0

        # Margin loss for other classes
        margin_violations = F.relu(self.margin - proto_dists) * other_class_mask
        inter_loss = margin_violations.sum() / (B * (self.num_classes - 1))

        # ====================================================================
        # 5. VOLUME DIVERSITY
        # ====================================================================
        volume_diversity_loss = -simplex_volumes.std()
        complexity_reward = -simplex_volumes.mean()

        total_loss = (
                1.0 * intra_loss +
                2.0 * inter_loss +
                0.5 * volume_diversity_loss +
                0.1 * complexity_reward
        )

        return {
            'total': total_loss,
            'intra_class_loss': intra_loss,
            'inter_class_loss': inter_loss,
            'volume_diversity_loss': volume_diversity_loss,
            'complexity_reward': complexity_reward,
            'mean_proto_dist': own_proto_dists.mean()
        }


# ============================================================
# INTEGRATION EXAMPLE
# ============================================================

class DualStreamWithGeometricDiversity(nn.Module):
    """
    Example integration into dual-stream training.
    """

    def __init__(self, base_model, loss_type='batched', **loss_kwargs):
        super().__init__()
        self.model = base_model

        if loss_type == 'batched':
            self.geometric_diversity_loss = BatchedGeometricDiversityLoss(**loss_kwargs)
        elif loss_type == 'prototypical':
            self.geometric_diversity_loss = PrototypicalGeometricLoss(**loss_kwargs)
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")

    def forward(self, images, labels):
        """
        Forward pass with geometric diversity loss.
        """
        # Get model outputs
        output = self.model(images, return_features=True)

        # Compute task loss (classification)
        task_loss = F.cross_entropy(output['logits'], labels)

        # Compute geometric diversity loss
        # Note: Need to compute simplex volumes from geometric stream
        geom_stream = output['geometric_stream']

        # Simple volume proxy: norm of geometric tokens
        simplex_volumes = torch.norm(geom_stream, dim=2).mean(dim=1)

        diversity_losses = self.geometric_diversity_loss(
            geom_stream,
            simplex_volumes,
            labels
        )

        # Combined loss
        total_loss = task_loss + 0.5 * diversity_losses['total']

        return {
            'loss': total_loss,
            'task_loss': task_loss,
            'diversity_loss': diversity_losses['total'],
            'logits': output['logits'],
            **diversity_losses
        }


# ============================================================
# USAGE IN TRAINER
# ============================================================

"""
In your trainer __init__:

self.geometric_diversity_loss = BatchedGeometricDiversityLoss(
    margin=2.0,              # Minimum desired inter-class distance
    temperature=0.1,
    intra_weight=1.0,        # Weight for intra-class compactness
    inter_weight=2.0,        # Weight for inter-class separation
    volume_weight=0.5,       # Weight for volume diversity
    complexity_weight=0.1    # Weight for 3D complexity reward
)

In your train_epoch:

# After getting model output
diversity_losses = self.geometric_diversity_loss(
    output['geometric_stream'],
    geometric_losses['simplex_volumes'],  # Or compute from stream
    labels
)

total_loss = (
    self.config.task_loss_weight * task_loss +
    self.config.flow_loss_weight * geometric_losses['flow_loss'] +
    self.config.coherence_loss_weight * geometric_losses['coherence_loss'] +
    self.config.multiscale_loss_weight * geometric_losses['multiscale_loss'] +
    0.5 * diversity_losses['total']  # NEW: Geometric diversity
)

# Log the breakdown
if self.writer:
    self.writer.add_scalar('diversity/intra_class', diversity_losses['intra_class_loss'], step)
    self.writer.add_scalar('diversity/inter_class', diversity_losses['inter_class_loss'], step)
    self.writer.add_scalar('diversity/mean_intra_dist', diversity_losses['mean_intra_dist'], step)
    self.writer.add_scalar('diversity/mean_inter_dist', diversity_losses['mean_inter_dist'], step)
"""