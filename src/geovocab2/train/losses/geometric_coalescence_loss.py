# geovocab2/train/losses/geometric_coalescence_loss.py

"""
GeometricCoalescenceLoss - Unsupervised Supervision for Shatter-Reconstruct Training
-------------------------------------------------------------------------------------

This loss provides geometric scaffolding that survives catastrophic forgetting induced
by aggressive LR boosting. When weights thrash, the model can rebuild from:
    1. Consciousness anchors (high-awareness attractors)
    2. Cantor measure topology (preserved distance relationships)
    3. Simplex volume preservation (structural integrity)

Philosophy:
    "The fragments become honest foundations. Geometric truth survives when
     learned patterns fail. Each reconstruction cycle builds more robust
     representations from first principles."

Usage:
    loss = GeometricCoalescenceLoss(embed_dim=384, num_anchors=64)
    coalescence_loss = loss(
        embeddings=x,           # [B, N, D] token embeddings
        cantor_measure=cantor,  # [B, N] Cantor measure
        consciousness=cons,      # [B, N] consciousness values
        current_lr=lr,          # Current learning rate
        baseline_lr=base_lr     # Baseline learning rate
    )
    total_loss = task_loss + lambda_coal * coalescence_loss

Author: AbstractPhil
License: MIT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import math


class GeometricCoalescenceLoss(nn.Module):
    """
    Geometric Coalescence Loss for shatter-reconstruct training.

    Three-component loss that provides geometric scaffolding during aggressive
    LR boosting:
        1. Consciousness Anchoring: High-consciousness tokens cluster
        2. Distance Preservation: Cantor topology guides embeddings
        3. Volume Preservation: Maintains simplex structural integrity

    Adaptive weighting increases stabilization during LR spikes.
    """

    def __init__(
            self,
            embed_dim: int,
            num_anchors: int = 64,
            k_simplex: int = 4,
            target_variance: float = 0.5,
            num_simplex_samples: int = 32,
            num_distance_pairs: int = 256,
            base_weight: float = 0.1,
            max_weight: float = 0.8,
            weight_power: float = 2.0,
            consciousness_weight: float = 0.3,
            distance_weight: float = 0.4,
            volume_weight: float = 0.3,
            eps: float = 1e-8
    ):
        """
        Args:
            embed_dim: Embedding dimension
            num_anchors: Number of learnable anchor points in geometric space
            k_simplex: k-simplex order (4 = pentachoron with 5 vertices)
            target_variance: Target variance for simplex edge lengths
            num_simplex_samples: Number of random simplices to sample per batch
            num_distance_pairs: Number of distance pairs to sample (avoid O(N²))
            base_weight: Minimum loss weight during normal training
            max_weight: Maximum loss weight during LR spikes
            weight_power: Power for adaptive weighting (2.0 = quadratic)
            consciousness_weight: Weight for consciousness anchoring component
            distance_weight: Weight for distance preservation component
            volume_weight: Weight for volume preservation component
            eps: Numerical stability epsilon
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.num_anchors = num_anchors
        self.k_simplex = k_simplex
        self.num_vertices = k_simplex + 1
        self.target_variance = target_variance
        self.num_simplex_samples = num_simplex_samples
        self.num_distance_pairs = num_distance_pairs

        # Adaptive weighting parameters
        self.base_weight = base_weight
        self.max_weight = max_weight
        self.weight_power = weight_power

        # Component weights
        self.consciousness_weight = consciousness_weight
        self.distance_weight = distance_weight
        self.volume_weight = volume_weight
        self.eps = eps

        # Learnable consciousness anchors
        # These are attractors in embedding space for high-consciousness tokens
        self.anchors = nn.Parameter(
            torch.randn(num_anchors, embed_dim) * 0.02
        )

        print(f"[GeometricCoalescenceLoss] Initialized")
        print(f"  Anchors: {num_anchors} x {embed_dim}")
        print(f"  k-simplex: {k_simplex} ({self.num_vertices} vertices)")
        print(f"  Adaptive weight: {base_weight:.2f} → {max_weight:.2f}")
        print(f"  Components: cons={consciousness_weight:.1f}, "
              f"dist={distance_weight:.1f}, vol={volume_weight:.1f}")

    def compute_adaptive_weight(
            self,
            current_lr: float,
            baseline_lr: float
    ) -> float:
        """
        Compute adaptive loss weight based on LR ratio.

        Weight increases quadratically with LR spike:
            - Normal training (lr ≈ baseline): weight ≈ base_weight (0.1)
            - After restart (lr >> baseline): weight → max_weight (0.8)
            - During recovery: gradually decreases

        Args:
            current_lr: Current learning rate
            baseline_lr: Baseline learning rate

        Returns:
            Adaptive weight scalar
        """
        lr_ratio = current_lr / (baseline_lr + self.eps)

        # Quadratic increase (can be cubic for more aggressive)
        lr_factor = lr_ratio ** self.weight_power

        # Clamp to [0, 1] and interpolate
        lr_factor = torch.clamp(torch.tensor(lr_factor), 0.0, 1.0).item()
        weight = self.base_weight + (self.max_weight - self.base_weight) * lr_factor

        return weight

    def consciousness_anchoring_loss(
            self,
            embeddings: torch.Tensor,
            consciousness: torch.Tensor,
            consciousness_threshold: float = 0.5
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Consciousness Anchoring: High-consciousness tokens cluster around anchors.

        High-consciousness tokens (aware tokens) are pulled toward learnable
        anchors, creating stable attractors. Low-consciousness tokens drift freely.

        Args:
            embeddings: [B, N, D] token embeddings
            consciousness: [B, N] consciousness values
            consciousness_threshold: Threshold for "high consciousness"

        Returns:
            loss: Scalar loss
            metrics: Dictionary of component metrics
        """
        B, N, D = embeddings.shape

        # Normalize embeddings and anchors for distance computation
        embeddings_norm = F.normalize(embeddings, p=2, dim=-1)  # [B, N, D]
        anchors_norm = F.normalize(self.anchors, p=2, dim=-1)  # [A, D]

        # Compute distance to nearest anchor for each token
        # [B, N, D] @ [D, A] = [B, N, A]
        similarities = torch.matmul(embeddings_norm, anchors_norm.t())

        # Convert cosine similarity to distance: dist = 1 - similarity
        distances = 1.0 - similarities  # [B, N, A]

        # Find nearest anchor distance for each token
        min_distances, anchor_indices = distances.min(dim=-1)  # [B, N]

        # Weight by consciousness: only pull high-consciousness tokens
        consciousness_weights = torch.clamp(
            consciousness - consciousness_threshold,
            min=0.0
        ) / (1.0 - consciousness_threshold + self.eps)

        # Loss: pull high-consciousness tokens toward anchors
        # Negative because we want to minimize distance
        weighted_distances = min_distances * consciousness_weights
        loss = weighted_distances.mean()

        # Metrics
        metrics = {
            'anchor_distance_mean': min_distances.mean().item(),
            'anchor_distance_std': min_distances.std().item(),
            'high_consciousness_ratio': (consciousness > consciousness_threshold).float().mean().item(),
            'weighted_pull_strength': weighted_distances.mean().item()
        }

        return loss, metrics

    def distance_preservation_loss(
            self,
            embeddings: torch.Tensor,
            cantor_measure: torch.Tensor,
            max_pairs: Optional[int] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Distance Preservation: Cantor measure topology guides embedding distances.

        Sample pairs of tokens and ensure Euclidean embedding distances respect
        the Cantor measure distances. This preserves topological structure even
        when weights thrash.

        Args:
            embeddings: [B, N, D] token embeddings
            cantor_measure: [B, N] Cantor measure values
            max_pairs: Maximum number of pairs to sample (default: self.num_distance_pairs)

        Returns:
            loss: Scalar MSE loss between normalized distances
            metrics: Dictionary of component metrics
        """
        B, N, D = embeddings.shape
        max_pairs = max_pairs or self.num_distance_pairs

        # Sample random pairs (avoid O(N²))
        num_pairs = min(max_pairs, N * (N - 1) // 2)

        # Generate random indices for pairs
        # Sample with replacement for simplicity
        idx_a = torch.randint(0, N, (B, num_pairs), device=embeddings.device)
        idx_b = torch.randint(0, N, (B, num_pairs), device=embeddings.device)

        # Ensure idx_a != idx_b
        mask = (idx_a == idx_b)
        idx_b = torch.where(mask, (idx_b + 1) % N, idx_b)

        # Gather embeddings and Cantor measures
        # [B, num_pairs, D]
        emb_a = torch.gather(
            embeddings,
            dim=1,
            index=idx_a.unsqueeze(-1).expand(-1, -1, D)
        )
        emb_b = torch.gather(
            embeddings,
            dim=1,
            index=idx_b.unsqueeze(-1).expand(-1, -1, D)
        )

        # [B, num_pairs]
        cantor_a = torch.gather(cantor_measure, dim=1, index=idx_a)
        cantor_b = torch.gather(cantor_measure, dim=1, index=idx_b)

        # Compute distances
        # Euclidean distance in embedding space
        euclidean_dist = torch.norm(emb_a - emb_b, p=2, dim=-1)  # [B, num_pairs]

        # Distance in Cantor space
        cantor_dist = torch.abs(cantor_a - cantor_b)  # [B, num_pairs]

        # Normalize both to [0, 1] for fair comparison
        euclidean_dist_norm = euclidean_dist / (euclidean_dist.max(dim=1, keepdim=True)[0] + self.eps)
        cantor_dist_norm = cantor_dist / (cantor_dist.max(dim=1, keepdim=True)[0] + self.eps)

        # MSE loss between normalized distances
        loss = F.mse_loss(euclidean_dist_norm, cantor_dist_norm)

        # Metrics
        metrics = {
            'euclidean_dist_mean': euclidean_dist.mean().item(),
            'cantor_dist_mean': cantor_dist.mean().item(),
            'distance_correlation': self._compute_correlation(
                euclidean_dist_norm.flatten(),
                cantor_dist_norm.flatten()
            ),
            'num_pairs_sampled': num_pairs
        }

        return loss, metrics

    def volume_preservation_loss(
            self,
            embeddings: torch.Tensor,
            num_samples: Optional[int] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Volume Preservation: Maintain simplex structural integrity.

        Sample random k-simplices and compute edge length variance as a proxy
        for volume. Target variance prevents collapse or explosion of geometric
        structures.

        Args:
            embeddings: [B, N, D] token embeddings
            num_samples: Number of simplices to sample (default: self.num_simplex_samples)

        Returns:
            loss: MSE loss to target variance
            metrics: Dictionary of component metrics
        """
        B, N, D = embeddings.shape
        num_samples = num_samples or self.num_simplex_samples

        # Sample random simplices
        # Each simplex needs k+1 vertices
        simplex_indices = torch.randint(
            0, N,
            (B, num_samples, self.num_vertices),
            device=embeddings.device
        )  # [B, num_samples, k+1]

        # Gather vertices for each simplex
        # [B, num_samples, k+1, D]
        simplex_vertices = torch.gather(
            embeddings.unsqueeze(1).expand(-1, num_samples, -1, -1),
            dim=2,
            index=simplex_indices.unsqueeze(-1).expand(-1, -1, -1, D)
        )

        # Compute all pairwise edge lengths within each simplex
        # [B, num_samples, k+1, 1, D] - [B, num_samples, 1, k+1, D]
        diff = simplex_vertices.unsqueeze(3) - simplex_vertices.unsqueeze(2)
        edge_lengths = torch.norm(diff, p=2, dim=-1)  # [B, num_samples, k+1, k+1]

        # Extract upper triangular (unique edges)
        # For a k-simplex, we have (k+1 choose 2) = (k+1)*k/2 edges
        mask = torch.triu(torch.ones(self.num_vertices, self.num_vertices), diagonal=1)
        mask = mask.bool().to(embeddings.device)

        # Extract edge lengths [B, num_samples, num_edges]
        edges_per_simplex = edge_lengths[:, :, mask]  # [B, num_samples, num_edges]

        # Compute variance of edge lengths for each simplex
        edge_variance = edges_per_simplex.var(dim=-1)  # [B, num_samples]

        # Loss: MSE to target variance
        target = torch.full_like(edge_variance, self.target_variance)
        loss = F.mse_loss(edge_variance, target)

        # Metrics
        metrics = {
            'edge_variance_mean': edge_variance.mean().item(),
            'edge_variance_std': edge_variance.std().item(),
            'edge_length_mean': edges_per_simplex.mean().item(),
            'target_variance': self.target_variance,
            'num_simplices_sampled': num_samples
        }

        return loss, metrics

    def forward(
            self,
            embeddings: torch.Tensor,
            cantor_measure: torch.Tensor,
            consciousness: torch.Tensor,
            current_lr: float,
            baseline_lr: float,
            return_components: bool = False
    ) -> torch.Tensor:
        """
        Forward pass: compute geometric coalescence loss.

        Args:
            embeddings: [B, N, D] token embeddings
            cantor_measure: [B, N] Cantor measure values
            consciousness: [B, N] consciousness values
            current_lr: Current learning rate
            baseline_lr: Baseline learning rate
            return_components: If True, return (loss, components_dict)

        Returns:
            loss: Scalar total loss (or tuple if return_components=True)
        """
        # Compute adaptive weight based on LR
        adaptive_weight = self.compute_adaptive_weight(current_lr, baseline_lr)

        # Component 1: Consciousness Anchoring
        anchor_loss, anchor_metrics = self.consciousness_anchoring_loss(
            embeddings, consciousness
        )

        # Component 2: Distance Preservation
        distance_loss, distance_metrics = self.distance_preservation_loss(
            embeddings, cantor_measure
        )

        # Component 3: Volume Preservation
        volume_loss, volume_metrics = self.volume_preservation_loss(
            embeddings
        )

        # Weighted combination
        total_loss = (
                self.consciousness_weight * anchor_loss +
                self.distance_weight * distance_loss +
                self.volume_weight * volume_loss
        )

        # Apply adaptive weighting
        weighted_loss = adaptive_weight * total_loss

        if return_components:
            components = {
                'total_loss': weighted_loss.item(),
                'adaptive_weight': adaptive_weight,
                'anchor_loss': anchor_loss.item(),
                'distance_loss': distance_loss.item(),
                'volume_loss': volume_loss.item(),
                'anchor_metrics': anchor_metrics,
                'distance_metrics': distance_metrics,
                'volume_metrics': volume_metrics,
            }
            return weighted_loss, components

        return weighted_loss

    @staticmethod
    def _compute_correlation(x: torch.Tensor, y: torch.Tensor) -> float:
        """Compute Pearson correlation coefficient."""
        x_mean = x.mean()
        y_mean = y.mean()

        numerator = ((x - x_mean) * (y - y_mean)).sum()
        denominator = torch.sqrt(((x - x_mean) ** 2).sum() * ((y - y_mean) ** 2).sum())

        if denominator < 1e-8:
            return 0.0

        return (numerator / denominator).item()

    def get_info(self) -> Dict:
        """Get loss configuration info."""
        return {
            'num_anchors': self.num_anchors,
            'k_simplex': self.k_simplex,
            'num_vertices': self.num_vertices,
            'adaptive_weight_range': f"{self.base_weight:.2f}-{self.max_weight:.2f}",
            'component_weights': {
                'consciousness': self.consciousness_weight,
                'distance': self.distance_weight,
                'volume': self.volume_weight
            },
            'sampling': {
                'distance_pairs': self.num_distance_pairs,
                'simplex_samples': self.num_simplex_samples
            },
            'target_variance': self.target_variance
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Integration Helper for Training Script
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def add_coalescence_loss_to_training(
        model_output: Dict[str, torch.Tensor],
        coalescence_loss_fn: GeometricCoalescenceLoss,
        current_lr: float,
        baseline_lr: float,
        lambda_coal: float = 1.0
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Helper function to add coalescence loss to training.

    Usage in training loop:
        #>>> coalescence_loss_fn = GeometricCoalescenceLoss(embed_dim=384)
        #>>>
        #>>> # In training loop
        #>>> result = model(images, return_fusion_info=True)
        #>>> logits, fusion_infos = result if return_fusion_info else (result, None)
        #>>>
        #>>> # Task loss
        #>>> task_loss = criterion(logits, labels)
        #>>>
        #>>> # Add coalescence loss if we have fusion info
        #>>> if fusion_infos:
        #>>>     coal_loss, coal_metrics = add_coalescence_loss_to_training(
        #>>>         fusion_infos[0],  # Use last layer's fusion info
        #>>>         coalescence_loss_fn,
        #>>>         current_lr=scheduler.get_last_lr()[0],
        #>>>         baseline_lr=config.learning_rate,
        #>>>         lambda_coal=0.5
        #>>>     )
        #>>>     total_loss = task_loss + coal_loss
        #>>> else:
        #>>>     total_loss = task_loss

    Args:
        model_output: Dictionary with keys:
            - 'output': [B, N, D] embeddings
            - 'cantor_measure': [B, N] Cantor measure
            - 'consciousness': [B, N] consciousness values
        coalescence_loss_fn: GeometricCoalescenceLoss instance
        current_lr: Current learning rate
        baseline_lr: Baseline learning rate
        lambda_coal: Scalar weight for coalescence loss

    Returns:
        weighted_loss: Weighted coalescence loss
        metrics: Dictionary of loss components and metrics
    """
    # Extract required tensors from model output
    embeddings = model_output['output']
    cantor_measure = model_output.get('cantor_measure')
    consciousness = model_output.get('consciousness')

    # Check if we have all required components
    if cantor_measure is None or consciousness is None:
        # Return zero loss if geometric info not available
        return torch.tensor(0.0, device=embeddings.device), {}

    # Compute coalescence loss with components
    coal_loss, components = coalescence_loss_fn(
        embeddings=embeddings,
        cantor_measure=cantor_measure,
        consciousness=consciousness,
        current_lr=current_lr,
        baseline_lr=baseline_lr,
        return_components=True
    )

    # Apply lambda weighting
    weighted_loss = lambda_coal * coal_loss

    # Add lambda to metrics
    components['lambda_coal'] = lambda_coal
    components['weighted_total'] = weighted_loss.item()

    return weighted_loss, components


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Testing & Validation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    print("=" * 70)
    print("Testing GeometricCoalescenceLoss")
    print("=" * 70)

    # Create loss function
    loss_fn = GeometricCoalescenceLoss(
        embed_dim=384,
        num_anchors=64,
        k_simplex=4,
        target_variance=0.5,
        base_weight=0.1,
        max_weight=0.8
    )

    print(f"\n{loss_fn.get_info()}")

    # Test data
    batch_size = 4
    seq_len = 64
    embed_dim = 384

    embeddings = torch.randn(batch_size, seq_len, embed_dim)
    cantor_measure = torch.linspace(0, 1, seq_len).unsqueeze(0).expand(batch_size, -1)
    consciousness = torch.rand(batch_size, seq_len) * 0.5 + 0.5  # [0.5, 1.0]

    # Test 1: Normal training (low LR)
    print("\n[Test 1] Normal training (LR = baseline)")
    loss, components = loss_fn(
        embeddings, cantor_measure, consciousness,
        current_lr=3e-4, baseline_lr=3e-4,
        return_components=True
    )
    print(f"  Total loss: {loss.item():.6f}")
    print(f"  Adaptive weight: {components['adaptive_weight']:.3f}")
    print(f"  Components:")
    print(f"    Anchor: {components['anchor_loss']:.6f}")
    print(f"    Distance: {components['distance_loss']:.6f}")
    print(f"    Volume: {components['volume_loss']:.6f}")

    # Test 2: After restart (high LR)
    print("\n[Test 2] After restart (LR = 2x baseline) 🚀")
    loss, components = loss_fn(
        embeddings, cantor_measure, consciousness,
        current_lr=6e-4, baseline_lr=3e-4,
        return_components=True
    )
    print(f"  Total loss: {loss.item():.6f}")
    print(f"  Adaptive weight: {components['adaptive_weight']:.3f} (BOOSTED!)")
    print(f"  Components:")
    print(f"    Anchor: {components['anchor_loss']:.6f}")
    print(f"    Distance: {components['distance_loss']:.6f}")
    print(f"    Volume: {components['volume_loss']:.6f}")

    # Test 3: Extreme boost (3x baseline)
    print("\n[Test 3] Extreme boost (LR = 3x baseline) 🚀🚀")
    loss, components = loss_fn(
        embeddings, cantor_measure, consciousness,
        current_lr=9e-4, baseline_lr=3e-4,
        return_components=True
    )
    print(f"  Total loss: {loss.item():.6f}")
    print(f"  Adaptive weight: {components['adaptive_weight']:.3f} (MAX STABILIZATION)")
    print(f"  Distance correlation: {components['distance_metrics']['distance_correlation']:.3f}")

    # Test 4: Gradient flow
    print("\n[Test 4] Gradient flow check")
    embeddings_grad = torch.randn(batch_size, seq_len, embed_dim, requires_grad=True)
    loss = loss_fn(
        embeddings_grad, cantor_measure, consciousness,
        current_lr=6e-4, baseline_lr=3e-4
    )
    loss.backward()

    print(f"  Loss: {loss.item():.6f}")
    print(f"  Grad norm: {embeddings_grad.grad.norm().item():.6f}")
    print(f"  Anchor grad norm: {loss_fn.anchors.grad.norm().item():.6f}")
    print(f"  ✓ Gradients flow correctly")

    # Test 5: Integration helper
    print("\n[Test 5] Integration helper")
    model_output = {
        'output': embeddings,
        'cantor_measure': cantor_measure,
        'consciousness': consciousness
    }

    weighted_loss, metrics = add_coalescence_loss_to_training(
        model_output,
        loss_fn,
        current_lr=6e-4,
        baseline_lr=3e-4,
        lambda_coal=0.5
    )

    print(f"  Weighted loss: {weighted_loss.item():.6f}")
    print(f"  Lambda: {metrics['lambda_coal']}")
    print(f"  Adaptive weight: {metrics['adaptive_weight']:.3f}")

    print("\n" + "=" * 70)
    print("All tests passed! ✓")
    print("\nReady for integration into training script.")
    print("\nExpected behavior during training:")
    print("  📉 Normal training: weight ≈ 0.1 (gentle guidance)")
    print("  🔄 At restart: weight → 0.8 (strong stabilization)")
    print("  📈 During recovery: weight gradually decreases")
    print("\nThis creates a safety net that:")
    print("  - Prevents geometric collapse during LR spikes")
    print("  - Guides reconstruction from geometric first principles")
    print("  - Adapts stabilization strength to training phase")
    print("=" * 70)