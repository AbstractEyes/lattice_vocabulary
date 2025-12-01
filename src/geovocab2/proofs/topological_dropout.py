"""
Topological Dropout - Structure-Preserving Regularization

Key insight: Standard dropout creates random holes everywhere, breaking feature structure.
Topological dropout drops entire routes/channels/patches, keeping surviving features intact.

Experimental Results (Fashion-MNIST):
- ScheduledTopologicalDropout: 93.01% (1.66% gap) - Best balance
- SpatialTopologicalDropout: 92.78% (0.62% gap) - Best generalization
- Standard dropout: 92.82% (1.23% gap) - Baseline
- Importance-weighted: 90.48% (5.17% gap) - HARMFUL, not included

Usage:
    from geofractal.model.layers.dropout.topological import (
        TopologicalDropout,
        ScheduledTopologicalDropout,
        SpatialTopologicalDropout,
        WormholeDropout,
    )
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class TopologicalDropout(nn.Module):
    """
    Structure-preserving dropout: drops entire routes/channels, not individual neurons.

    For CNNs: treats channels as routes (drops entire feature maps)
    For Attention: treats heads/routes as units
    For MLPs: treats feature groups as routes

    Args:
        drop_prob: Probability of dropping each route (default: 0.1)
        min_keep: Minimum routes to keep (default: 1)
        route_dim: Which dimension contains routes (default: 1 for [B, C, H, W])
        scale: Whether to scale surviving routes to preserve expected value (default: True)

    Example:
        #>>> dropout = TopologicalDropout(drop_prob=0.2, route_dim=1)
        #>>> x = torch.randn(32, 64, 14, 14)  # [B, C, H, W]
        #>>> out = dropout(x)  # Drops ~20% of channels entirely
    """

    def __init__(
            self,
            drop_prob: float = 0.1,
            min_keep: int = 1,
            route_dim: int = 1,
            scale: bool = True,
    ):
        super().__init__()
        self.drop_prob = drop_prob
        self.min_keep = min_keep
        self.route_dim = route_dim
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0:
            return x

        num_routes = x.shape[self.route_dim]
        num_keep = max(self.min_keep, int(num_routes * (1 - self.drop_prob)))

        # Random selection of routes to keep
        mask = torch.zeros(num_routes, device=x.device, dtype=x.dtype)
        perm = torch.randperm(num_routes, device=x.device)[:num_keep]
        mask[perm] = 1.0

        # Scale to preserve expected value
        if self.scale:
            mask = mask * (num_routes / num_keep)

        # Reshape for broadcast
        shape = [1] * x.dim()
        shape[self.route_dim] = num_routes

        return x * mask.view(shape)

    def extra_repr(self) -> str:
        return f"drop_prob={self.drop_prob}, min_keep={self.min_keep}, route_dim={self.route_dim}"


class ScheduledTopologicalDropout(nn.Module):
    """
    Topological dropout with warmup schedule.

    Starts with no dropout, linearly increases to target over warmup period.
    Designed to let model learn structure before applying regularization.

    For DavidBeans: Aligns with router crystallization phase (~E30-35).

    Args:
        drop_prob: Target dropout probability after warmup (default: 0.2)
        min_keep: Minimum routes to keep (default: 1)
        route_dim: Which dimension contains routes (default: 1)
        warmup_steps: Steps to reach full dropout (default: 1000)
        scale: Whether to scale surviving routes (default: True)

    Example:
        >>> dropout = ScheduledTopologicalDropout(drop_prob=0.2, warmup_steps=500)
        >>> # Early training: mild dropout
        >>> # After 500 steps: full 0.2 dropout
    """

    def __init__(
            self,
            drop_prob: float = 0.2,
            min_keep: int = 1,
            route_dim: int = 1,
            warmup_steps: int = 1000,
            scale: bool = True,
    ):
        super().__init__()
        self.target_drop_prob = drop_prob
        self.min_keep = min_keep
        self.route_dim = route_dim
        self.warmup_steps = warmup_steps
        self.scale = scale

        # Track training progress
        self.register_buffer('step', torch.tensor(0, dtype=torch.long))

    @property
    def current_drop_prob(self) -> float:
        """Current dropout probability based on training progress."""
        if self.warmup_steps <= 0:
            return self.target_drop_prob
        progress = min(1.0, self.step.item() / self.warmup_steps)
        return self.target_drop_prob * progress

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            self.step += 1

        drop_prob = self.current_drop_prob
        if not self.training or drop_prob == 0:
            return x

        num_routes = x.shape[self.route_dim]
        num_keep = max(self.min_keep, int(num_routes * (1 - drop_prob)))

        mask = torch.zeros(num_routes, device=x.device, dtype=x.dtype)
        perm = torch.randperm(num_routes, device=x.device)[:num_keep]
        mask[perm] = 1.0

        if self.scale:
            mask = mask * (num_routes / num_keep)

        shape = [1] * x.dim()
        shape[self.route_dim] = num_routes

        return x * mask.view(shape)

    def reset_schedule(self):
        """Reset warmup schedule (call at start of training)."""
        self.step.zero_()

    def set_step(self, step: int):
        """Manually set training step."""
        self.step.fill_(step)

    def extra_repr(self) -> str:
        return (f"drop_prob={self.target_drop_prob}, warmup_steps={self.warmup_steps}, "
                f"current={self.current_drop_prob:.3f}, step={self.step.item()}")


class SpatialTopologicalDropout(nn.Module):
    """
    Drops entire spatial patches instead of individual pixels.

    Complements channel-wise dropout by enforcing spatial redundancy.
    Showed best generalization (0.62% train-test gap) in experiments.

    Args:
        drop_prob: Probability of dropping each patch (default: 0.1)
        patch_size: Size of patches to drop (default: 2)
        scale: Whether to scale surviving patches (default: True)

    Example:
        >>> dropout = SpatialTopologicalDropout(drop_prob=0.2, patch_size=4)
        >>> x = torch.randn(32, 64, 28, 28)  # [B, C, H, W]
        >>> out = dropout(x)  # Drops ~20% of 4x4 patches
    """

    def __init__(
            self,
            drop_prob: float = 0.1,
            patch_size: int = 2,
            scale: bool = True,
    ):
        super().__init__()
        self.drop_prob = drop_prob
        self.patch_size = patch_size
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0:
            return x

        if x.dim() != 4:
            return x  # Only for [B, C, H, W] tensors

        B, C, H, W = x.shape
        pH = H // self.patch_size
        pW = W // self.patch_size

        if pH == 0 or pW == 0:
            return x  # Image too small for patches

        # Create patch mask [B, 1, pH, pW]
        mask = (torch.rand(B, 1, pH, pW, device=x.device, dtype=x.dtype) > self.drop_prob).float()

        # Scale surviving patches
        if self.scale:
            keep_ratio = mask.mean()
            if keep_ratio > 0:
                mask = mask / keep_ratio

        # Upsample mask to full resolution
        mask = F.interpolate(mask, size=(H, W), mode='nearest')

        return x * mask

    def extra_repr(self) -> str:
        return f"drop_prob={self.drop_prob}, patch_size={self.patch_size}"


class WormholeDropout(nn.Module):
    """
    Combined dropout strategy optimized for wormhole attention.

    Combines:
    1. Scheduled route dropout (warmup aligned to crystallization)
    2. Optional spatial dropout for patch embeddings

    Designed for DavidBeans architecture where routers crystallize ~E30-35.

    Args:
        route_drop_prob: Target route dropout after warmup (default: 0.15)
        spatial_drop_prob: Spatial patch dropout, 0 to disable (default: 0.0)
        num_routes: Number of wormhole routes (default: 8)
        min_routes_keep: Minimum routes to keep (default: 2)
        warmup_epochs: Epochs before full dropout (default: 35)
        steps_per_epoch: Steps per epoch for schedule (default: 469 for CIFAR-100)
        patch_size: Spatial dropout patch size (default: 2)

    Example:
        #>>> dropout = WormholeDropout(
        #...     route_drop_prob=0.15,
        #...     warmup_epochs=35,
        #...     steps_per_epoch=469,
        #... )
        #>>> # In training loop:
        #>>> dropout.set_epoch(current_epoch)
        #>>> x = dropout(x, route_dim=-2)
    """

    def __init__(
            self,
            route_drop_prob: float = 0.15,
            spatial_drop_prob: float = 0.0,
            num_routes: int = 8,
            min_routes_keep: int = 2,
            warmup_epochs: int = 35,
            steps_per_epoch: int = 469,
            patch_size: int = 2,
    ):
        super().__init__()
        self.num_routes = num_routes
        self.warmup_epochs = warmup_epochs

        warmup_steps = warmup_epochs * steps_per_epoch

        self.route_dropout = ScheduledTopologicalDropout(
            drop_prob=route_drop_prob,
            min_keep=min_routes_keep,
            route_dim=-2,  # Default for [B, S, R, D] tensors
            warmup_steps=warmup_steps,
        )

        self.spatial_dropout = None
        if spatial_drop_prob > 0:
            self.spatial_dropout = SpatialTopologicalDropout(
                drop_prob=spatial_drop_prob,
                patch_size=patch_size,
            )

    def forward(
            self,
            x: torch.Tensor,
            route_dim: int = -2,
            apply_spatial: bool = False,
    ) -> torch.Tensor:
        """
        Apply dropout to tensor.

        Args:
            x: Input tensor
            route_dim: Dimension containing routes (default: -2)
            apply_spatial: Whether to apply spatial dropout (for 4D tensors)

        Returns:
            Dropped-out tensor
        """
        # Apply spatial dropout if requested and available
        if apply_spatial and self.spatial_dropout is not None and x.dim() == 4:
            x = self.spatial_dropout(x)

        # Apply route dropout
        # Temporarily override route_dim if different from default
        original_route_dim = self.route_dropout.route_dim
        self.route_dropout.route_dim = route_dim
        x = self.route_dropout(x)
        self.route_dropout.route_dim = original_route_dim

        return x

    def set_epoch(self, epoch: int, steps_per_epoch: Optional[int] = None):
        """
        Set current epoch for schedule calculation.

        Args:
            epoch: Current epoch (0-indexed)
            steps_per_epoch: Override steps per epoch if needed
        """
        if steps_per_epoch is not None:
            step = epoch * steps_per_epoch
        else:
            # Estimate from warmup config
            step = epoch * (self.route_dropout.warmup_steps // max(1, self.warmup_epochs))
        self.route_dropout.set_step(step)

    def reset(self):
        """Reset dropout schedules."""
        self.route_dropout.reset_schedule()

    @property
    def current_drop_prob(self) -> float:
        """Current route dropout probability."""
        return self.route_dropout.current_drop_prob

    def extra_repr(self) -> str:
        spatial = f", spatial={self.spatial_dropout.drop_prob}" if self.spatial_dropout else ""
        return (f"routes={self.num_routes}, warmup_epochs={self.warmup_epochs}, "
                f"current_drop={self.current_drop_prob:.3f}{spatial}")


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def apply_topological_dropout(
        x: torch.Tensor,
        drop_prob: float,
        route_dim: int,
        training: bool,
        min_keep: int = 1,
        scale: bool = True,
) -> torch.Tensor:
    """
    Functional interface for topological dropout.

    Args:
        x: Input tensor
        drop_prob: Dropout probability
        route_dim: Route dimension
        training: Whether in training mode
        min_keep: Minimum routes to keep
        scale: Whether to scale output

    Returns:
        Dropped-out tensor
    """
    if not training or drop_prob == 0:
        return x

    num_routes = x.shape[route_dim]
    num_keep = max(min_keep, int(num_routes * (1 - drop_prob)))

    mask = torch.zeros(num_routes, device=x.device, dtype=x.dtype)
    perm = torch.randperm(num_routes, device=x.device)[:num_keep]
    mask[perm] = 1.0

    if scale:
        mask = mask * (num_routes / num_keep)

    shape = [1] * x.dim()
    shape[route_dim] = num_routes

    return x * mask.view(shape)