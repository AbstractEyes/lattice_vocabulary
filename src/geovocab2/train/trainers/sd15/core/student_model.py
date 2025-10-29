"""
StudentModel: Wrapper for trainable student models.
Handles block tracking, auxiliary heads, and feature extraction.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Callable
from abc import ABC, abstractmethod


class StudentModel(nn.Module, ABC):
    """
    Base class for student models.

    Students are trainable models that:
    - Learn from teacher targets
    - Maintain block/layer structure for feature matching
    - Support auxiliary prediction heads
    - Track intermediate features
    """

    def __init__(self, block_names: Optional[List[str]] = None):
        super().__init__()
        self.block_names = block_names or []
        self._hooks: List[Any] = []
        self._feature_bank: Dict[str, torch.Tensor] = {}

        # Auxiliary heads (e.g., local flow heads)
        self.aux_heads: nn.ModuleDict = nn.ModuleDict()

    def setup_feature_hooks(self, block_names: Optional[List[str]] = None) -> StudentModel:
        """
        Setup hooks to capture intermediate features.
        Override to customize hook placement.

        Args:
            block_names: Names of blocks to capture (uses self.block_names if None)

        Returns:
            self for chaining
        """
        return self

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def clear_features(self):
        """Clear captured features."""
        self._feature_bank.clear()

    def get_features(self) -> Dict[str, torch.Tensor]:
        """Get captured features."""
        return self._feature_bank.copy()

    @abstractmethod
    def forward_with_features(self, *args, **kwargs) -> Tuple[Any, Dict[str, torch.Tensor]]:
        """
        Forward pass that returns both outputs and intermediate features.

        Returns:
            (outputs, features_dict)
        """
        pass

    def add_aux_head(self, name: str, head: nn.Module):
        """
        Add auxiliary prediction head.

        Args:
            name: Name for the head (e.g., "local_flow_down_0")
            head: The head module
        """
        self.aux_heads[name] = head

    def get_aux_predictions(
            self,
            features: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Get predictions from auxiliary heads.

        Args:
            features: Feature dictionary

        Returns:
            Dictionary of auxiliary predictions
        """
        predictions = {}
        for name, head in self.aux_heads.items():
            # Extract block name from head name (e.g., "local_flow_down_0" -> "down_0")
            block_name = name.split('_', 2)[-1] if '_' in name else name
            if block_name in features:
                predictions[name] = head(features[block_name])
        return predictions

    def compute_independent_loss(
            self,
            outputs: Any,
            targets: Any,
            **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute student's independent loss (e.g., direct prediction loss).
        Override for task-specific losses.

        Returns:
            (loss, loss_breakdown_dict)
        """
        return torch.zeros((), device=next(self.parameters()).device), {}


class SD15Student(StudentModel):
    """
    Student UNet for SD1.5 distillation.
    Supports local flow heads for per-block velocity prediction.
    """

    def __init__(
            self,
            unet: nn.Module,
            block_names: Tuple[str, ...] = (
                    "down_0", "down_1", "down_2", "down_3",
                    "mid",
                    "up_0", "up_1", "up_2", "up_3"
            ),
            use_local_flow_heads: bool = False
    ):
        super().__init__(list(block_names))
        self.unet = unet
        self.use_local_flow_heads = use_local_flow_heads

        self.setup_feature_hooks(list(block_names))

    def setup_feature_hooks(self, block_names: Optional[List[str]] = None) -> SD15Student:
        """Setup hooks on UNet blocks."""
        block_names = block_names or self.block_names

        def make_hook(name: str):
            def hook(module, input, output):
                out = output[0] if isinstance(output, (tuple, list)) else output
                self._feature_bank[name] = out

            return hook

        # Hook down blocks
        for i, block in enumerate(self.unet.down_blocks):
            name = f"down_{i}"
            if name in block_names:
                handle = block.register_forward_hook(make_hook(name))
                self._hooks.append(handle)

        # Hook mid block
        if "mid" in block_names:
            handle = self.unet.mid_block.register_forward_hook(make_hook("mid"))
            self._hooks.append(handle)

        # Hook up blocks
        for i, block in enumerate(self.unet.up_blocks):
            name = f"up_{i}"
            if name in block_names:
                handle = block.register_forward_hook(make_hook(name))
                self._hooks.append(handle)

        return self

    def _ensure_local_flow_heads(self, features: Dict[str, torch.Tensor]):
        """Create local flow heads if needed and not yet created."""
        if not self.use_local_flow_heads:
            return

        # Get target dtype from UNet
        target_dtype = next(self.unet.parameters()).dtype

        for name, feat in features.items():
            head_name = f"local_flow_{name}"
            if head_name not in self.aux_heads:
                # Create 1x1 conv to predict local velocity (4 channels for latent)
                head = nn.Conv2d(feat.shape[1], 4, kernel_size=1)
                head = head.to(dtype=target_dtype, device=feat.device)
                self.add_aux_head(head_name, head)

    def forward_with_features(
            self,
            x_t: torch.Tensor,
            timesteps: torch.LongTensor,
            encoder_hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through student UNet with feature capture.

        Returns:
            (v_prediction, features_dict)
        """
        self.clear_features()

        # Forward through UNet
        v_pred = self.unet(
            x_t,
            timesteps,
            encoder_hidden_states=encoder_hidden_states
        ).sample

        # Get captured features
        features = self.get_features()

        # Ensure local flow heads exist if enabled
        self._ensure_local_flow_heads(features)

        return v_pred, features

    def forward(self, x_t: torch.Tensor, timesteps: torch.LongTensor,
                encoder_hidden_states: torch.Tensor):
        """Simple forward without feature tracking."""
        return self.unet(x_t, timesteps, encoder_hidden_states=encoder_hidden_states).sample

    def compute_local_flow_loss(
            self,
            features: Dict[str, torch.Tensor],
            target_v: torch.Tensor,
            weight: float = 1.0
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute local flow prediction losses.

        Args:
            features: Student features
            target_v: Target velocity field from teacher
            weight: Loss weight

        Returns:
            (total_loss, loss_breakdown)
        """
        if not self.use_local_flow_heads:
            return torch.zeros((), device=target_v.device), {}

        total_loss = torch.zeros((), device=target_v.device)
        breakdown = {}

        # Get local predictions
        local_preds = self.get_aux_predictions(features)

        for name, v_local in local_preds.items():
            # Downsample target to match local prediction size
            target_local = F.interpolate(
                target_v,
                size=v_local.shape[-2:],
                mode='bilinear',
                align_corners=False
            )

            # MSE loss
            loss = F.mse_loss(v_local, target_local)
            total_loss = total_loss + weight * loss

            breakdown[f"{name}_mse"] = float(loss.item())

        breakdown['local_flow_total'] = float(total_loss.item())
        return total_loss, breakdown


class GenericStudent(StudentModel):
    """
    Generic student wrapper for any trainable model.
    """

    def __init__(self, model: nn.Module, block_names: Optional[List[str]] = None):
        super().__init__(block_names)
        self.model = model

    def forward_with_features(self, *args, **kwargs) -> Tuple[Any, Dict[str, torch.Tensor]]:
        """Forward pass (no feature extraction by default)."""
        outputs = self.model(*args, **kwargs)
        return outputs, {}

    def forward(self, *args, **kwargs):
        """Simple forward."""
        return self.model(*args, **kwargs)


# Utility functions for feature matching

def spatial_pool(x: torch.Tensor, method: str = "mean") -> torch.Tensor:
    """Pool spatial dimensions of feature tensor."""
    if method == "mean":
        return x.mean(dim=(2, 3))
    elif method == "max":
        return x.amax(dim=(2, 3))
    elif method == "adaptive_avg":
        return F.adaptive_avg_pool2d(x, (1, 1)).flatten(1)
    elif method == "adaptive_max":
        return F.adaptive_max_pool2d(x, (1, 1)).flatten(1)
    else:
        raise ValueError(f"Unknown pooling method: {method}")


def feature_distillation_loss(
        student_features: Dict[str, torch.Tensor],
        teacher_features: Dict[str, torch.Tensor],
        block_weights: Optional[Dict[str, float]] = None,
        pooling: str = "mean",
        loss_type: str = "cosine"  # or "mse", "kl"
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute feature distillation loss between student and teacher.

    Args:
        student_features: Student feature dict
        teacher_features: Teacher feature dict
        block_weights: Per-block weights
        pooling: Spatial pooling method
        loss_type: Type of loss (cosine, mse, kl)

    Returns:
        (total_loss, loss_breakdown)
    """
    if block_weights is None:
        block_weights = {k: 1.0 for k in student_features.keys()}

    total_loss = torch.zeros((), device=next(iter(student_features.values())).device)
    breakdown = {}

    for name in student_features.keys():
        if name not in teacher_features:
            continue

        s_feat = spatial_pool(student_features[name], pooling)
        t_feat = spatial_pool(teacher_features[name], pooling)

        if loss_type == "cosine":
            # Cosine similarity loss
            s_norm = F.normalize(s_feat, dim=-1)
            t_norm = F.normalize(t_feat, dim=-1)
            loss = 1.0 - (s_norm * t_norm).sum(-1).mean()
        elif loss_type == "mse":
            loss = F.mse_loss(s_feat, t_feat)
        elif loss_type == "kl":
            # KL divergence (features should be in log-space or use softmax)
            s_prob = F.softmax(s_feat, dim=-1)
            t_prob = F.softmax(t_feat, dim=-1)
            loss = F.kl_div(
                s_prob.log().clamp_min(-10),
                t_prob,
                reduction='batchmean'
            )
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

        weight = block_weights.get(name, 1.0)
        total_loss = total_loss + weight * loss
        breakdown[f"kd_{name}"] = float(loss.item())

    breakdown['kd_total'] = float(total_loss.item())
    return total_loss, breakdown