"""
AssistantModel: Wrapper for auxiliary models that provide guidance, assessment, or auxiliary losses.
Examples: David (geometric assessor), discriminators, auxiliary classifiers, etc.
"""
from __future__ import annotations
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any, Callable
from abc import ABC, abstractmethod


class AssistantModel(nn.Module, ABC):
    """
    Base class for assistant models.

    Assistants provide auxiliary signals like:
    - Quality assessment (David scoring features)
    - Adversarial signals (discriminators)
    - Auxiliary predictions (multi-task heads)
    - Guidance signals (feature alignment targets)
    """

    def __init__(self, freeze: bool = True):
        super().__init__()
        self.freeze = freeze
        self._hooks: List[Any] = []

    def setup_hooks(self, target_model: nn.Module) -> AssistantModel:
        """
        Setup any hooks needed on the target model.
        Override this if assistant needs to monitor intermediate features.

        Returns self for chaining.
        """
        return self

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    @abstractmethod
    def assess(self, **kwargs) -> Dict[str, Any]:
        """
        Perform assessment on inputs.

        Returns:
            Dictionary with assessment results (scores, predictions, etc.)
        """
        pass

    @abstractmethod
    def compute_losses(self, **kwargs) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute auxiliary losses from assistant.

        Returns:
            (total_loss, loss_breakdown_dict)
        """
        pass

    def modify_outputs(self, outputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optionally modify model outputs based on assistant's assessment.
        Override if assistant should influence outputs.

        Args:
            outputs: Model outputs dictionary

        Returns:
            Modified outputs dictionary
        """
        return outputs

    def __enter__(self):
        """Context manager support for temporary hook setup."""
        if self.freeze:
            self.eval()
        return self

    def __exit__(self, *args):
        """Context manager cleanup."""
        self.remove_hooks()


class DavidAssistant(AssistantModel):
    """
    David-based geometric assessor assistant.
    Provides block-level quality scores and guidance signals.
    """

    def __init__(
        self,
        david_model: nn.Module,
        pooling: str = "mean",
        freeze: bool = True
    ):
        super().__init__(freeze=freeze)
        self.david = david_model
        self.pooling = pooling

        if freeze:
            for param in self.david.parameters():
                param.requires_grad_(False)
            self.david.eval()

    def _spatial_pool(self, x: torch.Tensor, name: str) -> torch.Tensor:
        """Pool spatial dimensions."""
        if self.pooling == "mean":
            return x.mean(dim=(2, 3))
        elif self.pooling == "max":
            return x.amax(dim=(2, 3))
        elif self.pooling == "adaptive":
            # Down/mid blocks use mean, up blocks use max
            if name.startswith("down") or name == "mid":
                return x.mean(dim=(2, 3))
            else:
                return x.amax(dim=(2, 3))
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")

    @torch.no_grad()
    def assess(
        self,
        features: Dict[str, torch.Tensor],
        timesteps: torch.LongTensor,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Assess feature quality using David.

        Args:
            features: Dictionary of block features {block_name: tensor}
            timesteps: Timestep values

        Returns:
            Dictionary with:
                - e_t: timestep errors per block
                - e_p: pattern errors per block
                - coh: coherence scores per block
        """
        # Pool spatial dimensions
        pooled = {name: self._spatial_pool(feat, name)
                  for name, feat in features.items()}

        # Forward through David
        outputs = self.david(pooled, timesteps.float())

        # Extract assessment metrics
        e_t = self._extract_timestep_errors(outputs, timesteps)
        e_p = self._extract_pattern_errors(outputs)
        coh = self._extract_coherence(outputs, pooled)

        return {
            'timestep_errors': e_t,
            'pattern_errors': e_p,
            'coherence': coh,
            'raw_outputs': outputs
        }

    def _extract_timestep_errors(
        self,
        outputs: Dict[str, Any],
        timesteps: torch.LongTensor
    ) -> Dict[str, float]:
        """Extract timestep prediction errors."""
        errors = {}

        # Find timestep logits key
        ts_key = None
        for key in ["timestep_logits", "logits_timestep", "timestep_head_logits"]:
            if key in outputs:
                ts_key = key
                break

        if ts_key is None:
            # No timestep prediction, return zeros
            return {name: 0.0 for name in outputs.get('block_names', [])}

        logits = outputs[ts_key]
        t_bins = (timesteps // 10).to(logits.device if isinstance(logits, torch.Tensor) else 'cuda')

        if isinstance(logits, dict):
            # Per-block logits
            for name, L in logits.items():
                ce = nn.functional.cross_entropy(L, t_bins, reduction='mean')
                errors[name] = float(ce.item())
        else:
            # Single head - broadcast to all blocks
            ce = nn.functional.cross_entropy(logits, t_bins, reduction='mean')
            for name in outputs.get('block_names', ['global']):
                errors[name] = float(ce.item())

        return errors

    def _extract_pattern_errors(self, outputs: Dict[str, Any]) -> Dict[str, float]:
        """Extract pattern prediction errors (or entropy as proxy)."""
        errors = {}

        # Find pattern logits key
        pt_key = None
        for key in ["pattern_logits", "logits_pattern", "pattern_head_logits"]:
            if key in outputs:
                pt_key = key
                break

        if pt_key is None:
            return {name: 0.0 for name in outputs.get('block_names', [])}

        logits = outputs[pt_key]

        if isinstance(logits, dict):
            for name, L in logits.items():
                P = L.softmax(-1)
                ent = -(P * P.clamp_min(1e-9).log()).sum(-1).mean()
                # Normalize entropy by max possible
                errors[name] = float(ent.item() / torch.log(torch.tensor(P.shape[-1])))
        else:
            P = logits.softmax(-1)
            ent = -(P * P.clamp_min(1e-9).log()).sum(-1).mean()
            for name in outputs.get('block_names', ['global']):
                errors[name] = float(ent.item() / torch.log(torch.tensor(P.shape[-1])))

        return errors

    def _extract_coherence(
        self,
        outputs: Dict[str, Any],
        features: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """Extract coherence scores (e.g., Cantor alphas)."""
        coherence = {}

        # Try to get Cantor alphas if David exposes them
        try:
            if hasattr(self.david, 'get_cantor_alphas'):
                alphas = self.david.get_cantor_alphas()
                if alphas:
                    avg_alpha = sum(alphas.values()) / len(alphas)
                    for name in features.keys():
                        coherence[name] = float(alphas.get(name, avg_alpha))
                    return coherence
        except Exception:
            pass

        # Default: assume full coherence
        for name in features.keys():
            coherence[name] = 1.0

        return coherence

    def compute_losses(
        self,
        features: Dict[str, torch.Tensor],
        timesteps: torch.LongTensor,
        **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute auxiliary losses from David assessment.

        Note: David is frozen, so this typically returns zero loss.
        Override if you want to add specific David-based losses.
        """
        # David is frozen - no direct loss
        # But we can compute assessment for logging
        with torch.no_grad():
            assessment = self.assess(features, timesteps)

        loss_dict = {
            'david_avg_timestep_error': sum(assessment['timestep_errors'].values()) / max(len(assessment['timestep_errors']), 1),
            'david_avg_pattern_error': sum(assessment['pattern_errors'].values()) / max(len(assessment['pattern_errors']), 1),
            'david_avg_coherence': sum(assessment['coherence'].values()) / max(len(assessment['coherence']), 1)
        }

        return torch.zeros((), device=timesteps.device), loss_dict

    def compute_adaptive_weights(
        self,
        features: Dict[str, torch.Tensor],
        timesteps: torch.LongTensor,
        base_weights: Dict[str, float],
        alpha: float = 0.5,
        beta: float = 0.25,
        delta: float = 0.25,
        lambda_min: float = 0.5,
        lambda_max: float = 3.0
    ) -> Dict[str, float]:
        """
        Compute adaptive per-block weights based on David assessment.

        Formula: λ_b = w_b * (1 + α·e_t + β·e_p + δ·(1−coh))

        Args:
            features: Block features
            timesteps: Timesteps
            base_weights: Base weights per block
            alpha: Timestep error weight
            beta: Pattern error weight
            delta: Incoherence weight
            lambda_min: Minimum lambda
            lambda_max: Maximum lambda

        Returns:
            Adaptive weights per block
        """
        assessment = self.assess(features, timesteps)

        e_t = assessment['timestep_errors']
        e_p = assessment['pattern_errors']
        coh = assessment['coherence']

        lambdas = {}
        for name, base in base_weights.items():
            val = base * (
                1.0
                + alpha * e_t.get(name, 0.0)
                + beta * e_p.get(name, 0.0)
                + delta * (1.0 - coh.get(name, 1.0))
            )
            lambdas[name] = max(lambda_min, min(lambda_max, val))

        return lambdas


class NullAssistant(AssistantModel):
    """
    Null assistant that does nothing.
    Useful for ablation studies or when no assistant is needed.
    """

    def assess(self, **kwargs) -> Dict[str, Any]:
        return {}

    def compute_losses(self, **kwargs) -> Tuple[torch.Tensor, Dict[str, float]]:
        return torch.zeros((), device='cuda'), {}