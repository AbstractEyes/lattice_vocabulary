"""
TeacherModel: Wrapper for frozen teacher models that provide training targets.
Examples: Pretrained diffusion models, large language models, etc.
"""
from __future__ import annotations
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any, Callable
from abc import ABC, abstractmethod


class TeacherModel(nn.Module, ABC):
    """
    Base class for teacher models.

    Teachers are frozen models that provide:
    - Training targets (predictions, features, logits)
    - Feature representations for distillation
    - Quality baselines
    """

    def __init__(self, freeze: bool = True):
        super().__init__()
        self.freeze = freeze
        self._hooks: List[Any] = []
        self._feature_bank: Dict[str, torch.Tensor] = {}

    def setup(self):
        """Setup teacher (freeze params, set eval mode)."""
        if self.freeze:
            for param in self.parameters():
                param.requires_grad_(False)
            self.eval()
        return self

    def setup_feature_hooks(self, block_names: List[str]) -> TeacherModel:
        """
        Setup hooks to capture intermediate features.
        Override this to customize hook placement.

        Args:
            block_names: Names of blocks to capture

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
    @torch.no_grad()
    def forward_with_features(self, *args, **kwargs) -> Tuple[Any, Dict[str, torch.Tensor]]:
        """
        Forward pass that returns both outputs and intermediate features.

        Returns:
            (outputs, features_dict)
        """
        pass

    @torch.no_grad()
    def get_targets(self, *args, **kwargs) -> Any:
        """
        Get training targets from teacher.
        Override this to customize what targets are extracted.
        """
        outputs, features = self.forward_with_features(*args, **kwargs)
        return outputs

    def __enter__(self):
        """Context manager support."""
        return self

    def __exit__(self, *args):
        """Context manager cleanup."""
        self.remove_hooks()
        self.clear_features()


class SD15Teacher(TeacherModel):
    """
    Stable Diffusion 1.5 teacher for flow matching distillation.
    """

    def __init__(
        self,
        unet: nn.Module,
        text_encoder: nn.Module,
        tokenizer: Any,
        scheduler: Any,
        block_names: Tuple[str, ...] = (
            "down_0", "down_1", "down_2", "down_3",
            "mid",
            "up_0", "up_1", "up_2", "up_3"
        ),
        freeze: bool = True
    ):
        super().__init__(freeze=freeze)
        self.unet = unet
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.scheduler = scheduler
        self.block_names = block_names

        self.setup()
        self.setup_feature_hooks(list(block_names))

    def setup_feature_hooks(self, block_names: List[str]) -> SD15Teacher:
        """Setup hooks on UNet blocks."""
        def make_hook(name: str):
            def hook(module, input, output):
                # Handle tuple outputs from blocks
                out = output[0] if isinstance(output, (tuple, list)) else output
                self._feature_bank[name] = out.detach()
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

    @torch.no_grad()
    def encode_text(self, prompts: List[str]) -> torch.Tensor:
        """Encode text prompts to embeddings."""
        tokens = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )
        return self.text_encoder(tokens.input_ids.to(next(self.parameters()).device))[0]

    @torch.no_grad()
    def forward_with_features(
        self,
        x_t: torch.Tensor,
        timesteps: torch.LongTensor,
        encoder_hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through UNet with feature capture.

        Returns:
            (noise_prediction, features_dict)
        """
        self.clear_features()

        # Forward through UNet
        noise_pred = self.unet(
            x_t,
            timesteps,
            encoder_hidden_states=encoder_hidden_states
        ).sample

        # Get captured features
        features = self.get_features()

        return noise_pred, features

    @torch.no_grad()
    def get_v_target(
        self,
        x_t: torch.Tensor,
        timesteps: torch.LongTensor,
        encoder_hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """
        Get v-prediction target for flow matching.

        v = α·ε - σ·x₀
        where x₀ = (x_t - σ·ε) / α
        """
        # Get noise prediction
        eps_pred, _ = self.forward_with_features(x_t, timesteps, encoder_hidden_states)

        # Get alpha and sigma from scheduler
        alpha, sigma = self.alpha_sigma(timesteps)

        # Predict x0
        x0_pred = (x_t - sigma * eps_pred) / (alpha + 1e-8)

        # Compute v
        v = alpha * eps_pred - sigma * x0_pred

        return v

    def alpha_sigma(self, timesteps: torch.LongTensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get alpha and sigma values for given timesteps."""
        alphas_cumprod = self.scheduler.alphas_cumprod.to(timesteps.device)

        ac = alphas_cumprod[timesteps]
        alpha = ac.sqrt().view(-1, 1, 1, 1)
        sigma = (1.0 - ac).sqrt().view(-1, 1, 1, 1)

        return alpha, sigma


class GenericTeacher(TeacherModel):
    """
    Generic teacher wrapper for any model.
    Useful for classification, regression, or other tasks.
    """

    def __init__(self, model: nn.Module, freeze: bool = True):
        super().__init__(freeze=freeze)
        self.model = model
        self.setup()

    @torch.no_grad()
    def forward_with_features(self, *args, **kwargs) -> Tuple[Any, Dict[str, torch.Tensor]]:
        """Forward pass (no feature extraction by default)."""
        outputs = self.model(*args, **kwargs)
        return outputs, {}

    @torch.no_grad()
    def forward(self, *args, **kwargs):
        """Simple forward."""
        return self.model(*args, **kwargs)