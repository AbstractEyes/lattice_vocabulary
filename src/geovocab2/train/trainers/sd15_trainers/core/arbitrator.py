"""
Arbitrator: Central coordination system for models, data, and training logic.
Delegates responsibilities and maintains system coherence.
"""
from __future__ import annotations
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass

from geovocab2.train.trainers.distillery.trainer_core import Trainer, LossComposer, TrainerConfig


@dataclass
class ArbitratorConfig:
    """Configuration for the Arbitrator."""
    # Optimizer
    optimizer_type: str = "AdamW"
    lr: float = 1e-4
    weight_decay: float = 1e-3
    betas: tuple = (0.9, 0.999)

    # Scheduler
    scheduler_type: str = "CosineAnnealingLR"  # or "None", "StepLR", etc.
    scheduler_kwargs: Dict[str, Any] = None

    # Model coordination
    freeze_teacher: bool = True
    freeze_assistant: bool = True

    def __post_init__(self):
        if self.scheduler_kwargs is None:
            self.scheduler_kwargs = {}


class Arbitrator:
    """
    Central coordinator for training system.

    Responsibilities:
    - Model management (teacher, student, assistant)
    - Data delegation
    - Loss coordination
    - Training orchestration
    """

    def __init__(
            self,
            cfg: ArbitratorConfig,
            trainer_cfg: TrainerConfig,
            device: str = "cuda"
    ):
        self.cfg = cfg
        self.trainer_cfg = trainer_cfg
        self.device = device

        # Models (set by user)
        self.teacher: Optional[nn.Module] = None
        self.student: Optional[nn.Module] = None
        self.assistant: Optional[nn.Module] = None

        # Trainer components
        self.trainer: Optional[Trainer] = None
        self.loss_composer: Optional[LossComposer] = None

        # Data factory
        self.data_factory: Optional[Callable] = None

    def register_teacher(self, teacher: nn.Module):
        """Register and configure teacher model."""
        self.teacher = teacher.to(self.device)
        if self.cfg.freeze_teacher:
            for param in self.teacher.parameters():
                param.requires_grad_(False)
            self.teacher.eval()
        return self

    def register_student(self, student: nn.Module):
        """Register and configure student model."""
        self.student = student.to(self.device)
        return self

    def register_assistant(self, assistant: nn.Module):
        """Register and configure assistant model (e.g., David)."""
        self.assistant = assistant.to(self.device)
        if self.cfg.freeze_assistant:
            for param in self.assistant.parameters():
                param.requires_grad_(False)
            self.assistant.eval()
        return self

    def register_data_factory(self, factory: Callable):
        """
        Register data factory.
        Factory should return a DataLoader when called.
        """
        self.data_factory = factory
        return self

    def register_loss_composer(self, composer: LossComposer):
        """Register loss composer."""
        self.loss_composer = composer
        return self

    def build_trainer(self) -> Trainer:
        """Build trainer with all components."""
        if self.data_factory is None:
            raise RuntimeError("Data factory not registered")
        if self.loss_composer is None:
            raise RuntimeError("Loss composer not registered")

        self.trainer = Trainer(
            cfg=self.trainer_cfg,
            data_factory=self.data_factory,
            loss_composer=self.loss_composer,
            device=self.device
        )

        # Set up optimizer
        optimizer = self._build_optimizer()
        self.trainer.set_optimizer(optimizer)

        # Set up scheduler
        if self.cfg.scheduler_type != "None":
            scheduler = self._build_scheduler(optimizer)
            self.trainer.set_scheduler(scheduler)

        # Set trainable params for gradient clipping
        trainable_params = self._get_trainable_parameters()
        self.trainer.set_trainable_params(trainable_params)

        return self.trainer

    def _build_optimizer(self) -> torch.optim.Optimizer:
        """Build optimizer for trainable parameters."""
        params = self._get_trainable_parameters()

        if self.cfg.optimizer_type == "AdamW":
            return torch.optim.AdamW(
                params,
                lr=self.cfg.lr,
                weight_decay=self.cfg.weight_decay,
                betas=self.cfg.betas
            )
        elif self.cfg.optimizer_type == "Adam":
            return torch.optim.Adam(
                params,
                lr=self.cfg.lr,
                weight_decay=self.cfg.weight_decay,
                betas=self.cfg.betas
            )
        elif self.cfg.optimizer_type == "SGD":
            return torch.optim.SGD(
                params,
                lr=self.cfg.lr,
                weight_decay=self.cfg.weight_decay,
                momentum=0.9
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.cfg.optimizer_type}")

    def _build_scheduler(self, optimizer: torch.optim.Optimizer):
        """Build learning rate scheduler."""
        if self.cfg.scheduler_type == "CosineAnnealingLR":
            # Default: anneal over all steps
            T_max = self.cfg.scheduler_kwargs.get(
                'T_max',
                self.trainer_cfg.epochs * (self.trainer_cfg.batch_size or 1000)
            )
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=T_max,
                **{k: v for k, v in self.cfg.scheduler_kwargs.items() if k != 'T_max'}
            )
        elif self.cfg.scheduler_type == "StepLR":
            return torch.optim.lr_scheduler.StepLR(
                optimizer,
                **self.cfg.scheduler_kwargs
            )
        elif self.cfg.scheduler_type == "ReduceLROnPlateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                **self.cfg.scheduler_kwargs
            )
        else:
            raise ValueError(f"Unknown scheduler: {self.cfg.scheduler_type}")

    def _get_trainable_parameters(self) -> List[nn.Parameter]:
        """Get all trainable parameters from student and unfrozen models."""
        params = []

        # Student params (main trainable model)
        if self.student is not None:
            params.extend([p for p in self.student.parameters() if p.requires_grad])

        # Assistant params (if not frozen)
        if self.assistant is not None and not self.cfg.freeze_assistant:
            params.extend([p for p in self.assistant.parameters() if p.requires_grad])

        # Teacher params (if not frozen - unusual but possible)
        if self.teacher is not None and not self.cfg.freeze_teacher:
            params.extend([p for p in self.teacher.parameters() if p.requires_grad])

        return params

    def create_forward_fn(self, custom_forward: Optional[Callable] = None) -> Callable:
        """
        Create forward function for training loop.

        If custom_forward is provided, uses that.
        Otherwise, creates default forward that runs teacher, student, assistant.
        """
        if custom_forward is not None:
            return custom_forward

        def default_forward(batch, **kwargs):
            """Default forward pass through all models."""
            outputs = {}

            # Teacher forward (no grad)
            if self.teacher is not None:
                with torch.no_grad():
                    outputs['teacher'] = self.teacher(batch)

            # Student forward (with grad)
            if self.student is not None:
                outputs['student'] = self.student(batch)

            # Assistant forward (usually no grad)
            if self.assistant is not None:
                if self.cfg.freeze_assistant:
                    with torch.no_grad():
                        outputs['assistant'] = self.assistant(batch)
                else:
                    outputs['assistant'] = self.assistant(batch)

            return outputs

        return default_forward

    def train(self, custom_forward: Optional[Callable] = None, **forward_kwargs):
        """
        Start training with optional custom forward function.

        Args:
            custom_forward: Optional custom forward function
            **forward_kwargs: Additional kwargs passed to forward and losses
        """
        if self.trainer is None:
            self.build_trainer()

        forward_fn = self.create_forward_fn(custom_forward)

        # Add models to forward_kwargs for easy access in losses
        forward_kwargs.update({
            'teacher': self.teacher,
            'student': self.student,
            'assistant': self.assistant
        })

        self.trainer.train(forward_fn, **forward_kwargs)

    def get_state_dict(self) -> Dict[str, Any]:
        """Get full training state."""
        state = {
            'cfg': self.cfg,
            'trainer_cfg': self.trainer_cfg,
        }

        if self.student is not None:
            state['student'] = self.student.state_dict()
        if self.assistant is not None and not self.cfg.freeze_assistant:
            state['assistant'] = self.assistant.state_dict()
        if self.teacher is not None and not self.cfg.freeze_teacher:
            state['teacher'] = self.teacher.state_dict()

        if self.trainer is not None:
            state['global_step'] = self.trainer.global_step
            if self.trainer.optimizer is not None:
                state['optimizer'] = self.trainer.optimizer.state_dict()
            if self.trainer.scheduler is not None:
                state['scheduler'] = self.trainer.scheduler.state_dict()

        return state

    def load_state_dict(self, state: Dict[str, Any]):
        """Load training state."""
        if 'student' in state and self.student is not None:
            self.student.load_state_dict(state['student'])
        if 'assistant' in state and self.assistant is not None:
            self.assistant.load_state_dict(state['assistant'])
        if 'teacher' in state and self.teacher is not None:
            self.teacher.load_state_dict(state['teacher'])

        if 'optimizer' in state and self.trainer is not None and self.trainer.optimizer is not None:
            self.trainer.optimizer.load_state_dict(state['optimizer'])
        if 'scheduler' in state and self.trainer is not None and self.trainer.scheduler is not None:
            self.trainer.scheduler.load_state_dict(state['scheduler'])
        if 'global_step' in state and self.trainer is not None:
            self.trainer.global_step = state['global_step']