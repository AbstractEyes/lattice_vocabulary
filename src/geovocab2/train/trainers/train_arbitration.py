"""
TrainArbitrator - Complete Model Coordination Layer
Author: AbstractPhil + Claude Sonnet 4.5

Inherits from TrainerCore and adds full model coordination capabilities.
Handles teacher/student/assistant registration, orchestration, and provides
sensible defaults for abstract methods.

This is the complete middle layer that SD15FlowTrainer and other trainers inherit from.

Location: geovocab2/train/trainers/train_arbitrator.py
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from geovocab2.train.trainers.trainer_core import TrainerCore, TrainerCoreConfig, LossComposer


# =====================================================================================
# Configuration
# =====================================================================================

@dataclass
class TrainArbitratorConfig(TrainerCoreConfig):
    """
    Configuration for TrainArbitrator.
    Extends TrainerCoreConfig with model coordination settings.
    """
    # Optimizer
    optimizer_type: str = "AdamW"
    betas: Tuple[float, float] = (0.9, 0.999)

    # Scheduler
    scheduler_type: str = "CosineAnnealingLR"  # or "None", "StepLR", "ReduceLROnPlateau"
    scheduler_kwargs: Dict[str, Any] = field(default_factory=dict)

    # Model coordination
    freeze_teacher: bool = True
    freeze_assistant: bool = True

    # Data factory (callable that returns DataLoader)
    data_factory: Optional[Callable[[], DataLoader]] = None


# =====================================================================================
# TrainArbitrator (Complete Implementation)
# =====================================================================================

class TrainArbitrator(TrainerCore):
    """
    Complete training arbitrator that coordinates multiple models.

    Inherits from TrainerCore and adds:
    - Teacher/Student/Assistant model registration
    - Model freezing/unfreezing
    - Optimizer and scheduler creation
    - Coordinated forward passes
    - Default implementations of abstract methods

    Subclasses (like SD15FlowTrainer) override:
    - _initialize_models() to set up specific models
    - get_loss() to define task-specific losses
    - get_datasets() to provide custom data (or use data_factory)
    - Custom forward logic if needed
    """

    def __init__(
        self,
        name: str = "arbitrator",
        uid: str = "arb.default",
        cfg: Optional[TrainArbitratorConfig] = None
    ):
        """Initialize arbitrator with optional config."""
        if cfg is None:
            cfg = TrainArbitratorConfig()

        super().__init__(name, uid, cfg)
        self.arb_cfg: TrainArbitratorConfig = cfg

        # Models (set by subclasses or via register methods)
        self.teacher: Optional[nn.Module] = None
        self.student: Optional[nn.Module] = None
        self.assistant: Optional[nn.Module] = None

        # Custom forward function (can be set by subclass)
        self._custom_forward: Optional[Callable] = None

        # Initialize models (subclasses override this)
        self._initialize_models()

    def _initialize_models(self):
        """
        Initialize models. Override in subclass to set up teacher/student/assistant.

        Example:
            def _initialize_models(self):
                self.teacher = MyTeacher().to(self.device)
                self.student = MyStudent().to(self.device)
                self.assistant = MyAssistant().to(self.device)
                self._register_teacher(self.teacher)
                self._register_student(self.student)
                self._register_assistant(self.assistant)
        """
        pass  # Subclasses implement

    def to(self, device) -> TrainArbitrator:
        """Transfer all models to device."""
        super().to(device)

        if self.teacher is not None:
            self.teacher = self.teacher.to(self.device)

        if self.student is not None:
            self.student = self.student.to(self.device)

        if self.assistant is not None:
            self.assistant = self.assistant.to(self.device)

        return self

    # ---------------------------------------------------------------------------------
    # Model Registration (Called by subclasses)
    # ---------------------------------------------------------------------------------

    def _register_teacher(self, teacher: nn.Module):
        """
        Register and configure teacher model.
        Called by subclass during initialization.
        """
        self.teacher = teacher.to(self.device)
        if self.arb_cfg.freeze_teacher:
            for param in self.teacher.parameters():
                param.requires_grad_(False)
            self.teacher.eval()
        return self

    def _register_student(self, student: nn.Module):
        """
        Register and configure student model.
        Called by subclass during initialization.
        """
        self.student = student.to(self.device)
        return self

    def _register_assistant(self, assistant: nn.Module):
        """
        Register and configure assistant model.
        Called by subclass during initialization.
        """
        self.assistant = assistant.to(self.device)
        if self.arb_cfg.freeze_assistant:
            for param in self.assistant.parameters():
                param.requires_grad_(False)
            self.assistant.eval()
        return self

    # ---------------------------------------------------------------------------------
    # Optimizer and Scheduler Setup
    # ---------------------------------------------------------------------------------

    def _build_optimizer(self):
        """
        Build optimizer for trainable parameters.
        Called automatically in train() if optimizer not set.
        """
        if self.optimizer is not None:
            return self.optimizer

        params = self._get_trainable_params()

        if self.arb_cfg.optimizer_type == "AdamW":
            self.optimizer = torch.optim.AdamW(
                params,
                lr=self.cfg.lr,
                weight_decay=self.cfg.weight_decay,
                betas=self.arb_cfg.betas
            )
        elif self.arb_cfg.optimizer_type == "Adam":
            self.optimizer = torch.optim.Adam(
                params,
                lr=self.cfg.lr,
                weight_decay=self.cfg.weight_decay,
                betas=self.arb_cfg.betas
            )
        elif self.arb_cfg.optimizer_type == "SGD":
            self.optimizer = torch.optim.SGD(
                params,
                lr=self.cfg.lr,
                weight_decay=self.cfg.weight_decay,
                momentum=0.9
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.arb_cfg.optimizer_type}")

        return self.optimizer

    def _build_scheduler(self):
        """
        Build learning rate scheduler.
        Called automatically in train() if scheduler not set.
        """
        if self.scheduler is not None:
            return self.scheduler

        if self.arb_cfg.scheduler_type == "None" or self.arb_cfg.scheduler_type is None:
            return None

        if self.optimizer is None:
            self._build_optimizer()

        if self.arb_cfg.scheduler_type == "CosineAnnealingLR":
            # Estimate total steps
            loaders = self.get_datasets()
            if isinstance(loaders, tuple):
                train_loader = loaders[0]
            else:
                train_loader = loaders

            T_max = self.arb_cfg.scheduler_kwargs.get(
                'T_max',
                self.cfg.epochs * len(train_loader) if hasattr(train_loader, '__len__') else 10000
            )
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=T_max,
                **{k: v for k, v in self.arb_cfg.scheduler_kwargs.items() if k != 'T_max'}
            )
        elif self.arb_cfg.scheduler_type == "StepLR":
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                **self.arb_cfg.scheduler_kwargs
            )
        elif self.arb_cfg.scheduler_type == "ReduceLROnPlateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                **self.arb_cfg.scheduler_kwargs
            )
        else:
            raise ValueError(f"Unknown scheduler: {self.arb_cfg.scheduler_type}")

        return self.scheduler

    # ---------------------------------------------------------------------------------
    # TrainerCore Abstract Method Implementations
    # ---------------------------------------------------------------------------------

    def _get_trainable_params(self) -> List[nn.Parameter]:
        """Get all trainable parameters from registered models."""
        params = []

        # Student params (main trainable model)
        if self.student is not None:
            params.extend([p for p in self.student.parameters() if p.requires_grad])

        # Assistant params (if not frozen)
        if self.assistant is not None and not self.arb_cfg.freeze_assistant:
            params.extend([p for p in self.assistant.parameters() if p.requires_grad])

        # Teacher params (if not frozen - unusual but possible)
        if self.teacher is not None and not self.arb_cfg.freeze_teacher:
            params.extend([p for p in self.teacher.parameters() if p.requires_grad])

        if not params:
            # Fallback: collect from any model
            for model in [self.student, self.assistant, self.teacher]:
                if model is not None:
                    params.extend([p for p in model.parameters() if p.requires_grad])

        return params

    def _forward_pass(self, batch, **kwargs):
        """
        Default forward pass through all registered models.
        Subclasses should override for custom behavior or set _custom_forward.

        Returns:
            Dict with outputs from teacher, student, assistant
        """
        # Use custom forward if provided
        if self._custom_forward is not None:
            return self._custom_forward(batch, **kwargs)

        outputs = {}

        # Add models to kwargs for easy access
        kwargs.update({
            'teacher': self.teacher,
            'student': self.student,
            'assistant': self.assistant
        })

        # Teacher forward (no grad)
        if self.teacher is not None:
            with torch.no_grad():
                outputs['teacher'] = self._teacher_forward(batch, **kwargs)

        # Student forward (with grad)
        if self.student is not None:
            outputs['student'] = self._student_forward(batch, **kwargs)

        # Assistant forward (usually no grad)
        if self.assistant is not None:
            if self.arb_cfg.freeze_assistant:
                with torch.no_grad():
                    outputs['assistant'] = self._assistant_forward(batch, **kwargs)
            else:
                outputs['assistant'] = self._assistant_forward(batch, **kwargs)

        return outputs

    def _teacher_forward(self, batch, **kwargs):
        """
        Teacher forward pass.
        Override in subclass for custom teacher logic.
        """
        if self.teacher is None:
            return None

        # Try common interfaces
        if hasattr(self.teacher, 'forward_with_features'):
            return self.teacher.forward_with_features(batch, **kwargs)
        elif callable(self.teacher):
            return self.teacher(batch)
        return None

    def _student_forward(self, batch, **kwargs):
        """
        Student forward pass.
        Override in subclass for custom student logic.
        """
        if self.student is None:
            return None

        # Try common interfaces
        if hasattr(self.student, 'forward_with_features'):
            return self.student.forward_with_features(batch, **kwargs)
        elif callable(self.student):
            return self.student(batch)
        return None

    def _assistant_forward(self, batch, **kwargs):
        """
        Assistant forward pass.
        Override in subclass for custom assistant logic.
        """
        if self.assistant is None:
            return None

        # Try common interfaces
        if hasattr(self.assistant, 'assess'):
            return self.assistant.assess(**kwargs)
        elif callable(self.assistant):
            return self.assistant(batch)
        return None

    # ---------------------------------------------------------------------------------
    # TrainBase Abstract Methods (Provide defaults or delegate to subclass)
    # ---------------------------------------------------------------------------------

    def get_model(self) -> Any:
        """
        Return the primary trainable model (student).
        Subclasses can override to return specific model.
        """
        return self.student

    def get_loss(self) -> LossComposer:
        """
        Get loss composer. MUST be overridden by subclass.

        Subclass example:
            def get_loss(self):
                composer = LossComposer()
                composer.register("main_loss", self._main_loss_fn, 1.0)
                composer.register("aux_loss", self._aux_loss_fn, 0.5)
                return composer
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement get_loss() to define training losses"
        )

    def get_datasets(self) -> Any:
        """
        Get datasets/dataloaders.

        Default behavior:
        1. If data_factory is configured, use it
        2. Otherwise, subclass must override

        Returns:
            DataLoader or Tuple[DataLoader, ...]
        """
        if self.arb_cfg.data_factory is not None:
            return self.arb_cfg.data_factory()

        raise NotImplementedError(
            f"{self.__class__.__name__} must implement get_datasets() or provide data_factory in config"
        )

    def validate(self, **kwargs) -> Dict[str, Any]:
        """
        Default validation implementation.
        Subclasses can override for custom validation logic.

        Returns validation metrics.
        """
        if not hasattr(self, '_validate_impl'):
            # No custom validation - return empty results
            return {'val_loss': 0.0, 'message': 'No validation implemented'}

        return self._validate_impl(**kwargs)

    def test(self, **kwargs) -> Dict[str, Any]:
        """
        Default test implementation.
        Subclasses can override for custom test logic.

        Returns test metrics.
        """
        if not hasattr(self, '_test_impl'):
            # No custom test - return empty results
            return {'test_loss': 0.0, 'message': 'No test implemented'}

        return self._test_impl(**kwargs)

    # ---------------------------------------------------------------------------------
    # Training Lifecycle (Override train() to setup optimizer/scheduler)
    # ---------------------------------------------------------------------------------

    def train(self, **kwargs) -> Dict[str, Any]:
        """
        Training with automatic optimizer/scheduler setup.
        Subclasses can override for custom setup logic.
        """
        # Build optimizer and scheduler if not already built
        if self.optimizer is None:
            self._build_optimizer()

        if self.scheduler is None:
            self._build_scheduler()

        # Call parent train (which calls TrainerCore's training loop)
        return super().train(**kwargs)

    # ---------------------------------------------------------------------------------
    # Custom Forward Function Support
    # ---------------------------------------------------------------------------------

    def set_custom_forward(self, forward_fn: Callable):
        """
        Set a custom forward function for the training loop.

        Args:
            forward_fn: Function with signature (batch, **kwargs) -> outputs_dict

        Example:
            def my_forward(batch, teacher, student, **kwargs):
                # Custom logic here
                return {'outputs': ...}

            trainer.set_custom_forward(my_forward)
        """
        self._custom_forward = forward_fn
        return self

    # ---------------------------------------------------------------------------------
    # State Management
    # ---------------------------------------------------------------------------------

    def get_state_dict(self) -> Dict[str, Any]:
        """Get full training state including all models."""
        state = {
            'cfg': self.cfg,
            'arb_cfg': self.arb_cfg,
            'global_step': self.global_step
        }

        # Model states
        if self.student is not None:
            state['student'] = self.student.state_dict()

        if self.assistant is not None and not self.arb_cfg.freeze_assistant:
            state['assistant'] = self.assistant.state_dict()

        if self.teacher is not None and not self.arb_cfg.freeze_teacher:
            state['teacher'] = self.teacher.state_dict()

        # Optimizer/scheduler states
        if self.optimizer is not None:
            state['optimizer'] = self.optimizer.state_dict()

        if self.scheduler is not None:
            state['scheduler'] = self.scheduler.state_dict()

        return state

    def load_state_dict(self, state: Dict[str, Any], strict: bool = True):
        """
        Load training state.

        Args:
            state: State dictionary from get_state_dict()
            strict: Whether to strictly enforce state dict keys
        """
        # Load configs (if present)
        if 'cfg' in state:
            # Optionally update config
            pass

        if 'arb_cfg' in state:
            # Optionally update arbitrator config
            pass

        # Load model states
        if 'student' in state and self.student is not None:
            self.student.load_state_dict(state['student'], strict=strict)

        if 'assistant' in state and self.assistant is not None:
            self.assistant.load_state_dict(state['assistant'], strict=strict)

        if 'teacher' in state and self.teacher is not None:
            self.teacher.load_state_dict(state['teacher'], strict=strict)

        # Load optimizer/scheduler states
        if 'optimizer' in state and self.optimizer is not None:
            self.optimizer.load_state_dict(state['optimizer'])

        if 'scheduler' in state and self.scheduler is not None:
            self.scheduler.load_state_dict(state['scheduler'])

        # Load global step
        if 'global_step' in state:
            self.global_step = state['global_step']

        return self

    def info(self) -> Dict[str, Any]:
        """Extended trainer metadata."""
        base_info = super().info()
        base_info.update({
            "description": "Training arbitrator with model coordination",
            "optimizer": self.arb_cfg.optimizer_type,
            "scheduler": self.arb_cfg.scheduler_type,
            "freeze_teacher": self.arb_cfg.freeze_teacher,
            "freeze_assistant": self.arb_cfg.freeze_assistant,
            "has_teacher": self.teacher is not None,
            "has_student": self.student is not None,
            "has_assistant": self.assistant is not None,
            "trainable_params": len(self._get_trainable_params()) if self.student else 0
        })
        return base_info

    # ---------------------------------------------------------------------------------
    # Utility Methods
    # ---------------------------------------------------------------------------------

    def count_parameters(self, only_trainable: bool = False) -> int:
        """Count total parameters in all models."""
        total = 0

        for model in [self.student, self.teacher, self.assistant]:
            if model is not None:
                if only_trainable:
                    total += sum(p.numel() for p in model.parameters() if p.requires_grad)
                else:
                    total += sum(p.numel() for p in model.parameters())

        return total

    def freeze_model(self, model_name: str):
        """Freeze a model by name ('teacher', 'student', 'assistant')."""
        model = getattr(self, model_name, None)
        if model is not None:
            for param in model.parameters():
                param.requires_grad_(False)
            model.eval()

    def unfreeze_model(self, model_name: str):
        """Unfreeze a model by name ('teacher', 'student', 'assistant')."""
        model = getattr(self, model_name, None)
        if model is not None:
            for param in model.parameters():
                param.requires_grad_(True)
            model.train()

    def __repr__(self) -> str:
        """String representation."""
        models = []
        if self.teacher: models.append("teacher")
        if self.student: models.append("student")
        if self.assistant: models.append("assistant")

        return (
            f"{self.__class__.__name__}("
            f"name='{self.name}', "
            f"uid='{self.uid}', "
            f"models={models}, "
            f"device='{self.device}'"
            f")"
        )