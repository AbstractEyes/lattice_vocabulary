"""
TrainerCore - Base TrainBase Implementation
Author: AbstractPhil + Claude Sonnet 4.5

Core training infrastructure that inherits from TrainBase.
Provides basic training loop, loss composition, and optimization.

This is the foundation that TrainArbitrator will build upon.

Location: geovocab2/train/trainer_core.py
"""
from __future__ import annotations
from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Callable, Optional, Any

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from geovocab2.train.train_base import TrainBase


# =====================================================================================
# Configuration
# =====================================================================================

@dataclass
class TrainerCoreConfig:
    """Base configuration for core trainer."""
    run_name: str = "training_run"
    out_dir: str = "./runs"

    # Training params
    epochs: int = 10
    batch_size: int = 32

    # Optimization
    lr: float = 1e-4
    weight_decay: float = 1e-3
    grad_clip: float = 1.0

    # AMP
    amp: bool = True

    # Logging
    log_every: int = 50

    def __post_init__(self):
        Path(self.out_dir).mkdir(parents=True, exist_ok=True)


# =====================================================================================
# Loss Composer
# =====================================================================================

class LossComposer:
    """
    Composes multiple loss functions with weights.
    Losses are callables that return (loss_value, loss_dict).
    """
    def __init__(self):
        self.losses: Dict[str, Callable] = {}
        self.weights: Dict[str, float] = {}

    def register(self, name: str, loss_fn: Callable, weight: float = 1.0):
        """Register a loss function with a weight."""
        self.losses[name] = loss_fn
        self.weights[name] = weight

    def compute(self, **kwargs) -> tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute all registered losses.
        Returns (total_loss, loss_dict).
        """
        device = kwargs.get('device', 'cuda')
        total = torch.zeros((), device=device)
        breakdown = {}

        for name, loss_fn in self.losses.items():
            loss_val, loss_info = loss_fn(**kwargs)
            weighted = self.weights[name] * loss_val
            total = total + weighted

            # Store both weighted and raw
            breakdown[f"{name}"] = float(loss_val.item())
            breakdown[f"{name}_weighted"] = float(weighted.item())

            # Add any additional info from loss_fn
            if loss_info:
                for k, v in loss_info.items():
                    breakdown[f"{name}_{k}"] = v

        breakdown["total"] = float(total.item())
        return total, breakdown


# =====================================================================================
# TrainerCore (Inherits TrainBase)
# =====================================================================================

class TrainerCore(TrainBase):
    """
    Core training infrastructure inheriting from TrainBase.

    Provides:
    - Basic training loop
    - Loss composition via LossComposer
    - Optimization and scheduling
    - Logging infrastructure

    Subclasses (like TrainArbitrator) override to add:
    - Model coordination
    - Advanced delegation
    - Custom forward passes
    """

    def __init__(
        self,
        name: str,
        uid: str,
        cfg: TrainerCoreConfig
    ):
        """Initialize core trainer."""
        super().__init__(name, uid)
        self.cfg = cfg
        self.device = torch.device("cpu")

        # Components (set by subclasses)
        self.loss_composer: Optional[LossComposer] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None

        # AMP
        self.scaler = GradScaler(enabled=cfg.amp)

        # Logging
        self.writer = SummaryWriter(log_dir=str(Path(cfg.out_dir) / cfg.run_name))
        self.global_step = 0

    def to(self, device) -> TrainerCore:
        """Transfer trainer to device."""
        self.device = torch.device(device)
        return self

    # ---------------------------------------------------------------------------------
    # TrainBase Abstract Methods (Must be implemented by subclasses)
    # ---------------------------------------------------------------------------------

    @abstractmethod
    def get_model(self) -> Any:
        """Return the model to train. Override in subclass."""
        pass

    @abstractmethod
    def get_loss(self) -> LossComposer:
        """Return loss composer. Override in subclass."""
        pass

    @abstractmethod
    def get_datasets(self) -> Any:
        """Return datasets. Override in subclass."""
        pass

    # ---------------------------------------------------------------------------------
    # Training Infrastructure (Implemented here, used by subclasses)
    # ---------------------------------------------------------------------------------

    def train(self, **kwargs) -> Dict[str, Any]:
        """
        Core training loop implementation.
        Subclasses can override to customize behavior.
        """
        epochs = kwargs.get('epochs', self.cfg.epochs)

        # Get components from subclass
        if self.loss_composer is None:
            self.loss_composer = self.get_loss()

        # Get datasets
        loaders = self.get_datasets()
        if isinstance(loaders, tuple):
            train_loader = loaders[0]
        else:
            train_loader = loaders

        # Training loop
        for epoch in range(epochs):
            self._train_epoch(epoch, train_loader, **kwargs)

        self.writer.close()
        return {"status": "training_complete"}

    def validate(self, **kwargs) -> Dict[str, Any]:
        """
        Validation loop implementation.
        Subclasses can override to customize behavior.
        """
        loaders = self.get_datasets()
        if isinstance(loaders, tuple) and len(loaders) > 1:
            val_loader = loaders[1]
        else:
            raise ValueError("No validation loader available")

        return self._eval_loop(val_loader, "validation", **kwargs)

    def test(self, **kwargs) -> Dict[str, Any]:
        """
        Test loop implementation.
        Subclasses can override to customize behavior.
        """
        loaders = self.get_datasets()
        if isinstance(loaders, tuple) and len(loaders) > 2:
            test_loader = loaders[2]
        else:
            raise ValueError("No test loader available")

        return self._eval_loop(test_loader, "test", **kwargs)

    # ---------------------------------------------------------------------------------
    # Internal Training Methods
    # ---------------------------------------------------------------------------------

    def _train_epoch(self, epoch: int, train_loader, **kwargs):
        """Train one epoch. Can be overridden for custom training logic."""
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.cfg.epochs}")
        epoch_losses = {}

        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            batch = self._to_device(batch)

            # Forward pass (subclass defines how)
            with autocast(enabled=self.cfg.amp):
                outputs = self._forward_pass(batch, **kwargs)

                # Compute losses
                loss, loss_dict = self.loss_composer.compute(
                    batch=batch,
                    outputs=outputs,
                    device=self.device,
                    **kwargs
                )

            # Backward pass
            self.optimizer.zero_grad(set_to_none=True)

            if self.cfg.amp:
                self.scaler.scale(loss).backward()
                if self.cfg.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(
                        self._get_trainable_params(),
                        self.cfg.grad_clip
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.cfg.grad_clip > 0:
                    nn.utils.clip_grad_norm_(
                        self._get_trainable_params(),
                        self.cfg.grad_clip
                    )
                self.optimizer.step()

            if self.scheduler is not None:
                self.scheduler.step()

            # Logging
            self._log_losses(loss_dict, epoch_losses)
            pbar.set_postfix({k: f"{v:.4f}" for k, v in list(loss_dict.items())[:3]})

            self.global_step += 1

            # Cleanup
            del loss, outputs, batch
            torch.cuda.empty_cache()

        # Epoch summary
        self._log_epoch_summary(epoch, epoch_losses)

    def _eval_loop(self, loader, name: str, **kwargs) -> Dict[str, Any]:
        """Evaluation loop for validation/test."""
        print(f"\nRunning {name}...")

        model = self.get_model()
        model.eval()

        total_loss = 0.0
        loss_breakdown = {}

        with torch.no_grad():
            for batch in loader:
                batch = self._to_device(batch)

                outputs = self._forward_pass(batch, **kwargs)
                loss, breakdown = self.loss_composer.compute(
                    batch=batch,
                    outputs=outputs,
                    device=self.device,
                    **kwargs
                )

                total_loss += loss.item()
                for k, v in breakdown.items():
                    loss_breakdown[k] = loss_breakdown.get(k, 0.0) + v

        n = len(loader)
        avg_loss = total_loss / n
        avg_breakdown = {k: v/n for k, v in loss_breakdown.items()}

        print(f"{name.capitalize()} Loss: {avg_loss:.6f}")
        for k, v in list(avg_breakdown.items())[:5]:
            print(f"  {k}: {v:.6f}")

        model.train()
        return {f"{name}_loss": avg_loss, **avg_breakdown}

    @abstractmethod
    def _forward_pass(self, batch, **kwargs):
        """
        Execute forward pass through model(s).
        MUST be implemented by subclass.

        Returns:
            outputs dict containing model predictions
        """
        pass

    @abstractmethod
    def _get_trainable_params(self):
        """
        Get trainable parameters for gradient clipping.
        MUST be implemented by subclass.
        """
        pass

    # ---------------------------------------------------------------------------------
    # Helper Methods
    # ---------------------------------------------------------------------------------

    def _to_device(self, batch):
        """Move batch to device."""
        if isinstance(batch, dict):
            return {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}
        elif isinstance(batch, (list, tuple)):
            return [v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for v in batch]
        elif isinstance(batch, torch.Tensor):
            return batch.to(self.device)
        return batch

    def _log_losses(self, loss_dict: Dict[str, float], epoch_accumulator: Dict[str, List[float]]):
        """Log losses to tensorboard and accumulate for epoch summary."""
        if self.global_step % self.cfg.log_every == 0:
            for name, value in loss_dict.items():
                self.writer.add_scalar(f"train/{name}", value, self.global_step)

        for name, value in loss_dict.items():
            if name not in epoch_accumulator:
                epoch_accumulator[name] = []
            epoch_accumulator[name].append(value)

    def _log_epoch_summary(self, epoch: int, epoch_losses: Dict[str, List[float]]):
        """Log epoch summary."""
        summary = {k: sum(v)/len(v) for k, v in epoch_losses.items()}

        print(f"\n[Epoch {epoch+1}] Summary:")
        for name, value in summary.items():
            print(f"  {name}: {value:.4f}")
            self.writer.add_scalar(f"epoch/{name}", value, epoch+1)

    def info(self) -> Dict[str, Any]:
        """Trainer metadata."""
        return {
            "name": self.name,
            "uid": self.uid,
            "description": "Core training infrastructure",
            "device": str(self.device),
            "epochs": self.cfg.epochs,
            "batch_size": self.cfg.batch_size,
            "lr": self.cfg.lr
        }