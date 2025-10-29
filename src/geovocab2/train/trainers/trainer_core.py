"""
Core trainer class with composable loss system and data factory pattern.
Handles the training loop, optimization, and metric logging.
"""
from __future__ import annotations
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from typing import Dict, List, Callable, Optional, Any
from dataclasses import dataclass
from pathlib import Path

from geovocab2.train.train_base import TrainBase


@dataclass
class TrainerConfig:
    """Base configuration for trainer."""
    run_name: str = "training_run"
    out_dir: str = "./runs"

    # Training params
    epochs: int = 10
    batch_size: int = 32
    num_workers: int = 2

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
        total = torch.zeros((), device=kwargs.get('device', 'cuda'))
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


class Trainer:
    """
    Core trainer that orchestrates training loop.
    Accepts models, data factory, and loss composer.
    """

    def __init__(
            self,
            name: str,
            uid: str,
            cfg: TrainerConfig,
            data_factory: Callable[[], DataLoader],
            loss_composer: LossComposer,
            device: str = "cuda"
    ):
        super().__init__(name=name, uid=uid)
        self.cfg = cfg
        self.device = device

        # Data
        self.data_factory = data_factory
        self.loader = None

        # Losses
        self.loss_composer = loss_composer

        # Optimizer & scheduler (set by Arbitrator)
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None

        # AMP
        self.scaler = GradScaler(enabled=cfg.amp)

        # Logging
        self.writer = SummaryWriter(log_dir=str(Path(cfg.out_dir) / cfg.run_name))
        self.global_step = 0

    def set_optimizer(self, optimizer: torch.optim.Optimizer):
        """Set optimizer (called by Arbitrator)."""
        self.optimizer = optimizer

    def set_scheduler(self, scheduler: torch.optim.lr_scheduler._LRScheduler):
        """Set scheduler (called by Arbitrator)."""
        self.scheduler = scheduler

    def train(self, forward_fn: Callable, **forward_kwargs):
        """
        Main training loop.

        Args:
            forward_fn: Function that takes batch and returns model outputs
            **forward_kwargs: Additional kwargs passed to forward_fn and losses
        """
        if self.optimizer is None:
            raise RuntimeError("Optimizer not set. Call set_optimizer() first.")

        # Create data loader
        self.loader = self.data_factory()

        for epoch in range(self.cfg.epochs):
            self._train_epoch(epoch, forward_fn, **forward_kwargs)

        self.writer.close()

    def _train_epoch(self, epoch: int, forward_fn: Callable, **forward_kwargs):
        """Train one epoch."""
        pbar = tqdm(self.loader, desc=f"Epoch {epoch + 1}/{self.cfg.epochs}")
        epoch_losses = {}

        for batch_idx, batch in enumerate(pbar):
            # Move batch to device if needed
            batch = self._to_device(batch)

            # Forward pass with AMP
            with autocast(enabled=self.cfg.amp):
                # Get model outputs
                outputs = forward_fn(batch, **forward_kwargs)

                # Compute losses
                loss, loss_dict = self.loss_composer.compute(
                    batch=batch,
                    outputs=outputs,
                    device=self.device,
                    **forward_kwargs
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
        # Log to tensorboard
        if self.global_step % self.cfg.log_every == 0:
            for name, value in loss_dict.items():
                self.writer.add_scalar(f"train/{name}", value, self.global_step)

        # Accumulate for epoch summary
        for name, value in loss_dict.items():
            if name not in epoch_accumulator:
                epoch_accumulator[name] = []
            epoch_accumulator[name].append(value)

    def _log_epoch_summary(self, epoch: int, epoch_losses: Dict[str, List[float]]):
        """Log epoch summary."""
        summary = {k: sum(v) / len(v) for k, v in epoch_losses.items()}

        print(f"\n[Epoch {epoch + 1}] Summary:")
        for name, value in summary.items():
            print(f"  {name}: {value:.4f}")
            self.writer.add_scalar(f"epoch/{name}", value, epoch + 1)

    def _get_trainable_params(self):
        """Get all trainable parameters (override if needed)."""
        # This should be set by Arbitrator
        if hasattr(self, '_trainable_params'):
            return self._trainable_params
        raise RuntimeError("Trainable params not set by Arbitrator")

    def set_trainable_params(self, params):
        """Set trainable parameters for gradient clipping."""
        self._trainable_params = params