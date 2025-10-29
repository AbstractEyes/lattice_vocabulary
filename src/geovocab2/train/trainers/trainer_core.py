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
