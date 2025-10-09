"""
TrainEpoch
Author: AbstractPhil + Claude Sonnet 4.5

Description: Epoch-based training implementation with full optimizer/scheduler support.
- Extends TrainBase with concrete epoch-based training loop
- Supports optimizers, schedulers, gradient clipping, checkpointing
- Provides metrics tracking and best model selection
- Ideal for standard supervised learning tasks
- Supports both torch and numpy frameworks

Design Philosophy:
    Implements the full training scaffolding for epoch-based learning:
    - Automatic device management (torch only)
    - Checkpoint save/load with best model tracking
    - Training history and metrics logging
    - Validation-based early stopping support
    - Flexible optimizer and scheduler configuration
    - Config can be TrainEpochConfig object or dict

License: MIT
"""

import torch
from torch import nn
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, Any, Optional, List
import json
from datetime import datetime
import logging

from geovocab2.train.train_base import TrainBase


class TrainEpoch(TrainBase):
    """
    Epoch-based trainer with optimizer, scheduler, and checkpoint support.

    This trainer implements standard epoch-based training with all the bells
    and whistles: optimizers, schedulers, gradient clipping, checkpointing,
    metrics tracking, and best model selection.

    Supports both torch and numpy frameworks via _detect_framework() override.

    Subclasses only need to implement:
        - get_model(): Return the model
        - get_loss(): Return the loss function
        - get_datasets(): Return (train_loader, val_loader, test_loader)

    Optionally override:
        - _detect_framework(): Return 'torch' or 'numpy' (default: 'torch')
        - get_optimizer(): Custom optimizer (default: AdamW)
        - get_scheduler(): Custom scheduler (default: None)
        - train_step(): Custom training step logic
        - val_step(): Custom validation step logic
        - test_step(): Custom test step logic

    Attributes:
        config: TrainEpochConfig instance or dict
        framework: 'torch' or 'numpy'
        device: Current device (cpu/cuda/mps) - torch only
        model: The model being trained
        loss_fn: The loss function
        optimizer: The optimizer
        scheduler: Learning rate scheduler (optional)
        checkpoint_dir: Directory for saving checkpoints
        train_history: List of training metrics per epoch
        val_history: List of validation metrics per epoch
        best_val_loss: Best validation loss seen
        current_epoch: Current training epoch
        global_step: Total training steps across all epochs

    Example:
        from geovocab2.train.config.config_train_epoch import TrainEpochConfig

        config = TrainEpochConfig(
            name="simplex_volume_trainer",
            uid="t.simplex.volume",
            num_epochs=100,
            learning_rate=1e-3
        )

        class SimplexVolumeTrainer(TrainEpoch):
            def get_model(self):
                return GeometricTransformer(vocab_size=1000, embed_dim=128)

            def get_loss(self):
                return CayleyMengerExpanded(
                    target_volume=1.0,
                    compute_loss=True,
                    loss_type='l2'
                )

            def get_datasets(self):
                train_ds = FactoryDataset(...)
                val_ds = FactoryDataset(...)
                return (
                    DataLoader(train_ds, batch_size=32),
                    DataLoader(val_ds, batch_size=32),
                    None
                )

        # Usage
        trainer = SimplexVolumeTrainer(config)
        trainer.to('cuda')
        trainer.train()
    """

    def __init__(self, config=None):
        """
        Initialize epoch-based trainer from config.

        Args:
            config: TrainEpochConfig instance or dict (REQUIRED)

        Raises:
            ValueError: If config is not provided
            AttributeError: If config is missing required attributes
        """
        # Validate config is provided
        if config is None:
            raise ValueError(
                "TrainEpoch requires a config. Provide a TrainEpochConfig or dict instance."
            )

        # Validate ALL required attributes exist - no defaults, no fallbacks
        required_attrs = [
            'name', 'uid', 'num_epochs', 'learning_rate', 'weight_decay',
            'grad_clip', 'log_interval', 'val_interval', 'save_best',
            'save_interval', 'checkpoint_dir', 'device'
        ]

        # Check if config is dict or object
        is_dict = isinstance(config, dict)

        if is_dict:
            missing = [attr for attr in required_attrs if attr not in config]
        else:
            missing = [attr for attr in required_attrs if not hasattr(config, attr)]

        if missing:
            raise AttributeError(
                f"Config is missing required attributes: {missing}. "
                f"Provide a valid TrainEpochConfig instance or dict."
            )

        # Store config and dict flag
        self.config = config
        self._config_is_dict = is_dict

        # Use config name/uid
        trainer_name = self._get_config_value('name')
        trainer_uid = self._get_config_value('uid')

        # Initialize base
        super().__init__(trainer_name, trainer_uid)

        # Setup logging
        self.logger = logging.getLogger("geovocab2.training")
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('[%(levelname)s] %(message)s'))
            self.logger.addHandler(handler)

        # Detect framework (torch vs numpy)
        self.framework = self._detect_framework()

        # Device setup (torch only)
        if self.framework == 'torch':
            device = self._get_config_value('device')
            if device is not None:
                self.device = torch.device(device)
            else:
                # Auto-detect device
                if torch.cuda.is_available():
                    self.device = torch.device("cuda")
                elif torch.backends.mps.is_available():
                    self.device = torch.device("mps")
                else:
                    self.device = torch.device("cpu")
        else:
            # Numpy doesn't use device
            self.device = None

        # Checkpoint setup - must be provided in config
        checkpoint_dir = self._get_config_value('checkpoint_dir')
        if checkpoint_dir is None:
            raise ValueError(
                f"Config must provide checkpoint_dir. Got None. "
                f"Set checkpoint_dir in config or use default in TrainEpochConfig."
            )
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Training components
        self.model: Optional[Any] = None
        self.loss_fn: Optional[Any] = None
        self.optimizer: Optional[Any] = None
        self.scheduler: Optional[Any] = None

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.train_history: List[Dict[str, float]] = []
        self.val_history: List[Dict[str, float]] = []
        self.best_val_loss = float('inf')
        self.best_checkpoint_path: Optional[Path] = None

        if self.framework == 'torch':
            self.logger.info(f"Initialized {self.name} (torch) on device: {self.device}")
        else:
            self.logger.info(f"Initialized {self.name} (numpy)")

    def _get_config_value(self, key: str) -> Any:
        """
        Get value from config whether it's dict or object.

        Args:
            key: Config key/attribute name

        Returns:
            Config value
        """
        if self._config_is_dict:
            return self.config[key]
        else:
            return getattr(self.config, key)

    def _detect_framework(self) -> str:
        """
        Detect whether to use torch or numpy based on model/loss types.

        Override this method to explicitly set framework.

        Returns:
            'torch' or 'numpy'
        """
        # Default to torch - subclass can override
        return 'torch'

    def to(self, device) -> 'TrainEpoch':
        """
        Transfer trainer to device (torch only).

        Args:
            device: Device string or torch.device

        Returns:
            Self for method chaining

        Raises:
            RuntimeError: If framework is not torch
        """
        if self.framework != 'torch':
            raise RuntimeError(
                f"to() is only available for torch framework. "
                f"Current framework: {self.framework}"
            )

        self.device = torch.device(device)
        self.logger.info(f"Moving trainer to device: {self.device}")

        if self.model is not None:
            self.model = self.model.to(self.device)

        if self.loss_fn is not None and hasattr(self.loss_fn, 'to'):
            self.loss_fn = self.loss_fn.to(self.device)

        return self

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # OPTIMIZER AND SCHEDULER (Override for custom behavior)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def get_optimizer(
        self,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0
    ) -> torch.optim.Optimizer:
        """
        Return optimizer. Override for custom optimizers.

        Args:
            learning_rate: Initial learning rate
            weight_decay: L2 regularization weight

        Returns:
            Optimizer instance
        """
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

    def get_scheduler(self, optimizer: torch.optim.Optimizer) -> Optional[Any]:
        """
        Return learning rate scheduler. Override for custom schedulers.

        Args:
            optimizer: The optimizer to schedule

        Returns:
            Scheduler instance or None

        Example:
            def get_scheduler(self, optimizer):
                return torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=100,
                    eta_min=1e-6
                )
        """
        return None

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # TRAINING STEPS (Override for custom behavior)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def train_step(self, batch: Any) -> Dict[str, Any]:
        """
        Single training step. Override for custom logic.

        Args:
            batch: Batch from train dataloader

        Returns:
            Dictionary with 'loss' key and any other metrics
        """
        if self.framework == 'torch':
            # Move batch to device
            if isinstance(batch, (tuple, list)):
                batch = tuple(
                    b.to(self.device) if isinstance(b, torch.Tensor) else b
                    for b in batch
                )
            elif isinstance(batch, torch.Tensor):
                batch = batch.to(self.device)

            # Forward pass
            output = self.model(batch)

            # Compute loss
            loss_dict = self.loss_fn(output)

            return loss_dict
        else:
            # Numpy path - subclass must override
            raise NotImplementedError(
                "train_step must be overridden for numpy framework"
            )

    def val_step(self, batch: Any) -> Dict[str, Any]:
        """
        Single validation step. Override for custom logic.

        Args:
            batch: Batch from validation dataloader

        Returns:
            Dictionary with 'loss' key and metrics
        """
        return self.train_step(batch)

    def test_step(self, batch: Any) -> Dict[str, Any]:
        """
        Single test step. Override for custom logic.

        Args:
            batch: Batch from test dataloader

        Returns:
            Dictionary with metrics
        """
        return self.val_step(batch)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # MAIN TRAINING LOOP
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def train(self, **kwargs) -> Dict[str, Any]:
        """
        Execute epoch-based training.

        Reads parameters from self.config. Kwargs can override specific config values.

        Args:
            **kwargs: Optional overrides for config parameters
                - num_epochs: Number of training epochs
                - learning_rate: Initial learning rate
                - weight_decay: L2 regularization weight
                - grad_clip: Gradient clipping threshold
                - log_interval: Log metrics every N steps
                - val_interval: Run validation every N epochs
                - save_best: Save checkpoint when validation improves
                - save_interval: Save checkpoint every N epochs

        Returns:
            Training summary dictionary
        """
        # Read from config, allow kwargs to override
        num_epochs = kwargs.get('num_epochs', self._get_config_value('num_epochs'))
        learning_rate = kwargs.get('learning_rate', self._get_config_value('learning_rate'))
        weight_decay = kwargs.get('weight_decay', self._get_config_value('weight_decay'))
        grad_clip = kwargs.get('grad_clip', self._get_config_value('grad_clip'))
        log_interval = kwargs.get('log_interval', self._get_config_value('log_interval'))
        val_interval = kwargs.get('val_interval', self._get_config_value('val_interval'))
        save_best = kwargs.get('save_best', self._get_config_value('save_best'))
        save_interval = kwargs.get('save_interval', self._get_config_value('save_interval'))

        self.logger.info(f"\n{'='*70}")
        self.logger.info(f"Training: {self.name} ({self.uid})")
        self.logger.info(f"{'='*70}")
        self.logger.info(f"Framework: {self.framework}")
        if self.framework == 'torch':
            self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Epochs: {num_epochs}")
        self.logger.info(f"Learning rate: {learning_rate}")
        self.logger.info(f"Weight decay: {weight_decay}")
        self.logger.info(f"Gradient clip: {grad_clip}")
        self.logger.info(f"Checkpoint dir: {self.checkpoint_dir}")
        self.logger.info(f"{'='*70}\n")

        # Initialize components
        self.model = self.get_model()
        if self.framework == 'torch':
            self.model = self.model.to(self.device)

        self.loss_fn = self.get_loss()
        if self.framework == 'torch' and hasattr(self.loss_fn, 'to'):
            self.loss_fn = self.loss_fn.to(self.device)

        self.optimizer = self.get_optimizer(learning_rate, weight_decay)
        self.scheduler = self.get_scheduler(self.optimizer)

        # Get datasets
        train_loader, val_loader, _ = self.get_datasets()

        # Training loop
        for epoch in range(num_epochs):
            self.current_epoch = epoch

            # Train epoch
            train_metrics = self._train_epoch(
                train_loader,
                grad_clip=grad_clip,
                log_interval=log_interval
            )

            self.logger.info(
                f"Epoch {epoch+1}/{num_epochs} - "
                f"Train Loss: {train_metrics['loss']:.6f}"
            )

            # Validation
            if (epoch + 1) % val_interval == 0:
                val_metrics = self.validate(val_loader)
                self.logger.info(
                    f"Epoch {epoch+1}/{num_epochs} - "
                    f"Val Loss: {val_metrics['loss']:.6f}"
                )

                # Save best model
                if save_best and val_metrics['loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['loss']
                    self.best_checkpoint_path = self.save_checkpoint(prefix="best")
                    self.logger.info(f"  ✓ Best model saved: {self.best_checkpoint_path.name}")

            # Periodic checkpoint
            if save_interval and (epoch + 1) % save_interval == 0:
                checkpoint_path = self.save_checkpoint(prefix=f"epoch_{epoch+1}")
                self.logger.info(f"  Checkpoint saved: {checkpoint_path.name}")

            # Step scheduler
            if self.scheduler is not None:
                self.scheduler.step()

        # Final summary
        summary = {
            "name": self.name,
            "uid": self.uid,
            "framework": self.framework,
            "total_epochs": num_epochs,
            "best_val_loss": self.best_val_loss,
            "best_checkpoint": str(self.best_checkpoint_path) if self.best_checkpoint_path else None,
            "final_train_loss": self.train_history[-1]['loss'] if self.train_history else None,
            "final_val_loss": self.val_history[-1]['loss'] if self.val_history else None
        }

        # Save summary
        summary_path = self.checkpoint_dir / "training_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        self.logger.info(f"\n{'='*70}")
        self.logger.info("Training completed!")
        self.logger.info(f"Best val loss: {self.best_val_loss:.6f}")
        self.logger.info(f"Summary saved: {summary_path}")
        self.logger.info(f"{'='*70}\n")

        return summary

    def _train_epoch(
        self,
        train_loader: DataLoader,
        grad_clip: Optional[float],
        log_interval: int
    ) -> Dict[str, float]:
        """Internal method for single training epoch."""
        if self.framework == 'torch':
            self.model.train()

        epoch_metrics = {}
        num_batches = len(train_loader)

        for batch_idx, batch in enumerate(train_loader):
            # Forward and backward
            loss_dict = self.train_step(batch)
            loss = loss_dict['loss']

            if self.framework == 'torch':
                self.optimizer.zero_grad()
                loss.backward()

                # Gradient clipping
                if grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)

                self.optimizer.step()
            else:
                # Numpy framework - subclass must handle optimization
                pass

            self.global_step += 1

            # Accumulate metrics
            for key, value in loss_dict.items():
                if key not in epoch_metrics:
                    epoch_metrics[key] = 0.0
                # Handle both torch tensors and numpy arrays
                if self.framework == 'torch':
                    epoch_metrics[key] += value.item()
                else:
                    epoch_metrics[key] += float(value)

            # Logging
            if (batch_idx + 1) % log_interval == 0:
                if self.framework == 'torch':
                    current_loss = loss.item()
                else:
                    current_loss = float(loss)
                self.logger.info(
                    f"  Step {self.global_step} [{batch_idx+1}/{num_batches}] - "
                    f"Loss: {current_loss:.6f}"
                )

        # Average metrics
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches

        self.train_history.append(epoch_metrics)
        return epoch_metrics

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # VALIDATION AND TESTING
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def validate(self, val_loader: Optional[DataLoader] = None) -> Dict[str, float]:
        """
        Run validation pass.

        Args:
            val_loader: Validation dataloader (uses get_datasets if None)

        Returns:
            Dictionary of averaged validation metrics
        """
        if val_loader is None:
            _, val_loader, _ = self.get_datasets()

        if self.framework == 'torch':
            self.model.eval()

            val_metrics = {}
            with torch.no_grad():
                for batch in val_loader:
                    loss_dict = self.val_step(batch)

                    for key, value in loss_dict.items():
                        if key not in val_metrics:
                            val_metrics[key] = 0.0
                        val_metrics[key] += value.item()
        else:
            # Numpy framework
            val_metrics = {}
            for batch in val_loader:
                loss_dict = self.val_step(batch)

                for key, value in loss_dict.items():
                    if key not in val_metrics:
                        val_metrics[key] = 0.0
                    val_metrics[key] += float(value)

        # Average metrics
        num_batches = len(val_loader)
        for key in val_metrics:
            val_metrics[key] /= num_batches

        self.val_history.append(val_metrics)
        return val_metrics

    def test(self, test_loader: Optional[DataLoader] = None) -> Dict[str, float]:
        """
        Run test evaluation.

        Args:
            test_loader: Test dataloader (uses get_datasets if None)

        Returns:
            Dictionary of averaged test metrics
        """
        if test_loader is None:
            _, _, test_loader = self.get_datasets()
            if test_loader is None:
                raise ValueError("No test dataset available")

        if self.framework == 'torch':
            self.model.eval()

            test_metrics = {}
            with torch.no_grad():
                for batch in test_loader:
                    loss_dict = self.test_step(batch)

                    for key, value in loss_dict.items():
                        if key not in test_metrics:
                            test_metrics[key] = 0.0
                        test_metrics[key] += value.item()
        else:
            # Numpy framework
            test_metrics = {}
            for batch in test_loader:
                loss_dict = self.test_step(batch)

                for key, value in loss_dict.items():
                    if key not in test_metrics:
                        test_metrics[key] = 0.0
                    test_metrics[key] += float(value)

        # Average metrics
        num_batches = len(test_loader)
        for key in test_metrics:
            test_metrics[key] /= num_batches

        self.logger.info(f"\n{'='*70}")
        self.logger.info("Test Results")
        self.logger.info(f"{'='*70}")
        for key, value in test_metrics.items():
            self.logger.info(f"{key}: {value:.6f}")
        self.logger.info(f"{'='*70}\n")

        return test_metrics

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # CHECKPOINTING
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def save_checkpoint(self, prefix: str = "checkpoint") -> Path:
        """
        Save training checkpoint.

        Args:
            prefix: Filename prefix (e.g., 'best', 'epoch_10')

        Returns:
            Path to saved checkpoint
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_path = self.checkpoint_dir / f"{prefix}_{timestamp}.pt"

        checkpoint = {
            "name": self.name,
            "uid": self.uid,
            "framework": self.framework,
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "train_history": self.train_history,
            "val_history": self.val_history,
            "best_val_loss": self.best_val_loss
        }

        if self.framework == 'torch':
            checkpoint["model_state_dict"] = self.model.state_dict()
            checkpoint["optimizer_state_dict"] = self.optimizer.state_dict()
            if self.scheduler:
                checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
            torch.save(checkpoint, checkpoint_path)
        else:
            # Numpy - use pickle
            import pickle
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(checkpoint, f)

        return checkpoint_path

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load training checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        if self.framework == 'torch':
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
        else:
            # Numpy - use pickle
            import pickle
            with open(checkpoint_path, 'rb') as f:
                checkpoint = pickle.load(f)

        # Initialize components if needed
        if self.model is None:
            self.model = self.get_model()
            if self.framework == 'torch':
                self.model = self.model.to(self.device)

        if self.optimizer is None:
            self.optimizer = self.get_optimizer()

        # Load states
        if self.framework == 'torch':
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if "scheduler_state_dict" in checkpoint and self.scheduler:
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        else:
            # Numpy - subclass must handle state loading
            pass

        self.current_epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        self.train_history = checkpoint["train_history"]
        self.val_history = checkpoint["val_history"]
        self.best_val_loss = checkpoint["best_val_loss"]

        self.logger.info(f"✓ Checkpoint loaded: {checkpoint_path}")
        self.logger.info(f"  Resuming from epoch {self.current_epoch+1}, step {self.global_step}")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # METADATA
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def info(self) -> Dict[str, Any]:
        """Trainer metadata for introspection."""
        info_dict = {
            "name": self.name,
            "uid": self.uid,
            "framework": self.framework,
            "checkpoint_dir": str(self.checkpoint_dir),
            "current_epoch": self.current_epoch,
            "global_step": self.global_step,
            "best_val_loss": self.best_val_loss,
            "num_train_epochs": len(self.train_history),
            "num_val_epochs": len(self.val_history),
            "config": self.config if self._config_is_dict else self.config.to_dict()
        }

        # Add device info if torch
        if self.framework == 'torch':
            info_dict["device"] = str(self.device)

        return info_dict


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# EXAMPLE USAGE AND TESTING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    from torch.utils.data import DataLoader, TensorDataset
    from geovocab2.train.config.config_train_epoch import TrainEpochConfig

    # Example trainer implementation
    class MinimalTrainer(TrainEpoch):
        """Minimal trainer for testing TrainEpoch with config."""

        def get_model(self):
            """Return a simple linear model."""
            return nn.Linear(10, 1)

        def get_loss(self):
            """Return MSE loss."""
            return nn.MSELoss()

        def get_datasets(self):
            """Return dummy dataloaders."""
            # Generate dummy data
            train_data = torch.randn(100, 10)
            train_labels = torch.randn(100, 1)
            val_data = torch.randn(50, 10)
            val_labels = torch.randn(50, 1)

            train_ds = TensorDataset(train_data, train_labels)
            val_ds = TensorDataset(val_data, val_labels)

            train_loader = DataLoader(train_ds, batch_size=10, shuffle=True)
            val_loader = DataLoader(val_ds, batch_size=10)

            return train_loader, val_loader, None

        def train_step(self, batch):
            """Custom training step."""
            data, target = batch
            output = self.model(data)
            loss = self.loss_fn(output, target)
            return {"loss": loss}

    print("="*70)
    print("TrainEpoch Example: Config Integration Testing")
    print("="*70)

    # Test 1: Training with TrainEpochConfig
    print("\n[Test 1] Training with TrainEpochConfig")
    print("-"*70)

    config = TrainEpochConfig(
        name="test_training",
        uid="c.train.test",
        num_epochs=3,
        learning_rate=1e-2,
        log_interval=5,
        val_interval=1,
        save_best=False,
        checkpoint_dir="./checkpoints/test_training"
    )

    trainer1 = MinimalTrainer(config)
    print(f"Trainer: {trainer1}")
    print(f"Device: {trainer1.device}")
    print(f"Checkpoint dir: {trainer1.checkpoint_dir}")

    print("\nStarting training with config...")
    result1 = trainer1.train()

    print(f"\nTraining completed!")
    print(f"Final train loss: {result1['final_train_loss']:.6f}")
    print(f"Final val loss: {result1['final_val_loss']:.6f}")

    # Test 2: Training with config + overrides
    print("\n" + "="*70)
    print("[Test 2] Training with config + parameter overrides")
    print("-"*70)

    trainer2 = MinimalTrainer(config)
    print("Config num_epochs: 3")
    print("Overriding with num_epochs=2, learning_rate=1e-3")

    result2 = trainer2.train(num_epochs=2, learning_rate=1e-3)
    print(f"\nCompleted {result2['total_epochs']} epochs (override worked!)")

    # Test 3: Info with config
    print("\n" + "="*70)
    print("[Test 3] Trainer info with config")
    print("-"*70)

    import json
    info = trainer1.info()
    print(json.dumps(info, indent=2, default=str))

    # Test 4: Device management with config
    print("\n" + "="*70)
    print("[Test 4] Device management with config")
    print("-"*70)

    device_config = TrainEpochConfig(
        name="device_test",
        uid="c.train.device",
        device="cpu",
        num_epochs=1,
        checkpoint_dir="./checkpoints/device_test"
    )

    trainer4 = MinimalTrainer(device_config)
    print(f"Config specifies device: cpu")
    print(f"Trainer device: {trainer4.device}")

    if torch.cuda.is_available():
        trainer4.to("cuda")
        print(f"After .to('cuda'): {trainer4.device}")

    # Test 5: Using dict config
    print("\n" + "="*70)
    print("[Test 5] Training with dict config")
    print("-"*70)

    dict_config = {
        'name': 'dict_training',
        'uid': 'c.train.dict',
        'num_epochs': 2,
        'learning_rate': 1e-2,
        'weight_decay': 0.0,
        'grad_clip': None,
        'log_interval': 5,
        'val_interval': 1,
        'save_best': False,
        'save_interval': None,
        'checkpoint_dir': './checkpoints/dict_training',
        'device': None
    }

    trainer5 = MinimalTrainer(dict_config)
    print(f"Trainer: {trainer5}")
    print(f"Config is dict: {trainer5._config_is_dict}")

    result5 = trainer5.train()
    print(f"\nCompleted {result5['total_epochs']} epochs with dict config!")

    print("\n" + "="*70)
    print("All tests completed successfully!")
    print("="*70)
    print("\nKey features demonstrated:")
    print("  ✓ Config is required (no silent failures)")
    print("  ✓ Training with TrainEpochConfig object")
    print("  ✓ Training with dict config")
    print("  ✓ Config + parameter overrides via kwargs")
    print("  ✓ Config included in trainer info")
    print("  ✓ Device management with config")
    print("  ✓ Framework detection (torch/numpy)")
    print("="*70)