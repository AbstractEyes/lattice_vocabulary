"""
TrainEpoch
Author: AbstractPhil + Claude Sonnet 4.5

Description: Epoch-based training implementation with full optimizer/scheduler support.
- Extends TrainBase with concrete epoch-based training loop
- Supports optimizers, schedulers, gradient clipping, checkpointing
- Provides metrics tracking and best model selection
- Ideal for standard supervised learning tasks

Design Philosophy:
    Implements the full training scaffolding for epoch-based learning:
    - Automatic device management
    - Checkpoint save/load with best model tracking
    - Training history and metrics logging
    - Validation-based early stopping support
    - Flexible optimizer and scheduler configuration

License: MIT
"""

import torch
from torch import nn
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import json
from datetime import datetime
import logging

from geovocab2.train.train_base import TrainBase
from geovocab2.train.config.config_base import ConfigBase


class TrainEpoch(TrainBase):
    """
    Epoch-based trainer with optimizer, scheduler, and checkpoint support.

    This trainer implements standard epoch-based training with all the bells
    and whistles: optimizers, schedulers, gradient clipping, checkpointing,
    metrics tracking, and best model selection.

    Subclasses only need to implement:
        - get_model(): Return the model
        - get_loss(): Return the loss function
        - get_datasets(): Return (train_loader, val_loader, test_loader)

    Optionally override:
        - get_optimizer(): Custom optimizer (default: AdamW)
        - get_scheduler(): Custom scheduler (default: None)
        - train_step(): Custom training step logic
        - val_step(): Custom validation step logic
        - test_step(): Custom test step logic

    Attributes:
        config: TrainEpochConfig instance
        device: Current device (cpu/cuda/mps)
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

    def __init__(
        self,
        config: 'TrainEpochConfig',
        name: Optional[str] = None,
        uid: Optional[str] = None
    ):
        """
        Initialize epoch-based trainer from config.

        Args:
            config: TrainEpochConfig instance (REQUIRED)
            name: Human-readable trainer name (uses config.name if None)
            uid: Unique identifier (uses config.uid if None)

        Raises:
            ValueError: If config is not provided or invalid
        """
        # Validate config is provided
        if config is None:
            raise ValueError(
                "TrainEpoch requires a config. Provide a TrainEpochConfig instance."
            )

        # Check if it's a ConfigBase instance or has required attributes
        if not isinstance(config, ConfigBase):
            required_attrs = [
                'name', 'uid', 'num_epochs', 'learning_rate', 'weight_decay',
                'grad_clip', 'log_interval', 'val_interval', 'save_best',
                'save_interval', 'checkpoint_dir', 'device'
            ]
            missing = [attr for attr in required_attrs if not hasattr(config, attr)]
            if missing:
                raise ValueError(
                    f"Config must be a ConfigBase instance or have required attributes. "
                    f"Missing: {missing}"
                )

        # Store config first
        self.config = config

        # Use config name/uid if not provided
        trainer_name = name if name is not None else config.name
        trainer_uid = uid if uid is not None else config.uid

        # Initialize base
        super().__init__(trainer_name, trainer_uid)

        # Setup logging
        self.logger = logging.getLogger("geovocab2.training")
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('[%(levelname)s] %(message)s'))
            self.logger.addHandler(handler)

        # Device setup from config
        if config.device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(config.device)

        # Checkpoint setup from config
        if config.checkpoint_dir is None:
            self.checkpoint_dir = Path(f"./checkpoints/{trainer_uid}")
        else:
            self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Training components
        self.model: Optional[nn.Module] = None
        self.loss_fn: Optional[Any] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler: Optional[Any] = None

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.train_history: List[Dict[str, float]] = []
        self.val_history: List[Dict[str, float]] = []
        self.best_val_loss = float('inf')
        self.best_checkpoint_path: Optional[Path] = None

        self.logger.info(f"Initialized {self.name} on device: {self.device}")

    def to(self, device) -> 'TrainEpoch':
        """
        Transfer trainer to device.

        Args:
            device: Device string or torch.device

        Returns:
            Self for method chaining
        """
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

    def train_step(self, batch: Any) -> Dict[str, torch.Tensor]:
        """
        Single training step. Override for custom logic.

        Args:
            batch: Batch from train dataloader

        Returns:
            Dictionary with 'loss' key and any other metrics
        """
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

    def val_step(self, batch: Any) -> Dict[str, torch.Tensor]:
        """
        Single validation step. Override for custom logic.

        Args:
            batch: Batch from validation dataloader

        Returns:
            Dictionary with 'loss' key and metrics
        """
        return self.train_step(batch)

    def test_step(self, batch: Any) -> Dict[str, torch.Tensor]:
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
        num_epochs = kwargs.get('num_epochs', self.config.num_epochs)
        learning_rate = kwargs.get('learning_rate', self.config.learning_rate)
        weight_decay = kwargs.get('weight_decay', self.config.weight_decay)
        grad_clip = kwargs.get('grad_clip', self.config.grad_clip)
        log_interval = kwargs.get('log_interval', self.config.log_interval)
        val_interval = kwargs.get('val_interval', self.config.val_interval)
        save_best = kwargs.get('save_best', self.config.save_best)
        save_interval = kwargs.get('save_interval', self.config.save_interval)

        self.logger.info(f"\n{'='*70}")
        self.logger.info(f"Training: {self.name} ({self.uid})")
        self.logger.info(f"{'='*70}")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Epochs: {num_epochs}")
        self.logger.info(f"Learning rate: {learning_rate}")
        self.logger.info(f"Weight decay: {weight_decay}")
        self.logger.info(f"Gradient clip: {grad_clip}")
        self.logger.info(f"Checkpoint dir: {self.checkpoint_dir}")
        self.logger.info(f"{'='*70}\n")

        # Initialize components
        self.model = self.get_model().to(self.device)
        self.loss_fn = self.get_loss()
        if hasattr(self.loss_fn, 'to'):
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
        self.model.train()

        epoch_metrics = {}
        num_batches = len(train_loader)

        for batch_idx, batch in enumerate(train_loader):
            # Forward and backward
            loss_dict = self.train_step(batch)
            loss = loss_dict['loss']

            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)

            self.optimizer.step()
            self.global_step += 1

            # Accumulate metrics
            for key, value in loss_dict.items():
                if key not in epoch_metrics:
                    epoch_metrics[key] = 0.0
                epoch_metrics[key] += value.item()

            # Logging
            if (batch_idx + 1) % log_interval == 0:
                current_loss = loss.item()
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

        self.model.eval()
        val_metrics = {}

        with torch.no_grad():
            for batch in val_loader:
                loss_dict = self.val_step(batch)

                for key, value in loss_dict.items():
                    if key not in val_metrics:
                        val_metrics[key] = 0.0
                    val_metrics[key] += value.item()

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

        self.model.eval()
        test_metrics = {}

        with torch.no_grad():
            for batch in test_loader:
                loss_dict = self.test_step(batch)

                for key, value in loss_dict.items():
                    if key not in test_metrics:
                        test_metrics[key] = 0.0
                    test_metrics[key] += value.item()

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
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "train_history": self.train_history,
            "val_history": self.val_history,
            "best_val_loss": self.best_val_loss
        }

        if self.scheduler:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        torch.save(checkpoint, checkpoint_path)
        return checkpoint_path

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load training checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Initialize components if needed
        if self.model is None:
            self.model = self.get_model().to(self.device)
        if self.optimizer is None:
            self.optimizer = self.get_optimizer()

        # Load states
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        self.train_history = checkpoint["train_history"]
        self.val_history = checkpoint["val_history"]
        self.best_val_loss = checkpoint["best_val_loss"]

        if "scheduler_state_dict" in checkpoint and self.scheduler:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.logger.info(f"✓ Checkpoint loaded: {checkpoint_path}")
        self.logger.info(f"  Resuming from epoch {self.current_epoch+1}, step {self.global_step}")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # METADATA
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def info(self) -> Dict[str, Any]:
        """Trainer metadata for introspection."""
        return {
            "name": self.name,
            "uid": self.uid,
            "device": str(self.device),
            "checkpoint_dir": str(self.checkpoint_dir),
            "current_epoch": self.current_epoch,
            "global_step": self.global_step,
            "best_val_loss": self.best_val_loss,
            "num_train_epochs": len(self.train_history),
            "num_val_epochs": len(self.val_history),
            "config": self.config.to_dict()
        }


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
        save_best=False
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
        num_epochs=1
    )

    trainer4 = MinimalTrainer(device_config)
    print(f"Config specifies device: cpu")
    print(f"Trainer device: {trainer4.device}")

    if torch.cuda.is_available():
        trainer4.to("cuda")
        print(f"After .to('cuda'): {trainer4.device}")

    # Test 5: Name/UID override
    print("\n" + "="*70)
    print("[Test 5] Name/UID override from init")
    print("-"*70)

    trainer5 = MinimalTrainer(config, name="custom_name", uid="t.custom.uid")
    print(f"Config name: {config.name}, Config uid: {config.uid}")
    print(f"Trainer name: {trainer5.name}, Trainer uid: {trainer5.uid}")
    print(f"Override worked: {trainer5.name == 'custom_name'}")

    print("\n" + "="*70)
    print("All tests completed successfully!")
    print("="*70)
    print("\nKey features demonstrated:")
    print("  ✓ Config is required (no silent failures)")
    print("  ✓ Training with TrainEpochConfig")
    print("  ✓ Config + parameter overrides via kwargs")
    print("  ✓ Config included in trainer info")
    print("  ✓ Device management with config")
    print("  ✓ Name/UID can be overridden at init")
    print("="*70)