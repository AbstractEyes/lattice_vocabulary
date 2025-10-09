"""
TrainBase
Author: AbstractPhil + Claude Sonnet 4.5

Description: Minimal ABC training contract for geometric lattice vocabulary training.
- Mirrors FormulaBase design: pure contract, shallow inheritance
- Subclasses implement all training logic
- Provides metadata structure only (name, uid)

Design Philosophy:
    Enforce a contract without imposing implementation. Subclasses define:
    1. get_model() - model architecture
    2. get_loss() - loss computation
    3. get_datasets() - data loading
    4. train() - training loop
    5. validate() - validation logic
    6. test() - test evaluation
    7. to() - device/dtype transfer

License: MIT
"""

from abc import ABC, abstractmethod
from typing import Any, Dict


class TrainBase(ABC):
    """
    Minimal training contract for geometric lattice vocabularies.

    This base class enforces a contract for training implementations while
    maintaining maximum flexibility. Subclasses implement all concrete logic.

    Attributes:
        name: Human-readable trainer name
        uid: Unique identifier for registry/logging (e.g., "t.simplex.volume")

    Methods:
        get_model(): Return model architecture
        get_loss(): Return loss function/module
        get_datasets(): Return data loaders
        train(): Execute training loop
        validate(): Execute validation pass
        test(): Execute test evaluation
        to(): Transfer trainer state to device/dtype
        info(): Trainer metadata for introspection

    Logging Recommendation:
        Subclasses should create their own logger in __init__:
            self.logger = logging.getLogger("geovocab2.training")
        Or use custom backends (wandb, tensorboard, mlflow, etc.)

    Example:
        class SimplexTrainer(TrainBase):
            def __init__(self):
                super().__init__("simplex_volume", "t.simplex.volume")
                self.device = torch.device("cpu")
                self.model = None
                self.loss_fn = None

            def to(self, device):
                self.device = torch.device(device)
                if self.model:
                    self.model = self.model.to(self.device)
                if self.loss_fn and hasattr(self.loss_fn, 'to'):
                    self.loss_fn = self.loss_fn.to(self.device)
                return self

            def get_model(self):
                return MyModel().to(self.device)

            def get_loss(self):
                loss = CayleyMengerExpanded()
                return loss.to(self.device) if hasattr(loss, 'to') else loss

            def get_datasets(self):
                return train_loader, val_loader, test_loader

            def train(self, **kwargs):
                pass

            def validate(self, **kwargs):
                pass

            def test(self, **kwargs):
                pass
    """

    def __init__(self, name: str, uid: str):
        """
        Initialize trainer with identifying metadata.

        Args:
            name: Human-readable trainer name
            uid: Unique identifier (recommend hierarchical format like "t.category.name")
        """
        self.name = name
        self.uid = uid

    @abstractmethod
    def get_model(self) -> Any:
        """
        Return the model to train.

        Returns:
            Model architecture (typically torch.nn.Module)

        Note:
            Subclass determines device placement, initialization, etc.
        """
        pass

    @abstractmethod
    def get_loss(self) -> Any:
        """
        Return the loss function or module.

        Returns:
            Loss function/module (often a FormulaBase subclass)

        Note:
            Can return callable, nn.Module, or FormulaBase instance.
        """
        pass

    @abstractmethod
    def get_datasets(self) -> Any:
        """
        Return datasets or dataloaders for training.

        Returns:
            Data structures needed for training (loaders, datasets, etc.)

        Note:
            Return format is subclass-defined. Common patterns:
            - Tuple[DataLoader, DataLoader, DataLoader] for train/val/test
            - Dict[str, DataLoader] for named splits
            - Single DataLoader if only training needed
        """
        pass

    @abstractmethod
    def train(self, *args, **kwargs) -> Any:
        """
        Execute training loop.

        Args:
            *args: Training-specific positional arguments
            **kwargs: Training-specific keyword arguments

        Returns:
            Training results (subclass-defined format)

        Note:
            Subclass defines all training logic: epochs, optimization,
            checkpointing, logging, etc.
        """
        pass

    @abstractmethod
    def validate(self, *args, **kwargs) -> Any:
        """
        Execute validation pass.

        Args:
            *args: Validation-specific positional arguments
            **kwargs: Validation-specific keyword arguments

        Returns:
            Validation results (subclass-defined format)

        Note:
            Subclass defines validation logic and metrics.
        """
        pass

    @abstractmethod
    def test(self, *args, **kwargs) -> Any:
        """
        Execute test evaluation.

        Args:
            *args: Test-specific positional arguments
            **kwargs: Test-specific keyword arguments

        Returns:
            Test results (subclass-defined format)

        Note:
            Subclass defines test logic and metrics.
        """
        pass

    @abstractmethod
    def to(self, *args, **kwargs) -> 'TrainBase':
        """
        Transfer trainer state to device/dtype.

        Args:
            *args: Typically device (e.g., 'cuda', 'cpu', torch.device)
            **kwargs: Optional dtype, memory_format, etc.

        Returns:
            Self for method chaining

        Note:
            Subclass should handle:
            - Moving model to device
            - Moving loss function to device (if applicable)
            - Storing device for future tensor allocation
            - Any other stateful components

        Example:
            def to(self, device):
                self.device = torch.device(device)
                if self.model:
                    self.model = self.model.to(self.device)
                if self.loss_fn and hasattr(self.loss_fn, 'to'):
                    self.loss_fn = self.loss_fn.to(self.device)
                return self
        """
        pass

    def info(self) -> Dict[str, Any]:
        """
        Metadata about the trainer for introspection and cataloging.

        Subclasses can override to provide rich metadata about training
        configuration, hyperparameters, model architecture, etc.

        Returns:
            Dictionary containing trainer metadata with at least:
                - name: Trainer name
                - uid: Unique identifier
                - description: Human-readable description

        Example:
            def info(self) -> Dict[str, Any]:
                return {
                    "name": self.name,
                    "uid": self.uid,
                    "description": "Trains simplex volume prediction",
                    "model": "GeometricTransformer",
                    "loss": "CayleyMengerExpanded",
                    "optimizer": "AdamW",
                    "device": str(self.device)
                }
        """
        return {
            "name": self.name,
            "uid": self.uid,
            "description": "No description provided"
        }

    def __repr__(self) -> str:
        """
        String representation for debugging and logging.

        Returns:
            Human-readable string identifying the trainer instance
        """
        return f"{self.__class__.__name__}(name='{self.name}', uid='{self.uid}')"

    def __str__(self) -> str:
        """
        User-friendly string representation.

        Returns:
            Trainer name and class
        """
        return f"Trainer[{self.name}] ({self.__class__.__name__})"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# EXAMPLE USAGE AND TESTING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    import logging
    import torch

    # Example: Minimal trainer implementation with device management
    class MinimalTrainer(TrainBase):
        """Example trainer showing minimal contract fulfillment."""

        def __init__(self):
            super().__init__("minimal_example", "t.example.minimal")
            # Subclass handles logging setup
            self.logger = logging.getLogger("geovocab2.training")
            self.logger.setLevel(logging.INFO)
            if not self.logger.handlers:
                handler = logging.StreamHandler()
                handler.setFormatter(logging.Formatter('[%(levelname)s] %(message)s'))
                self.logger.addHandler(handler)

            # Device management
            self.device = torch.device("cpu")
            self.model = None
            self.loss_fn = None

        def to(self, device):
            """Transfer trainer to device."""
            self.device = torch.device(device)
            self.logger.info(f"Moving trainer to device: {self.device}")

            if self.model is not None:
                self.model = self.model.to(self.device)
                self.logger.info(f"  Model moved to {self.device}")

            if self.loss_fn is not None and hasattr(self.loss_fn, 'to'):
                self.loss_fn = self.loss_fn.to(self.device)
                self.logger.info(f"  Loss function moved to {self.device}")

            return self

        def get_model(self):
            """Return a dummy model."""
            import torch.nn as nn
            self.logger.info("Creating model: Linear(10, 1)")
            return nn.Linear(10, 1).to(self.device)

        def get_loss(self):
            """Return a dummy loss."""
            import torch.nn as nn
            self.logger.info("Creating loss: MSELoss")
            return nn.MSELoss()

        def get_datasets(self):
            """Return dummy data loaders."""
            from torch.utils.data import DataLoader, TensorDataset

            self.logger.info("Creating datasets (100 samples, batch_size=10)")

            dummy_data = torch.randn(100, 10)
            dummy_labels = torch.randn(100, 1)
            dataset = TensorDataset(dummy_data, dummy_labels)

            return (
                DataLoader(dataset, batch_size=10),
                DataLoader(dataset, batch_size=10),
                DataLoader(dataset, batch_size=10)
            )

        def train(self, epochs=5):
            """Minimal training loop."""
            self.logger.info(f"Starting training for {epochs} epochs")

            self.model = self.get_model()
            self.loss_fn = self.get_loss()
            train_loader, _, _ = self.get_datasets()

            import torch.optim as optim
            optimizer = optim.Adam(self.model.parameters())

            for epoch in range(epochs):
                total_loss = 0.0
                for batch_idx, (data, target) in enumerate(train_loader):
                    # Move batch to device
                    data = data.to(self.device)
                    target = target.to(self.device)

                    optimizer.zero_grad()
                    output = self.model(data)
                    loss = self.loss_fn(output, target)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

                avg_loss = total_loss / len(train_loader)
                self.logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.6f}")

            return {"final_loss": avg_loss}

        def validate(self):
            """Minimal validation."""
            self.logger.info("Running validation...")
            _, val_loader, _ = self.get_datasets()

            self.model.eval()
            total_loss = 0.0

            with torch.no_grad():
                for data, target in val_loader:
                    data = data.to(self.device)
                    target = target.to(self.device)
                    output = self.model(data)
                    loss = self.loss_fn(output, target)
                    total_loss += loss.item()

            avg_loss = total_loss / len(val_loader)
            self.logger.info(f"Validation Loss: {avg_loss:.6f}")
            return {"val_loss": avg_loss}

        def test(self):
            """Minimal testing."""
            self.logger.info("Running test evaluation...")
            _, _, test_loader = self.get_datasets()

            self.model.eval()
            total_loss = 0.0

            with torch.no_grad():
                for data, target in test_loader:
                    data = data.to(self.device)
                    target = target.to(self.device)
                    output = self.model(data)
                    loss = self.loss_fn(output, target)
                    total_loss += loss.item()

            avg_loss = total_loss / len(test_loader)
            self.logger.info(f"Test Loss: {avg_loss:.6f}")
            return {"test_loss": avg_loss}

        def info(self):
            """Rich metadata."""
            return {
                "name": self.name,
                "uid": self.uid,
                "description": "Minimal example trainer with device management",
                "model": "Linear(10, 1)",
                "loss": "MSELoss",
                "optimizer": "Adam",
                "device": str(self.device)
            }

    # Test the example trainer
    print("="*70)
    print("TrainBase Example: MinimalTrainer with Device Management")
    print("="*70)

    trainer = MinimalTrainer()
    print(f"\n{trainer}")

    print("\nTrainer Info (CPU):")
    info = trainer.info()
    for key, value in info.items():
        print(f"  {key}: {value}")

    # Test device transfer
    print("\n" + "-"*70)
    print("Testing device transfer...")
    print("-"*70)

    # Initialize model on CPU
    trainer.model = trainer.get_model()
    print(f"Model initialized on: {next(trainer.model.parameters()).device}")

    # Move to CUDA if available
    if torch.cuda.is_available():
        trainer.to("cuda")
        print(f"After .to('cuda'): {next(trainer.model.parameters()).device}")
        trainer.to("cpu")
        print(f"After .to('cpu'): {next(trainer.model.parameters()).device}")
    else:
        print("CUDA not available, skipping GPU test")

    print("\n" + "-"*70)
    print("Running training on CPU...")
    print("-"*70)
    trainer.train(epochs=2)
    trainer.validate()

    print("\n" + "="*70)
    print("Example completed successfully!")
    print("="*70)