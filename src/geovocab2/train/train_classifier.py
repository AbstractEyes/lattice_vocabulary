"""
TrainSimpleClassifier
Author: AbstractPhil + Claude Sonnet 4.5

Description: Complete training example for SimpleClassifier using TrainEpoch.
- Demonstrates full training pipeline
- Uses synthetic data for reproducibility
- Educational example with intentional gotcha for students to debug

Design Philosophy:
    Students should be able to run this immediately, but results won't be
    perfect. They need to investigate and fix the issues - learning by doing.

Note to Students:
    This code runs without errors, but you might notice the training isn't
    converging as well as it should. Can you figure out why?
    Hint: Check the data preprocessing and hyperparameters carefully!

License: MIT
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from geovocab2.train.train_epoch import TrainEpoch
from geovocab2.train.config.config_train_epoch import TrainEpochConfig
from geovocab2.train.model.simple_classifier import SimpleClassifier


class TrainSimpleClassifier(TrainEpoch):
    """
    Trainer for SimpleClassifier demonstrating complete training pipeline.

    Creates synthetic multiclass data and trains a classifier.
    Includes all necessary components: model, loss, data, optimizer.

    Note:
        This implementation has a subtle issue that prevents optimal training.
        Students should run it, observe the results, and debug!

    Args:
        config: TrainEpochConfig instance

    Example:
        # Create config
        config = TrainEpochConfig(
            name="classifier_training",
            uid="c.train.classifier",
            num_epochs=20,
            learning_rate=0.01,  # Hmm, is this right?
            checkpoint_dir="./checkpoints/classifier"
        )

        # Create and run trainer
        trainer = TrainSimpleClassifier(config)
        results = trainer.train()

        # Check results
        print(f"Final validation accuracy: {results['final_val_acc']:.2%}")
    """

    def __init__(self, config):
        """Initialize trainer with config."""
        super().__init__(config)

        # Data parameters (can be overridden by subclass)
        self.num_samples = 1000
        self.num_features = 20
        self.num_classes = 5
        self.batch_size = 32

    def get_model(self):
        """Create SimpleClassifier model."""
        return SimpleClassifier(
            input_dim=self.num_features,
            hidden_dims=[64, 32],
            num_classes=self.num_classes,
            dropout=0.3  # A bit high, but should be okay... right?
        )

    def get_loss(self):
        """Return CrossEntropyLoss for classification."""
        return nn.CrossEntropyLoss()

    def get_datasets(self):
        """
        Generate synthetic classification data.

        Creates separable clusters with some noise for each class.
        """
        # Set seed for reproducibility
        np.random.seed(42)
        torch.manual_seed(42)

        # Generate synthetic data
        X_train, y_train = self._generate_data(self.num_samples)
        X_val, y_val = self._generate_data(self.num_samples // 5)
        X_test, y_test = self._generate_data(self.num_samples // 5)

        # SUSPICIOUS: No normalization here...
        # Most people normalize their data before training, but maybe it's fine?

        # Create datasets
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.LongTensor(y_train)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val),
            torch.LongTensor(y_val)
        )
        test_dataset = TensorDataset(
            torch.FloatTensor(X_test),
            torch.LongTensor(y_test)
        )

        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False
        )

        return train_loader, val_loader, test_loader

    def _generate_data(self, num_samples):
        """
        Generate synthetic classification data.

        Creates clusters in feature space for each class.
        """
        X = []
        y = []

        samples_per_class = num_samples // self.num_classes

        for class_idx in range(self.num_classes):
            # Create cluster center
            center = np.random.randn(self.num_features) * 5

            # Generate samples around center with noise
            class_samples = center + np.random.randn(
                samples_per_class,
                self.num_features
            ) * 2  # Noise scale

            X.append(class_samples)
            y.extend([class_idx] * samples_per_class)

        X = np.vstack(X)
        y = np.array(y)

        # Shuffle
        indices = np.random.permutation(len(y))
        X = X[indices]
        y = y[indices]

        return X, y

    def train_step(self, batch):
        """
        Custom training step with accuracy tracking.

        Args:
            batch: (data, labels) tuple

        Returns:
            Dictionary with loss and accuracy
        """
        data, labels = batch

        # Forward pass
        logits = self.model(data)
        loss = self.loss_fn(logits, labels)

        # Calculate accuracy
        predictions = torch.argmax(logits, dim=-1)
        accuracy = (predictions == labels).float().mean()

        return {
            "loss": loss,
            "accuracy": accuracy
        }

    def val_step(self, batch):
        """Validation step with accuracy tracking."""
        return self.train_step(batch)

    def validate(self, val_loader=None):
        """Enhanced validation with accuracy reporting."""
        metrics = super().validate(val_loader)

        # Log accuracy if available
        if "accuracy" in metrics:
            self.logger.info(f"  Validation Accuracy: {metrics['accuracy']:.2%}")

        return metrics


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FIXED VERSION (For reference)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TrainSimpleClassifierFixed(TrainSimpleClassifier):
    """
    Fixed version of the trainer with proper preprocessing.

    Students: Compare this to the original. What's different?
    """

    def get_datasets(self):
        """Generate and PROPERLY NORMALIZE data."""
        # Generate data
        train_loader, val_loader, test_loader = super().get_datasets()

        # Extract data for normalization
        train_data = []
        train_labels = []
        for batch_data, batch_labels in train_loader:
            train_data.append(batch_data)
            train_labels.append(batch_labels)

        train_data = torch.cat(train_data, dim=0)
        train_labels = torch.cat(train_labels, dim=0)

        # Compute normalization statistics from training data
        mean = train_data.mean(dim=0, keepdim=True)
        std = train_data.std(dim=0, keepdim=True) + 1e-8  # Avoid division by zero

        # Normalize ALL datasets using training statistics
        def normalize_loader(loader, mean, std):
            data_list = []
            labels_list = []
            for batch_data, batch_labels in loader:
                normalized_data = (batch_data - mean) / std
                data_list.append(normalized_data)
                labels_list.append(batch_labels)

            data = torch.cat(data_list, dim=0)
            labels = torch.cat(labels_list, dim=0)

            dataset = TensorDataset(data, labels)
            return DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=isinstance(loader.dataset, TensorDataset)
            )

        # Create normalized loaders
        train_loader_norm = normalize_loader(train_loader, mean, std)
        val_loader_norm = normalize_loader(val_loader, mean, std)
        test_loader_norm = normalize_loader(test_loader, mean, std)

        return train_loader_norm, val_loader_norm, test_loader_norm


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN EXECUTION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    print("=" * 70)
    print("TrainSimpleClassifier: Complete Training Example")
    print("=" * 70)

    # Create config
    config = TrainEpochConfig(
        name="classifier_training",
        uid="c.train.classifier",
        num_epochs=15,
        learning_rate=0.01,  # Students: Is this optimal?
        weight_decay=0.0,
        grad_clip=None,
        log_interval=5,
        val_interval=1,
        save_best=True,
        save_interval=None,
        checkpoint_dir="./checkpoints/classifier",
        device=None  # Auto-detect
    )

    print("\n[Training Original Version]")
    print("-" * 70)
    print("Running training with potential issues...")
    print("Students: Pay attention to the validation accuracy!")
    print()

    # Train original version
    trainer = TrainSimpleClassifier(config)
    results = trainer.train()

    print("\n" + "=" * 70)
    print("Original Training Complete")
    print("=" * 70)
    print(f"Best validation loss: {results['best_val_loss']:.4f}")
    print(f"Final train loss: {results['final_train_loss']:.4f}")
    print(f"Final val loss: {results['final_val_loss']:.4f}")

    # Test the trained model
    _, _, test_loader = trainer.get_datasets()
    test_results = trainer.test(test_loader)
    print(f"Test accuracy: {test_results['accuracy']:.2%}")

    print("\n" + "=" * 70)
    print("Challenge for Students:")
    print("=" * 70)
    print("The code above runs without errors, but the results aren't great.")
    print("Can you figure out what's wrong?")
    print()
    print("Things to investigate:")
    print("  1. Is the learning rate appropriate?")
    print("  2. Is the data properly preprocessed?")
    print("  3. Are the model hyperparameters optimal?")
    print("  4. Is the dropout rate too high?")
    print()
    print("Try running TrainSimpleClassifierFixed to see the difference!")
    print("=" * 70)

    # Optionally run fixed version for comparison
    print("\n[Training Fixed Version for Comparison]")
    print("-" * 70)

    # Use lower learning rate for fixed version
    fixed_config = TrainEpochConfig(
        name="classifier_training_fixed",
        uid="c.train.classifier.fixed",
        num_epochs=15,
        learning_rate=0.001,  # Better learning rate
        weight_decay=1e-5,  # Add regularization
        grad_clip=1.0,  # Add gradient clipping
        log_interval=5,
        val_interval=1,
        save_best=True,
        save_interval=None,
        checkpoint_dir="./checkpoints/classifier_fixed",
        device=None
    )

    trainer_fixed = TrainSimpleClassifierFixed(fixed_config)
    results_fixed = trainer_fixed.train()

    print("\n" + "=" * 70)
    print("Fixed Training Complete")
    print("=" * 70)
    print(f"Best validation loss: {results_fixed['best_val_loss']:.4f}")
    print(f"Final train loss: {results_fixed['final_train_loss']:.4f}")
    print(f"Final val loss: {results_fixed['final_val_loss']:.4f}")

    # Test the fixed model
    _, _, test_loader_fixed = trainer_fixed.get_datasets()
    test_results_fixed = trainer_fixed.test(test_loader_fixed)
    print(f"Test accuracy: {test_results_fixed['accuracy']:.2%}")

    print("\n" + "=" * 70)
    print("Comparison:")
    print("=" * 70)
    print(f"Original test accuracy:  {test_results['accuracy']:.2%}")
    print(f"Fixed test accuracy:     {test_results_fixed['accuracy']:.2%}")
    print(
        f"Improvement:             {(test_results_fixed['accuracy'] - test_results['accuracy']) * 100:.1f} percentage points")
    print()
    print("Key fixes:")
    print("  ✓ Added data normalization (zero mean, unit variance)")
    print("  ✓ Reduced learning rate (0.01 → 0.001)")
    print("  ✓ Added weight decay for regularization")
    print("  ✓ Added gradient clipping for stability")
    print("=" * 70)