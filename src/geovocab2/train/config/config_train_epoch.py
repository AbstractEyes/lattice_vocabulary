"""
TrainEpochConfig
Author: AbstractPhil + Claude Sonnet 4.5

Description: Dataclass configuration for TrainEpoch training.
- Provides all parameters needed for epoch-based training
- Clean defaults for standard training scenarios
- Easy to override and pass to TrainEpoch.train()

Design Philosophy:
    Simple dataclass with all TrainEpoch.train() parameters.
    Use as-is or extend for custom training configurations.

License: MIT
"""

from dataclasses import dataclass
from typing import Optional
from geovocab2.train.config.config_base import BaseConfig


@dataclass
class TrainEpochConfig(BaseConfig):
    """
    Configuration for TrainEpoch training.

    Contains all parameters for the TrainEpoch.train() method plus
    trainer initialization parameters.

    Attributes:
        # Metadata
        name: Configuration name
        uid: Configuration unique identifier

        # Trainer initialization
        checkpoint_dir: Directory for saving checkpoints
        device: Device string ('cpu', 'cuda', 'mps', or None for auto)

        # Training loop parameters
        num_epochs: Number of training epochs
        learning_rate: Initial learning rate
        weight_decay: L2 regularization weight
        grad_clip: Gradient clipping threshold (None = no clipping)
        log_interval: Log metrics every N steps
        val_interval: Run validation every N epochs
        save_best: Save checkpoint when validation improves
        save_interval: Save checkpoint every N epochs (None = disabled)

    Example:
        # Create config with defaults
        config = TrainEpochConfig(
            name="my_training",
            uid="c.train.my_experiment",
            num_epochs=100,
            learning_rate=1e-3
        )

        # Pass to trainer
        trainer = MyTrainer()
        trainer.train(**config.to_dict())

        # Or extract specific params
        trainer.train(
            num_epochs=config.num_epochs,
            learning_rate=config.learning_rate,
            grad_clip=config.grad_clip
        )
    """

    # Metadata
    name: str = "train_epoch_config"
    uid: str = "c.train.epoch"

    # Trainer initialization
    checkpoint_dir: Optional[str] = None
    device: Optional[str] = None

    # Training loop parameters
    num_epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    grad_clip: Optional[float] = None
    log_interval: int = 10
    val_interval: int = 1
    save_best: bool = True
    save_interval: Optional[int] = None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# EXAMPLE USAGE AND TESTING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    import json

    print("=" * 70)
    print("TrainEpochConfig Example")
    print("=" * 70)

    # Example 1: Default configuration
    print("\n[1] Default configuration")
    print("-" * 70)
    default_config = TrainEpochConfig()
    print(f"Config: {default_config}")
    print(f"\nKey parameters:")
    print(f"  num_epochs: {default_config.num_epochs}")
    print(f"  learning_rate: {default_config.learning_rate}")
    print(f"  grad_clip: {default_config.grad_clip}")
    print(f"  save_best: {default_config.save_best}")

    # Example 2: Custom configuration
    print("\n[2] Custom configuration")
    print("-" * 70)
    custom_config = TrainEpochConfig(
        name="simplex_training",
        uid="c.train.simplex",
        num_epochs=200,
        learning_rate=1e-4,
        weight_decay=1e-5,
        grad_clip=1.0,
        checkpoint_dir="./checkpoints/simplex",
        device="cuda",
        val_interval=5,
        save_interval=50
    )
    print(f"Config: {custom_config}")

    # Example 3: Dictionary conversion
    print("\n[3] Dictionary conversion")
    print("-" * 70)
    config_dict = custom_config.to_dict()
    print(json.dumps(config_dict, indent=2))

    # Example 4: Loading from dictionary
    print("\n[4] Loading from dictionary")
    print("-" * 70)
    loaded_config = TrainEpochConfig(**config_dict)
    print(f"Loaded: {loaded_config}")
    print(f"Matches original: {loaded_config == custom_config}")

    # Example 5: Usage pattern with trainer
    print("\n[5] Usage pattern with trainer")
    print("-" * 70)

    # Simulate trainer usage
    print("# Create config")
    print("config = TrainEpochConfig(")
    print("    name='my_experiment',")
    print("    num_epochs=100,")
    print("    learning_rate=1e-3,")
    print("    grad_clip=1.0")
    print(")")
    print()
    print("# Pass to trainer")
    print("trainer = MyTrainer()")
    print("trainer.train(**config.to_dict())")
    print()
    print("# Or use specific fields")
    print("trainer.train(")
    print("    num_epochs=config.num_epochs,")
    print("    learning_rate=config.learning_rate")
    print(")")

    # Example 6: Multiple configurations for experiments
    print("\n[6] Multiple configurations for experiments")
    print("-" * 70)

    configs = [
        TrainEpochConfig(
            name="experiment_1",
            uid="c.train.exp1",
            learning_rate=1e-3,
            grad_clip=1.0
        ),
        TrainEpochConfig(
            name="experiment_2",
            uid="c.train.exp2",
            learning_rate=1e-4,
            grad_clip=0.5
        ),
        TrainEpochConfig(
            name="experiment_3",
            uid="c.train.exp3",
            learning_rate=1e-3,
            weight_decay=1e-5,
            grad_clip=1.0
        )
    ]

    print("Created 3 experiment configurations:")
    for i, cfg in enumerate(configs, 1):
        print(f"  {i}. {cfg.name}: lr={cfg.learning_rate}, "
              f"wd={cfg.weight_decay}, clip={cfg.grad_clip}")

    # Example 7: Extract training parameters only
    print("\n[7] Extract training parameters only")
    print("-" * 70)

    # Get only the parameters needed for train() method
    train_params = {
        k: v for k, v in custom_config.to_dict().items()
        if k not in ['name', 'uid', 'checkpoint_dir', 'device']
    }
    print("Parameters for trainer.train():")
    print(json.dumps(train_params, indent=2))

    # Example 8: Presets for common scenarios
    print("\n[8] Common configuration presets")
    print("-" * 70)

    # Fast training
    fast_config = TrainEpochConfig(
        name="fast_training",
        uid="c.train.fast",
        num_epochs=10,
        learning_rate=1e-2,
        log_interval=5,
        val_interval=1
    )
    print(f"Fast: {fast_config.num_epochs} epochs, lr={fast_config.learning_rate}")

    # Careful training
    careful_config = TrainEpochConfig(
        name="careful_training",
        uid="c.train.careful",
        num_epochs=500,
        learning_rate=1e-4,
        weight_decay=1e-6,
        grad_clip=0.5,
        val_interval=5,
        save_interval=100
    )
    print(f"Careful: {careful_config.num_epochs} epochs, lr={careful_config.learning_rate}")

    # Production training
    production_config = TrainEpochConfig(
        name="production_training",
        uid="c.train.production",
        num_epochs=1000,
        learning_rate=1e-3,
        weight_decay=1e-5,
        grad_clip=1.0,
        val_interval=10,
        save_best=True,
        save_interval=100,
        checkpoint_dir="./checkpoints/production"
    )
    print(f"Production: {production_config.num_epochs} epochs, lr={production_config.learning_rate}")

    print("\n" + "=" * 70)
    print("All examples completed successfully!")
    print("=" * 70)
    print("\nKey features:")
    print("  • All TrainEpoch.train() parameters included")
    print("  • Sensible defaults for quick prototyping")
    print("  • Easy dictionary conversion for passing to trainer")
    print("  • Simple to create multiple experiment configs")
    print("=" * 70)