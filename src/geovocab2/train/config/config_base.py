"""
ConfigBase
Author: AbstractPhil + Claude Sonnet 4.5

Description: Minimal dataclass configuration container for training and formula systems.
- Uses Python dataclasses for zero-boilerplate configuration
- Provides name/uid metadata plus automatic dictionary conversion
- Subclasses just add fields - no methods required

Design Philosophy:
    Maximum simplicity using dataclasses. Subclasses only need to:
    1. Add @dataclass decorator
    2. Define their fields with defaults
    That's it. Everything else is automatic.

Do's:
    - Ignore my config, make your own any time you want
    - Stick to the ConfigBase pattern and everything falls into place, mine or not.
    - Use @dataclass for all configs
    - Serialize if you want
    - Just add fields, no methods needed
    - Use type hints for all fields
    - Use Optional[T] for nullable fields, pytorch likes that
    - Use as_dict() for dictionary conversion, it works until it doesn't.
        - If it doesn't - override or replace it. It's not rocket science.

Don't:
    - Let AI tell you to use complex libraries
    - Add unnecessary dependencies
    - Over-engineer with metaclasses or custom serialization
    - Add custom __init__, __repr__, __eq__ methods
    - Manually implement dictionary conversion
    - Use complex inheritance hierarchies
    - Use enums or other complex types unless serializable

This is Python, you have everything you need. Don't let AI build you into a corner.

License: MIT
"""

from dataclasses import dataclass, asdict, field
from typing import Optional, Dict, Any


@dataclass
class BaseConfig:
    """
    Minimal dataclass configuration container.

    Provides name/uid metadata and automatic dictionary conversion.
    Subclasses inherit all dataclass functionality automatically.

    Attributes:
        name: Human-readable configuration name
        uid: Unique identifier for registry/logging

    Methods:
        to_dict(): Convert to dictionary (uses dataclasses.asdict)

    Example:
        @dataclass
        class TrainingConfig(ConfigBase):
            name: str = "training_config"
            uid: str = "c.training.default"
            learning_rate: float = 1e-3
            batch_size: int = 32
            num_epochs: int = 100
            grad_clip: Optional[float] = None

        # Usage
        config = TrainingConfig(learning_rate=1e-4, batch_size=64)
        print(config.learning_rate)  # 1e-4
        config_dict = config.to_dict()  # Full dictionary
    """

    name: str = "config_base"
    uid: str = "c.base"

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.

        Returns:
            Dictionary containing all configuration fields
        """
        return asdict(self)


ConfigBase = BaseConfig  # Alias for convenience


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# EXAMPLE USAGE AND TESTING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    import json


    # Example 1: Training configuration
    @dataclass
    class TrainingConfig(BaseConfig):
        """Training configuration with optimizer parameters."""
        name: str = "training_config"
        uid: str = "c.training.default"
        learning_rate: float = 1e-3
        batch_size: int = 32
        num_epochs: int = 100
        weight_decay: float = 0.0
        grad_clip: Optional[float] = None


    # Example 2: Formula configuration
    @dataclass
    class FormulaConfig(BaseConfig):
        """Formula configuration for Cayley-Menger."""
        name: str = "cayley_menger_config"
        uid: str = "c.formula.cayley"
        target_volume: float = 1.0
        loss_type: str = "l2"
        eps: float = 1e-10
        validate_input: bool = True


    # Example 3: Model configuration
    @dataclass
    class ModelConfig(BaseConfig):
        """Model architecture configuration."""
        name: str = "transformer_config"
        uid: str = "c.model.transformer"
        vocab_size: int = 1000
        embed_dim: int = 128
        num_heads: int = 8
        num_layers: int = 6
        dropout: float = 0.1


    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # TEST SUITE
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    print("=" * 70)
    print("ConfigBase Example: Dataclass Configurations")
    print("=" * 70)

    # Test 1: TrainingConfig
    print("\n[1] TrainingConfig - Basic usage")
    print("-" * 70)
    config = TrainingConfig(learning_rate=1e-4, batch_size=64)
    print(f"Config: {config}")
    print(f"\nAccess fields:")
    print(f"  learning_rate: {config.learning_rate}")
    print(f"  batch_size: {config.batch_size}")
    print(f"  num_epochs: {config.num_epochs}")

    # Test 2: to_dict()
    print("\n[2] Dictionary conversion")
    print("-" * 70)
    config_dict = config.to_dict()
    print(json.dumps(config_dict, indent=2))

    # Test 3: Instantiation from dict
    print("\n[3] Instantiation from dictionary")
    print("-" * 70)
    new_config = TrainingConfig(**config_dict)
    print(f"Loaded: {new_config}")
    print(f"Matches original: {new_config == config}")

    # Test 4: Modify and compare
    print("\n[4] Modification and comparison")
    print("-" * 70)
    modified = TrainingConfig(learning_rate=1e-4, batch_size=64, num_epochs=200)
    print(f"Original epochs: {config.num_epochs}")
    print(f"Modified epochs: {modified.num_epochs}")
    print(f"Configs equal: {config == modified}")

    # Test 5: FormulaConfig
    print("\n[5] FormulaConfig")
    print("-" * 70)
    formula_config = FormulaConfig(target_volume=0.5, loss_type="l1")
    print(f"Config: {formula_config}")
    print(f"Dictionary: {json.dumps(formula_config.to_dict(), indent=2)}")

    # Test 6: ModelConfig
    print("\n[6] ModelConfig")
    print("-" * 70)
    model_config = ModelConfig(embed_dim=256, num_layers=12)
    print(f"Config: {model_config}")
    print(f"Embed dim: {model_config.embed_dim}")
    print(f"Num layers: {model_config.num_layers}")

    # Test 7: Default values
    print("\n[7] Default values")
    print("-" * 70)
    default_config = TrainingConfig()
    print(f"Default config: {default_config}")
    print(f"Default learning_rate: {default_config.learning_rate}")
    print(f"Default batch_size: {default_config.batch_size}")

    # Test 8: Nested usage pattern
    print("\n[8] Practical usage pattern")
    print("-" * 70)


    @dataclass
    class ExperimentConfig(BaseConfig):
        """Complete experiment configuration."""
        name: str = "experiment_config"
        uid: str = "c.experiment.default"

        # Sub-configs (as dicts for simplicity)
        training: Dict[str, Any] = field(default_factory=lambda: TrainingConfig().to_dict())
        model: Dict[str, Any] = field(default_factory=lambda: ModelConfig().to_dict())
        formula: Dict[str, Any] = field(default_factory=lambda: FormulaConfig().to_dict())


    experiment = ExperimentConfig()
    print(f"Experiment config created")
    print(f"Training LR: {experiment.training['learning_rate']}")
    print(f"Model embed_dim: {experiment.model['embed_dim']}")
    print(f"Formula target: {experiment.formula['target_volume']}")

    print("\n" + "=" * 70)
    print("All examples completed successfully!")
    print("=" * 70)
    print("\nKey takeaways:")
    print("  • Zero boilerplate - just add fields")
    print("  • Automatic __init__, __repr__, __eq__")
    print("  • Type hints built-in")
    print("  • Dictionary conversion with to_dict()")
    print("  • Simple inheritance - just extend and add fields")
    print("=" * 70)