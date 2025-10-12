"""
David Configuration System - Add this to the top of david.py after imports

Follows ConfigBase pattern: Simple dataclass configs with minimal boilerplate.
"""

import json
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional
from geovocab2.core.config_base import ConfigBase


# ============================================================================
# DAVID CONFIGURATION
# ============================================================================

@dataclass
class DavidArchitectureConfig(ConfigBase):
    """
    Configuration for David's multi-scale crystal classifier architecture.

    Follows ConfigBase pattern: Just add fields, everything else is automatic.
    Uses dataclass for zero-boilerplate configuration.
    """

    # ConfigBase metadata
    name: str = "david_architecture"
    uid: str = "c.david.architecture"

    # Core architecture
    feature_dim: int = 512
    num_classes: int = 1000
    scales: List[int] = field(default_factory=lambda: [256, 512, 768, 1024])
    sharing_mode: str = "partial_shared"  # fully_shared, partial_shared, decoupled, hierarchical
    fusion_mode: str = "hierarchical_tree"  # attention, gated, hierarchical_tree, deep_efficiency, etc.

    # Projection head configuration
    use_belly: bool = True  # Use 2x expansion bottleneck in projection heads
    belly_expand: float = 2.0  # Expansion factor for belly (if use_belly=True)

    # Shared feature extraction (for FULLY_SHARED and PARTIAL_SHARED modes)
    shared_feature_dim: int = 768
    shared_layers: int = 2
    shared_dropout: float = 0.1

    # Fusion configuration
    fusion_temperature: float = 1.0  # Temperature for attention/gated fusion
    fusion_dropout: float = 0.1

    # Hierarchical tree gating (when fusion_mode="hierarchical_tree")
    tree_depth: int = 3

    # Deep efficiency gating (when fusion_mode="deep_efficiency")
    num_experts: int = 3
    compression_ratio: int = 4

    # Progressive training
    progressive_training: bool = True
    scale_warmup_epochs: Optional[Dict[int, int]] = None

    def __post_init__(self):
        """Auto-generate warmup schedule if not provided."""
        if self.scale_warmup_epochs is None and self.progressive_training:
            self.scale_warmup_epochs = {}
            for i, scale in enumerate(self.scales):
                self.scale_warmup_epochs[scale] = i * 3

    @classmethod
    def from_json(cls, path: str) -> 'DavidArchitectureConfig':
        """Load config from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        # Convert string keys back to ints for scale_warmup_epochs
        if 'scale_warmup_epochs' in data and data['scale_warmup_epochs']:
            data['scale_warmup_epochs'] = {
                int(k): v for k, v in data['scale_warmup_epochs'].items()
            }
        return cls(**data)

    def to_json(self, path: str):
        """Save config to JSON file."""
        data = self.to_dict()
        # Convert int keys to strings for JSON serialization
        if data.get('scale_warmup_epochs'):
            data['scale_warmup_epochs'] = {
                str(k): v for k, v in data['scale_warmup_epochs'].items()
            }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)


# ============================================================================
# PRESET CONFIGURATIONS
# ============================================================================

class DavidPresets:
    """
    Preset configurations for common use cases.
    Simple factory functions that return DavidArchitectureConfig instances.
    """

    @staticmethod
    def small_fast() -> DavidArchitectureConfig:
        """Small, fast model for prototyping."""
        return DavidArchitectureConfig(
            name="david_small_fast",
            uid="c.david.small_fast",
            feature_dim=512,
            scales=[256, 512],
            sharing_mode="fully_shared",
            fusion_mode="weighted_sum",
            use_belly=False,
            shared_feature_dim=512,
            shared_layers=1,
            progressive_training=False,
        )

    @staticmethod
    def balanced() -> DavidArchitectureConfig:
        """Balanced config - good accuracy/speed tradeoff."""
        return DavidArchitectureConfig(
            name="david_balanced",
            uid="c.david.balanced",
            feature_dim=512,
            scales=[256, 512, 768, 1024],
            sharing_mode="partial_shared",
            fusion_mode="hierarchical_tree",
            use_belly=True,
            progressive_training=True,
        )

    @staticmethod
    def high_accuracy() -> DavidArchitectureConfig:
        """Maximum accuracy configuration."""
        return DavidArchitectureConfig(
            name="david_high_accuracy",
            uid="c.david.high_accuracy",
            feature_dim=512,
            scales=[256, 512, 768, 1024, 1280],
            sharing_mode="decoupled",
            fusion_mode="deep_efficiency",
            use_belly=True,
            belly_expand=2.5,
            num_experts=5,
            compression_ratio=2,
            progressive_training=True,
            scale_warmup_epochs={256: 0, 512: 3, 768: 6, 1024: 9, 1280: 12}
        )

    @staticmethod
    def hierarchical_refinement() -> DavidArchitectureConfig:
        """Hierarchical mode - coarse to fine refinement."""
        return DavidArchitectureConfig(
            name="david_hierarchical",
            uid="c.david.hierarchical",
            feature_dim=512,
            scales=[256, 512, 768, 1024],
            sharing_mode="hierarchical",
            fusion_mode="progressive",
            use_belly=True,
            progressive_training=True,
        )

    @staticmethod
    def clip_vit_b16() -> DavidArchitectureConfig:
        """CLIP ViT-B/16 optimized config."""
        return DavidArchitectureConfig(
            name="david_clip_vit_b16",
            uid="c.david.clip_vit_b16",
            feature_dim=512,
            scales=[256, 512, 768, 1024],
            sharing_mode="partial_shared",
            fusion_mode="hierarchical_tree",
            use_belly=True,
            progressive_training=True,
        )

    @staticmethod
    def clip_vit_l14() -> DavidArchitectureConfig:
        """CLIP ViT-L/14 optimized config."""
        return DavidArchitectureConfig(
            name="david_clip_vit_l14",
            uid="c.david.clip_vit_l14",
            feature_dim=768,
            scales=[384, 768, 1024, 1280],
            sharing_mode="partial_shared",
            fusion_mode="deep_efficiency",
            use_belly=True,
            shared_feature_dim=1024,
            num_experts=4,
            progressive_training=True,
        )

    @staticmethod
    def clip_vit_h14() -> DavidArchitectureConfig:
        """CLIP ViT-H/14 optimized config."""
        return DavidArchitectureConfig(
            name="david_clip_vit_h14",
            uid="c.david.clip_vit_h14",
            feature_dim=1024,
            scales=[512, 768, 1024, 1536],
            sharing_mode="partial_shared",
            fusion_mode="deep_efficiency",
            use_belly=True,
            shared_feature_dim=1280,
            num_experts=5,
            compression_ratio=3,
            progressive_training=True,
        )


# ============================================================================
# MODIFIED DAVID CLASS WITH CONFIG SUPPORT
# ============================================================================

# Add this method to the David class (after __init__):

def __init__(
        self,
        feature_dim: Optional[int] = None,
        num_classes: Optional[int] = None,
        scales: Optional[List[int]] = None,
        sharing_mode: Optional[SharingMode] = None,
        fusion_mode: Optional[FusionMode] = None,
        shared_feature_dim: Optional[int] = None,
        shared_layers: Optional[int] = None,
        shared_dropout: Optional[float] = None,
        fusion_temperature: Optional[float] = None,
        fusion_dropout: Optional[float] = None,
        tree_depth: Optional[int] = None,
        num_experts: Optional[int] = None,
        compression_ratio: Optional[int] = None,
        progressive_training: Optional[bool] = None,
        scale_warmup_epochs: Optional[Dict[int, int]] = None,
        use_belly: Optional[bool] = None,
        belly_expand: Optional[float] = None,
        config: Optional[DavidArchitectureConfig] = None
):
    """
    Initialize David with multi-scale architecture.

    Can be initialized either:
    1. From explicit parameters (backward compatible)
    2. From a DavidArchitectureConfig object
    3. Mix of both (config provides defaults, explicit params override)

    Args:
        config: DavidArchitectureConfig object (if provided, serves as defaults)
        ... (all other parameters same as before)
    """
    super().__init__()

    # If config provided, use it as defaults
    if config is not None:
        feature_dim = feature_dim or config.feature_dim
        num_classes = num_classes or config.num_classes
        scales = scales or config.scales

        # Convert string modes to enums if needed
        if sharing_mode is None:
            sharing_mode = getattr(SharingMode, config.sharing_mode.upper())
        if fusion_mode is None:
            fusion_mode = getattr(FusionMode, config.fusion_mode.upper())

        shared_feature_dim = shared_feature_dim or config.shared_feature_dim
        shared_layers = shared_layers or config.shared_layers
        shared_dropout = shared_dropout or config.shared_dropout
        fusion_temperature = fusion_temperature or config.fusion_temperature
        fusion_dropout = fusion_dropout or config.fusion_dropout
        tree_depth = tree_depth or config.tree_depth
        num_experts = num_experts or config.num_experts
        compression_ratio = compression_ratio or config.compression_ratio
        progressive_training = progressive_training if progressive_training is not None else config.progressive_training
        scale_warmup_epochs = scale_warmup_epochs or config.scale_warmup_epochs
        use_belly = use_belly if use_belly is not None else config.use_belly
        belly_expand = belly_expand or config.belly_expand
    else:
        # Set defaults if no config and no explicit values
        feature_dim = feature_dim or 512
        num_classes = num_classes or 1000
        scales = scales or [256, 512, 768, 1024]
        sharing_mode = sharing_mode or SharingMode.PARTIAL_SHARED
        fusion_mode = fusion_mode or FusionMode.GATED
        shared_feature_dim = shared_feature_dim or 768
        shared_layers = shared_layers or 2
        shared_dropout = shared_dropout or 0.1
        fusion_temperature = fusion_temperature or 1.0
        fusion_dropout = fusion_dropout or 0.1
        tree_depth = tree_depth or 3
        num_experts = num_experts or 3
        compression_ratio = compression_ratio or 4
        progressive_training = progressive_training if progressive_training is not None else True
        scale_warmup_epochs = scale_warmup_epochs or {s: 0 for s in scales}
        use_belly = use_belly if use_belly is not None else True
        belly_expand = belly_expand or 2.0

    self.feature_dim = feature_dim
    self.num_classes = num_classes
    self.scales = scales
    self.sharing_mode = sharing_mode
    self.fusion_mode = fusion_mode
    self.progressive_training = progressive_training
    self.scale_warmup_epochs = scale_warmup_epochs
    self.use_belly = use_belly
    self.belly_expand = belly_expand

    # David's memory
    self.current_epoch = 0
    self.scale_accuracies = {s: [] for s in self.scales}

    # Build David's neural architecture
    self._build_architecture(
        shared_feature_dim=shared_feature_dim,
        shared_layers=shared_layers,
        shared_dropout=shared_dropout
    )

    # Build David's fusion strategy
    self._build_fusion(
        shared_feature_dim=shared_feature_dim,
        temperature=fusion_temperature,
        dropout=fusion_dropout,
        tree_depth=tree_depth,
        num_experts=num_experts,
        compression_ratio=compression_ratio
    )

    # David's confidence in each scale
    self.register_buffer(
        "scale_weights",
        torch.tensor([1.0 for _ in self.scales])
    )


# Add these class methods to David:

@classmethod
def from_config(cls, config: DavidArchitectureConfig) -> 'David':
    """
    Create David from a configuration object.

    Args:
        config: DavidArchitectureConfig instance

    Returns:
        Initialized David model

    Example:
        >>> config = DavidPresets.balanced()
        >>> david = David.from_config(config)
    """
    return cls(config=config)


@classmethod
def from_preset(cls, preset_name: str) -> 'David':
    """
    Create David from a named preset.

    Args:
        preset_name: One of 'small_fast', 'balanced', 'high_accuracy',
                    'hierarchical_refinement', 'clip_vit_b16', 'clip_vit_l14',
                    'clip_vit_h14'

    Returns:
        Initialized David model

    Example:
        >>> david = David.from_preset('balanced')
    """
    preset_map = {
        'small_fast': DavidPresets.small_fast,
        'balanced': DavidPresets.balanced,
        'high_accuracy': DavidPresets.high_accuracy,
        'hierarchical_refinement': DavidPresets.hierarchical_refinement,
        'clip_vit_b16': DavidPresets.clip_vit_b16,
        'clip_vit_l14': DavidPresets.clip_vit_l14,
        'clip_vit_h14': DavidPresets.clip_vit_h14,
    }

    if preset_name not in preset_map:
        raise ValueError(
            f"Unknown preset '{preset_name}'. "
            f"Available presets: {list(preset_map.keys())}"
        )

    config = preset_map[preset_name]()
    return cls.from_config(config)


def get_config(self) -> DavidArchitectureConfig:
    """
    Extract current configuration from the model.

    Returns:
        DavidArchitectureConfig representing current model configuration
    """
    return DavidArchitectureConfig(
        feature_dim=self.feature_dim,
        num_classes=self.num_classes,
        scales=self.scales,
        sharing_mode=self.sharing_mode.value,
        fusion_mode=self.fusion_mode.value,
        use_belly=self.use_belly,
        belly_expand=self.belly_expand,
        progressive_training=self.progressive_training,
        scale_warmup_epochs=self.scale_warmup_epochs,
    )


# Update _build_architecture to use self.use_belly and self.belly_expand:

def _build_architecture(self, shared_feature_dim: int,
                        shared_layers: int, shared_dropout: float):
    """Build David's processing architecture based on sharing mode."""

    if self.sharing_mode == SharingMode.FULLY_SHARED:
        # David shares everything - maximum parameter efficiency
        self.shared_extractor = SharedFeatureExtractor(
            self.feature_dim,
            shared_feature_dim,
            shared_layers,
            shared_dropout
        )
        self.heads = nn.ModuleDict({
            str(scale): ScaleSpecificHead(
                shared_feature_dim,
                scale,
                use_belly=self.use_belly,
                belly_expand=self.belly_expand
            )
            for scale in self.scales
        })

    elif self.sharing_mode == SharingMode.PARTIAL_SHARED:
        # David shares a base, then specializes - balanced approach
        self.shared_base = nn.Linear(self.feature_dim, shared_feature_dim)
        self.heads = nn.ModuleDict({
            str(scale): ScaleSpecificHead(
                shared_feature_dim,
                scale,
                use_belly=self.use_belly,
                belly_expand=self.belly_expand
            )
            for scale in self.scales
        })

    elif self.sharing_mode == SharingMode.DECOUPLED:
        # David keeps scales independent - maximum flexibility
        self.heads = nn.ModuleDict({
            str(scale): ScaleSpecificHead(
                self.feature_dim,
                scale,
                use_belly=self.use_belly,
                belly_expand=self.belly_expand
            )
            for scale in self.scales
        })

    elif self.sharing_mode == SharingMode.HIERARCHICAL:
        # David refines progressively - coarse to fine
        for i, scale in enumerate(self.scales):
            if i == 0:
                # First scale processes directly
                setattr(self, f'head_{scale}',
                        ScaleSpecificHead(
                            self.feature_dim, scale,
                            use_belly=self.use_belly,
                            belly_expand=self.belly_expand
                        ))
            else:
                # Later scales refine previous outputs
                prev_scale = self.scales[i - 1]
                setattr(self, f'refine_{scale}', nn.Sequential(
                    nn.Linear(prev_scale + self.feature_dim, scale),
                    nn.ReLU()
                ))
                setattr(self, f'head_{scale}',
                        ScaleSpecificHead(
                            scale, scale,
                            use_belly=self.use_belly,
                            belly_expand=self.belly_expand
                        ))


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    """
    Examples following ConfigBase pattern: Simple dataclass configs.
    No over-engineering, just add fields and use them.
    """

    print("=" * 80)
    print("David Configuration Examples (ConfigBase Pattern)")
    print("=" * 80)

    # Example 1: Using presets
    print("\n[1] Using Presets")
    print("-" * 80)
    config = DavidPresets.balanced()
    print(f"Preset: {config.name}")
    print(f"UID: {config.uid}")
    print(f"Scales: {config.scales}")
    print(f"Sharing: {config.sharing_mode}")
    print(f"Use belly: {config.use_belly}")

    # Example 2: Custom config
    print("\n[2] Custom Configuration")
    print("-" * 80)
    custom = DavidArchitectureConfig(
        name="my_custom_david",
        uid="c.david.custom",
        feature_dim=512,
        num_classes=100,  # CIFAR-100
        scales=[128, 256, 512],
        sharing_mode="decoupled",
        use_belly=True,
        belly_expand=2.5,
    )
    print(f"Custom config: {custom.name}")
    print(f"Num classes: {custom.num_classes}")
    print(f"Belly expand: {custom.belly_expand}")

    # Example 3: to_dict() - ConfigBase method
    print("\n[3] Dictionary Conversion (ConfigBase.to_dict())")
    print("-" * 80)
    config_dict = custom.to_dict()
    print(f"Keys: {list(config_dict.keys())}")
    print(f"Feature dim: {config_dict['feature_dim']}")
    print(f"Scales: {config_dict['scales']}")

    # Example 4: Reconstruct from dict
    print("\n[4] Reconstruct from Dictionary")
    print("-" * 80)
    reconstructed = DavidArchitectureConfig(**config_dict)
    print(f"Matches original: {reconstructed == custom}")
    print(f"Name: {reconstructed.name}")

    # Example 5: Save/load JSON
    print("\n[5] Save/Load JSON")
    print("-" * 80)
    import tempfile
    import os

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_path = f.name

    try:
        # Save
        custom.to_json(temp_path)
        print(f"Saved to: {temp_path}")

        # Load
        loaded = DavidArchitectureConfig.from_json(temp_path)
        print(f"Loaded: {loaded.name}")
        print(f"Matches original: {loaded == custom}")
    finally:
        os.unlink(temp_path)

    # Example 6: Progressive training warmup
    print("\n[6] Progressive Training Warmup (auto-generated)")
    print("-" * 80)
    progressive = DavidArchitectureConfig(
        scales=[256, 512, 768, 1024],
        progressive_training=True,
        # scale_warmup_epochs auto-generated in __post_init__
    )
    print(f"Warmup schedule: {progressive.scale_warmup_epochs}")

    # Example 7: Override defaults
    print("\n[7] Override Defaults")
    print("-" * 80)
    preset = DavidPresets.balanced()
    overridden = DavidArchitectureConfig(
        **{**preset.to_dict(), 'num_classes': 100, 'use_belly': False}
    )
    print(f"Original num_classes: {preset.num_classes}")
    print(f"Overridden num_classes: {overridden.num_classes}")
    print(f"Original use_belly: {preset.use_belly}")
    print(f"Overridden use_belly: {overridden.use_belly}")

    print("\n" + "=" * 80)
    print("✅ All examples completed!")
    print("=" * 80)
    print("\nConfigBase Pattern Benefits:")
    print("  • Zero boilerplate with @dataclass")
    print("  • Automatic __init__, __repr__, __eq__")
    print("  • Simple to_dict() via ConfigBase")
    print("  • Easy JSON serialization")
    print("  • Clean inheritance - just add fields")
    print("  • No over-engineering!")
    print("=" * 80)