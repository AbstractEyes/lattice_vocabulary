"""
David Configuration System
===========================
Configuration and presets for David multi-scale crystal classifier.
Separated from model implementation for clean architecture.

Should be placed at: geovocab2/train/config/david_config.py

Author: AbstractPhil
"""

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional
from pathlib import Path


# ============================================================================
# ENUMS
# ============================================================================

class SharingMode(Enum):
    """Parameter sharing strategies across scales."""
    FULLY_SHARED = "fully_shared"
    PARTIAL_SHARED = "partial_shared"
    DECOUPLED = "decoupled"
    HIERARCHICAL = "hierarchical"


class FusionMode(Enum):
    """Multi-scale prediction fusion strategies."""
    WEIGHTED_SUM = "weighted_sum"
    ATTENTION = "attention"
    GATED = "gated"
    HIERARCHICAL_TREE = "hierarchical_tree"
    DEEP_EFFICIENCY = "deep_efficiency"
    MAX_CONFIDENCE = "max_confidence"
    PROGRESSIVE = "progressive"


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class DavidArchitectureConfig:
    """
    Complete configuration for David's architecture.

    All parameters needed to construct a David model instance.
    """

    # Metadata
    name: str = "david_architecture"
    uid: str = "c.david.architecture"

    # Core architecture
    feature_dim: int = 512
    num_classes: int = 1000
    scales: List[int] = field(default_factory=lambda: [256, 512, 768, 1024])

    # Sharing and fusion strategy
    sharing_mode: str = "partial_shared"
    fusion_mode: str = "hierarchical_tree"

    # Projection head configuration
    use_belly: bool = True
    belly_expand: float = 2.0
    projection_temperature: float = 0.07  # Temperature for logit scaling (CLIP default)

    # Shared feature extraction (FULLY_SHARED/PARTIAL_SHARED modes)
    shared_feature_dim: int = 768
    shared_layers: int = 2
    shared_dropout: float = 0.1

    # Fusion parameters
    fusion_temperature: float = 1.0
    fusion_dropout: float = 0.1

    # Hierarchical tree gating
    tree_depth: int = 3

    # Deep efficiency gating
    num_experts: int = 3
    compression_ratio: int = 4
    expert_dropout: float = 0.1
    attention_dropout: float = 0.1

    # Progressive training
    progressive_training: bool = True
    scale_warmup_epochs: Optional[Dict[int, int]] = None

    def __post_init__(self):
        """Auto-generate warmup schedule if not provided."""
        if self.scale_warmup_epochs is None and self.progressive_training:
            self.scale_warmup_epochs = {}
            for i, scale in enumerate(self.scales):
                self.scale_warmup_epochs[scale] = i * 3

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        data = {
            'name': self.name,
            'uid': self.uid,
            'feature_dim': self.feature_dim,
            'num_classes': self.num_classes,
            'scales': self.scales,
            'sharing_mode': self.sharing_mode,
            'fusion_mode': self.fusion_mode,
            'use_belly': self.use_belly,
            'belly_expand': self.belly_expand,
            'shared_feature_dim': self.shared_feature_dim,
            'shared_layers': self.shared_layers,
            'shared_dropout': self.shared_dropout,
            'fusion_temperature': self.fusion_temperature,
            'fusion_dropout': self.fusion_dropout,
            'tree_depth': self.tree_depth,
            'num_experts': self.num_experts,
            'compression_ratio': self.compression_ratio,
            'expert_dropout': self.expert_dropout,
            'attention_dropout': self.attention_dropout,
            'progressive_training': self.progressive_training,
            'scale_warmup_epochs': self.scale_warmup_epochs,
        }
        return data

    @classmethod
    def from_dict(cls, data: dict) -> 'DavidArchitectureConfig':
        """Create config from dictionary."""
        return cls(**data)

    def to_json(self, path: str):
        """Save config to JSON file."""
        data = self.to_dict()
        # Convert int keys to strings for JSON
        if data.get('scale_warmup_epochs'):
            data['scale_warmup_epochs'] = {
                str(k): v for k, v in data['scale_warmup_epochs'].items()
            }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def from_json(cls, path: str) -> 'DavidArchitectureConfig':
        """Load config from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        # Convert string keys back to ints
        if 'scale_warmup_epochs' in data and data['scale_warmup_epochs']:
            data['scale_warmup_epochs'] = {
                int(k): v for k, v in data['scale_warmup_epochs'].items()
            }
        return cls(**data)

    def __repr__(self):
        return (
            f"DavidArchitectureConfig(\n"
            f"  name='{self.name}',\n"
            f"  scales={self.scales},\n"
            f"  sharing_mode='{self.sharing_mode}',\n"
            f"  fusion_mode='{self.fusion_mode}',\n"
            f"  use_belly={self.use_belly}\n"
            f")"
        )


# ============================================================================
# PRESETS
# ============================================================================

class DavidPresets:
    """Factory for preset configurations."""

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
        """Balanced accuracy/speed tradeoff."""
        return DavidArchitectureConfig(
            name="david_balanced",
            uid="c.david.balanced",
            feature_dim=512,
            scales=[256, 512, 768, 1024],
            sharing_mode="partial_shared",
            fusion_mode="hierarchical_tree",
            use_belly=True,
            belly_expand=2.0,
            shared_feature_dim=768,
            shared_layers=2,
            tree_depth=3,
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
        """Hierarchical coarse-to-fine refinement."""
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
        """CLIP ViT-B/16 optimized."""
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
        """CLIP ViT-L/14 optimized."""
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
    def clip_vit_l14_deep() -> DavidArchitectureConfig:
        """CLIP ViT-L/14 with deeper shared layers."""
        return DavidArchitectureConfig(
            name="david_clip_vit_l14_deep",
            uid="c.david.clip_vit_l14_deep",
            feature_dim=768,
            scales=[256, 512, 768, 1024, 1280, 1536, 1792, 2048, 2304, 2560],
            sharing_mode="partial_shared",
            fusion_mode="deep_efficiency",
            use_belly=True,
            shared_feature_dim=1024,
            shared_layers=4,
            num_experts=4,
            progressive_training=True,
        )

    @staticmethod
    def clip_vit_l14_very_deep() -> DavidArchitectureConfig:
        """CLIP ViT-L/14 with very deep architecture."""
        return DavidArchitectureConfig(
            name="david_clip_vit_l14_very_deep",
            uid="c.david.clip_vit_l14_very_deep",
            feature_dim=768,
            scales=[256, 512, 768, 1024, 1280, 1536, 1792, 2048, 2304, 2560],
            sharing_mode="partial_shared",
            fusion_mode="deep_efficiency",
            use_belly=True,
            belly_expand=2.5,
            shared_feature_dim=1536,  # Richer shared space
            shared_layers=8,  # Much deeper
            num_experts=8,  # One expert per 1.25 scales
            progressive_training=True,
            scale_warmup_epochs={
                256: 0, 512: 1, 768: 2, 1024: 3, 1280: 4,
                1536: 5, 1792: 6, 2048: 7, 2304: 8, 2560: 9
            }
        )

    @staticmethod
    def clip_vit_l14_ultra_deep() -> DavidArchitectureConfig:
        """CLIP ViT-L/14 with ultra-deep architecture - GO BIG."""
        return DavidArchitectureConfig(
            name="david_clip_vit_l14_ultra_deep",
            uid="c.david.clip_vit_l14_ultra_deep",
            feature_dim=768,
            scales=[256, 512, 768, 1024, 1280, 1536, 1792, 2048, 2304, 2560],
            sharing_mode="partial_shared",
            fusion_mode="deep_efficiency",
            use_belly=True,
            belly_expand=3.0,  # Maximum expansion
            shared_feature_dim=2048,  # HUGE shared representation
            shared_layers=12,  # ResNet-level depth
            num_experts=10,  # One expert per scale
            progressive_training=True,
            scale_warmup_epochs={
                256: 0, 512: 1, 768: 2, 1024: 3, 1280: 4,
                1536: 5, 1792: 6, 2048: 7, 2304: 8, 2560: 9
            }
        )

    @staticmethod
    def clip_vit_h14() -> DavidArchitectureConfig:
        """CLIP ViT-H/14 optimized."""
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

    @staticmethod
    def get_preset(name: str) -> DavidArchitectureConfig:
        """Get preset by name."""
        presets = {
            'small_fast': DavidPresets.small_fast,
            'balanced': DavidPresets.balanced,
            'high_accuracy': DavidPresets.high_accuracy,
            'hierarchical_refinement': DavidPresets.hierarchical_refinement,
            'clip_vit_b16': DavidPresets.clip_vit_b16,
            'clip_vit_l14': DavidPresets.clip_vit_l14,
            'clip_vit_l14_deep': DavidPresets.clip_vit_l14_deep,
            'clip_vit_l14_very_deep': DavidPresets.clip_vit_l14_very_deep,
            'clip_vit_l14_ultra_deep': DavidPresets.clip_vit_l14_ultra_deep,
            'clip_vit_h14': DavidPresets.clip_vit_h14,
        }
        if name not in presets:
            raise ValueError(
                f"Unknown preset '{name}'. "
                f"Available: {list(presets.keys())}"
            )
        return presets[name]()

    @staticmethod
    def list_presets() -> List[str]:
        """List available preset names."""
        return [
            'small_fast',
            'balanced',
            'high_accuracy',
            'hierarchical_refinement',
            'clip_vit_b16',
            'clip_vit_l14',
            'clip_vit_h14',
        ]


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("David Configuration System")
    print("="*80)

    # Example 1: List presets
    print("\n[1] Available Presets:")
    for preset_name in DavidPresets.list_presets():
        config = DavidPresets.get_preset(preset_name)
        print(f"  • {preset_name:30s} - {len(config.scales)} scales, {config.fusion_mode}")

    # Example 2: Get preset
    print("\n[2] Load Preset:")
    config = DavidPresets.get_preset('balanced')
    print(f"  {config}")

    # Example 3: Custom config
    print("\n[3] Custom Configuration:")
    custom = DavidArchitectureConfig(
        name="my_custom",
        feature_dim=512,
        num_classes=100,
        scales=[128, 256, 512],
        use_belly=True,
        belly_expand=2.5,
    )
    print(f"  {custom}")

    # Example 4: Save/load JSON
    print("\n[4] JSON Serialization:")
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        path = f.name

    custom.to_json(path)
    print(f"  Saved to: {path}")

    loaded = DavidArchitectureConfig.from_json(path)
    print(f"  Loaded: {loaded.name}")
    print(f"  Match: {loaded.to_dict() == custom.to_dict()}")

    import os
    os.unlink(path)

    print("\n" + "="*80)