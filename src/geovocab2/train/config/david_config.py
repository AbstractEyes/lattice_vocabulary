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
    GEOMETRIC_ATTENTION = "geometric_attention"     # NEW: Geometric + Cayley-Menger attention
    CANTOR_SCALE = "cantor_scale"                   # NEW: Fractal-based scale routing


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

    # Geometric attention parameters (for GEOMETRIC_ATTENTION mode)
    geometric_num_heads: int = 4
    geometric_use_cayley: bool = True
    geometric_use_angular: bool = True
    geometric_scale_dim_aware: bool = True

    # Cantor scale fusion parameters (for CANTOR_SCALE mode)
    cantor_num_heads: int = 4
    cantor_depth: int = 8
    cantor_local_window: int = 3  # How many scales to attend to
    cantor_use_scale_embeddings: bool = True

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
            'projection_temperature': self.projection_temperature,
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
            'geometric_num_heads': self.geometric_num_heads,
            'geometric_use_cayley': self.geometric_use_cayley,
            'geometric_use_angular': self.geometric_use_angular,
            'geometric_scale_dim_aware': self.geometric_scale_dim_aware,
            'cantor_num_heads': self.cantor_num_heads,
            'cantor_depth': self.cantor_depth,
            'cantor_local_window': self.cantor_local_window,
            'cantor_use_scale_embeddings': self.cantor_use_scale_embeddings,
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
    def geometric_fusion() -> DavidArchitectureConfig:
        """Geometric attention with pentachoron-inspired routing."""
        return DavidArchitectureConfig(
            name="david_geometric_fusion",
            uid="c.david.geometric_fusion",
            feature_dim=512,
            scales=[256, 512, 768, 1024, 1280],
            sharing_mode="partial_shared",
            fusion_mode="geometric_attention",
            use_belly=True,
            belly_expand=2.0,
            shared_feature_dim=768,
            shared_layers=2,
            geometric_num_heads=4,
            geometric_use_cayley=True,
            geometric_use_angular=True,
            geometric_scale_dim_aware=True,
            fusion_temperature=0.07,
            progressive_training=True,
        )

    @staticmethod
    def cantor_routing() -> DavidArchitectureConfig:
        """Fractal-based scale routing with Cantor geometry."""
        return DavidArchitectureConfig(
            name="david_cantor_routing",
            uid="c.david.cantor_routing",
            feature_dim=512,
            scales=[256, 512, 768, 1024, 1280],
            sharing_mode="partial_shared",
            fusion_mode="cantor_scale",
            use_belly=True,
            belly_expand=2.0,
            shared_feature_dim=768,
            shared_layers=2,
            cantor_num_heads=4,
            cantor_depth=8,
            cantor_local_window=3,
            cantor_use_scale_embeddings=True,
            fusion_temperature=0.07,
            progressive_training=True,
        )

    @staticmethod
    def gated_expert_team() -> DavidArchitectureConfig:
        """Maximum accuracy configuration."""
        return DavidArchitectureConfig(
            name="david_gated_expert_team",
            uid="c.david.gated_expert_team",
            feature_dim=512,
            scales=[128, 256, 384, 448, 512, 576, 640, 768, 896],
            sharing_mode="decoupled",
            fusion_mode="deep_efficiency",
            use_belly=True,
            belly_expand=4,
            num_experts=8,
            compression_ratio=2,
            shared_feature_dim=1024,
            shared_layers=4,
            progressive_training=True,
            scale_warmup_epochs={
                128: 0,
                256: 0,
                384: 1,
                448: 1,
                512: 2,
                576: 3,
                640: 4,
                768: 5,
                896: 6
            }
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
    def clip_vit_b16_geometric() -> DavidArchitectureConfig:
        """CLIP ViT-B/16 with geometric attention."""
        return DavidArchitectureConfig(
            name="david_clip_vit_b16_geometric",
            uid="c.david.clip_vit_b16_geometric",
            feature_dim=512,
            scales=[256, 512, 768, 1024],
            sharing_mode="partial_shared",
            fusion_mode="geometric_attention",
            use_belly=True,
            geometric_num_heads=4,
            geometric_use_cayley=True,
            geometric_use_angular=True,
            progressive_training=True,
        )

    @staticmethod
    def clip_vit_b16_cantor() -> DavidArchitectureConfig:
        """CLIP ViT-B/16 with Cantor routing."""
        return DavidArchitectureConfig(
            name="david_clip_vit_b16_cantor",
            uid="c.david.clip_vit_b16_cantor",
            feature_dim=512,
            scales=[256, 512, 768, 1024],
            sharing_mode="decoupled",
            fusion_mode="cantor_scale",
            use_belly=True,
            cantor_num_heads=4,
            cantor_depth=8,
            cantor_local_window=3,
            progressive_training=True,
        )


    @staticmethod
    def clip_vit_b16_cantor_decoupled_moderate() -> DavidArchitectureConfig:
        """CLIP ViT-B/16 with Cantor routing."""
        return DavidArchitectureConfig(
            name="david_clip_vit_b16_cantor_decoupled_moderate",
            uid="c.david.clip_vit_b16_cantor_decoupled_moderate",
            feature_dim=512,
            scales=[256, 512, 768, 1024],
            sharing_mode="decoupled",
            fusion_mode="cantor_scale",
            use_belly=True,
            cantor_num_heads=8,
            cantor_depth=8,
            cantor_local_window=3,
            progressive_training=True,
        )


    @staticmethod
    def clip_vit_b16_cantor_decoupled_massive() -> DavidArchitectureConfig:
        """CLIP ViT-B/16 with Cantor routing."""
        return DavidArchitectureConfig(
            name="david_clip_vit_b16_cantor_decoupled_massive",
            uid="c.david.clip_vit_b16_cantor_decoupled_massive",
            feature_dim=512,
            scales=[512, 4096, 8192, 16384],
            sharing_mode="decoupled",
            fusion_mode="cantor_scale",
            use_belly=True,
            cantor_num_heads=8,
            cantor_depth=8,
            cantor_local_window=3,
            progressive_training=True,
        )

    @staticmethod
    def clip_vit_b16_cantor_big_window() -> DavidArchitectureConfig:
        """CLIP ViT-B/16 with Cantor routing."""
        return DavidArchitectureConfig(
            name="clip_vit_b16_cantor_big_window",
            uid="c.david.clip_vit_b16_cantor_big_window",
            feature_dim=512,
            scales=[256, 512, 768, 1024, 2048, 4096],
            sharing_mode="decoupled",
            fusion_mode="cantor_scale",
            use_belly=True,
            cantor_num_heads=8,
            cantor_depth=8,
            cantor_local_window=32,
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
    def clip_vit_l14_geometric() -> DavidArchitectureConfig:
        """CLIP ViT-L/14 with geometric attention."""
        return DavidArchitectureConfig(
            name="david_clip_vit_l14_geometric",
            uid="c.david.clip_vit_l14_geometric",
            feature_dim=768,
            scales=[384, 768, 1024, 1280, 1536],
            sharing_mode="partial_shared",
            fusion_mode="geometric_attention",
            use_belly=True,
            shared_feature_dim=1024,
            geometric_num_heads=6,
            geometric_use_cayley=True,
            geometric_use_angular=True,
            progressive_training=True,
        )

    @staticmethod
    def clip_vit_l14_cantor() -> DavidArchitectureConfig:
        """CLIP ViT-L/14 with Cantor routing."""
        return DavidArchitectureConfig(
            name="david_clip_vit_l14_cantor",
            uid="c.david.clip_vit_l14_cantor",
            feature_dim=768,
            scales=[384, 768, 1024, 1280, 1536],
            sharing_mode="partial_shared",
            fusion_mode="cantor_scale",
            use_belly=True,
            shared_feature_dim=1024,
            cantor_num_heads=6,
            cantor_depth=8,
            cantor_local_window=4,  # More scales = more neighbors
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
    def clip_vit_l14_deep_cantor() -> DavidArchitectureConfig:
        """CLIP ViT-L/14 deep with Cantor routing (10 scales)."""
        return DavidArchitectureConfig(
            name="david_clip_vit_l14_deep_cantor",
            uid="c.david.clip_vit_l14_deep_cantor",
            feature_dim=768,
            scales=[256, 512, 768, 1024, 1280, 1536, 1792, 2048, 2304, 2560],
            sharing_mode="partial_shared",
            fusion_mode="cantor_scale",
            use_belly=True,
            shared_feature_dim=1024,
            shared_layers=4,
            cantor_num_heads=8,
            cantor_depth=10,  # Deeper fractal for more scales
            cantor_local_window=5,
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
            shared_feature_dim=1536,
            shared_layers=8,
            num_experts=8,
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
            belly_expand=3.0,
            shared_feature_dim=2048,
            shared_layers=12,
            num_experts=10,
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
    def clip_vit_h14_geometric() -> DavidArchitectureConfig:
        """CLIP ViT-H/14 with geometric attention."""
        return DavidArchitectureConfig(
            name="david_clip_vit_h14_geometric",
            uid="c.david.clip_vit_h14_geometric",
            feature_dim=1024,
            scales=[512, 768, 1024, 1280, 1536, 1792],
            sharing_mode="partial_shared",
            fusion_mode="geometric_attention",
            use_belly=True,
            shared_feature_dim=1280,
            geometric_num_heads=8,
            geometric_use_cayley=True,
            geometric_use_angular=True,
            progressive_training=True,
        )

    @staticmethod
    def clip_vit_bigG() -> DavidArchitectureConfig:
        """CLIP ViT-bigG/14 optimized."""
        return DavidArchitectureConfig(
            name="david_clip_vit_bigg14",
            uid="c.david.clip_vit_bigg14",
            feature_dim=1280,
            scales=[384, 512, 768, 1024, 1280, 1536, 1792, 2048],
            sharing_mode="partial_shared",
            fusion_mode="deep_efficiency",
            use_belly=True,
            shared_feature_dim=1280,
            num_experts=8,
            compression_ratio=2,
            progressive_training=False,
        )

    @staticmethod
    def clip_vit_bigG_cantor() -> DavidArchitectureConfig:
        """CLIP ViT-bigG/14 with Cantor routing."""
        return DavidArchitectureConfig(
            name="david_clip_vit_bigg14_cantor",
            uid="c.david.clip_vit_bigg14_cantor",
            feature_dim=1280,
            scales=[384, 512, 768, 1024, 1280, 1536, 1792, 2048],
            sharing_mode="partial_shared",
            fusion_mode="cantor_scale",
            use_belly=True,
            shared_feature_dim=1536,
            cantor_num_heads=8,
            cantor_depth=10,
            cantor_local_window=5,
            progressive_training=True,
        )

    @staticmethod
    def clip_vit_bigG_cantor_decoupled() -> DavidArchitectureConfig:
        """CLIP ViT-bigG/14 with Cantor routing."""
        return DavidArchitectureConfig(
            name="david_clip_vit_bigg14_cantor_decoupled",
            uid="c.david.david_clip_vit_bigg14_cantor_decoupled",
            feature_dim=1280,
            scales=[384, 512, 768, 1024, 1280, 1536, 1792, 2048],
            sharing_mode="decoupled",
            fusion_mode="cantor_scale",
            use_belly=True,
            shared_feature_dim=1536,
            cantor_num_heads=8,
            cantor_depth=10,
            cantor_local_window=5,
            progressive_training=True,
        )

    @staticmethod
    def get_preset(name: str) -> DavidArchitectureConfig:
        """Get preset by name."""
        presets = {
            'small_fast': DavidPresets.small_fast,
            'balanced': DavidPresets.balanced,
            'high_accuracy': DavidPresets.high_accuracy,
            'geometric_fusion': DavidPresets.geometric_fusion,
            'cantor_routing': DavidPresets.cantor_routing,
            'hierarchical_refinement': DavidPresets.hierarchical_refinement,
            'gated_expert_team': DavidPresets.gated_expert_team,
            'clip_vit_b16': DavidPresets.clip_vit_b16,
            'clip_vit_b16_geometric': DavidPresets.clip_vit_b16_geometric,
            'clip_vit_b16_cantor': DavidPresets.clip_vit_b16_cantor,
            'clip_vit_b16_cantor_decoupled_moderate': DavidPresets.clip_vit_b16_cantor_decoupled_moderate,
            'clip_vit_b16_cantor_decoupled_massive': DavidPresets.clip_vit_b16_cantor_decoupled_massive,
            "clip_vit_b16_cantor_big_window": DavidPresets.clip_vit_b16_cantor_big_window,
            'clip_vit_l14': DavidPresets.clip_vit_l14,
            'clip_vit_l14_geometric': DavidPresets.clip_vit_l14_geometric,
            'clip_vit_l14_cantor': DavidPresets.clip_vit_l14_cantor,
            'clip_vit_l14_deep': DavidPresets.clip_vit_l14_deep,
            'clip_vit_l14_deep_cantor': DavidPresets.clip_vit_l14_deep_cantor,
            'clip_vit_l14_very_deep': DavidPresets.clip_vit_l14_very_deep,
            'clip_vit_l14_ultra_deep': DavidPresets.clip_vit_l14_ultra_deep,
            'clip_vit_h14': DavidPresets.clip_vit_h14,
            'clip_vit_h14_geometric': DavidPresets.clip_vit_h14_geometric,
            'clip_vit_bigg14': DavidPresets.clip_vit_bigG,
            'clip_vit_bigg14_cantor': DavidPresets.clip_vit_bigG_cantor,
            'clip_vit_bigg14_cantor_decoupled': DavidPresets.clip_vit_bigG_cantor_decoupled,
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
            'geometric_fusion',
            'cantor_routing',
            'hierarchical_refinement',
            'gated_expert_team',
            'clip_vit_b16',
            'clip_vit_b16_geometric',
            'clip_vit_b16_cantor',
            'clip_vit_b16_cantor_decoupled_moderate',
            'clip_vit_b16_cantor_decoupled_massive',
            'clip_vit_b16_cantor_big_window',
            'clip_vit_l14',
            'clip_vit_l14_geometric',
            'clip_vit_l14_cantor',
            'clip_vit_l14_deep',
            'clip_vit_l14_deep_cantor',
            'clip_vit_l14_very_deep',
            'clip_vit_l14_ultra_deep',
            'clip_vit_h14',
            'clip_vit_h14_geometric',
            'clip_vit_bigg14',
            'clip_vit_bigg14_cantor',
            'clip_vit_bigg14_cantor_decoupled',
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
        print(f"  • {preset_name:35s} - {len(config.scales)} scales, {config.fusion_mode}")

    # Example 2: Get geometric preset
    print("\n[2] Geometric Fusion Preset:")
    config = DavidPresets.get_preset('geometric_fusion')
    print(f"  {config}")
    print(f"  Geometric heads: {config.geometric_num_heads}")
    print(f"  Use Cayley: {config.geometric_use_cayley}")

    # Example 3: Get Cantor preset
    print("\n[3] Cantor Routing Preset:")
    config = DavidPresets.get_preset('cantor_routing')
    print(f"  {config}")
    print(f"  Cantor depth: {config.cantor_depth}")
    print(f"  Local window: {config.cantor_local_window}")

    # Example 4: Custom config with new modes
    print("\n[4] Custom Geometric Configuration:")
    custom = DavidArchitectureConfig(
        name="my_geometric",
        feature_dim=512,
        num_classes=100,
        scales=[128, 256, 512, 768],
        fusion_mode="geometric_attention",
        geometric_num_heads=8,
        geometric_use_cayley=True,
        geometric_use_angular=True,
    )
    print(f"  {custom}")

    print("\n" + "="*80)
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
    print("Config path cleaned up and temporary file deleted.")
    print("All tests completed successfully.")