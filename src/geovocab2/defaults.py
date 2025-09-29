import numpy as np
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

# Type definitions
TokenID = int
TokenStr = str
Crystal = np.ndarray  # Shape: (vertices, dim)
PooledVector = np.ndarray  # Shape: (dim,)

# Constants
EPS = 1e-12

# ============================================================================
# Crystal Type Taxonomy
# ============================================================================
class DimensionType(Enum):
    """Dimensional control for geometric tokens"""
    D1 = 1
    D2 = 2
    D3 = 3
    D4 = 4
    D5 = 5
    D6_PLUS = 6


class ContentType(Enum):
    """Content richness of crystal"""
    SPARSE = "sparse"
    ENRICHED = "enriched"
    TRAJECTORY = "trajectory"
    MAGNITUDE = "magnitude"
    VOLUME = "volume"
    HYBRID = "hybrid"


class FormulaType(Enum):
    """Mathematical formula basis"""
    ROSE_CAYLEY = "rose_cayley"
    CAYLEY_MENGER = "cayley_menger"
    CAYLEY = "cayley"
    MENGER = "menger"
    EULER = "euler"
    GRAHAM_INFINITE = "graham_infinite"
    GRAHAM_FINITE = "graham_finite"
    GRAHAM_MASKED = "graham_masked"
    HYBRID_V1V2 = "hybrid_v1v2"


class NormType(Enum):
    """Normalization strategies"""
    L1 = "l1"
    L2 = "l2"
    LINF = "linf"
    NONE = "none"


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class UnifiedCrystalConfig:
    """Unified configuration for the complete system"""
    # Dataset settings
    repo_id: str = "AbstractPhil/geometric-vocab"
    dataset_name: str = "unicode_100d"
    split: str = "train"

    # Dimensions
    embedding_dim: int = 100
    dimension_type: DimensionType = DimensionType.D5

    # Crystal properties
    content_type: ContentType = ContentType.HYBRID
    formula_type: FormulaType = FormulaType.HYBRID_V1V2
    norm_type: NormType = NormType.L2

    # Synthesis options
    enable_synthesis: bool = True
    use_definitions: bool = True
    use_character_composition: bool = True
    silent_synthesis: bool = False
    prefer_dataset: bool = True

    # Cache settings
    memory_cache_size: int = 10000
    disk_cache_path: Optional[Path] = None

    # Performance
    batch_size: int = 100
    num_threads: int = 4

    # Graham-specific
    graham_levels: Optional[int] = None
    graham_mask: Optional[np.ndarray] = None

    # Rose structure
    use_rose_structure: bool = False
    freeze_anchor: bool = True

