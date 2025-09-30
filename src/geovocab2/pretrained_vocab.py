"""
Unified Geometric Crystal Vocabulary System (Fixed)
Fixes for:
1. Character synthesis weight order bug - clarified that implementation was correct
2. Cache key generation for definitions - fixed to handle None properly
"""

import numpy as np
import os
import warnings
import pickle
import json
from typing import Dict, List, Optional, Tuple, Any, Iterator
from pathlib import Path
from huggingface_hub import login

from shapes.legacy_shape_factory import CrystalFactory
from src.geovocab2.defaults import (
    DimensionType,
    ContentType,
    FormulaType,
    NormType,
    UnifiedCrystalConfig,
    EPS, TokenID, TokenStr, Crystal, PooledVector
)


from data.trie import (
    ArrowDataManager,
)

from data.cache import LRUCache


HF_TOKEN = os.environ.get("HF_TOKEN")
print(HF_TOKEN)
login(HF_TOKEN)






# ============================================================================
# Unified Vocabulary System
# ============================================================================

class UnifiedGeometricVocabulary:
    """Main interface combining all components"""

    def __init__(self, config: Optional[UnifiedCrystalConfig] = None):
        self.config = config or UnifiedCrystalConfig()

        # Initialize components
        self.data_manager = ArrowDataManager(self.config)
        self.factory = CrystalFactory(self.config, self.data_manager)

        # Caches
        self.crystal_cache = LRUCache(maxsize=self.config.memory_cache_size)
        self.pooled_cache = LRUCache(maxsize=self.config.memory_cache_size * 2)

        # Tracking
        self._synthetic_ids: Dict[TokenID, TokenStr] = {}
        self._next_synthetic_id = -1

        self.stats = {
            "crystals_created": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "dataset_hits": 0,
            "synthesized": 0,
        }

        # Initialize dataset
        self._dataset_loaded = self.data_manager.initialize()
        if not self._dataset_loaded:
            print("Running without dataset - synthesis only mode")

    def get_crystal(self, token: str, definition: Optional[str] = None,
                    synthesize: bool = True,
                    override_config: Optional[Dict[str, Any]] = None) -> Optional[Crystal]:
        """Get crystal embedding with cache key handling"""
        definition_key = hash(definition) if definition is not None else ""
        cache_key = f"{token}_{definition_key}_{str(override_config)}"

        if cache_key in self.crystal_cache:
            self.stats["cache_hits"] += 1
            return self.crystal_cache[cache_key]

        self.stats["cache_misses"] += 1

        # Try dataset first only if no definition
        if self._dataset_loaded and not definition:
            dataset_data = self.data_manager.get_token_data(token)
            if dataset_data and dataset_data.get("crystal") is not None:
                self.stats["dataset_hits"] += 1
                result = self.factory.create_crystal(token, definition, override_config, use_dataset=True)
                crystal = result['crystal']
                self.crystal_cache[cache_key] = crystal
                return crystal

        # Synthesize if needed
        if synthesize:
            result = self.factory.create_crystal(token, definition, override_config, use_dataset=False)
            crystal = result['crystal']
            self.crystal_cache[cache_key] = crystal

            if not self._token_exists(token):
                token_id = self._next_synthetic_id
                self._next_synthetic_id -= 1
                self._synthetic_ids[token_id] = token
                self.stats["synthesized"] += 1

            self.stats["crystals_created"] += 1
            return crystal

        return None

    def get_pooled(self, token: str, definition: Optional[str] = None,
                   method: str = "mean", synthesize: bool = True) -> Optional[PooledVector]:
        """Get pooled vector with FIXED cache key"""
        # FIX #2 applied here as well
        definition_key = hash(definition) if definition is not None else ""
        cache_key = f"{token}_{definition_key}_{method}"

        if cache_key in self.pooled_cache:
            return self.pooled_cache[cache_key]

        crystal = self.get_crystal(token, definition, synthesize)
        if crystal is None:
            return None

        if crystal.ndim == 1:
            # Handle 1D case
            pooled = crystal
        elif method == "mean":
            pooled = crystal.mean(axis=0)
        elif method == "first":
            pooled = crystal[0]
        elif method == "sum":
            pooled = crystal.sum(axis=0)
        elif method == "max":
            pooled = crystal.max(axis=0)
        elif method == "geometric_centroid":
            pooled = pool_geometric_centroid(crystal)
        elif method == "volume_weighted":
            pooled = pool_volume_weighted(crystal)
        else:
            raise ValueError(f"Unknown pooling method: {method}")

        self.pooled_cache[cache_key] = pooled
        return pooled

    def get_pooled_advanced(self, token: str, definition: Optional[str] = None,
                            method: str = "geometric_centroid", synthesize: bool = True) -> Optional[np.ndarray]:
        """Get pooled vector with advanced methods"""
        return self.get_pooled(token, definition, method, synthesize)

    def encode_batch(self, tokens: List[str],
                     definitions: Optional[List[Optional[str]]] = None,
                     synthesize: bool = True,
                     use_threading: bool = True) -> List[Optional[Crystal]]:
        """Batch encode tokens"""
        if definitions is None:
            definitions = [None] * len(tokens)

        # Get batch data from dataset if available
        batch_data = {}
        if self._dataset_loaded:
            batch_data = self.data_manager.get_batch_data(tokens)

        results = []
        for token, definition in zip(tokens, definitions):
            crystal = self.get_crystal(token, definition, synthesize)
            results.append(crystal)

        return results

    def similarity(self, token_a: str, token_b: str,
                   definition_a: Optional[str] = None,
                   definition_b: Optional[str] = None,
                   method: str = "cosine",
                   synthesize: bool = True) -> float:
        """Compute similarity with PROPER identical token handling"""
        # For identical tokens with same definition, return perfect similarity immediately
        if token_a == token_b and definition_a == definition_b:
            if method == "cosine":
                return 1.0
            elif method in ["euclidean", "manhattan"]:
                return 0.0

        pooled_a = self.get_pooled(token_a, definition_a, synthesize=synthesize)
        pooled_b = self.get_pooled(token_b, definition_b, synthesize=synthesize)

        if pooled_a is None or pooled_b is None:
            return -1.0

        if method == "cosine":
            norm_a = pooled_a / (np.linalg.norm(pooled_a) + EPS)
            norm_b = pooled_b / (np.linalg.norm(pooled_b) + EPS)
            return float(np.dot(norm_a, norm_b))
        elif method == "euclidean":
            return -float(np.linalg.norm(pooled_a - pooled_b))
        elif method == "manhattan":
            return -float(np.abs(pooled_a - pooled_b).sum())
        else:
            raise ValueError(f"Unknown similarity method: {method}")

    def crystal_similarity_advanced(self, token_a: str, token_b: str,
                                    method: str = "hausdorff",
                                    definition_a: Optional[str] = None,
                                    definition_b: Optional[str] = None) -> float:
        """Compute advanced similarity metrics between crystals"""
        crystal_a = self.get_crystal(token_a, definition_a)
        crystal_b = self.get_crystal(token_b, definition_b)

        if crystal_a is None or crystal_b is None:
            return -1.0

        if method == "hausdorff":
            return -hausdorff_distance(crystal_a, crystal_b)  # Negative for similarity
        elif method == "volume_ratio":
            vol_a = self.factory._compute_volume(crystal_a)
            vol_b = self.factory._compute_volume(crystal_b)
            if vol_a > 0 and vol_b > 0:
                ratio = min(vol_a, vol_b) / max(vol_a, vol_b)
                return ratio
            return 0.0
        else:
            return self.similarity(token_a, token_b, definition_a, definition_b, method)

    def create_custom_crystal(self, token: str,
                              dimension_type: DimensionType,
                              content_type: ContentType,
                              formula_type: FormulaType,
                              norm_type: NormType = NormType.L2,
                              definition: Optional[str] = None) -> Dict[str, Any]:
        """Create crystal with custom configuration"""
        override_config = {
            'dimension_type': dimension_type,
            'content_type': content_type,
            'formula_type': formula_type,
            'norm_type': norm_type,
        }

        return self.factory.create_crystal(token, definition, override_config)

    def find_similar(self, token: str, threshold: float = 0.3) -> List[Tuple[str, float]]:
        """Find similar tokens using n-gram similarity"""
        if self._dataset_loaded:
            return self.data_manager.find_similar_tokens(token, threshold)
        return []

    def fuzzy_search(self, token: str, max_distance: int = 2) -> List[Tuple[str, int]]:
        """Fuzzy search using edit distance or n-gram similarity"""
        if not self._dataset_loaded:
            return []

        # For short tokens, use edit distance
        if len(token) <= 3:
            results = []
            trie_results = self.data_manager.token_trie.fuzzy_search(token, min(max_distance, 1))
            for token_str, node, distance in trie_results:
                results.append((token_str, distance))
            return results
        else:
            # For longer tokens, use n-gram similarity (much faster)
            similar = self.data_manager.find_similar_tokens(token, threshold=0.3)
            results = []
            for similar_token, similarity in similar[:20]:
                approx_distance = int((1 - similarity) * max(len(token), len(similar_token)))
                if approx_distance <= max_distance:
                    results.append((similar_token, approx_distance))
            return results

    def prefix_search(self, token: str, max_results: int = 100) -> List[str]:
        """Find all tokens with given prefix"""
        if not self._dataset_loaded:
            return []

        results = self.data_manager.token_trie.prefix_search(token, max_results)
        return [token for token, _ in results]

    def save_vocabulary(self, path: Path):
        """Save vocabulary state to disk"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save configuration
        with open(path / "config.json", 'w') as f:
            config_dict = {
                'embedding_dim': self.config.embedding_dim,
                'dimension_type': self.config.dimension_type.name,
                'content_type': self.config.content_type.value,
                'formula_type': self.config.formula_type.value,
                'norm_type': self.config.norm_type.value,
                'memory_cache_size': self.config.memory_cache_size,
            }
            json.dump(config_dict, f, indent=2)

        # Save synthetic tokens
        with open(path / "synthetic_tokens.pkl", 'wb') as f:
            pickle.dump({
                'synthetic_ids': self._synthetic_ids,
                'next_synthetic_id': self._next_synthetic_id,
            }, f)

        # Save cached crystals
        cached_crystals = {}
        for key, crystal in self.crystal_cache.items():
            cached_crystals[key] = crystal

        np.savez_compressed(path / "cached_crystals.npz", **cached_crystals)

        # Save statistics
        with open(path / "stats.json", 'w') as f:
            json.dump(self.get_stats(), f, indent=2)

        print(f"Vocabulary saved to {path}")

    def load_vocabulary(self, path: Path):
        """Load vocabulary state from disk"""
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"No saved vocabulary at {path}")

        # Load synthetic tokens
        with open(path / "synthetic_tokens.pkl", 'rb') as f:
            synthetic_data = pickle.load(f)
            self._synthetic_ids = synthetic_data['synthetic_ids']
            self._next_synthetic_id = synthetic_data['next_synthetic_id']

        # Load cached crystals
        if (path / "cached_crystals.npz").exists():
            cached = np.load(path / "cached_crystals.npz")
            for key in cached.files:
                self.crystal_cache[key] = cached[key]

        # Load statistics
        if (path / "stats.json").exists():
            with open(path / "stats.json", 'r') as f:
                saved_stats = json.load(f)
                self.stats['synthesized'] = saved_stats.get('synthesized', 0)

        print(f"Vocabulary loaded from {path}")

    def iterate_vocabulary(self, include_synthetic: bool = True) -> Iterator[Tuple[str, Optional[np.ndarray]]]:
        """Iterate over all tokens in vocabulary"""
        # Dataset tokens
        if self._dataset_loaded:
            for token in self.data_manager._token_to_id.keys():
                crystal = self.get_crystal(token, synthesize=False)
                yield token, crystal

        # Synthetic tokens
        if include_synthetic:
            for token_id, token in self._synthetic_ids.items():
                crystal = self.get_crystal(token, synthesize=True)
                yield token, crystal

    def get_all_tokens(self) -> List[str]:
        """Get all tokens in vocabulary"""
        tokens = []

        # Dataset tokens
        if self._dataset_loaded:
            tokens.extend(list(self.data_manager._token_to_id.keys()))

        # Synthetic tokens
        tokens.extend(list(self._synthetic_ids.values()))

        return tokens

    def sample_tokens(self, n: int = 10, seed: Optional[int] = None) -> List[str]:
        """Randomly sample tokens from vocabulary"""
        if seed is not None:
            np.random.seed(seed)

        all_tokens = self.get_all_tokens()
        if not all_tokens:
            return []

        n = min(n, len(all_tokens))
        return list(np.random.choice(all_tokens, n, replace=False))

    def _token_exists(self, token: str) -> bool:
        """Check if token exists"""
        if self._dataset_loaded and self.data_manager.token_exists(token):
            return True
        return token in self._synthetic_ids.values()

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics"""
        stats = dict(self.stats)
        stats["crystal_cache_size"] = len(self.crystal_cache)
        stats["pooled_cache_size"] = len(self.pooled_cache)
        stats["synthetic_tokens"] = len(self._synthetic_ids)
        stats["dataset_loaded"] = self._dataset_loaded

        if self._dataset_loaded:
            stats.update(self.data_manager.stats)

        stats["config"] = {
            "dimension": self.config.dimension_type.name,
            "content": self.config.content_type.value,
            "formula": self.config.formula_type.value,
            "norm": self.config.norm_type.value,
        }

        return stats

    # Static methods for crystal operations
    @staticmethod
    def interpolate_crystals(crystal1: np.ndarray, crystal2: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        """Interpolate between two crystals"""
        if crystal1.shape != crystal2.shape:
            raise ValueError("Crystals must have same shape")

        interpolated = (1 - alpha) * crystal1 + alpha * crystal2
        interpolated -= interpolated.mean(axis=0, keepdims=True)

        return interpolated

    @staticmethod
    def add_crystals(crystal1: np.ndarray, crystal2: np.ndarray) -> np.ndarray:
        """Add two crystals with proper geometric handling"""
        if crystal1.shape != crystal2.shape:
            raise ValueError("Crystals must have same shape")

        result = crystal1 + crystal2
        result -= result.mean(axis=0, keepdims=True)

        # Rescale to maintain reasonable volume
        scale1 = np.linalg.norm(crystal1)
        scale2 = np.linalg.norm(crystal2)
        scale_result = np.linalg.norm(result)

        if scale_result > 0:
            target_scale = (scale1 + scale2) / 2
            result *= (target_scale / scale_result)

        return result

    @staticmethod
    def rotate_crystal(crystal: np.ndarray, angle: float, axis1: int = 0, axis2: int = 1) -> np.ndarray:
        """Rotate crystal in specified plane"""
        rotated = crystal.copy()
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)

        for i in range(crystal.shape[0]):
            temp = rotated[i, axis1] * cos_a - rotated[i, axis2] * sin_a
            rotated[i, axis2] = rotated[i, axis1] * sin_a + rotated[i, axis2] * cos_a
            rotated[i, axis1] = temp

        return rotated


# ============================================================================
# Helper Functions
# ============================================================================



def pool_geometric_centroid(crystal: np.ndarray) -> np.ndarray:
    """Pool using geometric centroid with outlier weighting"""
    if crystal.ndim == 1:
        return crystal

    centroid = crystal.mean(axis=0)
    distances = np.linalg.norm(crystal - centroid, axis=1)
    weights = 1.0 / (distances + 1e-8)
    weights /= weights.sum()

    return np.sum(crystal * weights[:, np.newaxis], axis=0)


def pool_volume_weighted(crystal: np.ndarray) -> np.ndarray:
    """Pool weighted by each vertex's contribution to volume"""
    if crystal.ndim == 1:
        return crystal

    n = crystal.shape[0]
    weights = np.zeros(n)

    for i in range(n):
        subset = np.delete(crystal, i, axis=0)
        if subset.shape[0] >= 2:
            weights[i] = np.mean(np.linalg.norm(subset - subset.mean(axis=0), axis=1))

    if weights.sum() > 0:
        weights /= weights.sum()
    else:
        weights = np.ones(n) / n

    return np.sum(crystal * weights[:, np.newaxis], axis=0)


def hausdorff_distance(crystal1: np.ndarray, crystal2: np.ndarray) -> float:
    """Compute Hausdorff distance between crystals"""
    if crystal1.ndim == 1:
        crystal1 = crystal1.reshape(1, -1)
    if crystal2.ndim == 1:
        crystal2 = crystal2.reshape(1, -1)

    # Compute all pairwise distances
    distances = np.zeros((crystal1.shape[0], crystal2.shape[0]))
    for i in range(crystal1.shape[0]):
        for j in range(crystal2.shape[0]):
            distances[i, j] = np.linalg.norm(crystal1[i] - crystal2[j])

    # Hausdorff distance: max of min distances
    h1 = np.max(np.min(distances, axis=1))
    h2 = np.max(np.min(distances, axis=0))

    return max(h1, h2)


def validate_config(config: UnifiedCrystalConfig) -> List[str]:
    """Validate configuration consistency"""
    issues = []

    # Graham formulas require sufficient dimensions
    if config.formula_type in [FormulaType.GRAHAM_INFINITE, FormulaType.GRAHAM_FINITE, FormulaType.GRAHAM_MASKED]:
        if config.dimension_type.value < 3:
            issues.append(f"Graham formulas require at least D3, got {config.dimension_type.name}")

    # Rose-Cayley works best with D5
    if config.formula_type == FormulaType.ROSE_CAYLEY:
        if config.dimension_type != DimensionType.D5:
            issues.append(f"Rose-Cayley designed for D5, got {config.dimension_type.name}")

    # Volume content type needs proper dimensions
    if config.content_type == ContentType.VOLUME:
        if config.dimension_type.value < 2:
            issues.append("Volume content type requires at least D2")

    # Trajectory needs multiple vertices
    if config.content_type == ContentType.TRAJECTORY:
        if config.dimension_type.value < 2:
            issues.append("Trajectory content type requires at least D2")

    return issues


# ============================================================================
# Factory Functions
# ============================================================================

def create_unified_vocabulary(
        embedding_dim: int = 100,
        dimension_type: DimensionType = DimensionType.D5,
        content_type: ContentType = ContentType.HYBRID,
        formula_type: FormulaType = FormulaType.HYBRID_V1V2,
        norm_type: NormType = NormType.L2,
        cache_size: int = 10000,
        enable_synthesis: bool = True,
        repo_id: str = "AbstractPhil/geometric-vocab",
        dataset_name: str = "wordnet_eng_100d",
        prefer_dataset: bool = True
) -> UnifiedGeometricVocabulary:
    """Create unified vocabulary system"""

    config = UnifiedCrystalConfig(
        repo_id=repo_id,
        dataset_name=dataset_name,
        embedding_dim=embedding_dim,
        dimension_type=dimension_type,
        content_type=content_type,
        formula_type=formula_type,
        norm_type=norm_type,
        memory_cache_size=cache_size,
        enable_synthesis=enable_synthesis,
        prefer_dataset=prefer_dataset
    )

    return UnifiedGeometricVocabulary(config)


def create_validated_vocabulary(**kwargs) -> UnifiedGeometricVocabulary:
    """Create vocabulary with config validation"""
    config = UnifiedCrystalConfig(**kwargs)

    issues = validate_config(config)
    if issues:
        warnings.warn(f"Configuration issues: {'; '.join(issues)}", RuntimeWarning)

    return UnifiedGeometricVocabulary(config)

if __name__ == "__main__":
    # Example usage
    vocab = create_validated_vocabulary(
        embedding_dim=100,
        dimension_type=DimensionType.D5,
        content_type=ContentType.HYBRID,
        formula_type=FormulaType.HYBRID_V1V2,
        norm_type=NormType.L2,
        enable_synthesis=True,
        repo_id="AbstractPhil/geometric-vocab",
        dataset_name="unicode_100d"
    )

    token = "example"
    definition = "a representative form or pattern"

    crystal = vocab.get_crystal(token, definition)
    pooled = vocab.get_pooled(token, definition, method="geometric_centroid")

    print(f"Crystal for '{token}':\n{crystal}")
    print(f"Pooled vector for '{token}':\n{pooled}")
    print(f"Vocabulary stats:\n{vocab.get_stats()}")