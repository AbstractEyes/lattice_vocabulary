"""
Unified Geometric Crystal Vocabulary System (Fixed)
Fixes for:
1. Character synthesis weight order bug - clarified that implementation was correct
2. Cache key generation for definitions - fixed to handle None properly
"""

import numpy as np
import torch
import hashlib
import warnings
import time
import threading
import pickle
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Union, Tuple, Protocol, TypeVar, Generic, Callable, Any, Iterator, Set
from enum import Enum
from pathlib import Path
from collections import OrderedDict, defaultdict
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor

# Optional imports
try:
    import pyarrow as pa
    import pyarrow.parquet as pq

    HAS_ARROW = True
except ImportError:
    HAS_ARROW = False
    warnings.warn("PyArrow not found. Arrow features will be unavailable.", ImportWarning)

try:
    import datasets
    from datasets import Dataset

    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False
    warnings.warn("datasets library not found. Dataset loading will be unavailable.", ImportWarning)

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


# ============================================================================
# Trie Infrastructure for O(k) Lookups
# ============================================================================

class TrieNode:
    """Node in lexical tree - stores only row indices, not data"""

    def __init__(self):
        self.children: Dict[str, TrieNode] = {}
        self.row_indices: List[int] = []  # Arrow table row indices
        self.token_id: Optional[TokenID] = None
        self.is_terminal: bool = False
        # Cache for frequently accessed data
        self.embedding_cache: Optional[Crystal] = None
        self.volume_cache: Optional[float] = None


class LexicalTrie:
    """Trie for O(k) token lookups - maps tokens to row indices only"""

    def __init__(self):
        self.root = TrieNode()
        self._size = 0
        self._lock = threading.RLock()

    def insert(self, token: str, row_index: int, token_id: TokenID):
        """Insert token with row index - no data loading"""
        with self._lock:
            node = self.root

            for char in token:
                if char not in node.children:
                    node.children[char] = TrieNode()
                node = node.children[char]

            node.row_indices.append(row_index)
            node.token_id = token_id
            node.is_terminal = True

            if len(node.row_indices) == 1:
                self._size += 1

    def search(self, token: str) -> Optional[TrieNode]:
        """O(k) search for token"""
        with self._lock:
            node = self.root

            for char in token:
                if char not in node.children:
                    return None
                node = node.children[char]

            return node if node.is_terminal else None

    def prefix_search(self, prefix: str, max_results: int = 100) -> List[Tuple[str, TrieNode]]:
        """Find all tokens with given prefix"""
        with self._lock:
            node = self.root

            for char in prefix:
                if char not in node.children:
                    return []
                node = node.children[char]

            results = []
            self._collect_terminals(node, prefix, results, max_results)
            return results

    def _collect_terminals(self, node: TrieNode, current: str, results: List, max_results: int):
        """Collect terminal nodes recursively"""
        if len(results) >= max_results:
            return

        if node.is_terminal:
            results.append((current, node))

        for char, child in node.children.items():
            if len(results) >= max_results:
                break
            self._collect_terminals(child, current + char, results, max_results)

    def fuzzy_search(self, token: str, max_distance: int = 2) -> List[Tuple[str, TrieNode, int]]:
        """Fuzzy search using edit distance with bounded search"""
        results = []
        with self._lock:
            max_depth = len(token) + max_distance
            self._fuzzy_dfs_bounded(self.root, "", token, max_distance, results, max_depth, 0)
        return sorted(results, key=lambda x: x[2])[:20]

    def _fuzzy_dfs_bounded(self, node, current: str, target: str, max_dist: int,
                           results: List, max_depth: int, depth: int):
        """DFS with pruning and depth limit for fuzzy search"""
        if depth > max_depth:
            return

        if abs(len(current) - len(target)) > max_dist:
            return

        if node.is_terminal:
            dist = self._edit_distance(current, target)
            if dist <= max_dist:
                results.append((current, node, dist))

        if len(results) < 100:
            for char, child in node.children.items():
                self._fuzzy_dfs_bounded(child, current + char, target, max_dist,
                                        results, max_depth, depth + 1)

    @staticmethod
    def _edit_distance(s1: str, s2: str) -> int:
        """Levenshtein distance calculation"""
        if len(s1) < len(s2):
            s1, s2 = s2, s1

        if len(s2) == 0:
            return len(s1)

        prev_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            curr_row = [i + 1]
            for j, c2 in enumerate(s2):
                curr_row.append(min(
                    prev_row[j + 1] + 1,  # deletion
                    curr_row[j] + 1,  # insertion
                    prev_row[j] + (c1 != c2)  # substitution
                ))
            prev_row = curr_row

        return prev_row[-1]

    def size(self) -> int:
        """Number of unique tokens"""
        return self._size


# ============================================================================
# N-gram Index for Similarity
# ============================================================================

class NgramIndex:
    """N-gram index for approximate matching without embeddings"""

    def __init__(self, n: int = 3):
        self.n = n
        self.ngram_to_tokens: Dict[str, Set[str]] = defaultdict(set)
        self.token_to_ngrams: Dict[str, Set[str]] = {}
        self._lock = threading.RLock()

    def add_token(self, token: str):
        """Add token to index"""
        with self._lock:
            if token in self.token_to_ngrams:
                return

            ngrams = self._extract_ngrams(token)
            self.token_to_ngrams[token] = ngrams

            for ngram in ngrams:
                self.ngram_to_tokens[ngram].add(token)

    def _extract_ngrams(self, token: str) -> Set[str]:
        """Extract character n-grams"""
        padded = f"#{token}#"
        ngrams = set()

        for i in range(len(padded) - self.n + 1):
            ngrams.add(padded[i:i + self.n])

        return ngrams

    def find_similar(self, token: str, threshold: float = 0.3) -> List[Tuple[str, float]]:
        """Find similar tokens using Jaccard similarity"""
        with self._lock:
            query_ngrams = self._extract_ngrams(token)
            candidates = set()

            for ngram in query_ngrams:
                candidates.update(self.ngram_to_tokens.get(ngram, set()))

            results = []
            for candidate in candidates:
                if candidate == token:
                    continue

                candidate_ngrams = self.token_to_ngrams[candidate]

                intersection = len(query_ngrams & candidate_ngrams)
                union = len(query_ngrams | candidate_ngrams)

                if union > 0:
                    similarity = intersection / union
                    if similarity >= threshold:
                        results.append((candidate, similarity))

            return sorted(results, key=lambda x: x[1], reverse=True)


# ============================================================================
# Cache Implementation
# ============================================================================

class LRUCache(OrderedDict):
    """Thread-safe LRU cache"""

    def __init__(self, maxsize: int = 128):
        super().__init__()
        self.maxsize = maxsize
        self._lock = threading.RLock()

    def __getitem__(self, key):
        with self._lock:
            value = super().__getitem__(key)
            self.move_to_end(key)
            return value

    def __setitem__(self, key, value):
        with self._lock:
            if key in self:
                self.move_to_end(key)
            super().__setitem__(key, value)
            if len(self) > self.maxsize:
                oldest = next(iter(self))
                del self[oldest]

    def get(self, key, default=None):
        with self._lock:
            if key in self:
                return self[key]
            return default

    def clear(self):
        """Clear the cache"""
        with self._lock:
            super().clear()


# ============================================================================
# Arrow Dataset Manager (Lazy Loading)
# ============================================================================

class ArrowDataManager:
    """
    Manages Arrow table with lazy loading via Trie.
    Only builds index, doesn't load embeddings until needed.
    """

    def __init__(self, config: UnifiedCrystalConfig):
        self.config = config
        self._dataset: Optional[Dataset] = None
        self._arrow_table: Optional[pa.Table] = None

        # Lightweight indices
        self.token_trie = LexicalTrie()
        self.ngram_index = NgramIndex(n=3)

        # Token mappings
        self._token_to_id: Dict[TokenStr, TokenID] = {}
        self._id_to_token: Dict[TokenID, TokenStr] = {}

        # Column references for lazy access
        self._token_column = None
        self._id_column = None
        self._crystal_column = None
        self._volume_column = None

        self.stats = {
            "rows_loaded": 0,
            "unique_tokens": 0,
            "trie_lookups": 0,
            "arrow_fetches": 0,
        }

        self._initialized = False
        self._lock = threading.RLock()

    def initialize(self) -> bool:
        """Initialize dataset and build lightweight index"""
        if self._initialized:
            return True

        with self._lock:
            if self._initialized:
                return True

            try:
                if not HAS_DATASETS:
                    warnings.warn("datasets library not available", RuntimeWarning)
                    return False

                print(f"Loading dataset {self.config.repo_id}/{self.config.dataset_name}...")

                # Load dataset
                self._dataset = datasets.load_dataset(
                    self.config.repo_id,
                    name=self.config.dataset_name,
                    split=self.config.split
                )

                # Get Arrow table reference
                if hasattr(self._dataset, '_data'):
                    self._arrow_table = self._dataset._data
                elif hasattr(self._dataset, 'data'):
                    if hasattr(self._dataset.data, 'table'):
                        self._arrow_table = self._dataset.data.table
                    else:
                        self._arrow_table = self._dataset.data
                else:
                    # Fallback: convert to Arrow
                    if HAS_ARROW:
                        self._arrow_table = pa.Table.from_pandas(self._dataset.to_pandas())

                # Store column references for lazy access
                self._token_column = self._arrow_table.column("token")
                self._id_column = self._arrow_table.column("token_id")

                if "crystal" in self._arrow_table.column_names:
                    self._crystal_column = self._arrow_table.column("crystal")

                if "volume" in self._arrow_table.column_names:
                    self._volume_column = self._arrow_table.column("volume")

                print(f"Building trie index for {len(self._arrow_table)} rows...")
                self._build_index()

                self._initialized = True
                print(f"Dataset ready: {self.stats['unique_tokens']} unique tokens indexed")
                return True

            except Exception as e:
                warnings.warn(f"Failed to load dataset: {e}", RuntimeWarning)
                self._initialized = False
                return False

    def _build_index(self):
        """Build lightweight Trie index - no data loading"""
        total_rows = len(self._arrow_table)
        chunk_size = 10000

        for i in range(0, total_rows, chunk_size):
            end = min(i + chunk_size, total_rows)

            # Use Arrow's slicing for efficient columnar access
            token_slice = self._token_column[i:end]
            id_slice = self._id_column[i:end]

            # Convert to Python objects for indexing
            tokens = token_slice.to_pylist()
            ids = id_slice.to_pylist()

            for j, (token, token_id) in enumerate(zip(tokens, ids)):
                row_idx = i + j
                token_str = str(token)

                # Add to Trie - just the index, no data
                self.token_trie.insert(token_str, row_idx, token_id)

                # Add to mappings
                if token_str not in self._token_to_id:
                    self.ngram_index.add_token(token_str)
                    self._token_to_id[token_str] = token_id
                    self._id_to_token[token_id] = token_str

        self.stats["rows_loaded"] = total_rows
        self.stats["unique_tokens"] = self.token_trie.size()

    def get_token_data(self, token: TokenStr) -> Optional[Dict[str, Any]]:
        """Lazy fetch token data using Trie lookup + Arrow columnar access"""
        if not self._initialized:
            return None

        self.stats["trie_lookups"] += 1

        # O(k) Trie lookup
        node = self.token_trie.search(token)
        if node is None:
            return None

        # Return cached if available
        if node.embedding_cache is not None:
            return {
                "token": token,
                "token_id": node.token_id,
                "crystal": node.embedding_cache,
                "volume": node.volume_cache or 1.0,
                "cached": True
            }

        # Lazy fetch from Arrow using columnar access
        if node.row_indices:
            idx = node.row_indices[0]
            self.stats["arrow_fetches"] += 1

            try:
                crystal_data = None
                if self._crystal_column is not None:
                    crystal_val = self._crystal_column[idx].as_py()
                    if crystal_val is not None:
                        crystal_array = np.array(crystal_val, dtype=np.float32)
                        # Handle flattened embeddings from dataset
                        if crystal_array.ndim == 1:
                            expected_size = 5 * self.config.embedding_dim
                            if crystal_array.size == expected_size:
                                crystal_data = crystal_array.reshape(5, self.config.embedding_dim)
                            else:
                                crystal_data = crystal_array
                        else:
                            crystal_data = crystal_array

                volume_data = 1.0
                if self._volume_column is not None:
                    volume_val = self._volume_column[idx].as_py()
                    if volume_val is not None:
                        volume_data = float(volume_val)

                # Cache in Trie node
                if crystal_data is not None:
                    node.embedding_cache = crystal_data
                    node.volume_cache = volume_data

                return {
                    "token": token,
                    "token_id": node.token_id,
                    "crystal": crystal_data,
                    "volume": volume_data,
                    "cached": False
                }

            except Exception as e:
                warnings.warn(f"Arrow fetch failed: {e}", RuntimeWarning)
                return None

        return None

    def get_batch_data(self, tokens: List[TokenStr]) -> Dict[TokenStr, Dict[str, Any]]:
        """Batch fetch using columnar lazy access"""
        if not self._initialized:
            return {}

        result: Dict[TokenStr, Dict[str, Any]] = {}
        indices_to_fetch: List[int] = []
        token_to_idx: Dict[str, int] = {}

        # Collect indices via Trie
        for token in tokens:
            node = self.token_trie.search(token)
            if node:
                if node.embedding_cache is not None:
                    result[token] = {
                        "token": token,
                        "token_id": node.token_id,
                        "crystal": node.embedding_cache,
                        "volume": node.volume_cache or 1.0,
                        "cached": True,
                    }
                elif node.row_indices:
                    idx = node.row_indices[0]
                    token_to_idx[token] = idx
                    indices_to_fetch.append(idx)

        if indices_to_fetch:
            try:
                for token, idx in token_to_idx.items():
                    crystal_data = None
                    if self._crystal_column is not None:
                        crystal_val = self._crystal_column[idx].as_py()
                        if crystal_val is not None:
                            crystal_data = np.array(crystal_val, dtype=np.float32)
                            if crystal_data.ndim == 1:
                                expected = 5 * self.config.embedding_dim
                                if crystal_data.size == expected:
                                    crystal_data = crystal_data.reshape(5, self.config.embedding_dim)

                    volume_data = 1.0
                    if self._volume_column is not None:
                        volume_val = self._volume_column[idx].as_py()
                        if volume_val is not None:
                            volume_data = float(volume_val)

                    result[token] = {
                        "token": token,
                        "token_id": self._token_to_id.get(token, -1),
                        "crystal": crystal_data,
                        "volume": volume_data,
                        "cached": False,
                    }
            except Exception as e:
                warnings.warn(f"Batch fetch error: {e}", RuntimeWarning)

        return result

    def token_exists(self, token: TokenStr) -> bool:
        """O(k) existence check via Trie"""
        if not self._initialized:
            return False
        return self.token_trie.search(token) is not None

    def find_similar_tokens(self, token: TokenStr, threshold: float = 0.3) -> List[Tuple[TokenStr, float]]:
        """Find similar tokens using n-gram index"""
        if not self._initialized:
            return []
        return self.ngram_index.find_similar(token, threshold)


# ============================================================================
# Crystal Factory
# ============================================================================

def regular_simplex_5():
    """Create regular 4-simplex vertices in 5D with unit edge length"""
    E = np.eye(5, dtype=np.float64)
    centroid = np.mean(E, axis=0, keepdims=True)
    S = E - centroid
    edge_length = np.linalg.norm(S[0] - S[1])
    S = S / edge_length
    return S.astype(np.float32)


class CrystalFactory:
    """Factory for creating crystals with various configurations"""

    def __init__(self, config: UnifiedCrystalConfig, data_manager: Optional[ArrowDataManager] = None):
        self.config = config
        self.dim = config.embedding_dim
        self.data_manager = data_manager
        self._initialize_handlers()
        self._char_cache: Dict[str, np.ndarray] = {}

    def _initialize_handlers(self):
        """Initialize handler mappings"""
        self.dimension_handlers = {
            DimensionType.D1: self._build_1d,
            DimensionType.D2: self._build_2d,
            DimensionType.D3: self._build_3d,
            DimensionType.D4: self._build_4d,
            DimensionType.D5: self._build_5d,
            DimensionType.D6_PLUS: self._build_nd,
        }

        self.formula_handlers = {
            FormulaType.ROSE_CAYLEY: self._apply_rose_cayley,
            FormulaType.CAYLEY_MENGER: self._apply_cayley_menger,
            FormulaType.CAYLEY: self._apply_cayley,
            FormulaType.MENGER: self._apply_menger,
            FormulaType.EULER: self._apply_euler,
            FormulaType.GRAHAM_INFINITE: self._apply_graham_infinite,
            FormulaType.GRAHAM_FINITE: self._apply_graham_finite,
            FormulaType.GRAHAM_MASKED: self._apply_graham_masked,
            FormulaType.HYBRID_V1V2: self._apply_hybrid_v1v2,
        }

        self.content_handlers = {
            ContentType.SPARSE: self._content_sparse,
            ContentType.ENRICHED: self._content_enriched,
            ContentType.TRAJECTORY: self._content_trajectory,
            ContentType.MAGNITUDE: self._content_magnitude,
            ContentType.VOLUME: self._content_volume,
            ContentType.HYBRID: self._content_hybrid,
        }

    def create_crystal(self, token: str, definition: Optional[str] = None,
                       override_config: Optional[Dict[str, Any]] = None,
                       use_dataset: bool = True) -> Dict[str, Any]:
        """Create crystal with optional dataset lookup"""
        config = self.config
        if override_config:
            config = replace(config, **override_config)

        # Try dataset first if available
        if use_dataset and self.data_manager and config.prefer_dataset and not definition:
            dataset_data = self.data_manager.get_token_data(token)
            if dataset_data and dataset_data.get("crystal") is not None:
                base_crystal = dataset_data["crystal"]

                # Handle different data formats
                if base_crystal.ndim == 1:
                    center = base_crystal
                    base_crystal = self._pentachoron_from_center(center, token)
                elif base_crystal.shape[0] != config.dimension_type.value:
                    base_crystal = self._reshape_crystal(base_crystal, config.dimension_type)

                # Apply transformations
                formula_crystal = self.formula_handlers[config.formula_type](base_crystal, token, definition)
                content_crystal = self.content_handlers[config.content_type](formula_crystal, token, definition)

                # Smart normalization based on content type
                if config.content_type == ContentType.VOLUME:
                    final_crystal = content_crystal
                elif config.norm_type != NormType.NONE:
                    if config.norm_type == NormType.L2:
                        current_scale = np.sqrt(np.mean(content_crystal * content_crystal))
                        target_scale = np.sqrt(config.embedding_dim) * 0.1
                        if current_scale > EPS:
                            final_crystal = content_crystal * (target_scale / current_scale)
                        else:
                            final_crystal = content_crystal
                    else:
                        final_crystal = self._apply_normalization(content_crystal, config.norm_type)
                else:
                    final_crystal = content_crystal

                final_crystal = final_crystal - final_crystal.mean(axis=0, keepdims=True)

                metadata = self._compute_metadata(final_crystal, token, definition)
                metadata["source"] = "dataset"

                return {
                    'crystal': final_crystal.astype(np.float32),
                    'volume': metadata["volume"],
                    'metadata': metadata,
                    'config': config
                }

        # Synthesize
        base_crystal = self.dimension_handlers[config.dimension_type](token, definition)
        base_crystal = base_crystal - base_crystal.mean(axis=0, keepdims=True)

        formula_crystal = self.formula_handlers[config.formula_type](base_crystal, token, definition)
        content_crystal = self.content_handlers[config.content_type](formula_crystal, token, definition)

        # Smart normalization
        if config.content_type == ContentType.VOLUME:
            final_crystal = content_crystal
        elif config.norm_type != NormType.NONE:
            if config.norm_type == NormType.L2:
                current_scale = np.sqrt(np.mean(content_crystal * content_crystal))
                target_scale = np.sqrt(config.embedding_dim) * 0.1
                if current_scale > EPS:
                    final_crystal = content_crystal * (target_scale / current_scale)
                else:
                    final_crystal = content_crystal
            else:
                final_crystal = self._apply_normalization(content_crystal, config.norm_type)
        else:
            final_crystal = content_crystal

        final_crystal = final_crystal - final_crystal.mean(axis=0, keepdims=True)

        metadata = self._compute_metadata(final_crystal, token, definition)
        metadata["source"] = "synthesized"

        return {
            'crystal': final_crystal.astype(np.float32),
            'volume': metadata['volume'],
            'metadata': metadata,
            'config': config
        }

    # Dimension builders
    def _build_1d(self, token: str, definition: Optional[str]) -> Crystal:
        return self._text_to_vec(definition or token).reshape(1, -1)

    def _build_2d(self, token: str, definition: Optional[str]) -> Crystal:
        X = np.zeros((2, self.dim), dtype=np.float32)
        X[0] = self._text_to_vec(token)
        X[1] = self._text_to_vec(definition or token[::-1])
        return X

    def _build_3d(self, token: str, definition: Optional[str]) -> Crystal:
        X = np.zeros((3, self.dim), dtype=np.float32)
        X[0] = self._text_to_vec(token)
        X[1] = self._text_to_vec(token[:len(token) // 2] if len(token) > 1 else token)
        X[2] = self._text_to_vec(definition or token)
        return X

    def _build_4d(self, token: str, definition: Optional[str]) -> Crystal:
        X = np.zeros((4, self.dim), dtype=np.float32)
        base = self._text_to_vec(token)
        for i in range(4):
            X[i] = self._rotate_vector(base, i * np.pi / 2)
        return X

    def _build_5d(self, token: str, definition: Optional[str]) -> Crystal:
        if definition:
            return self._build_5d_with_full_v1(token, definition)
        elif self.config.use_character_composition:
            return self._build_5d_from_characters(token)
        else:
            return self._build_5d_deterministic(token)

    def _build_nd(self, token: str, definition: Optional[str]) -> Crystal:
        n = self.config.dimension_type.value
        X = np.zeros((n, self.dim), dtype=np.float32)
        for i in range(n):
            binary = format(i, f'0{min(n, 32)}b')
            X[i] = self._text_to_vec(token + binary)
        return X

    def _build_5d_with_full_v1(self, token: str, definition: str) -> np.ndarray:
        """Full V1-style synthesis with projections and cardinal axes"""
        S5 = regular_simplex_5()

        # Build cardinal axes from definition
        C4 = self._cardinal_axes_from_definition_full(definition, token)

        # Build orthonormal frame
        Q5 = self._orthonormal_frame_5_full(C4, token, definition)

        # Project simplex through frame
        b = (S5 @ Q5.T).astype(np.float32)

        # Base trajectory from definition
        v0 = self._text_to_vec(definition)

        # Length-based scaling
        L = len(definition.encode('utf-8', errors='ignore'))
        base = float(np.clip(np.log1p(L) * 0.5, 0.5, 2.0))

        # Gamma factors for vertex scaling
        gamma = np.array([base, base * 0.9, -0.8 * base, base * 1.1, 1.2 * base], dtype=np.float32)

        # Project simplex vertices onto cardinal axes
        proj = np.zeros((5, 4), dtype=np.float32)
        for i in range(5):
            for k in range(4):
                proj[i, k] = float(np.dot(C4[k], b[i]))

        # Definition-weighted delta factors
        base_vec = np.array([L + 1, 1, 1, 1], dtype=np.float64)
        base_vec = base_vec / base_vec.sum()

        delta = np.tile(base_vec[None, :], (5, 1)).astype(np.float32)
        delta[1, 1] *= 1.2
        delta[2, 2] *= 1.5
        delta[3, 3] *= 1.2
        delta = delta / (delta.sum(axis=1, keepdims=True) + EPS)

        # Construct crystal vertices with full projections
        X = np.zeros((5, self.dim), dtype=np.float32)
        for i in range(5):
            xi = gamma[i] * b[i]
            for k in range(4):
                xi += delta[i, k] * proj[i, k] * C4[k]
            X[i] = v0 + xi

        # Center the crystal
        X -= X.mean(axis=0, keepdims=True)

        # Gentle scaling based on content
        if self.config.norm_type == NormType.L1:
            current_scale = np.mean(np.abs(X))
            target_scale = np.clip(np.log1p(L) * 0.3, 0.4, 2.0)
        else:
            current_scale = np.sqrt(np.mean(X * X))
            target_scale = np.clip(np.log1p(L) * 0.2, 0.25, 1.5)

        if current_scale > EPS:
            X *= (target_scale / current_scale)

        return X

    def _build_5d_from_characters(self, token: str) -> Crystal:
        """Character-based synthesis with exponential positional weighting
        NOTE: exp(-0.3 * i) gives weight 1.0 to first char (i=0), decreasing for later chars
        This is the intended behavior - first characters are most important."""
        char_vecs = []

        for char in token:
            if char in self._char_cache:
                char_vecs.append(self._char_cache[char])
            else:
                # Try dataset lookup
                if self.data_manager and self.data_manager.token_exists(char):
                    char_data = self.data_manager.get_token_data(char)
                    if char_data and char_data.get("crystal") is not None:
                        crystal_data = char_data["crystal"]
                        if crystal_data.ndim > 1:
                            pooled = crystal_data.mean(axis=0)
                        else:
                            pooled = crystal_data
                        self._char_cache[char] = pooled
                        char_vecs.append(pooled)
                        continue

                # Synthesize character
                char_vec = self._create_char_embedding(char)
                self._char_cache[char] = char_vec
                char_vecs.append(char_vec)

        if not char_vecs:
            return self._build_5d_deterministic(token)

        # Exponential decay weights: first char gets weight ~1.0, decreasing exponentially
        weights = np.array([np.exp(-0.3 * i) for i in range(len(char_vecs))])
        weights /= weights.sum()

        # Build center with interaction to ensure different tokens with same chars differ
        center = np.zeros(self.dim, dtype=np.float32)
        interaction_hash = self._sha_u64(token + "_char_interaction")

        for i, (weight, vec) in enumerate(zip(weights, char_vecs)):
            # Perturbation based on token-specific hash
            perturbation = 1.0 + ((interaction_hash >> (i * 4)) & 0xF) / 100.0 - 0.075
            center += weight * vec * perturbation

        return self._pentachoron_from_center(center, token)

    def _build_5d_deterministic(self, token: str) -> Crystal:
        """Pure deterministic synthesis"""
        center = self._deterministic_center(token)
        return self._pentachoron_from_center(center, token)

    def _pentachoron_from_center(self, center: np.ndarray, token: str) -> Crystal:
        """Build geometrically correct pentachoron with reasonable initial scale"""
        dim = len(center)
        token_hash = self._sha_u64(token)

        # Standard 4-simplex vertices in 5D (before embedding)
        simplex_5d = np.array([
            [1, 0, 0, 0, 0],
            [-0.25, 0.968, 0, 0, 0],
            [-0.25, -0.323, 0.913, 0, 0],
            [-0.25, -0.323, -0.457, 0.791, 0],
            [-0.25, -0.323, -0.457, -0.395, 0.686]
        ], dtype=np.float32)

        # Create random orthonormal basis in embedding dimension
        np.random.seed(token_hash % 2 ** 32)

        if dim >= 5:
            Q = np.random.randn(dim, 5).astype(np.float32)
            Q, _ = np.linalg.qr(Q)
            Q = Q[:, :5]
        else:
            Q = np.random.randn(dim, min(dim, 5)).astype(np.float32)
            Q, _ = np.linalg.qr(Q)
            simplex_5d = simplex_5d[:, :dim]

        # Embed simplex in the space
        X = (simplex_5d @ Q.T).astype(np.float32)

        # Scale appropriately for the embedding dimension
        token_length = len(token.encode('utf-8'))
        base_scale = np.sqrt(dim) * 0.5
        length_factor = 1.0 + np.log1p(token_length) * 0.2
        scale = base_scale * length_factor

        X = X * scale

        # Add center and token-specific variations
        state = token_hash
        for i in range(5):
            state = (state * 0x9E3779B97F4A7C15) & ((1 << 64) - 1)

            perturbation = np.zeros(dim, dtype=np.float32)
            for j in range(min(dim, 10)):
                perturbation[state % dim] += (state % 1000 - 500) / 5000.0
                state = (state * 1099511628211) & ((1 << 64) - 1)

            X[i] = X[i] + center + perturbation

        # Center the crystal
        X = X - X.mean(axis=0, keepdims=True)

        return X

    def _cardinal_axes_from_definition_full(self, def_text: str, token: str) -> np.ndarray:
        """Full V1 cardinal axes with QR decomposition"""
        v_def = self._text_to_vec(def_text).astype(np.float64)

        C = np.zeros((4, self.dim), dtype=np.float64)
        built = 0

        # First axis: definition direction
        n = float(np.linalg.norm(v_def) if self.config.norm_type == NormType.L2 else np.abs(v_def).sum())
        if n > EPS:
            C[0] = v_def / n
            built = 1

        # Build remaining axes deterministically
        state = self._sha_u64(token) ^ 0xD1F2C3B4A5968778
        mask = (1 << 64) - 1

        while built < 4:
            h = np.zeros(self.dim, dtype=np.float64)
            for _ in range(8):
                state ^= 0x9E3779B97F4A7C15
                state = (state * 1099511628211) & mask
                h[state % self.dim] += 1.0

            # Orthogonalize against existing axes
            vk = h
            for j in range(built):
                vk -= np.dot(vk, C[j]) * C[j]

            n = float(np.linalg.norm(vk) if self.config.norm_type == NormType.L2 else np.abs(vk).sum())
            if n <= EPS:
                idx = (state >> 5) % self.dim
                vk = np.zeros(self.dim)
                vk[idx] = 1.0
                for j in range(built):
                    vk -= np.dot(vk, C[j]) * C[j]
                n = float(np.linalg.norm(vk) if self.config.norm_type == NormType.L2 else np.abs(vk).sum())

            C[built] = vk / (n + EPS)
            built += 1

        # QR decomposition for numerical stability
        M = C.T
        Qr, _ = np.linalg.qr(M, mode='reduced')
        return Qr.T.astype(np.float32)

    def _orthonormal_frame_5_full(self, C4: np.ndarray, token: str, def_text: str) -> np.ndarray:
        """Full orthonormal frame extension with QR"""
        Q = np.zeros((self.dim, 5), dtype=np.float64)

        # Copy cardinal axes
        for k in range(4):
            Q[:, k] = C4[k].astype(np.float64)

        # Fifth axis: orthogonal to all cardinals
        v5 = self._text_to_vec(def_text if def_text else token).astype(np.float64)

        # Orthogonalize
        for k in range(4):
            v5 -= np.dot(v5, Q[:, k]) * Q[:, k]

        n = float(np.linalg.norm(v5) if self.config.norm_type == NormType.L2 else np.abs(v5).sum())
        if n <= EPS:
            # Deterministic fallback
            state = self._sha_u64(token) ^ 0xABCDEF9876543210
            mask = (1 << 64) - 1
            h = np.zeros(self.dim, dtype=np.float64)

            for _ in range(12):
                state ^= 0x9E3779B97F4A7C15
                state = (state * 1099511628211) & mask
                h[state % self.dim] += 1.0

            v5 = h
            for k in range(4):
                v5 -= np.dot(v5, Q[:, k]) * Q[:, k]
            n = float(np.linalg.norm(v5) if self.config.norm_type == NormType.L2 else np.abs(v5).sum())

            if n <= EPS:
                idx = (state >> 7) % self.dim
                v5 = np.zeros(self.dim)
                v5[idx] = 1.0
                for k in range(4):
                    v5 -= np.dot(v5, Q[:, k]) * Q[:, k]
                n = float(np.linalg.norm(v5) if self.config.norm_type == NormType.L2 else np.abs(v5).sum())

        Q[:, 4] = v5 / (n + EPS)

        # Final QR for numerical stability
        Qr, _ = np.linalg.qr(Q, mode='reduced')
        return Qr.astype(np.float32)

    # Formula applications
    def _apply_rose_cayley(self, crystal: Crystal, token: str, definition: Optional[str]) -> Crystal:
        """Rose-Cayley with proper semantic distance preservation"""
        if crystal.shape[0] != 5:
            return crystal

        rose = crystal.copy()
        anchor = rose[0].copy()

        for i in range(1, 5):
            # Semantic distance scaling
            semantic_scale = 1.0 / (1.0 + i * 0.25)

            # Angular position in rose pattern
            angle = i * (2 * np.pi / 4)

            # Apply scaling from anchor
            direction = rose[i] - anchor
            rose[i] = anchor + direction * semantic_scale

            # Apply rotation in primary plane
            if rose.shape[1] >= 2:
                cos_a = np.cos(angle)
                sin_a = np.sin(angle)

                rel_pos = rose[i] - anchor
                rotated = rel_pos.copy()
                rotated[0] = cos_a * rel_pos[0] - sin_a * rel_pos[1]
                rotated[1] = sin_a * rel_pos[0] + cos_a * rel_pos[1]
                rose[i] = anchor + rotated

        # Re-center
        rose = rose - rose.mean(axis=0, keepdims=True)

        return rose

    def _apply_cayley_menger(self, crystal: Crystal, token: str, definition: Optional[str]) -> Crystal:
        n = crystal.shape[0]
        crystal = crystal - crystal.mean(axis=0, keepdims=True)

        distances = []
        for i in range(n):
            for j in range(i + 1, n):
                distances.append(np.linalg.norm(crystal[i] - crystal[j]))

        if distances:
            target_edge = 1.5
            current_edge = np.median(distances)
            if current_edge > EPS:
                crystal *= (target_edge / current_edge)

        return crystal

    def _apply_cayley(self, crystal: Crystal, token: str, definition: Optional[str]) -> Crystal:
        """Pure Cayley determinant optimization"""
        n = crystal.shape[0]

        # Build distance matrix
        D = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    D[i, j] = np.linalg.norm(crystal[i] - crystal[j]) ** 2

        # Optimize Cayley determinant
        det = np.linalg.det(D)

        # Scale to optimize determinant
        if abs(det) > EPS:
            target_det = (2.0 ** n) * (n ** n)
            scale = np.power(abs(target_det / det), 1.0 / (2 * n))
            crystal *= scale
        else:
            # Degenerate - add small perturbation
            crystal += np.random.randn(*crystal.shape) * 0.01

        return crystal

    def _apply_menger(self, crystal: Crystal, token: str, definition: Optional[str]) -> Crystal:
        """Menger curvature optimization"""
        n = crystal.shape[0]

        if n >= 4:
            curvatures = []

            for i in range(n - 3):
                p1, p2, p3, p4 = crystal[i:i + 4]

                edges = [
                    np.linalg.norm(p2 - p1),
                    np.linalg.norm(p3 - p2),
                    np.linalg.norm(p4 - p3),
                    np.linalg.norm(p4 - p1),
                    np.linalg.norm(p3 - p1),
                    np.linalg.norm(p4 - p2)
                ]

                if min(edges) > EPS:
                    curv = 1.0 / min(edges)
                    curvatures.append(curv)

            if curvatures:
                target_curv = 0.5
                avg_curv = np.mean(curvatures)

                if avg_curv > EPS:
                    scale = target_curv / avg_curv
                    crystal *= scale

        return crystal

    def _apply_euler(self, crystal: Crystal, token: str, definition: Optional[str]) -> Crystal:
        angles = [np.pi / 6, np.pi / 4, np.pi / 3]

        for angle in angles[:1]:
            cos_a = np.cos(angle)
            sin_a = np.sin(angle)

            for i in range(crystal.shape[0]):
                temp = crystal[i, 0] * cos_a - crystal[i, 1] * sin_a
                crystal[i, 1] = crystal[i, 0] * sin_a + crystal[i, 1] * cos_a
                crystal[i, 0] = temp

        return crystal

    def _apply_graham_infinite(self, crystal: Crystal, token: str, definition: Optional[str]) -> Crystal:
        """Graham infinite with controlled tetration growth"""
        n_vertices = crystal.shape[0]

        for i in range(n_vertices):
            if i == 0:
                g_level = 3
            elif i == 1:
                g_level = 27
            elif i == 2:
                g_level = min(3 ** 27, 1e15)  # Cap for stability
            elif i == 3:
                g_level = 1e20
            else:
                g_level = 10 ** (10 + i * 2)

            scale = np.log1p(g_level) / ((i + 1) * 2.0)
            crystal[i] *= scale

        return crystal

    def _apply_graham_finite(self, crystal: Crystal, token: str, definition: Optional[str]) -> Crystal:
        """Graham finite with bounded growth"""
        n_vertices = crystal.shape[0]

        for i in range(n_vertices):
            if i == 0:
                g_level = 3
            elif i == 1:
                g_level = 27
            elif i == 2:
                g_level = 7625
            else:
                g_level = 1000 * (i + 1) ** 3

            scale = np.log1p(g_level) / (i + 2)
            crystal[i] *= scale

        return crystal

    def _apply_graham_masked(self, crystal: Crystal, token: str, definition: Optional[str]) -> Crystal:
        mask = self.config.graham_mask if self.config.graham_mask is not None else np.random.binomial(1, 0.7,
                                                                                                      crystal.shape)
        graham = self._apply_graham_finite(crystal, token, definition)
        return crystal * (1 - mask) + graham * mask

    def _apply_hybrid_v1v2(self, crystal: Crystal, token: str, definition: Optional[str]) -> Crystal:
        if definition:
            v_def = self._text_to_vec(definition)
            for i in range(crystal.shape[0]):
                crystal[i] = crystal[i] * 0.7 + v_def * 0.3
        return crystal

    # Content applications
    def _content_sparse(self, crystal: Crystal, token: str, definition: Optional[str]) -> Crystal:
        for i in range(crystal.shape[0]):
            char = token[i % len(token)] if token else ' '
            crystal[i] = self._create_char_embedding(char)
        return crystal

    def _content_enriched(self, crystal: Crystal, token: str, definition: Optional[str]) -> Crystal:
        if definition:
            def_vec = self._text_to_vec(definition)
            for i in range(crystal.shape[0]):
                crystal[i] = crystal[i] * 0.8 + def_vec * 0.2
        return crystal

    def _content_trajectory(self, crystal: Crystal, token: str, definition: Optional[str]) -> Crystal:
        """Create smooth trajectory through vertices"""
        if crystal.shape[0] <= 1:
            return crystal

        trajectory = crystal.copy()
        alpha = 0.7  # smoothing factor

        for i in range(1, crystal.shape[0]):
            trajectory[i] = alpha * trajectory[i - 1] + (1 - alpha) * crystal[i]

        return trajectory

    def _content_magnitude(self, crystal: Crystal, token: str, definition: Optional[str]) -> Crystal:
        magnitudes = np.linalg.norm(crystal, axis=1)
        target_mag = np.mean(magnitudes)
        for i in range(crystal.shape[0]):
            if magnitudes[i] > EPS:
                crystal[i] *= (target_mag / magnitudes[i])
        return crystal

    def _content_volume(self, crystal: Crystal, token: str, definition: Optional[str]) -> Crystal:
        """Volume regularization - scale to target volume"""
        vol = self._compute_volume(crystal)
        target_vol = 1.0

        if vol > EPS:
            # Scale to achieve target volume
            n = crystal.shape[0] - 1  # Dimension of simplex
            if n > 0:
                scale = np.power(target_vol / vol, 1.0 / n)
                crystal *= scale

        return crystal

    def _content_hybrid(self, crystal: Crystal, token: str, definition: Optional[str]) -> Crystal:
        if definition:
            crystal = self._content_enriched(crystal, token, definition)
        if crystal.shape[0] > 3:
            crystal = self._content_trajectory(crystal, token, definition)
        return self._content_volume(crystal, token, definition)

    # Utilities
    def _apply_normalization(self, crystal: Crystal, norm_type: NormType) -> Crystal:
        """Apply normalization to the whole crystal structure"""
        if norm_type == NormType.NONE:
            return crystal

        # Handle both 1D and 2D crystals
        if crystal.ndim == 1:
            if norm_type == NormType.L1:
                total_norm = np.abs(crystal).sum()
            elif norm_type == NormType.L2:
                total_norm = np.linalg.norm(crystal)
            elif norm_type == NormType.LINF:
                total_norm = np.max(np.abs(crystal))
            else:
                return crystal

            if total_norm > EPS:
                return crystal / total_norm
            return crystal

        # Normalize the entire crystal structure
        if norm_type == NormType.L1:
            total_norm = np.abs(crystal).sum()
            if total_norm > EPS:
                return crystal / total_norm
        elif norm_type == NormType.L2:
            # Frobenius norm for matrices
            total_norm = np.linalg.norm(crystal, 'fro')
            if total_norm > EPS:
                return crystal / total_norm
        elif norm_type == NormType.LINF:
            max_val = np.max(np.abs(crystal))
            if max_val > EPS:
                return crystal / max_val

        return crystal

    def _text_to_vec(self, text: str) -> np.ndarray:
        """FNV hash-based text vectorization"""
        acc = np.zeros(self.dim, dtype=np.float64)
        b = text.encode('utf-8', errors='ignore')
        state = 1469598103934665603
        FNV = 1099511628211
        mask = (1 << 64) - 1

        for by in b:
            state ^= by
            state = (state * FNV) & mask
            acc[state % self.dim] += 1.0

        n = float(np.linalg.norm(acc))
        return (acc / n if n > EPS else acc).astype(np.float32)

    def _sha_u64(self, s: str) -> int:
        h = hashlib.sha256(s.encode('utf-8')).digest()
        return int.from_bytes(h[:8], 'little', signed=False)

    def _create_char_embedding(self, char: str) -> np.ndarray:
        seed = ord(char) if len(char) == 1 else hash(char)
        np.random.seed(seed % 2 ** 32)
        vec = np.random.randn(self.dim).astype(np.float32)
        return vec / (np.abs(vec).sum() + EPS)

    def _deterministic_center(self, token: str) -> np.ndarray:
        state = self._sha_u64(token)
        np.random.seed(state % 2 ** 32)
        vec = np.random.randn(self.dim).astype(np.float32)
        return vec / (np.abs(vec).sum() + EPS)

    def _rotate_vector(self, vec: np.ndarray, angle: float) -> np.ndarray:
        rotated = vec.copy()
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        temp = rotated[0] * cos_a - rotated[1] * sin_a
        rotated[1] = rotated[0] * sin_a + rotated[1] * cos_a
        rotated[0] = temp
        return rotated

    def _reshape_crystal(self, crystal: Crystal, target_dim: DimensionType) -> Crystal:
        current_n = crystal.shape[0]
        target_n = target_dim.value

        if current_n == target_n:
            return crystal

        if target_n < current_n:
            indices = np.linspace(0, current_n - 1, target_n, dtype=int)
            return crystal[indices]
        else:
            new_crystal = np.zeros((target_n, crystal.shape[1]), dtype=np.float32)
            for i in range(min(current_n, target_n)):
                new_crystal[i] = crystal[i]
            for i in range(current_n, target_n):
                idx1 = i % current_n
                idx2 = (i + 1) % current_n
                alpha = (i - current_n) / (target_n - current_n)
                new_crystal[i] = crystal[idx1] * (1 - alpha) + crystal[idx2] * alpha
            return new_crystal

    def _compute_volume(self, crystal: Crystal) -> float:
        """CORRECTED Cayley-Menger volume calculation"""
        n = crystal.shape[0]

        # Handle special case for D1 (single vertex has no volume)
        if n == 1:
            return 0.0

        if n < 2:
            return 0.0

        # Build squared distance matrix
        d2 = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(i + 1, n):
                v = crystal[i] - crystal[j]
                d2[i, j] = d2[j, i] = float(np.dot(v, v))

        if n == 2:
            return np.sqrt(d2[0, 1])

        # Cayley-Menger matrix
        M = np.ones((n + 1, n + 1), dtype=np.float64)
        M[0, 0] = 0.0
        M[1:, 1:] = d2

        det = np.linalg.det(M)

        # CRITICAL FIX: Correct volume formulas for (n-1)-simplex
        k = n - 1

        if k == 1:  # Line
            vol2 = abs(det) / 4.0
        elif k == 2:  # Triangle
            vol2 = -det / 16.0
        elif k == 3:  # Tetrahedron
            vol2 = det / 288.0
        elif k == 4:  # 4-simplex
            vol2 = -det / 9216.0
        elif k == 5:  # 5-simplex
            vol2 = det / 460800.0
        elif k == 6:  # 6-simplex
            vol2 = -det / 33177600.0
        else:
            # General formula
            sign = (-1) ** (k + 1)
            factorial_k = 1
            for i in range(1, k + 1):
                factorial_k *= i
            divisor = (2 ** k) * (factorial_k ** 2)
            vol2 = sign * det / divisor

        return float(np.sqrt(abs(vol2))) if abs(vol2) > EPS else 0.0

    def _compute_metadata(self, crystal: Crystal, token: str, definition: Optional[str]) -> Dict[str, Any]:
        metadata = {
            'token': token,
            'has_definition': definition is not None,
        }

        # Handle both 1D and 2D crystals
        if crystal.ndim == 1:
            metadata['n_vertices'] = 1
            metadata['embedding_dim'] = crystal.shape[0]
            metadata['volume'] = 0.0
        else:
            metadata['n_vertices'] = crystal.shape[0]
            metadata['embedding_dim'] = crystal.shape[1]
            metadata['volume'] = self._compute_volume(crystal)

            # Edge statistics
            distances = []
            for i in range(crystal.shape[0]):
                for j in range(i + 1, crystal.shape[0]):
                    distances.append(np.linalg.norm(crystal[i] - crystal[j]))

            if distances:
                metadata['edge_mean'] = float(np.mean(distances))
                metadata['edge_std'] = float(np.std(distances))

        return metadata

    def validate_and_fix_crystal(self, crystal: np.ndarray) -> np.ndarray:
        """Validate and attempt to fix crystal issues"""
        validation = validate_crystal(crystal)

        if not validation['valid']:
            warnings.warn(f"Invalid crystal: {validation['errors']}", RuntimeWarning)
            # Attempt to fix
            crystal = np.nan_to_num(crystal, nan=0.0, posinf=1.0, neginf=-1.0)

        # Re-center if needed
        crystal = crystal - crystal.mean(axis=0, keepdims=True)

        return crystal


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
        """Get crystal embedding with FIXED cache key handling"""
        # FIX #2: Use proper cache key when definition is provided
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

def validate_crystal(crystal: np.ndarray) -> Dict[str, Any]:
    """Validate crystal geometry"""
    validation = {
        'valid': True,
        'errors': [],
        'warnings': []
    }

    # Check shape
    if crystal.ndim != 2:
        validation['valid'] = False
        validation['errors'].append(f"Crystal must be 2D, got {crystal.ndim}D")
        return validation

    n_vertices, dim = crystal.shape

    # Check centering
    centroid = crystal.mean(axis=0)
    centroid_norm = np.linalg.norm(centroid)
    if centroid_norm > 0.1:
        validation['warnings'].append(f"Crystal not centered: centroid norm = {centroid_norm:.4f}")

    # Check edge lengths
    if n_vertices > 1:
        distances = []
        for i in range(n_vertices):
            for j in range(i + 1, n_vertices):
                distances.append(np.linalg.norm(crystal[i] - crystal[j]))

        if distances:
            min_dist = min(distances)
            max_dist = max(distances)

            if min_dist < 1e-6:
                validation['valid'] = False
                validation['errors'].append(f"Degenerate crystal: min edge = {min_dist:.8f}")

            if max_dist / (min_dist + 1e-8) > 100:
                validation['warnings'].append(f"High edge ratio: {max_dist / min_dist:.2f}")

    # Check for NaN or Inf
    if np.any(np.isnan(crystal)):
        validation['valid'] = False
        validation['errors'].append("Crystal contains NaN values")

    if np.any(np.isinf(crystal)):
        validation['valid'] = False
        validation['errors'].append("Crystal contains Inf values")

    # Check volume (if applicable)
    if 2 <= n_vertices <= 6:
        factory = CrystalFactory(UnifiedCrystalConfig())
        volume = factory._compute_volume(crystal)

        if volume < 1e-10:
            validation['warnings'].append(f"Near-degenerate volume: {volume:.8f}")

        validation['volume'] = volume

    return validation


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