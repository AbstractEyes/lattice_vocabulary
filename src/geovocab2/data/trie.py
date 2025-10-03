from __future__ import annotations
import numpy as np
import os
import warnings
import threading
from typing import Dict, List, Optional, Tuple, Any, Set
from collections import defaultdict
from src.geovocab2.defaults import TokenID, TokenStr, Crystal, UnifiedCrystalConfig

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

    def fuzzy_search(self, token: str, max_distance: int = 2, max_returned: int = 20) -> List[Tuple[str, TrieNode, int]]:
        """Fuzzy search using edit distance with bounded search"""
        results = []
        with self._lock:
            max_depth = len(token) + max_distance
            self._fuzzy_dfs_bounded(self.root, "", token, max_distance, results, max_depth, 0)
        return sorted(results, key=lambda x: x[2])[:max_returned]

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


if __name__ == "__main__":

    # Example usage
    config = UnifiedCrystalConfig(
        repo_id="AbstractPhil/geometric-vocab",
        dataset_name="unicode_100d",
        split="train"
    )
    manager = ArrowDataManager(config)
    if manager.initialize():
        token = "example_token"
        data = manager.get_token_data(token)
        if data:
            print(f"Token: {data['token']}, ID: {data['token_id']}, Crystal shape: {data['crystal'].shape if data['crystal'] is not None else 'None'}")
        else:
            print(f"Token '{token}' not found.")