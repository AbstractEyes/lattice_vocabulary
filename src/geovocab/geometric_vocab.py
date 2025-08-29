from __future__ import annotations
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Union, Tuple, Optional, Callable, Any, List
import warnings
from collections import defaultdict

# Optional dependencies for spatial indexing
try:
    import faiss

    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

try:
    from sklearn.neighbors import NearestNeighbors

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class SpatialIndex:
    """Spatial indexing for fast similarity search."""

    def __init__(self, vectors: np.ndarray, token_ids: List[int], method: str = "auto"):
        self.token_ids = np.array(token_ids)
        self.method = method
        self._index = None

        if method == "auto":
            if FAISS_AVAILABLE and vectors.shape[0] > 1000:
                method = "faiss"
            elif SKLEARN_AVAILABLE:
                method = "sklearn"
            else:
                method = "linear"

        self._build_index(vectors, method)

    def _build_index(self, vectors: np.ndarray, method: str):
        if method == "faiss" and FAISS_AVAILABLE:
            # L1 distance approximation using L2 index with normalized vectors
            vectors_l2 = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8)
            self._index = faiss.IndexFlatIP(vectors_l2.shape[1])  # Inner product for normalized vectors
            self._index.add(vectors_l2.astype(np.float32))
            self.method = "faiss"

        elif method == "sklearn" and SKLEARN_AVAILABLE:
            # Use manhattan distance for true L1
            self._index = NearestNeighbors(
                metric='manhattan',
                algorithm='ball_tree',
                n_jobs=-1
            ).fit(vectors)
            self.method = "sklearn"
        else:
            # Fallback to linear search
            self._vectors = vectors
            self.method = "linear"

    def search_radius(self, query_vector: np.ndarray, max_distance: float, max_results: int = 1000) -> Tuple[
        List[int], List[float]]:
        """Find all points within max_distance using L1 metric."""
        if self.method == "sklearn":
            indices = self._index.radius_neighbors([query_vector], radius=max_distance)[1][0]
            if len(indices) > max_results:
                # Compute actual distances and take closest
                distances = np.sum(np.abs(self._vectors[indices] - query_vector), axis=1)
                top_k = np.argsort(distances)[:max_results]
                indices = indices[top_k]
            distances = np.sum(np.abs(self._vectors[indices] - query_vector), axis=1)
            return self.token_ids[indices].tolist(), distances.tolist()

        elif self.method == "faiss":
            # Approximate search using cosine similarity
            query_l2 = query_vector / (np.linalg.norm(query_vector) + 1e-8)
            similarities, indices = self._index.search(query_l2.reshape(1, -1).astype(np.float32), max_results)
            # Filter by converting similarity threshold to approximate distance
            threshold_sim = 1.0 - max_distance  # rough approximation
            mask = similarities[0] >= threshold_sim
            return self.token_ids[indices[0][mask]].tolist(), (1.0 - similarities[0][mask]).tolist()

        else:  # linear
            distances = np.sum(np.abs(self._vectors - query_vector), axis=1)
            mask = distances <= max_distance
            if np.sum(mask) > max_results:
                indices = np.argsort(distances)[:max_results]
                mask = np.zeros_like(distances, dtype=bool)
                mask[indices] = True
            return self.token_ids[mask].tolist(), distances[mask].tolist()


class GeometricVocab(ABC):
    """
    Optimized geometric vocabulary with spatial indexing and caching.
    """

    def __init__(self, dim: int):
        self.dim = int(dim)
        self._token_to_id: Dict[str, int] = {}
        self._id_to_token: Dict[int, str] = {}
        self._id_to_vec: Dict[int, np.ndarray] = {}
        self._id_to_volume: Dict[int, float] = {}
        self._id_to_provenance: Dict[int, dict] = {}
        self._valid_token_ids: set[int] = set()

        # Optimization caches
        self._normalized_cache: Dict[int, np.ndarray] = {}
        self._pooled_cache: Dict[int, np.ndarray] = {}
        self._spatial_index: Optional[SpatialIndex] = None
        self._index_dirty = False

        self._dense_ids: Optional[np.ndarray] = None            # shape [M]
        self._dense_tokens: Optional[List[str]] = None          # len M
        self._dense_pooled: Optional[np.ndarray] = None         # [M, D] float32
        self._dense_l1norm: Optional[np.ndarray] = None         # [M, D] float32
        self._dense_dirty: bool = True
        self._similarity_mode: str = "l1dot"  # "l1dot" | "cosine"

    def _invalidate_caches(self):
        """Invalidate caches when vocabulary changes."""
        self._normalized_cache.clear()
        self._pooled_cache.clear()
        self._spatial_index = None
        self._index_dirty = True


    def _ensure_spatial_index(self):
        """Build spatial index if needed."""
        if self._spatial_index is None or self._index_dirty:
            if len(self._valid_token_ids) < 10:
                return  # Too few tokens for indexing

            pooled_vectors = []
            token_ids = []
            for tid in sorted(self._valid_token_ids):
                pooled_vec = self._get_cached_pooled(tid)
                if pooled_vec is not None:
                    pooled_vectors.append(pooled_vec)
                    token_ids.append(tid)

            if pooled_vectors:
                self._spatial_index = SpatialIndex(
                    np.array(pooled_vectors),
                    token_ids,
                    method="auto"
                )
                self._index_dirty = False

    def _get_cached_pooled(self, token_id: int) -> Optional[np.ndarray]:
        """Get pooled vector with caching."""
        if token_id in self._pooled_cache:
            return self._pooled_cache[token_id]

        if token_id in self._id_to_vec:
            X = self._id_to_vec[token_id]
            pooled = X.mean(axis=0)
            self._pooled_cache[token_id] = pooled
            return pooled
        return None

    def _get_cached_normalized(self, token_id: int) -> Optional[np.ndarray]:
        """Get L1-normalized pooled vector with caching."""
        if token_id in self._normalized_cache:
            return self._normalized_cache[token_id]

        pooled = self._get_cached_pooled(token_id)
        if pooled is not None:
            normalized = pooled / (np.abs(pooled).sum() + 1e-8)
            self._normalized_cache[token_id] = normalized
            return normalized
        return None

    # --------------------------- abstract surface --------------------
    @abstractmethod
    def encode(self, token: str, *, return_id: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, int]]:
        raise NotImplementedError

    @abstractmethod
    def get_score(self, token_or_id: Union[str, int]) -> float:
        raise NotImplementedError

    # --------------------------- basic queries (optimized) -----------------------
    def decode(self, token_id: int, fallback: str = "<unk>") -> Optional[str]:
        if token_id in self._id_to_token:
            return self._id_to_token[token_id]
        return fallback if fallback in self._token_to_id else None

    def decode_with_provenance(self, token_id: int, fallback: str = "<unk>") -> Tuple[Optional[str], Optional[dict]]:
        tok = self.decode(token_id, fallback=fallback)
        prov = self._id_to_provenance.get(token_id)
        return tok, prov

    def provenance(self, token_or_id: Union[str, int]) -> Optional[dict]:
        tid = token_or_id if isinstance(token_or_id, int) else self._token_to_id.get(token_or_id)
        return self._id_to_provenance.get(tid)

    def embedding(self, token_or_id: Union[str, int]) -> Optional[np.ndarray]:
        tid = token_or_id if isinstance(token_or_id, int) else self._token_to_id.get(token_or_id)
        return self._id_to_vec.get(tid)

    def pooled(self, token_or_id: Union[str, int], method: str = "mean") -> Optional[np.ndarray]:
        tid = token_or_id if isinstance(token_or_id, int) else self._token_to_id.get(token_or_id)
        if tid is None:
            return None

        if method == "mean":
            return self._get_cached_pooled(tid)

        # Fallback for other methods
        X = self._id_to_vec.get(tid)
        if X is None:
            return None
        if method == "first":
            return X[0]
        if method == "sum":
            return X.sum(axis=0)
        raise ValueError(f"Invalid pooling method: {method}")

    # --------------------------- optimized similarity ---------------------
    def similarity(self, token_a: Union[str, int], token_b: Union[str, int]) -> float:
        """
        Optimized L1-normalized directional similarity using cached vectors.
        """
        tid_a = token_a if isinstance(token_a, int) else self._token_to_id.get(token_a)
        tid_b = token_b if isinstance(token_b, int) else self._token_to_id.get(token_b)

        if tid_a is None or tid_b is None:
            return -1.0

        a_norm = self._get_cached_normalized(tid_a)
        b_norm = self._get_cached_normalized(tid_b)

        if a_norm is None or b_norm is None:
            return -1.0

        return float(np.dot(a_norm, b_norm))

    def similarity_magnitude(self, token_a: Union[str, int], token_b: Union[str, int]) -> float:
        """
        Raw dot-product using cached pooled vectors.
        """
        tid_a = token_a if isinstance(token_a, int) else self._token_to_id.get(token_a)
        tid_b = token_b if isinstance(token_b, int) else self._token_to_id.get(token_b)

        if tid_a is None or tid_b is None:
            return -1.0

        a = self._get_cached_pooled(tid_a)
        b = self._get_cached_pooled(tid_b)

        if a is None or b is None:
            return -1.0

        return float(np.dot(a, b))

    # --------------------------- optimized spatial search ---------------------
    def extract_band(self, trajectory: np.ndarray, max_angle: float = 0.3, method: str = "pooled") -> Dict[
        str, np.ndarray]:
        """
        Optimized spatial search using indexing when available.
        """
        if trajectory.ndim == 2:
            direction = trajectory.mean(0)
        else:
            direction = trajectory
        direction = direction / (np.abs(direction).sum() + 1e-8)

        # Try spatial index first
        self._ensure_spatial_index()
        if self._spatial_index is not None:
            try:
                # Convert angle threshold to distance threshold (approximation)
                max_distance = max_angle * 2.0  # rough conversion
                token_ids, distances = self._spatial_index.search_radius(
                    direction, max_distance, max_results=1000
                )

                # Refine results with exact L1 similarity check
                out: Dict[str, np.ndarray] = {}
                for tid in token_ids:
                    tok = self._id_to_token.get(tid)
                    if tok is None:
                        continue
                    v_norm = self._get_cached_normalized(tid)
                    if v_norm is not None and float(np.dot(v_norm, direction)) >= 1.0 - max_angle:
                        out[tok] = self._id_to_vec[tid]
                return out

            except Exception as e:
                warnings.warn(f"Spatial index search failed: {e}, falling back to linear")

        # Fallback to linear search
        out: Dict[str, np.ndarray] = {}
        for tok, tid in self._token_to_id.items():
            v_norm = self._get_cached_normalized(tid)
            if v_norm is not None and float(np.dot(v_norm, direction)) >= 1.0 - max_angle:
                out[tok] = self._id_to_vec[tid]
        return out

    def find_similar_tokens(self, token: Union[str, int], k: int = 10, min_similarity: float = 0.5) -> List[
        Tuple[str, float]]:
        """
        Find k most similar tokens using spatial indexing when available.
        """
        tid = token if isinstance(token, int) else self._token_to_id.get(token)
        if tid is None:
            return []

        query_vec = self._get_cached_normalized(tid)
        if query_vec is None:
            return []

        self._ensure_spatial_index()
        if self._spatial_index is not None:
            try:
                # Use spatial index for approximate search
                max_distance = (1.0 - min_similarity) * 2.0
                token_ids, _ = self._spatial_index.search_radius(
                    query_vec, max_distance, max_results=k * 3  # Get extra for refinement
                )

                # Compute exact similarities and sort
                similarities = []
                for tid_cand in token_ids:
                    if tid_cand == tid:  # Skip self
                        continue
                    sim = self.similarity(tid, tid_cand)
                    if sim >= min_similarity:
                        tok = self._id_to_token.get(tid_cand)
                        if tok:
                            similarities.append((tok, sim))

                return sorted(similarities, key=lambda x: x[1], reverse=True)[:k]

            except Exception as e:
                warnings.warn(f"Spatial similarity search failed: {e}, falling back to linear")

        # Linear fallback
        similarities = []
        for tok_cand, tid_cand in self._token_to_id.items():
            if tid_cand == tid:
                continue
            sim = self.similarity(tid, tid_cand)
            if sim >= min_similarity:
                similarities.append((tok_cand, sim))

        return sorted(similarities, key=lambda x: x[1], reverse=True)[:k]

    # --------------------------- helpers exposed to callbacks --------
    def _helpers(self) -> Dict[str, Callable[..., np.ndarray]]:
        def _emb(x):
            e = self.embedding(x)
            return None if e is None else np.asarray(e, np.float32)

        def _poo(x):
            p = self.pooled(x)
            return None if p is None else np.asarray(p, np.float32)

        def _chars(s):
            return [self.pooled(c) for c in s] if isinstance(s, str) else None

        return {"embedding": _emb, "pooled": _poo, "chars_pooled": _chars}

    # --------------------------- DEFAULT create_crystal (unicode path) ----
    def _default_create_crystal(self, config: dict, callback: Callable[..., np.ndarray]) -> np.ndarray:
        """
        Deterministic default when user leaves callback/create_crystal=None.
        """
        pool_type = config.get("pool_type") or "unicode"
        H = config["helpers"]
        token_plain = str(config["data"]["token"])
        d = int(config["dim"])

        c_uni = self._compose_unicode_center(token_plain, H, pool_type, d)
        c_defs = self._compose_wordnet_center(config.get("additional_definitions", []), H, pool_type, d)

        if pool_type == "combination":
            parts = [v for v in (c_uni, c_defs) if v is not None]
            c = np.mean(np.stack(parts, 0), 0) if parts else np.zeros(d, np.float32)
        elif pool_type == "wordnet":
            c = c_defs if c_defs is not None else np.zeros(d, np.float32)
        else:
            c = c_uni if c_uni is not None else np.zeros(d, np.float32)

        # L1 normalization only
        l1 = float(np.abs(c).sum()) + 1e-8
        c = c / l1
        return self._deterministic_pentachoron(c)

    def _default_unicode_callback(self, name: str, **kwargs) -> np.ndarray:
        raise NotImplementedError("Default callback is not invoked directly.")

    # --------------------------- universal builders (overrideable) ---
    def _compose_unicode_center(
            self, token_plain: str, H, pool_type: Optional[str], dim: int
    ) -> Optional[np.ndarray]:
        """
        Build a center vector from the token's Unicode characters.
        """
        vecs: List[np.ndarray] = []
        for ch in token_plain:
            v = H["pooled"](ch)
            if v is None:
                continue
            v = np.asarray(v, np.float32)
            if v.shape[0] != dim:
                raise ValueError(f"Unicode pooled dim mismatch for '{ch}': got {v.shape[0]}, expected {dim}")
            vecs.append(v)

        if not vecs:
            return None

        stacked = np.stack(vecs, 0)

        if pool_type in (None, "unicode", "mean"):
            c = stacked.mean(axis=0)
        elif pool_type == "abs":
            c = np.abs(stacked).mean(axis=0)
        elif pool_type == "dot":
            c = stacked.mean(axis=0)
            c = c / (np.abs(c).sum() + 1e-8)  # L1 normalize
        elif pool_type == "mse":
            c = (stacked ** 2).mean(axis=0)
        elif pool_type == "max":
            c = stacked.max(axis=0)
        else:
            raise ValueError(f"Unsupported pool_type '{pool_type}'")

        return c.astype(np.float32, copy=False)

    def _compose_wordnet_center(
            self, definitions: List[str], H, pool_type: Optional[str], dim: int
    ) -> Optional[np.ndarray]:
        """Build a center vector from definition text characters."""
        vecs: List[np.ndarray] = []
        for text in definitions:
            for ch in str(text):
                v = H["pooled"](ch)
                if v is None:
                    continue
                v = np.asarray(v, np.float32)
                if v.shape[0] != dim:
                    raise ValueError(f"Definition pooled dim mismatch for '{ch}': got {v.shape[0]}, expected {dim}")
                vecs.append(v)

        if not vecs:
            return None

        stacked = np.stack(vecs, 0)

        if pool_type in (None, "unicode", "mean"):
            c = stacked.mean(axis=0)
        elif pool_type == "abs":
            c = np.abs(stacked).mean(axis=0)
        elif pool_type == "dot":
            c = stacked.mean(axis=0)
            c = c / (np.abs(c).sum() + 1e-8)  # L1 normalize
        elif pool_type == "mse":
            c = (stacked ** 2).mean(axis=0)
        elif pool_type == "max":
            c = stacked.max(axis=0)
        else:
            raise ValueError(f"Unsupported pool_type '{pool_type}'")

        return c.astype(np.float32, copy=False)

    def _deterministic_pentachoron(self, center_vec: np.ndarray) -> np.ndarray:
        """Universal pentachoron inflation (deterministic; overrideable)."""
        d = center_vec.shape[0]
        proposals = np.stack([
            center_vec,
            np.roll(center_vec, 1),
            np.roll(center_vec, 3) * np.sign(center_vec + 1e-8),
            np.roll(center_vec, 7) - center_vec,
            np.roll(center_vec, 11) + center_vec,
        ], 0).astype(np.float32)

        # L1 row norms
        norms = np.sum(np.abs(proposals), axis=1, keepdims=True) + 1e-8
        Q = proposals / norms

        # GS orthogonalization with L1 row renorm at each step
        for i in range(5):
            for j in range(i):
                Q[i] -= np.dot(Q[i], Q[j]) * Q[j]
            Q[i] /= (np.sum(np.abs(Q[i])) + 1e-8)

        gamma = np.array([1.0, 0.9, -0.8, 1.1, 1.2], np.float32)
        X = np.zeros((5, d), np.float32)
        for i in range(5):
            X[i] = center_vec + gamma[i] * Q[i]
        return X - X.mean(0, keepdims=True)

    # --------------------------- finalize + provenance (overrideable) ----
    def _finalize_crystal(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, np.float32, order='C')  # Ensure C-contiguous
        if X.shape != (5, self.dim):
            raise ValueError(f"Crystal must be shape (5, {self.dim}); got {X.shape}.")
        return X - X.mean(0, keepdims=True)

    def _auto_provenance_from_cfg(self, cfg: Dict[str, Any]) -> dict:
        token = cfg["data"]["token"]
        prov = {
            "source": "special/compose",
            "token": token,
            "pool_type": cfg.get("pool_type") or "unicode",
            "components": list(token),
            "additional_definitions": list(cfg.get("additional_definitions", [])),
        }
        if cfg.get("antonyms"):
            prov["antonyms"] = list(cfg["antonyms"])
        if cfg.get("inversion_formula") is not None:
            prov["inversion_formula"] = "user_supplied"
        return prov

    def _finalize_crystal_and_provenance(
            self, product: Union[np.ndarray, Dict[str, Any]], cfg: Dict[str, Any]
    ) -> Tuple[np.ndarray, dict]:
        # ndarray path
        if isinstance(product, np.ndarray):
            X = self._finalize_crystal(product)
            prov = self._auto_provenance_from_cfg(cfg)
            return X, prov

        # dict path
        if not isinstance(product, dict):
            raise TypeError(
                "create_crystal must return ndarray or dict with {'base':..., 'ops':..., 'provenance':...}.")
        base = np.asarray(product["base"], np.float32)
        X = base
        for op in product.get("ops", []):
            name = op.get("name")
            if name == "center":
                X -= X.mean(0, keepdims=True)
            elif name == "scale":
                X *= float(op.get("k", 1.0))
            elif name == "translate":
                t = np.asarray(op.get("t"), np.float32)
                if t.shape != (self.dim,):
                    raise ValueError(f"translate.t must be shape ({self.dim},)")
                X = X + t[None, :]
            elif name == "normalize_rows":
                n = np.sum(np.abs(X), axis=1, keepdims=True) + 1e-8
                X = X / n
            elif name == "align_to":
                v = np.asarray(op.get("v"), np.float32)
                if v.shape != (self.dim,):
                    raise ValueError(f"align_to.v must be shape ({self.dim},)")
                v = v / (np.abs(v).sum() + 1e-8)
                p = X.mean(0)
                p = p / (np.abs(p).sum() + 1e-8)
                alpha = float(op.get("alpha", 1.0))
                X = X + alpha * (v - p)[None, :]
            else:
                raise ValueError(f"Unsupported op '{name}'")
        prov = dict(product.get("provenance", {})) or self._auto_provenance_from_cfg(cfg)
        return self._finalize_crystal(X), prov

    # --------------------------- universal manifestation routine ----------
    def _manifest_special_tokens(
            self,
            base_set: Dict[str, int],
            create_crystal: Callable[[dict, Callable[..., np.ndarray]], Union[np.ndarray, Dict[str, Any]]],
            callback: Optional[Callable[..., np.ndarray]],
            create_config: Dict[str, Any],
    ) -> None:
        """Universal, deterministic manifestor. Subclasses call this after loading."""
        helpers = self._helpers()

        for name, tid in base_set.items():
            # Keep if already present
            if tid in self._id_to_vec:
                self._token_to_id[name] = tid
                self._id_to_token.setdefault(tid, name)
                self._valid_token_ids.add(tid)
                continue

            # Build per-token config
            cfg = {
                "dim": self.dim,
                "pool_type": create_config.get("pool_type", None),
                "special_tokens": create_config.get("special_tokens"),
                "additional_definitions": create_config.get("additional_definitions", []),
                "antonyms": create_config.get("antonyms"),
                "inversion_formula": create_config.get("inversion_formula"),
                "data": {"token": name.strip("<>").strip(), "token_id": tid, "origin": "special"},
                "helpers": helpers,
            }

            if create_crystal is None:
                create_crystal = self._default_create_crystal

            product = create_crystal(cfg, callback) if callback is not None else create_crystal(cfg,
                                                                                                self._default_unicode_callback)
            X, prov = self._finalize_crystal_and_provenance(product, cfg)

            # Register
            self._token_to_id[name] = tid
            self._id_to_token[tid] = name
            self._id_to_vec[tid] = X.astype(np.float32, copy=False, order='C')
            self._id_to_provenance[tid] = prov
            self._valid_token_ids.add(tid)
            self._id_to_volume.setdefault(tid, 1.0)

            # Aliases
            for alias in (cfg.get("special_tokens") or []):
                alias = str(alias)
                self._token_to_id[alias] = tid
                self._id_to_token.setdefault(tid, alias)
            if cfg.get("special_tokens"):
                self._id_to_provenance[tid].setdefault("aliases", list(cfg["special_tokens"]))

            # Antonyms
            antonyms = cfg.get("antonyms") or []
            invf = cfg.get("inversion_formula")
            if invf:
                for anti in antonyms:
                    if anti in base_set:
                        anti_id = base_set[anti]
                        if anti_id not in self._id_to_vec:
                            X_inv = invf(X, cfg)  # must be deterministic
                            X_inv = self._finalize_crystal(X_inv)
                            self._token_to_id[anti] = anti_id
                            self._id_to_token[anti_id] = anti
                            self._id_to_vec[anti_id] = X_inv.astype(np.float32, copy=False, order='C')
                            inv_prov = {
                                "source": "inversion",
                                "of_token": name,
                                "of_token_id": tid,
                                "pool_type": cfg.get("pool_type") or "unicode",
                                "components": prov.get("components", []),
                                "additional_definitions": cfg.get("additional_definitions", []),
                                "ops": ["invert"],
                            }
                            self._id_to_provenance[anti_id] = inv_prov
                            self._valid_token_ids.add(anti_id)
                            self._id_to_volume.setdefault(anti_id, 1.0)

        # Invalidate caches after adding tokens
        self._invalidate_caches()

    # --------------------------- basics -------------------------------
    def vocab_size(self) -> int:
        return len(self._token_to_id)

    def token_to_id(self, token: str) -> Optional[int]:
        return self._token_to_id.get(token)

    def id_to_token(self, token_id: int) -> Optional[str]:
        return self._id_to_token.get(token_id)

    def cache_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        return {
            "normalized_cache_size": len(self._normalized_cache),
            "pooled_cache_size": len(self._pooled_cache),
            "spatial_index_size": len(self._spatial_index.token_ids) if self._spatial_index else 0,
            "vocab_size": len(self._valid_token_ids)
        }

    def clear_caches(self):
        """Clear all caches to free memory."""
        self._invalidate_caches()

    def set_similarity_mode(self, mode: str = "l1dot"):
        """
        Set similarity space used by vector ops:
          - "l1dot": uses L1-normalized vectors + dot (current default behavior)
          - "cosine": uses L2-normalized vectors + dot == cosine
        """
        if mode not in ("l1dot", "cosine"):
            raise ValueError("mode must be 'l1dot' or 'cosine'")
        if mode != self._similarity_mode:
            self._similarity_mode = mode
            # normalization basis changed → recompute dense views
            self._dense_dirty = True
            self._normalized_cache.clear()

    def _ensure_dense_views(self):
        """Build contiguous [M,D] matrices for pooled + normalized; M = number of valid tokens."""
        if not self._dense_dirty and self._dense_ids is not None:
            return

        if not self._valid_token_ids:
            self._dense_ids = np.empty((0,), np.int64)
            self._dense_tokens = []
            self._dense_pooled = np.empty((0, self.dim), np.float32)
            self._dense_l1norm = np.empty((0, self.dim), np.float32)
            self._dense_dirty = False
            return

        tids = np.fromiter(sorted(self._valid_token_ids), dtype=np.int64)
        toks = [self._id_to_token.get(int(tid), f"<id:{int(tid)}>") for tid in tids]

        pooled_list = []
        for tid in tids:
            pv = self._get_cached_pooled(int(tid))
            if pv is None:
                # Should not happen for valid ids, but guard
                pv = np.zeros(self.dim, np.float32)
            pooled_list.append(pv.astype(np.float32, copy=False))
        P = np.stack(pooled_list, axis=0)  # [M,D] float32

        if self._similarity_mode == "l1dot":
            denom = np.sum(np.abs(P), axis=1, keepdims=True) + 1e-8
            N = P / denom
        else:  # cosine
            denom = np.sqrt(np.sum(P * P, axis=1, keepdims=True)) + 1e-9
            N = P / denom

        self._dense_ids = tids
        self._dense_tokens = toks
        self._dense_pooled = P
        self._dense_l1norm = N
        self._dense_dirty = False

    # --- exact vectorized similarity (dense) ---
    def _dense_topk_by_vector(self, q_vec: np.ndarray, k: int, exclude_tid: Optional[int] = None):
        """
        Returns (topk_indices_in_dense, topk_scores) using the current similarity_mode.
        q_vec must already be normalized according to mode.
        """
        self._ensure_dense_views()
        if self._dense_l1norm is None or self._dense_l1norm.shape[0] == 0:
            return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.float32)

        # dot with [M,D]
        scores = self._dense_l1norm @ q_vec.astype(np.float32)
        if exclude_tid is not None and self._dense_ids is not None:
            mask = (self._dense_ids == int(exclude_tid))
            if mask.any():
                scores[mask] = -np.inf

        m = scores.shape[0]
        kk = min(k, m)
        if kk <= 0:
            return np.empty((0,), np.int64), np.empty((0,), np.float32)

        # partial top-k
        idx_part = np.argpartition(scores, -(kk))[-kk:]
        idx_sorted = idx_part[np.argsort(scores[idx_part])[::-1]]
        return idx_sorted, scores[idx_sorted].astype(np.float32, copy=False)

    # --- public: exact cosine-angle / l1dot band extraction ---
    def extract_band_fast(
        self,
        trajectory: np.ndarray,
        *,
        max_angle: float = 0.3,
        k_cap: int = 4096
    ) -> Dict[str, np.ndarray]:
        """
        Exact, dense, vectorized band extraction.

        If similarity_mode == "cosine": interprets `max_angle` literally (radians)
          and keeps tokens with cos(theta) >= cos(max_angle).
        If similarity_mode == "l1dot": interprets `max_angle` as a dot threshold
          via: dot >= 1 - max_angle  (legacy semantics, kept for continuity).
        """
        # Build direction
        direction = trajectory.mean(0) if trajectory.ndim == 2 else trajectory
        direction = direction.astype(np.float32)

        # Normalize according to mode
        if self._similarity_mode == "cosine":
            denom = np.sqrt((direction * direction).sum()) + 1e-9
            q = direction / denom
            thr = float(np.cos(max_angle))
        else:
            denom = np.abs(direction).sum() + 1e-8
            q = direction / denom
            thr = 1.0 - float(max_angle)

        self._ensure_dense_views()
        if self._dense_l1norm is None or self._dense_l1norm.shape[0] == 0:
            return {}

        scores = (self._dense_l1norm @ q)  # [M]
        # keep mask by threshold
        keep = scores >= thr
        if not np.any(keep):
            return {}

        # cap results (optional)
        keep_idx = np.nonzero(keep)[0]
        if keep_idx.size > k_cap:
            # take highest scores among those above threshold
            part = np.argpartition(scores[keep_idx], -(k_cap))[-k_cap:]
            keep_idx = keep_idx[part[np.argsort(scores[keep_idx][part])[::-1]]]

        out: Dict[str, np.ndarray] = {}
        for idx in keep_idx.tolist():
            tid = int(self._dense_ids[idx])
            tok = self._id_to_token.get(tid)
            if tok is None:
                continue
            out[tok] = self._id_to_vec[tid]
        return out

    def find_similar_tokens_fast(
        self, token: Union[str, int], k: int = 10, min_similarity: float = 0.5
    ) -> List[Tuple[str, float]]:
        """
        Exact, dense, vectorized nearest-neighbor by dot in the current mode.
        """
        tid = token if isinstance(token, int) else self._token_to_id.get(token)
        if tid is None:
            return []

        # query vector normalized per mode
        if self._similarity_mode == "cosine":
            v = self._get_cached_pooled(tid)
            if v is None:
                return []
            q = v / (np.linalg.norm(v) + 1e-9)
        else:
            q = self._get_cached_normalized(tid)
            if q is None:
                return []

        idxs, scores = self._dense_topk_by_vector(q, k=k+32, exclude_tid=int(tid))  # overfetch slightly
        out: List[Tuple[str, float]] = []
        for i, s in zip(idxs.tolist(), scores.tolist()):
            tok = self._dense_tokens[i]
            if s >= min_similarity:
                out.append((tok, float(s)))
            if len(out) >= k:
                break
        return out