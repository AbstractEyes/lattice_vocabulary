from __future__ import annotations

import os
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Union, Optional, Tuple, Callable, Any

import numpy as np
import torch
from datasets import load_dataset

from .geometric_vocab import GeometricVocab


class PretrainedGeometricVocab(GeometricVocab):
    """
    Parquet-backed deterministic vocab with columnar load, duplicate-mean aggregation,
    pooled caching, vectorized finalize, and optional persisted pooled cache.

    Key capabilities:
      - Vectorized fast path for uniform 'crystal' rows (reshape → sort → reduceat → batch finalize).
      - Fallback path aggregates jagged rows with per-id mean, then batch finalize.
      - Optional pooled-only storage to reduce RAM footprint on huge vocabs.
      - Selectable storage dtype (e.g., float16 for pooled).
      - Strict shape mode to forbid automatic inference/pooled inflation.
      - Optional on-disk pooled cache ({ids.npy, pooled.npy, tokens.txt}) keyed by repo/split/dim/dtype.
    """

    def __init__(
        self,
        repo_id: str,
        dim: int,
        *,
        split: str = "wordnet_eng",
        base_set: Optional[Dict[str, int]] = None,
        create_config: Optional[Dict[str, Any]] = None,
        create_crystal: Optional[Callable[[dict, Callable[..., np.ndarray]], Union[np.ndarray, Dict[str, Any]]]] = None,
        callback: Optional[Callable[..., np.ndarray]] = None,
        manifest_specials: bool = True,
        # perf/robustness knobs
        store: str = "full",                 # "full" | "pooled" | "both"
        reshape_order: str = "C",
        vertex_count: int = 5,
        infer_dim: bool = True,
        strict_shapes: bool = False,
        # finalize/caching
        finalize_mode: str = "post_mean",    # "none" | "post_mean"
        cache_pooled: bool = True,
        # huge-vocab knobs
        storage_dtype: Union[str, np.dtype] = np.float32,  # e.g., np.float16 for pooled/both
        persist_cache_dir: Optional[Union[str, Path]] = None,
        reuse_persisted_pooled: bool = True,
        build_dense_views: bool = False,     # defer heavy dense structures in base, if any
        build_index: bool = False,           # defer spatial index
    ):
        super().__init__(dim)
        self.repo_id = str(repo_id)
        self._id_to_pooled: Dict[int, np.ndarray] = {}
        self._storage_dtype = np.dtype(storage_dtype)

        # Optional pooled cache reload (fast path for huge vocabs)
        self._cache_dir: Optional[Path] = Path(persist_cache_dir) if persist_cache_dir else None
        self._cache_key = self._make_cache_key(repo_id, split, dim, vertex_count, str(self._storage_dtype))

        # Attempt pooled-only warm load if available and allowed
        if self._cache_dir and reuse_persisted_pooled and cache_pooled and store in ("pooled", "both"):
            loaded = self._try_load_pooled_cache(self._cache_dir, self._cache_key)
            if loaded and manifest_specials and base_set:
                # Specials still need to be materialized; they’re small.
                self._manifest_special_tokens(
                    base_set=base_set,
                    create_crystal=create_crystal,
                    callback=callback,
                    create_config=create_config or {}
                )
                if not build_dense_views:
                    self._dense_dirty = True
                if not build_index:
                    self._spatial_index = None
                    self._index_dirty = False
                return  # early exit with pooled maps built

        # ---------- load split (columnar, minimal columns) ----------
        ds = load_dataset(self.repo_id, split=split)
        have = set(ds.column_names)
        wanted = ["token_id", "token", "crystal", "volume"]
        keep = [c for c in wanted if c in have]
        drop = [c for c in ds.column_names if c not in keep]
        if drop:
            ds = ds.remove_columns(drop)
        ds = ds.with_format("numpy", columns=keep)

        ids  = ds["token_id"] if "token_id" in keep else np.array([], dtype=np.int64)
        toks = ds["token"]    if "token"    in keep else np.array([], dtype=object)
        cryst= ds["crystal"]  if "crystal"  in keep else np.array([], dtype=object)
        vols = ds["volume"]   if "volume"   in keep else None

        ids  = np.asarray(ids).astype(np.int64, copy=False)
        toks = np.asarray(toks)
        vols_f = np.asarray(vols, dtype=np.float32) if vols is not None else None

        # --------- shape helpers (strict_shapes wired) ----------
        def _coerce(raw: Any) -> np.ndarray:
            X = np.asarray(raw, np.float32)
            if X.ndim == 2:
                V, D = int(X.shape[0]), int(X.shape[1])
                if V != vertex_count:
                    raise ValueError(f"Crystal has {V} vertices, expected {vertex_count}.")
                if D != self.dim:
                    if infer_dim and not strict_shapes:
                        self.dim = D
                    else:
                        raise ValueError(f"Dim mismatch: got {D}, expected {self.dim}.")
                return X

            if X.ndim == 1:
                n = int(X.size)
                # exact flat [V*D]
                if n == vertex_count * self.dim:
                    return np.reshape(X, (vertex_count, self.dim), order=reshape_order)
                # infer D if divisible by V (only if not strict)
                if infer_dim and not strict_shapes and n % vertex_count == 0:
                    self.dim = n // vertex_count
                    return np.reshape(X, (vertex_count, self.dim), order=reshape_order)
                # pooled [D] → deterministic inflate (only if not strict)
                if n == self.dim:
                    if strict_shapes:
                        raise ValueError(
                            f"Strict mode: refusing to inflate pooled vector length {n} "
                            f"into crystal [{vertex_count},{self.dim}]"
                        )
                    c = X / (np.abs(X).sum() + 1e-8)
                    return self._deterministic_pentachoron(c)

            raise ValueError(
                f"Unsupported crystal shape {X.shape if isinstance(X, np.ndarray) else type(X)} (ndim={X.ndim})."
            )

        def _finalize_if_needed_block(X_block: np.ndarray) -> np.ndarray:
            """
            X_block: (K,V,D) float32
            finalize_mode == 'post_mean' → mean-center rows per token.
            """
            if store in ("full", "both"):
                if finalize_mode == "none":
                    return np.ascontiguousarray(X_block, dtype=np.float32)
                elif finalize_mode == "post_mean":
                    return np.ascontiguousarray(X_block - X_block.mean(axis=1, keepdims=True), dtype=np.float32)
                else:
                    raise ValueError(f"finalize_mode must be 'none' or 'post_mean', got {finalize_mode!r}")
            # For 'pooled', caller will not store X_block anyway; keep contiguous
            return np.ascontiguousarray(X_block, dtype=np.float32)

        # ---------- FAST PATH: flat uniform crystals ----------
        fastpath_ok = False
        A = None  # (N, L) float32
        try:
            A = np.stack(cryst)  # may raise if jagged / object
            if A.ndim == 2 and A.dtype != object:
                A = A.astype(np.float32, copy=False)
                L = A.shape[1]
                if L % vertex_count == 0:
                    D_guess = L // vertex_count
                    if self.dim != D_guess:
                        if infer_dim and not strict_shapes:
                            self.dim = int(D_guess)
                        elif self.dim != D_guess:
                            raise ValueError(f"Dim mismatch: got D={D_guess}, expected dim={self.dim}.")
                    fastpath_ok = True
        except Exception:
            fastpath_ok = False

        if fastpath_ok and A is not None and len(ids) > 0:
            V = vertex_count
            D = self.dim
            A = A.reshape(-1, V, D, order=reshape_order)

            order_idx = np.argsort(ids, kind="stable")
            ids_sorted = ids[order_idx]
            A_sorted = A[order_idx]
            vols_sorted = vols_f[order_idx] if vols_f is not None else None

            uniq_ids, idx, counts = np.unique(ids_sorted, return_index=True, return_counts=True)
            sums = np.add.reduceat(A_sorted, idx, axis=0)                 # (K,V,D)
            means = sums / counts[:, None, None]                           # (K,V,D)

            if vols_sorted is not None:
                v_sums = np.add.reduceat(vols_sorted, idx)
                v_means = v_sums / counts.astype(np.float32)
            else:
                v_means = np.ones_like(uniq_ids, dtype=np.float32)

            # clear and prepare maps
            self._token_to_id.clear(); self._id_to_token.clear()
            self._id_to_vec.clear();   self._id_to_volume.clear()
            self._valid_token_ids.clear()
            self._id_to_pooled.clear()

            # batch pooled over vertices (K,D)
            if cache_pooled or store == "pooled":
                pooled_all = means.mean(axis=1).astype(np.float32, copy=False)  # (K,D)

            # batch finalize if storing full/both
            if store in ("full", "both"):
                X_store_block = _finalize_if_needed_block(means)  # (K,V,D)
            else:
                X_store_block = None

            toks_sorted = toks[order_idx]
            rep_toks = toks_sorted[idx]

            t2i = self._token_to_id; i2t = self._id_to_token
            i2v = self._id_to_vec;    i2vol = self._id_to_volume
            valid = self._valid_token_ids
            i2p = self._id_to_pooled

            uniq_ids_l = uniq_ids.tolist()
            rep_toks_l = rep_toks.tolist()
            v_means_l  = v_means.tolist()
            K = len(uniq_ids_l)

            for k in range(K):
                tid = int(uniq_ids_l[k])
                tok = str(rep_toks_l[k])

                if cache_pooled:
                    i2p[tid] = pooled_all[k].astype(self._storage_dtype, copy=False)

                t2i[tok] = tid
                i2t[tid] = tok

                if store in ("full", "both"):
                    i2v[tid] = X_store_block[k].astype(self._storage_dtype, copy=False)
                elif store == "pooled":
                    i2v[tid] = (i2p[tid] if cache_pooled else pooled_all[k].astype(self._storage_dtype, copy=False))

                i2vol[tid] = float(v_means_l[k])
                valid.add(tid)

        else:
            # ---------- FALLBACK: per-row coerce + dict mean ----------
            ids_int  = ids.tolist()
            toks_str = [str(x) for x in toks.tolist()]
            vols_list = (vols_f.tolist() if vols_f is not None else [1.0] * len(ids_int))

            x_sum: Dict[int, np.ndarray] = {}
            v_sum: Dict[int, float]      = {}
            n_cnt: Dict[int, int]        = {}
            tok_pref: Dict[int, str]     = {}

            for tid, tok, raw, vol in zip(ids_int, toks_str, cryst, vols_list):
                X = _coerce(raw)  # [V,D] float32
                if tid not in x_sum:
                    x_sum[tid]  = X.astype(np.float32, copy=True)
                    v_sum[tid]  = float(vol)
                    n_cnt[tid]  = 1
                    tok_pref[tid] = tok
                else:
                    x_sum[tid] += X
                    v_sum[tid] += float(vol)
                    n_cnt[tid] += 1

            self._token_to_id.clear(); self._id_to_token.clear()
            self._id_to_vec.clear();   self._id_to_volume.clear()
            self._valid_token_ids.clear()
            self._id_to_pooled.clear()

            tids_uniq = np.fromiter(x_sum.keys(), dtype=np.int64, count=len(x_sum))

            X_means_arr = np.stack(
                [x_sum[int(tid)] / float(n_cnt[int(tid)]) for tid in tids_uniq],
                axis=0
            ).astype(np.float32, copy=False)  # (K,V,D)

            vols_arr = np.array(
                [v_sum[int(tid)] / float(n_cnt[int(tid)]) for tid in tids_uniq],
                dtype=np.float32
            )
            toks_arr = [tok_pref[int(tid)] for tid in tids_uniq]

            if cache_pooled or store == "pooled":
                pooled_all = X_means_arr.mean(axis=1).astype(np.float32, copy=False)  # (K,D)

            if store in ("full", "both"):
                X_store_block = _finalize_if_needed_block(X_means_arr)  # (K,V,D)
            else:
                X_store_block = None

            t2i = self._token_to_id; i2t = self._id_to_token
            i2v = self._id_to_vec;    i2vol = self._id_to_volume
            valid = self._valid_token_ids
            i2p = self._id_to_pooled

            K = tids_uniq.shape[0]
            for k in range(K):
                tid = int(tids_uniq[k])
                tok = str(toks_arr[k])

                if cache_pooled:
                    i2p[tid] = pooled_all[k].astype(self._storage_dtype, copy=False)

                t2i[tok] = tid
                i2t[tid] = tok

                if store in ("full", "both"):
                    i2v[tid] = X_store_block[k].astype(self._storage_dtype, copy=False)
                elif store == "pooled":
                    i2v[tid] = (i2p[tid] if cache_pooled else pooled_all[k].astype(self._storage_dtype, copy=False))

                i2vol[tid] = float(vols_arr[k])
                valid.add(tid)

        # ---------- persist pooled cache (optional) ----------
        if self._cache_dir and cache_pooled and store in ("pooled", "both"):
            try:
                tids_sorted = np.fromiter(sorted(self._valid_token_ids), dtype=np.int64)
                pooled_mat  = np.stack([self._id_to_pooled[int(t)] for t in tids_sorted], axis=0).astype(self._storage_dtype, copy=False)
                tokens_sorted = [self._id_to_token[int(t)] for t in tids_sorted]
                self._save_pooled_cache(self._cache_dir, self._cache_key, tids_sorted, pooled_mat, tokens_sorted)
            except Exception:
                # Persistence failures are non-fatal by design.
                pass

        # ---------- specials ----------
        if manifest_specials and base_set:
            self._manifest_special_tokens(
                base_set=base_set,
                create_crystal=create_crystal,
                callback=callback,
                create_config=create_config or {}
            )

        # ---------- defer heavy structures ----------
        if not build_dense_views:
            self._dense_dirty = True
        if not build_index:
            self._spatial_index = None
            self._index_dirty = False

    # -------- override pooled() to use cache (if present) --------
    def pooled(self, token_or_id: Union[str, int], method: str = "mean") -> Optional[np.ndarray]:
        tid = token_or_id if isinstance(token_or_id, int) else self._token_to_id.get(token_or_id)
        if tid is not None and tid in self._id_to_pooled:
            return self._id_to_pooled[tid]
        return super().pooled(token_or_id, method=method)

    # -------- SP-like surface --------
    def encode(self, token: str, *, return_id: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, int]]:
        tid = self._token_to_id.get(token)
        if tid is None:
            unk_id = self._token_to_id.get("<unk>")
            if unk_id is None:
                raise KeyError(f"Token '{token}' not found and '<unk>' missing.")
            X = self._id_to_vec[unk_id]
            return (X, unk_id) if return_id else X
        X = self._id_to_vec[tid]
        return (X, tid) if return_id else X

    def get_score(self, token_or_id: Union[str, int]) -> float:
        tid = token_or_id if isinstance(token_or_id, int) else self._token_to_id.get(token_or_id, None)
        if tid is None or tid not in self._valid_token_ids:
            return -100.0
        vol = self._id_to_volume.get(tid, 1.0)
        return float(np.clip(vol / 10.0, 0.01, 1.0))

    # -------- Torch cache (zero-copy CPU, one transfer to device) ----------
    def cache(self, tokens: Union[List[str], Dict[str, int]], device: str = "cpu", dtype: torch.dtype = torch.float32):
        tok_list = list(tokens.keys()) if isinstance(tokens, dict) else list(tokens)
        crystals_np: List[np.ndarray] = []
        pooled_np:   List[np.ndarray] = []
        keep:        List[str]        = []

        emb = self.embedding
        poo = self.pooled

        for t in tok_list:
            X = emb(t)
            v = poo(t)
            if X is None or v is None:
                continue
            crystals_np.append(np.asarray(X, dtype=np.float32, order="C"))
            pooled_np.append(np.asarray(v, dtype=np.float32, order="C"))
            keep.append(t)

        if not crystals_np:
            raise ValueError("No valid tokens found in input.")

        crystals = torch.from_numpy(np.stack(crystals_np, 0)).to(dtype=dtype)
        pooled   = torch.from_numpy(np.stack(pooled_np,   0)).to(dtype=dtype)

        if device != "cpu":
            crystals = crystals.to(device, non_blocking=True)
            pooled   = pooled.to(device, non_blocking=True)

        return {"tokens": keep, "crystals": crystals, "pooled": pooled}

    # -------- Introspection --------
    def describe(self) -> Dict[str, Union[str, int]]:
        return {"repo": self.repo_id, "dimension": self.dim, "vocab_size": self.vocab_size()}

    # ===================== On-disk pooled cache helpers =====================
    def _make_cache_key(self, repo_id: str, split: str, dim: int, V: int, dtype_str: str) -> str:
        h = hashlib.sha1(f"{repo_id}|{split}|{dim}|{V}|{dtype_str}".encode("utf-8")).hexdigest()[:12]
        return h

    def _cache_paths(self, cache_dir: Path, cache_key: str) -> Tuple[Path, Path, Path]:
        base = cache_dir / f"pretrained_vocab_{cache_key}"
        return base.with_suffix(".ids.npy"), base.with_suffix(".pooled.npy"), base.with_suffix(".tokens.txt")

    def _try_load_pooled_cache(self, cache_dir: Path, cache_key: str) -> bool:
        try:
            ids_p, pooled_p, toks_p = self._cache_paths(cache_dir, cache_key)
            if not (ids_p.exists() and pooled_p.exists() and toks_p.exists()):
                return False

            ids = np.load(ids_p, mmap_mode="r")
            pooled = np.load(pooled_p, mmap_mode="r")  # stored dtype
            with open(toks_p, "r", encoding="utf-8") as f:
                toks = [line.rstrip("\n") for line in f]

            if ids.shape[0] != pooled.shape[0] or ids.shape[0] != len(toks):
                return False

            # Populate minimal maps quickly (pooled only)
            self._token_to_id.clear(); self._id_to_token.clear()
            self._id_to_vec.clear();   self._id_to_volume.clear()
            self._valid_token_ids.clear(); self._id_to_pooled.clear()

            for i in range(ids.shape[0]):
                tid = int(ids[i])
                tok = toks[i]
                self._token_to_id[tok] = tid
                self._id_to_token[tid] = tok
                # pooled cache; keep contiguous copy for safety when slicing memory-mapped arrays
                pv = np.asarray(pooled[i], dtype=self._storage_dtype, order="C")
                self._id_to_pooled[tid] = pv
                self._id_to_vec[tid] = pv  # in pooled mode, embedding is pooled
                self._id_to_volume[tid] = 1.0
                self._valid_token_ids.add(tid)
            return True
        except Exception:
            return False

    def _save_pooled_cache(
        self,
        cache_dir: Path,
        cache_key: str,
        ids: np.ndarray,
        pooled: np.ndarray,
        tokens: List[str]
    ) -> None:
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            ids_p, pooled_p, toks_p = self._cache_paths(cache_dir, cache_key)
            np.save(ids_p, np.asarray(ids, dtype=np.int64))
            np.save(pooled_p, np.asarray(pooled, dtype=self._storage_dtype))
            with open(toks_p, "w", encoding="utf-8") as f:
                for t in tokens:
                    f.write(f"{t}\n")
        except Exception:
            # Cache saving is best-effort.
            pass
