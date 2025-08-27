import numpy as np
import torch
from datasets import load_dataset
from typing import List, Dict, Union, Optional, Tuple, Callable, Any
from .geometric_vocab import GeometricVocab

class PretrainedGeometricVocab(GeometricVocab):
    """
    Parquet-backed deterministic vocab with columnar load, duplicate-mean aggregation,
    pooled caching, and fast path for flat crystals.
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
        # new perf knobs
        finalize_mode: str = "post_mean",    # "none" | "post_mean"
        cache_pooled: bool = True,
    ):
        super().__init__(dim)
        self.repo_id = str(repo_id)
        self._id_to_pooled: Dict[int, np.ndarray] = {}  # optional pooled cache

        # ---------- load split (columnar, minimal columns) ----------
        ds = load_dataset(self.repo_id, split=split)
        have = set(ds.column_names)
        wanted = ["token_id", "token", "crystal", "volume"]
        keep = [c for c in wanted if c in have]
        drop = [c for c in ds.column_names if c not in keep]
        if drop:
            ds = ds.remove_columns(drop)
        ds = ds.with_format("numpy", columns=keep)

        ids = ds["token_id"] if "token_id" in keep else np.array([], dtype=np.int64)
        toks = ds["token"]    if "token"    in keep else np.array([], dtype=object)
        cryst= ds["crystal"]  if "crystal"  in keep else np.array([], dtype=object)
        vols = ds["volume"]   if "volume"   in keep else None

        ids = np.asarray(ids).astype(np.int64, copy=False)
        toks = np.asarray(toks)

        # --------- shape helpers ----------
        def _coerce(raw: Any) -> np.ndarray:
            X = np.asarray(raw, np.float32)
            if X.ndim == 2:
                V, D = int(X.shape[0]), int(X.shape[1])
                if V != vertex_count:
                    raise ValueError(f"Crystal has {V} vertices, expected {vertex_count}.")
                if D != self.dim:
                    if infer_dim: self.dim = D
                    else: raise ValueError(f"Dim mismatch: got {D}, expected {self.dim}.")
                return X
            if X.ndim == 1:
                n = int(X.size)
                if n == vertex_count * self.dim:
                    return np.reshape(X, (vertex_count, self.dim), order=reshape_order)
                if infer_dim and n % vertex_count == 0:
                    self.dim = n // vertex_count
                    return np.reshape(X, (vertex_count, self.dim), order=reshape_order)
                if n == self.dim:
                    c = X / (np.abs(X).sum() + 1e-8)
                    return self._deterministic_pentachoron(c)
            raise ValueError(f"Unsupported crystal shape {X.shape if isinstance(X, np.ndarray) else type(X)}.")

        def _finalize_if_needed(X: np.ndarray) -> np.ndarray:
            if finalize_mode == "none":
                return np.asarray(X, np.float32, order="C")
            elif finalize_mode == "post_mean":
                return self._finalize_crystal(X)
            else:
                raise ValueError(f"finalize_mode must be 'none' or 'post_mean', got {finalize_mode!r}")

        vols_f = np.asarray(vols, dtype=np.float32) if vols is not None else None

        # ---------- FAST PATH: flat uniform crystals ----------
        # Try to stack into (N, L); succeeds when each row is the same length.
        fastpath_ok = False
        A = None  # (N, L) float32
        try:
            A = np.stack(cryst)  # may raise if jagged / object
            if A.ndim == 2 and A.dtype != object:
                A = A.astype(np.float32, copy=False)
                L = A.shape[1]
                if L % vertex_count == 0:
                    # infer or validate D
                    D = L // vertex_count
                    if self.dim != D:
                        if infer_dim:
                            self.dim = int(D)
                        else:
                            raise ValueError(f"Dim mismatch: got D={D}, expected dim={self.dim}.")
                    fastpath_ok = True
        except Exception:
            fastpath_ok = False

        if fastpath_ok and A is not None and len(ids) > 0:
            # reshape to (N, V, D)
            V = vertex_count
            D = self.dim
            A = A.reshape(-1, V, D, order=reshape_order)

            # sort by ids and reduceat to mean duplicates in pure NumPy
            order = np.argsort(ids, kind="stable")
            ids_sorted = ids[order]
            A_sorted = A[order]
            vols_sorted = vols_f[order] if vols_f is not None else None

            uniq_ids, idx, counts = np.unique(ids_sorted, return_index=True, return_counts=True)
            sums = np.add.reduceat(A_sorted, idx, axis=0)              # (K, V, D)
            means = sums / counts[:, None, None]                        # (K, V, D)

            if vols_sorted is not None:
                v_sums = np.add.reduceat(vols_sorted, idx)
                v_means = v_sums / counts.astype(np.float32)
            else:
                v_means = np.ones_like(uniq_ids, dtype=np.float32)

            # commit maps
            self._token_to_id.clear(); self._id_to_token.clear()
            self._id_to_vec.clear();   self._id_to_volume.clear(); self._valid_token_ids.clear()
            self._id_to_pooled.clear()

            # pick a representative token per id: first occurrence in sorted block
            toks_sorted = toks[order]
            rep_toks = toks_sorted[idx]

            for tid, tok, X_mean, v_m in zip(uniq_ids.tolist(), rep_toks.tolist(), means, v_means.tolist()):
                # cache pooled BEFORE finalize to preserve signal
                if cache_pooled:
                    self._id_to_pooled[tid] = X_mean.mean(axis=0).astype(np.float32, copy=False)
                X_store = _finalize_if_needed(X_mean)

                self._token_to_id[str(tok)] = tid
                self._id_to_token[tid] = str(tok)
                if store in ("full", "both"):
                    self._id_to_vec[tid] = np.asarray(X_store, np.float32, order="C")
                elif store == "pooled":
                    # store pooled as embedding if desired
                    self._id_to_vec[tid] = (self._id_to_pooled[tid] if cache_pooled
                                            else X_mean.mean(axis=0).astype(np.float32, copy=False))
                self._id_to_volume[tid] = float(v_m)
                self._valid_token_ids.add(tid)

        else:
            # ---------- FALLBACK: per-row coerce + dict mean ----------
            ids_int  = ids.tolist()
            toks_str = [str(x) for x in toks.tolist()]
            vols_f   = (vols_f.tolist() if vols_f is not None else [1.0] * len(ids_int))

            x_sum: Dict[int, np.ndarray] = {}
            v_sum: Dict[int, float]      = {}
            n_cnt: Dict[int, int]        = {}
            tok_pref: Dict[int, str]     = {}

            for tid, tok, raw, vol in zip(ids_int, toks_str, cryst, vols_f):
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
            self._id_to_vec.clear();   self._id_to_volume.clear(); self._valid_token_ids.clear()
            self._id_to_pooled.clear()

            for tid in x_sum.keys():  # order not critical; add sorted(tids) if you need determinism
                X_mean = x_sum[tid] / float(n_cnt[tid])
                if cache_pooled:
                    self._id_to_pooled[tid] = X_mean.mean(axis=0).astype(np.float32, copy=False)
                X_store = _finalize_if_needed(X_mean)

                tok    = tok_pref[tid]
                vol_m  = v_sum[tid] / float(n_cnt[tid])

                self._token_to_id[tok] = tid
                self._id_to_token[tid] = tok
                if store in ("full", "both"):
                    self._id_to_vec[tid] = np.asarray(X_store, np.float32, order="C")
                elif store == "pooled":
                    self._id_to_vec[tid] = (self._id_to_pooled[tid] if cache_pooled
                                            else X_mean.mean(axis=0).astype(np.float32, copy=False))
                self._id_to_volume[tid] = float(vol_m)
                self._valid_token_ids.add(tid)

        # ---------- specials ----------
        if manifest_specials and base_set:
            self._manifest_special_tokens(
                base_set=base_set,
                create_crystal=create_crystal,
                callback=callback,
                create_config=create_config or {}
            )

    # -------- override pooled() to use cache (if present) --------
    def pooled(self, token_or_id: Union[str, int], method: str = "mean") -> Optional[np.ndarray]:
        # Favor cached pooled when available; fallback to base (computes mean)
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

    # -------- Torch cache ----------
    def cache(self, tokens: Union[List[str], Dict[str, int]], device: str = "cpu", dtype: torch.dtype = torch.float32):
        tok_list = list(tokens.keys()) if isinstance(tokens, dict) else list(tokens)
        mats, pooled, keep = [], [], []
        for t in tok_list:
            X = self.embedding(t)
            v = self.pooled(t)
            if X is None or v is None:
                continue
            mats.append(torch.as_tensor(X, dtype=dtype))
            pooled.append(torch.as_tensor(v, dtype=dtype))
            keep.append(t)
        if not mats:
            raise ValueError("No valid tokens found in input.")
        return {
            "tokens": keep,
            "crystals": torch.stack(mats, 0).to(device),
            "pooled":   torch.stack(pooled, 0).to(device),
        }
