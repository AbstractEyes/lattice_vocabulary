import numpy as np
import torch
from datasets import load_dataset
from typing import List, Dict, Union, Optional, Tuple, Callable, Any
from .geometric_vocab import GeometricVocab


class PretrainedGeometricVocab(GeometricVocab):
    """
    Thin, deterministic accessor around a parquet-backed vocabulary.

    - Loads tensors and volumes from HF Datasets.
    - get_score(): returns -100.0 for anything outside pretrained ids.
    - encode(): returns [5,D] crystal (or crystal,id).
    - cache(): Torch batching for downstream.
    - For special tokens, call self._manifest_special_tokens(...) defined in base.
    """

    def __init__(
            self,
            repo_id: str,
            dim: int,
            *,
            split: str = "wordnet_eng",
            base_set: Optional[Dict[str, int]] = None,
            create_config: Optional[Dict[str, Any]] = None,
            create_crystal: Optional[
                Callable[[dict, Callable[..., np.ndarray]], Union[np.ndarray, Dict[str, Any]]]] = None,
            callback: Optional[Callable[..., np.ndarray]] = None,
            manifest_specials: bool = True,
            # perf/robustness knobs
            store: str = "full",  # "full" | "pooled" | "both"
            reshape_order: str = "C",
            vertex_count: int = 5,
            infer_dim: bool = True,
            strict_shapes: bool = False,
    ):
        super().__init__(dim)
        self.repo_id = str(repo_id)

        # ---------- load split (columnar, minimal columns) ----------
        ds = load_dataset(self.repo_id, split=split)
        have = set(ds.column_names)
        wanted = ["token_id", "token", "crystal", "volume"]
        keep = [c for c in wanted if c in have]
        drop = [c for c in ds.column_names if c not in keep]
        if drop:
            ds = ds.remove_columns(drop)
        ds = ds.with_format("numpy", columns=keep)

        ids = ds["token_id"] if "token_id" in keep else []
        toks = ds["token"] if "token" in keep else []
        cryst = ds["crystal"] if "crystal" in keep else []
        vols = ds["volume"] if "volume" in keep else None

        # ---------- shape coerce helper ----------
        def _coerce(raw: Any) -> np.ndarray:
            X = np.asarray(raw, np.float32)
            if X.ndim == 2:
                V, D = int(X.shape[0]), int(X.shape[1])
                if V != vertex_count:
                    raise ValueError(f"Crystal has {V} vertices, expected {vertex_count}.")
                if D != self.dim:
                    if infer_dim:
                        self.dim = D
                    else:
                        raise ValueError(f"Dim mismatch: got {D}, expected {self.dim}.")
                return X
            if X.ndim == 1:
                n = int(X.size)
                # flat [V*D]
                if n == vertex_count * self.dim:
                    return np.reshape(X, (vertex_count, self.dim), order=reshape_order)
                # infer D from length
                if infer_dim and n % vertex_count == 0:
                    self.dim = n // vertex_count
                    return np.reshape(X, (vertex_count, self.dim), order=reshape_order)
                # pooled [D] -> inflate deterministically
                if n == self.dim:
                    c = X / (np.abs(X).sum() + 1e-8)
                    return self._deterministic_pentachoron(c)
            raise ValueError(f"Unsupported crystal shape {X.shape if isinstance(X, np.ndarray) else type(X)}.")

        # Early probe (handles flat [5*D])
        if len(cryst) > 0:
            _ = _coerce(cryst[0])

        # ---------- mean-aggregate duplicates by token_id ----------
        ids_int = [int(x) for x in ids]
        toks_str = [str(x) for x in toks]
        vols_f = [float(v) for v in (vols if vols is not None else [1.0] * len(ids_int))]

        # Accumulators
        x_sum: Dict[int, np.ndarray] = {}
        v_sum: Dict[int, float] = {}
        n_cnt: Dict[int, int] = {}
        tok_pref: Dict[int, str] = {}

        for tid, tok, raw, vol in zip(ids_int, toks_str, cryst, vols_f):
            X = _coerce(raw)  # [5,D] float32
            if tid not in x_sum:
                # copy to own buffer to ensure independent accumulation
                x_sum[tid] = X.astype(np.float32, copy=True)
                v_sum[tid] = float(vol)
                n_cnt[tid] = 1
                tok_pref[tid] = tok
            else:
                x_sum[tid] += X
                v_sum[tid] += float(vol)
                n_cnt[tid] += 1
                # If token text differs across same id, keep first-seen deterministically.

        # ---------- commit maps from aggregated means ----------
        # (Rebuild cleanly to avoid any prefilled entries)
        self._token_to_id.clear()
        self._id_to_token.clear()
        self._id_to_vec.clear()
        self._id_to_volume.clear()
        self._valid_token_ids.clear()

        for tid in sorted(x_sum.keys()):
            X_mean = x_sum[tid] / float(n_cnt[tid])  # mean of [5,D]
            X_mean = self._finalize_crystal(X_mean)  # enforce centering & shape
            tok = tok_pref[tid]
            vol_m = v_sum[tid] / float(n_cnt[tid])  # mean volume

            self._token_to_id[tok] = tid
            self._id_to_token[tid] = tok

            if store in ("full", "both"):
                self._id_to_vec[tid] = np.asarray(X_mean, np.float32, order="C")
            elif store == "pooled":
                self._id_to_vec[tid] = X_mean.mean(axis=0).astype(np.float32, copy=False)

            self._id_to_volume[tid] = float(vol_m)
            self._valid_token_ids.add(tid)

        # ---------- specials (unchanged) ----------
        if manifest_specials and base_set:
            self._manifest_special_tokens(
                base_set=base_set,
                create_crystal=create_crystal,
                callback=callback,
                create_config=create_config or {}
            )

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
        # deterministic, bounded surrogate
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
            mats.append(torch.tensor(X, dtype=dtype))
            pooled.append(torch.tensor(v, dtype=dtype))
            keep.append(t)
        if not mats:
            raise ValueError("No valid tokens found in input.")
        return {
            "tokens": keep,
            "crystals": torch.stack(mats, 0).to(device),
            "pooled":   torch.stack(pooled, 0).to(device),
        }

    def _coerce_crystal_shape(
        self,
        raw: Any,
        *,
        vertex_count: int,
        reshape_order: str,
        infer_dim: bool,
        strict_shapes: bool
    ) -> np.ndarray:
        """
        Accepts raw crystal data and returns [vertex_count, self.dim] float32 C-order.

        Acceptable inputs:
          - [vertex_count, D]
          - [vertex_count * D] (flat)  -> reshaped to [vertex_count, D]
          - [D] (pooled center)       -> converted by deterministic pentachoron (fallback)
        """
        X = np.asarray(raw, dtype=np.float32)

        # Already [V, D]
        if X.ndim == 2:
            V, D = int(X.shape[0]), int(X.shape[1])
            if V != vertex_count:
                if strict_shapes:
                    raise ValueError(f"Crystal has {V} vertices, expected {vertex_count}.")
                # Gentle fallback: attempt to treat rows as vertices if divisible
                if V * D % vertex_count == 0 and infer_dim:
                    # e.g., [10, D] -> try to collapse/average into [5,D]? Not safe.
                    # Safer: hard error to avoid silent geometry change.
                    raise ValueError(f"Unexpected vertex rows {V}; refusing to coerce silently.")
                else:
                    raise ValueError(f"Crystal has {V} vertices, expected {vertex_count}.")
            # Update dim if needed
            if D != self.dim:
                if infer_dim:
                    self.dim = D
                else:
                    raise ValueError(f"Dim mismatch: got D={D}, expected dim={self.dim}.")
            # Ensure mean-centered (finalize handles centering)
            return X

        # Flat [V*D]
        if X.ndim == 1:
            n = int(X.size)
            # Exact match for flat crystal
            if n == vertex_count * self.dim:
                return np.reshape(X, (vertex_count, self.dim), order=reshape_order)

            # Infer D from total length if divisible
            if infer_dim and n % vertex_count == 0:
                inferred = n // vertex_count
                self.dim = int(inferred)
                return np.reshape(X, (vertex_count, self.dim), order=reshape_order)

            # Pooled [D]: inflate deterministically to [V, D]
            if n == self.dim:
                c = X / (np.abs(X).sum() + 1e-8)  # L1
                return self._deterministic_pentachoron(c)

            if strict_shapes:
                raise ValueError(
                    f"Cannot coerce crystal of length {n}. "
                    f"Expected {vertex_count*self.dim} (flat) or {self.dim} (pooled)."
                )
            # Conservative fallback: treat as pooled center with inferred D if reasonable
            if infer_dim and n > 0:
                self.dim = n
                c = X / (np.abs(X).sum() + 1e-8)
                return self._deterministic_pentachoron(c)

        raise ValueError(f"Unsupported crystal shape {X.shape} (ndim={X.ndim}).")


    # -------- Introspection --------
    def describe(self) -> Dict[str, Union[str, int]]:
        return {"repo": self.repo_id, "dimension": self.dim, "vocab_size": self.vocab_size()}
