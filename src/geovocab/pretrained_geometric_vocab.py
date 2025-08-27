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
        create_crystal: Optional[Callable[[dict, Callable[..., np.ndarray]], Union[np.ndarray, Dict[str, Any]]]] = None,
        callback: Optional[Callable[..., np.ndarray]] = None,
        manifest_specials: bool = True,
        # ----- NEW shape controls -----
        layout: str = "auto",             # "auto" | "flat" | "stacked" | "pooled"
        vertex_count: int = 5,
        infer_dim: bool = True,
        strict_shapes: bool = False,
        reshape_order: str = "C",
    ):
        super().__init__(dim)
        self.repo_id = str(repo_id)

        # ---------- Load parquet (columnar) ----------
        ds = load_dataset(self.repo_id, split=split)
        wanted = {"token_id", "token", "crystal", "volume"}
        drop = [c for c in ds.column_names if c not in wanted]
        if drop:
            ds = ds.remove_columns(drop)
        ds = ds.with_format("numpy")

        ids = ds["token_id"]
        toks = ds["token"]
        crystals = ds["crystal"]
        has_volume = "volume" in ds.column_names
        vols = ds["volume"] if has_volume else None

        # ---------- Optional early shape probe ----------
        # Find first non-empty to set/validate dimension proactively
        if len(crystals) > 0:
            probe = self._coerce_crystal_shape(
                crystals[0],
                vertex_count=vertex_count,
                reshape_order=reshape_order,
                infer_dim=infer_dim,
                strict_shapes=strict_shapes,
            )
            # Validate probe width against current self.dim (helper may have updated it)
            if probe.shape != (vertex_count, self.dim):
                raise ValueError(f"Probe mismatch: expected ({vertex_count},{self.dim}) got {probe.shape}.")

        # ---------- Bulk maps ----------
        ids_int = [int(x) for x in ids]
        toks_str = [str(x) for x in toks]
        self._token_to_id.update(zip(toks_str, ids_int))
        self._id_to_token.update(zip(ids_int, toks_str))

        # ---------- Coerce + register crystals ----------
        # (Reshape/convert each entry once; center finalize later only when needed.)
        for tid, raw in zip(ids_int, crystals):
            X = self._coerce_crystal_shape(
                raw,
                vertex_count=vertex_count,
                reshape_order=reshape_order,
                infer_dim=infer_dim,
                strict_shapes=strict_shapes,
            )
            # Ensure final [V,D] with mean-centering invariance
            self._id_to_vec[tid] = self._finalize_crystal(X)
            self._valid_token_ids.add(tid)

        # ---------- Volumes ----------
        if has_volume and vols is not None:
            self._id_to_volume.update(zip(ids_int, [float(v) for v in vols]))
        else:
            self._id_to_volume.update(zip(ids_int, [1.0] * len(ids_int)))

        # ---------- Specials ----------
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
