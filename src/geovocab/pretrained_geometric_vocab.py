from __future__ import annotations
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
    ):
        super().__init__(dim)
        self.repo_id = str(repo_id)

        # Load deterministic parquet
        ds = load_dataset(self.repo_id, split=split)

        for rec in ds:
            tid = int(rec["token_id"])
            tok = str(rec["token"])
            X = np.asarray(rec["crystal"], dtype=np.float32)
            self._token_to_id[tok] = tid
            self._id_to_token[tid] = tok
            self._id_to_vec[tid] = X
            self._id_to_volume[tid] = float(rec.get("volume", 1.0))
            self._valid_token_ids.add(tid)

        # Optionally manifest specials via base universal routine
        if manifest_specials and base_set:
            self._manifest_special_tokens(
                base_set=base_set,
                create_crystal=create_crystal,   # may be None -> base default
                callback=callback,               # may be None
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

    # -------- Introspection --------
    def describe(self) -> Dict[str, Union[str, int]]:
        return {"repo": self.repo_id, "dimension": self.dim, "vocab_size": self.vocab_size()}
