from __future__ import annotations
import numpy as np
import torch
from datasets import load_dataset
from typing import List, Dict, Union, Optional, Tuple, Callable, Any

from .geometric_vocab import GeometricVocab


class PretrainedGeometricVocab(GeometricVocab):
    """
    Deterministic accessor around a parquet-backed vocabulary.

    Special tokens are manifested by user-supplied `create_crystal(config, callback)`,
    with `callback` being the deterministic math provider.

    If left None, the default path is to form crystals directly from
    Unicode character composition (no RNG).
    """

    def __init__(
        self,
        repo_id: str,
        dim: int,
        *,
        base_set: Optional[Dict[str, int]] = None,
        create_config: Optional[Dict[str, Any]] = None,
        create_crystal: Optional[Callable[[dict, Callable[..., np.ndarray]], Union[np.ndarray, Dict[str, Any]]]] = None,
        callback: Optional[Callable[..., np.ndarray]] = None,
        manifest_specials: bool = True,
    ):
        super().__init__(dim)
        self.repo_id = str(repo_id)

        ds = load_dataset(self.repo_id, split="train")

        self._id_to_volume: Dict[int, float] = {}
        for rec in ds:
            tid = int(rec["token_id"])
            tok = str(rec["token"])
            X = np.asarray(rec["crystal"], dtype=np.float32)
            self._token_to_id[tok] = tid
            self._id_to_token[tid] = tok
            self._id_to_vec[tid] = X
            self._id_to_volume[tid] = float(rec.get("volume", 1.0))

        self._valid_token_ids = set(self._id_to_token.keys())

        if manifest_specials and base_set:
            if create_crystal is None:
                create_crystal = self._default_create_crystal
            if callback is None:
                callback = self._default_unicode_callback
            self._manifest_special_tokens(base_set, create_crystal, callback, create_config or {})

    # ---------- SentencePiece-compatible surface ----------
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

    # ---------- Torch cache ----------
    def cache(self, tokens: Union[List[str], Dict[str, int]], device: str = "cpu", dtype: torch.dtype = torch.float32):
        tok_list = list(tokens.keys()) if isinstance(tokens, dict) else list(tokens)
        mats, pooled, keep = [], [], []

        for t in tok_list:
            X = self.embedding(t); v = self.pooled(t)
            if X is None or v is None: continue
            mats.append(torch.tensor(X, dtype=dtype))
            pooled.append(torch.tensor(v, dtype=dtype))
            keep.append(t)

        if not mats: raise ValueError("No valid tokens found in input.")

        return {
            "tokens": keep,
            "crystals": torch.stack(mats, 0).to(device),   # [N, 5, D]
            "pooled": torch.stack(pooled, 0).to(device),   # [N, D]
        }

    def describe(self) -> Dict[str, Union[str, int]]:
        return {"repo": self.repo_id, "dimension": self.dim, "vocab_size": self.vocab_size()}

    # ---------- Special token manifestation ----------
    def _manifest_special_tokens(self, base_set, create_crystal, callback, create_config):
        helpers = self._helpers()

        for name, tid in base_set.items():
            if tid in self._id_to_vec:
                self._token_to_id[name] = tid
                self._id_to_token.setdefault(tid, name)
                self._valid_token_ids.add(tid)
                continue

            cfg = {
                "dim": self.dim,
                "pool_type": create_config.get("pool_type", None),  # None => "unicode"
                "special_tokens": create_config.get("special_tokens"),
                "additional_definitions": create_config.get("additional_definitions", []),
                "antonyms": create_config.get("antonyms"),
                "inversion_formula": create_config.get("inversion_formula"),
                "data": {"token": name.strip("<>").strip(), "token_id": tid, "origin": "special"},
                "helpers": helpers,
            }

            product = create_crystal(cfg, callback)
            X = self.__finalize_crystal(product)

            self._token_to_id[name] = tid
            self._id_to_token[tid] = name
            self._id_to_vec[tid] = X.astype(np.float32, copy=False)
            self._valid_token_ids.add(tid)
            self._id_to_volume.setdefault(tid, 1.0)

            # Aliases
            for alias in cfg.get("special_tokens") or []:
                self._token_to_id[alias] = tid
                self._id_to_token.setdefault(tid, alias)

            # Antonyms
            antonyms = cfg.get("antonyms") or []
            invf = cfg.get("inversion_formula")
            if invf:
                for anti in antonyms:
                    if anti in base_set:
                        anti_id = base_set[anti]
                        if anti_id not in self._id_to_vec:
                            X_inv = self.__finalize_crystal(invf(X, cfg))
                            self._token_to_id[anti] = anti_id
                            self._id_to_token[anti_id] = anti
                            self._id_to_vec[anti_id] = X_inv.astype(np.float32, copy=False)
                            self._valid_token_ids.add(anti_id)
                            self._id_to_volume.setdefault(anti_id, 1.0)

    # ---------- Default deterministic path ----------
    def _default_create_crystal(self, config: dict, callback: Callable[..., np.ndarray]) -> np.ndarray:
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

        c /= (np.linalg.norm(c) + 1e-8)
        return self._deterministic_pentachoron(c)

    def _default_unicode_callback(self, name: str, **kwargs) -> np.ndarray:
        raise NotImplementedError("Default callback not called directly.")

    # ---------- deterministic builders ----------
    def _compose_unicode_center(self, token_plain, H, pool_type, dim):
        vecs = []
        for ch in token_plain:
            v = H["pooled"](ch)
            if v is None: continue
            v = np.asarray(v, np.float32)
            if pool_type == "abs": v = np.abs(v)
            vecs.append(v)
        if not vecs: return None
        c = np.mean(np.stack(vecs, 0), 0)
        if pool_type == "dot": c /= (np.linalg.norm(c)+1e-8)
        return c

    def _compose_wordnet_center(self, definitions, H, pool_type, dim):
        vecs = []
        for text in definitions:
            for ch in str(text):
                v = H["pooled"](ch)
                if v is None: continue
                v = np.asarray(v, np.float32)
                if pool_type == "abs": v = np.abs(v)
                vecs.append(v)
        if not vecs: return None
        c = np.mean(np.stack(vecs, 0), 0)
        if pool_type == "dot": c /= (np.linalg.norm(c)+1e-8)
        return c

    def _deterministic_pentachoron(self, c: np.ndarray) -> np.ndarray:
        d = c.shape[0]
        proposals = np.stack([
            c,
            np.roll(c, 1),
            np.roll(c, 3) * np.sign(c + 1e-8),
            np.roll(c, 7) - c,
            np.roll(c, 11) + c,
        ], 0).astype(np.float32)

        norms = np.linalg.norm(proposals, 1, keepdims=True) + 1e-8
        Q = proposals / norms
        for i in range(5):
            for j in range(i):
                Q[i] -= np.dot(Q[i], Q[j]) * Q[j]
            Q[i] /= (np.linalg.norm(Q[i]) + 1e-8)

        gamma = np.array([1.0, 0.9, -0.8, 1.1, 1.2], np.float32)
        X = np.zeros((5, d), np.float32)
        for i in range(5): X[i] = c + gamma[i] * Q[i]
        return X - X.mean(0, keepdims=True)

    def __finalize_crystal(self, product: Union[np.ndarray, Dict[str, Any]]) -> np.ndarray:
        if isinstance(product, np.ndarray): return product
        if not isinstance(product, dict): raise TypeError("create_crystal must return ndarray or dict.")
        base = np.asarray(product["base"], np.float32)
        X = base
        for op in product.get("ops", []):
            if op["name"] == "center": X -= X.mean(0, keepdims=True)
            elif op["name"] == "scale": X *= float(op.get("k", 1.0))
        return X

    def _helpers(self) -> Dict[str, Callable[..., np.ndarray]]:
        return {
            "embedding": lambda x: np.asarray(self.embedding(x), np.float32) if self.embedding(x) is not None else None,
            "pooled": lambda x: np.asarray(self.pooled(x), np.float32) if self.pooled(x) is not None else None,
            "chars_pooled": lambda s: [self.pooled(c) for c in s] if isinstance(s, str) else None,
        }
