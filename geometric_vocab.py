from __future__ import annotations
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Union, Tuple, Optional, Callable, Any, List


class GeometricVocab(ABC):
    """
    Abstract, deterministic crystal vocabulary.

    Responsibilities:
      • Holds maps (token<->id, id->crystal, id->volume, id->provenance)
      • Provides universal deterministic utilities:
          - embedding/pooled/similarity/similarity_magnitude
          - extract_band
          - helpers exposed to user callbacks
          - DEFAULT unicode-based create_crystal path (no RNG)
          - deterministic pentachoron inflation and finalize(+provenance)
          - special-token manifestation routine (calls user create_crystal)
      • Subclasses MUST implement encode() and get_score().
      • Subclasses MAY override any protected utilities (_compose_*, _deterministic_*).

    ZERO stochasticity in this class.
    """

    # --------------------------- lifecycle ---------------------------
    def __init__(self, dim: int):
        self.dim = int(dim)
        self._token_to_id: Dict[str, int] = {}
        self._id_to_token: Dict[int, str] = {}
        self._id_to_vec: Dict[int, np.ndarray] = {}
        self._id_to_volume: Dict[int, float] = {}
        self._id_to_provenance: Dict[int, dict] = {}
        self._valid_token_ids: set[int] = set()

    # --------------------------- abstract surface --------------------
    @abstractmethod
    def encode(self, token: str, *, return_id: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, int]]:
        raise NotImplementedError

    @abstractmethod
    def get_score(self, token_or_id: Union[str, int]) -> float:
        raise NotImplementedError

    # --------------------------- basic queries -----------------------
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
        X = self.embedding(token_or_id)
        if X is None:
            return None
        if method == "mean": return X.mean(axis=0)
        if method == "first": return X[0]
        if method == "sum": return X.sum(axis=0)
        raise ValueError(f"Invalid pooling method: {method}")

    # --------------------------- rose similarity ---------------------
    def similarity(self, token_a: Union[str, int], token_b: Union[str, int]) -> float:
        a = self.pooled(token_a); b = self.pooled(token_b)
        if a is None or b is None: return -1.0
        a /= (np.linalg.norm(a) + 1e-8); b /= (np.linalg.norm(b) + 1e-8)
        return float(np.dot(a, b))

    def similarity_magnitude(self, token_a: Union[str, int], token_b: Union[str, int]) -> float:
        a = self.pooled(token_a); b = self.pooled(token_b)
        if a is None or b is None: return -1.0
        return float(np.dot(a, b))

    # --------------------------- trajectory band ---------------------
    def extract_band(self, trajectory: np.ndarray, max_angle: float = 0.3, method: str = "pooled") -> Dict[str, np.ndarray]:
        if trajectory.ndim == 2: direction = trajectory.mean(0)
        else: direction = trajectory
        direction /= (np.linalg.norm(direction) + 1e-8)

        out: Dict[str, np.ndarray] = {}
        for tok, tid in self._token_to_id.items():
            v = self.pooled(tid, method=method)
            if v is None: continue
            v /= (np.linalg.norm(v) + 1e-8)
            if float(np.dot(v, direction)) >= 1.0 - max_angle:
                out[tok] = self._id_to_vec[tid]
        return out

    # --------------------------- helpers exposed to callbacks --------
    def _helpers(self) -> Dict[str, Callable[..., np.ndarray]]:
        return {
            "embedding": lambda x: np.asarray(self.embedding(x), np.float32) if self.embedding(x) is not None else None,
            "pooled": lambda x: np.asarray(self.pooled(x),    np.float32) if self.pooled(x)    is not None else None,
            "chars_pooled": lambda s: [self.pooled(c) for c in s] if isinstance(s, str) else None,
        }

    # --------------------------- DEFAULT create_crystal (unicode path) ----
    def _default_create_crystal(self, config: dict, callback: Callable[..., np.ndarray]) -> np.ndarray:
        """
        Deterministic default when user leaves callback/create_crystal=None.
        Honors:
          - config['pool_type'] in {"mean","abs","dot","unicode","wordnet","combination", None}
          - config['additional_definitions'] (list[str])
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
        else:  # unicode / mean / abs / dot handled in builders; fall back to unicode
            c = c_uni if c_uni is not None else np.zeros(d, np.float32)

        c /= (np.linalg.norm(c) + 1e-8)
        return self._deterministic_pentachoron(c)

    def _default_unicode_callback(self, name: str, **kwargs) -> np.ndarray:
        raise NotImplementedError("Default callback is not invoked directly.")

    # --------------------------- universal builders (overrideable) ---
    def _compose_unicode_center(self, token_plain: str, H, pool_type: str, dim: int) -> Optional[np.ndarray]:
        vecs: List[np.ndarray] = []
        for ch in token_plain:
            v = H["pooled"](ch)
            if v is None: continue
            v = np.asarray(v, np.float32)
            if pool_type == "abs": v = np.abs(v)
            vecs.append(v)
        if not vecs: return None
        c = np.mean(np.stack(vecs, 0), 0)
        if pool_type == "dot": c /= (np.linalg.norm(c) + 1e-8)
        return c

    def _compose_wordnet_center(self, definitions: List[str], H, pool_type: str, dim: int) -> Optional[np.ndarray]:
        vecs: List[np.ndarray] = []
        for text in definitions:
            for ch in str(text):
                v = H["pooled"](ch)
                if v is None: continue
                v = np.asarray(v, np.float32)
                if pool_type == "abs": v = np.abs(v)
                vecs.append(v)
        if not vecs: return None
        c = np.mean(np.stack(vecs, 0), 0)
        if pool_type == "dot": c /= (np.linalg.norm(c) + 1e-8)
        return c

    def _deterministic_pentachoron(self, center_vec: np.ndarray) -> np.ndarray:
        """
        Universal pentachoron inflation (deterministic; overrideable).
        """
        d = center_vec.shape[0]
        proposals = np.stack([
            center_vec,
            np.roll(center_vec, 1),
            np.roll(center_vec, 3) * np.sign(center_vec + 1e-8),
            np.roll(center_vec, 7) - center_vec,
            np.roll(center_vec, 11) + center_vec,
        ], 0).astype(np.float32)

        norms = np.linalg.norm(proposals, 1, keepdims=True) + 1e-8
        Q = proposals / norms
        for i in range(5):
            for j in range(i):
                Q[i] -= np.dot(Q[i], Q[j]) * Q[j]
            Q[i] /= (np.linalg.norm(Q[i]) + 1e-8)

        gamma = np.array([1.0, 0.9, -0.8, 1.1, 1.2], np.float32)
        X = np.zeros((5, d), np.float32)
        for i in range(5): X[i] = center_vec + gamma[i] * Q[i]
        return X - X.mean(0, keepdims=True)

    # --------------------------- finalize + provenance (overrideable) ----
    def _finalize_crystal(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, np.float32)
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

        # dict path: {"base": [5,D], "ops": [...], "provenance": {...}}
        if not isinstance(product, dict):
            raise TypeError("create_crystal must return ndarray or dict with {'base':..., 'ops':..., 'provenance':...}.")
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
                if t.shape != (self.dim,): raise ValueError(f"translate.t must be shape ({self.dim},)")
                X = X + t[None, :]
            elif name == "normalize_rows":
                n = np.linalg.norm(X, 1, keepdims=True) + 1e-8
                X = X / n
            elif name == "align_to":
                v = np.asarray(op.get("v"), np.float32)
                if v.shape != (self.dim,): raise ValueError(f"align_to.v must be shape ({self.dim},)")
                v = v / (np.linalg.norm(v) + 1e-8)
                p = X.mean(0); p = p / (np.linalg.norm(p) + 1e-8)
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
        """
        Universal, deterministic manifestor. Subclasses call this after loading.
        """
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
                "pool_type": create_config.get("pool_type", None),  # None => "unicode"
                "special_tokens": create_config.get("special_tokens"),
                "additional_definitions": create_config.get("additional_definitions", []),
                "antonyms": create_config.get("antonyms"),
                "inversion_formula": create_config.get("inversion_formula"),
                "data": {"token": name.strip("<>").strip(), "token_id": tid, "origin": "special"},
                "helpers": helpers,
            }

            # Default path if user left create_crystal/callback None:
            if create_crystal is None:
                create_crystal = self._default_create_crystal
            # Note: default callback is not used directly; helpers suffice.

            product = create_crystal(cfg, callback) if callback is not None else create_crystal(cfg, self._default_unicode_callback)  # callback may be None
            X, prov = self._finalize_crystal_and_provenance(product, cfg)

            # Register
            self._token_to_id[name] = tid
            self._id_to_token[tid] = name
            self._id_to_vec[tid] = X.astype(np.float32, copy=False)
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
                            self._id_to_vec[anti_id] = X_inv.astype(np.float32, copy=False)
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

    # --------------------------- basics -------------------------------
    def vocab_size(self) -> int: return len(self._token_to_id)
    def token_to_id(self, token: str) -> Optional[int]: return self._token_to_id.get(token)
    def id_to_token(self, token_id: int) -> Optional[str]: return self._id_to_token.get(token_id)
