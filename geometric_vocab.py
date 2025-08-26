from __future__ import annotations
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Union, Tuple, Optional


class GeometricVocab(ABC):
    """
    Abstract, deterministic crystal vocabulary.
    Provides canonical utilities for embedding lookup,
    pooling, rose similarity, and band extraction.

    Subclasses must implement:
      • encode(token)
      • get_score(token_or_id)
    """

    def __init__(self, dim: int):
        self.dim = int(dim)
        self._token_to_id: Dict[str, int] = {}
        self._id_to_token: Dict[int, str] = {}
        self._id_to_vec: Dict[int, np.ndarray] = {}

    # ---------- abstract ----------
    @abstractmethod
    def encode(self, token: str, *, return_id: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, int]]:
        raise NotImplementedError

    @abstractmethod
    def get_score(self, token_or_id: Union[str, int]) -> float:
        raise NotImplementedError

    # ---------- core utils ----------
    def decode(self, token_id: int, fallback: str = "<unk>") -> Optional[str]:
        if token_id in self._id_to_token:
            return self._id_to_token[token_id]
        return fallback if fallback in self._token_to_id else None

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

    # ---------- rose similarity ----------
    def similarity(self, token_a: Union[str, int], token_b: Union[str, int]) -> float:
        a = self.pooled(token_a); b = self.pooled(token_b)
        if a is None or b is None: return -1.0
        a /= (np.linalg.norm(a) + 1e-8); b /= (np.linalg.norm(b) + 1e-8)
        return float(np.dot(a, b))

    def similarity_magnitude(self, token_a: Union[str, int], token_b: Union[str, int]) -> float:
        a = self.pooled(token_a); b = self.pooled(token_b)
        if a is None or b is None: return -1.0
        return float(np.dot(a, b))

    # ---------- trajectory band ----------
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

    # ---------- basics ----------
    def vocab_size(self) -> int: return len(self._token_to_id)
    def token_to_id(self, token: str) -> Optional[int]: return self._token_to_id.get(token)
    def id_to_token(self, token_id: int) -> Optional[str]: return self._id_to_token.get(token_id)
