# ============================================================================
# VocabBase (minimal interface)
# ============================================================================
from __future__ import annotations
import abc
from typing import Any, Dict, Tuple, Union
import numpy as np


class VocabBase(abc.ABC):
    """
    Minimal base class for vocabulary systems.
    Provides only the standard access avenues to be overridden or extended.

    Required overrides:
        encode(token, *, return_id=False)
        get_score(token_or_id)
    """

    def __init__(self, name: str = "vocab_base", fid: str = "vocab.base", dim: int = 512):
        self.name = name
        self.fid = fid
        self.dim = dim

    # ------------------------------------------------------------------
    # Primary abstract entry points
    # ------------------------------------------------------------------
    @abc.abstractmethod
    def encode(self, token: str, *, return_id: bool = False
               ) -> Union[np.ndarray, Tuple[np.ndarray, int]]:
        """Return crystal or embedding for token. Deterministic per subclass."""

    @abc.abstractmethod
    def get_score(self, token_or_id: Union[str, int]) -> float:
        """Return quality or reliability score for token or id."""

    # ------------------------------------------------------------------
    # Standard access avenues (thin stubs to override if needed)
    # ------------------------------------------------------------------
    def embedding(self, token_or_id: Union[str, int]) -> np.ndarray:
        """Default path: defer to encode(); subclasses may override."""
        tok = token_or_id if isinstance(token_or_id, str) else str(token_or_id)
        out = self.encode(tok)
        return out[0] if isinstance(out, tuple) else out

    def pooled(self, token_or_id: Union[str, int]) -> np.ndarray:
        """Return pooled vector (default: mean across first axis)."""
        x = self.embedding(token_or_id)
        return x.mean(axis=0) if x.ndim == 2 else x

    def similarity(self, a: Union[str, int], b: Union[str, int], *, metric: str = "cosine") -> float:
        """Compute simple similarity; override for advanced routing."""
        va, vb = self.pooled(a), self.pooled(b)
        if metric == "cosine":
            return float(np.dot(va, vb) / ((np.linalg.norm(va) * np.linalg.norm(vb)) + 1e-12))
        if metric == "dot":
            return float(np.dot(va, vb))
        return float(-np.linalg.norm(va - vb))

    # ------------------------------------------------------------------
    # Informational helpers
    # ------------------------------------------------------------------
    def summary(self) -> Dict[str, Any]:
        return {"name": self.name, "fid": self.fid, "dim": self.dim}

    def __repr__(self):
        return f"<VocabBase name={self.name} fid={self.fid} dim={self.dim}>"
