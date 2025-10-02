# geo_tensor.py
# specifically formatted as a dataclass for easy use and immutability
# concise and clear definition of geometric tensors for a very small footprint
from __future__ import annotations
from dataclasses import dataclass, replace
from torch import Tensor
from typing import Dict, Optional, Literal, Iterable, Any
import torch

# ========= BASE CONFIGURATION =========

Role = Literal["anchor", "need", "relation", "purpose", "observer", "generic"]
Tier = Literal["simple", "windowed", "barycentric", "hierarchical", "graph"]


@dataclass(frozen=True, slots=True)
class GeoTensor:
    fields: Dict[str, Tensor]
    tier: Tier = "simple"
    eps: float = 1e-12
    metadata: Optional[Dict[str, Any]] = None

    def has(self, n: str) -> bool:
        return n in self.fields

    def get(self, n: str) -> Tensor:
        t = self.fields.get(n)
        if t is None:
            raise KeyError(f"missing field {n}")
        return t

    def ensure(self) -> "GeoTensor":
        if self.tier == "windowed" and self.has("mask"):
            m = self.get("mask")
            x = self.get("x")
            if m.dim() == 2:
                m = m.unsqueeze(-1)
            if m.shape[:-1] != x.shape[:-1] or m.shape[-1] != 1:
                raise ValueError("mask must be [...,W,1] matching x")
        return self
