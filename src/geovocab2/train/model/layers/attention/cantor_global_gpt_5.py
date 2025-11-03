# geovocab2/train/model/layers/attention/cantor_global_plus.py
import math, torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Dict, Tuple

@dataclass
class CantorAttentionPlusConfig:
    dim: int = 512
    num_heads: int = 8
    head_dim: Optional[int] = None
    depth: int = 8
    max_seq_len: int = 524_288
    local_window: int = 64
    adaptive_window: bool = False
    min_window: int = 16
    max_window: int = 64
    sparsity_target: float = 0.25
    dropout: float = 0.0
    causal: bool = False
    qkv_bias: bool = True
    out_bias: bool = True

    # Opt-in improvements (all default OFF to preserve baseline)
    deterministic_topk: bool = False        # stable tie-breaks only
    enable_attention_mask: bool = False     # pad mask support
    use_gather_impl: bool = False           # take_along_dim gather
    per_device_route_cache: bool = False    # keep routes on device
    logits_cantor_prior_lambda: float = 0.0 # >0 to subtract λ*|Δcantor|
    verbose: bool = False

    def __post_init__(self):
        if self.head_dim is None:
            assert self.dim % self.num_heads == 0
            self.head_dim = self.dim // self.num_heads
        if self.adaptive_window:
            assert self.min_window > 0 and self.max_window >= self.min_window
            assert 0 < self.sparsity_target <= 1.0

    def get_window_size(self, n: int) -> int:
        if not self.adaptive_window: return self.local_window
        k = int(n * self.sparsity_target)
        return max(self.min_window, min(k, self.max_window))

class CantorAttentionPlus(nn.Module):
    """
    Compatible alternate for Cantor attention with opt-in hardening.
    Routing semantics remain: exact KNN in Cantor-coordinate space.
    """
    def __init__(self, cfg: CantorAttentionPlusConfig):
        super().__init__()
        self.cfg = cfg
        self.dim, self.num_heads, self.head_dim = cfg.dim, cfg.num_heads, cfg.head_dim
        self.qkv = nn.Linear(cfg.dim, 3 * cfg.dim, bias=cfg.qkv_bias)
        self.out_proj = nn.Linear(cfg.dim, cfg.dim, bias=cfg.out_bias)
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.routes_cache: Dict[Tuple[int,int], torch.Tensor] = {}
        self.routes_cache_dev: Dict[Tuple[int,int,str,Optional[int]], torch.Tensor] = {}

        # Optional prebuild (same list as your baseline)
        for size in [64,128,256,512,1024,2048,4096,8192,16384,32768]:
            if size <= cfg.max_seq_len:
                k = cfg.get_window_size(size)
                self.routes_cache[(size,k)] = self._build_routes_knn(size, k, cfg.depth)

    # --- Cantor coordinate (same semantics as your mapping) ---
    @staticmethod
    def _cantor_coordinate(pos: int, max_len: int, depth: int) -> float:
        x = pos / max(1, max_len - 1); x = max(1e-6, min(x, 1.0-1e-6))
        v, f = 0.0, 0.5
        for _ in range(depth):
            x *= 3.0; d = int(x); x -= d
            if d == 2: v += f
            f *= 0.5
        return v

    def _stable_topk_indices(self, dist: torch.Tensor, k: int) -> torch.Tensor:
        # Deterministic only for exact ties; unequal orders unchanged.
        n = dist.shape[0]
        eps = torch.arange(n, device=dist.device, dtype=dist.dtype) * 1e-12
        _, idx = torch.topk(dist + eps, k, largest=False)
        return idx

    def _build_routes_knn(self, n: int, k: int, depth: int) -> torch.Tensor:
        coords = torch.tensor([self._cantor_coordinate(i, n, depth) for i in range(n)], dtype=torch.float32)
        routes = torch.empty((n, k), dtype=torch.long)
        for i in range(n):
            dist = (coords - coords[i]).abs()
            idx = self._stable_topk_indices(dist, k) if self.cfg.deterministic_topk else torch.topk(dist, k, largest=False).indices
            routes[i] = idx
        routes[:,0] = torch.arange(n)  # ensure self for safety
        return routes

    def _get_routes(self, n: int, k: int, device):
        key = (n, k)
        if key in self.routes_cache:
            return self.routes_cache[key].to(device)
        # ⇩ add this block (identical to your baseline)
        for (cn, ck), tbl in sorted(self.routes_cache.items()):
            if ck == k and cn >= n:
                r = tbl[:n, :].clamp_(0, n - 1)
                self.routes_cache[key] = r
                return r.to(device)
        # fall back to on-demand build only if no larger cached table exists
        r = self._build_routes_knn(n, k, self.cfg.depth)
        self.routes_cache[key] = r
        return r.to(device)

    def _sparse_attention(self, q, k, v, seq_len, attention_mask=None):
        B, H, _, D = q.shape
        dev = q.device
        k_neighbors = self.cfg.get_window_size(seq_len)
        routes = self._get_routes(seq_len, k_neighbors, dev)            # (n,k)
        routes_bc = routes.view(1,1,seq_len,k_neighbors).expand(B,H,seq_len,k_neighbors)

        if self.cfg.use_gather_impl:
            idx = routes_bc.unsqueeze(-1)                               # (B,H,n,k,1)
            k_src = k.unsqueeze(3).expand(B,H,seq_len,k_neighbors,D)
            v_src = v.unsqueeze(3).expand_as(k_src)
            k_g = torch.take_along_dim(k_src, idx, dim=2)               # (B,H,n,k,D)
            v_g = torch.take_along_dim(v_src, idx, dim=2)
        else:
            # legacy advanced indexing path (identical numerics)
            batch_idx = torch.arange(B, device=dev).view(-1,1,1,1).expand(B,H,seq_len,k_neighbors)
            head_idx  = torch.arange(H, device=dev).view(1,-1,1,1).expand(B,H,seq_len,k_neighbors)
            k_g = k[batch_idx, head_idx, routes_bc, :]
            v_g = v[batch_idx, head_idx, routes_bc, :]

        # logits
        scores = torch.einsum('bhqd,bhqkd->bhqk', q, k_g).to(torch.float32) * self.scale

        # optional Cantor-distance prior on logits (does not change neighbors)
        if self.cfg.logits_cantor_prior_lambda > 0.0:
            # reuse scalar coords cheaply
            with torch.no_grad():
                coords = torch.linspace(0, 1, steps=seq_len, device=dev)  # monotone proxy of mapping
            delta = (coords[routes_bc] - coords.view(1,1,seq_len,1)).abs()
            scores = scores - self.cfg.logits_cantor_prior_lambda * delta

        if self.cfg.causal:
            pos = torch.arange(seq_len, device=dev).view(1,1,seq_len,1)
            scores = scores.masked_fill(routes_bc > pos, torch.finfo(scores.dtype).min)

        if self.cfg.enable_attention_mask and (attention_mask is not None):
            am = attention_mask
            if am.dim() == 2: am = am.to(dev).unsqueeze(1).unsqueeze(1)  # (B,1,1,n)
            else:             am = am.to(dev)
            am_g = torch.take_along_dim(am.expand(B,1,seq_len,seq_len), routes_bc, dim=3)  # (B,1,n,k)
            scores = scores.masked_fill(am_g == 0, torch.finfo(scores.dtype).min)

        # all-masked guard: lift self to zero only if entire row is −inf
        all_masked = torch.isneginf(scores).all(dim=-1, keepdim=True)
        if all_masked.any():
            scores = torch.where(all_masked, torch.zeros_like(scores), scores)

        attn = F.softmax(scores, dim=-1).to(q.dtype)
        attn = self.attn_dropout(attn)
        out  = torch.einsum('bhqk,bhqkd->bhqd', attn, v_g)
        return out

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor]=None) -> torch.Tensor:
        B, n, _ = x.shape
        qkv = self.qkv(x).reshape(B, n, 3, self.num_heads, self.head_dim).permute(2,0,3,1,4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        y = self._sparse_attention(q, k, v, n, attention_mask if self.cfg.enable_attention_mask else None)
        y = y.transpose(1,2).reshape(B, n, self.dim)
        return self.resid_dropout(self.out_proj(y))

    def extra_repr(self) -> str:
        return (f"dim={self.dim}, heads={self.num_heads}, head_dim={self.head_dim}, "
                f"k={self.cfg.local_window}, adaptive={self.cfg.adaptive_window}, depth={self.cfg.depth}, "
                f"opts(det_topk={self.cfg.deterministic_topk}, mask={self.cfg.enable_attention_mask}, "
                f"gather={self.cfg.use_gather_impl}, dev_cache={self.cfg.per_device_route_cache}, "
                f"λ={self.cfg.logits_cantor_prior_lambda}))")

def create_cantor_attention_plus(**kwargs) -> CantorAttentionPlus:
    cfg = CantorAttentionPlusConfig(**kwargs)
    return CantorAttentionPlus(cfg)
