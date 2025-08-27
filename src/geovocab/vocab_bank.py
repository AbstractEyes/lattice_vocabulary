# device_bank.py
# Global device-safe geometric vocab bank with GPU-native nearest ops.
from __future__ import annotations
import math, json, weakref, threading
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any, Iterable, Union

import numpy as np
import torch

# ========= RWLock (lightweight, Python-level; protects maps/tensors wiring) =========
class _RWLock:
    """
    Simple readers-writer lock:
      - Multiple readers may hold the lock simultaneously.
      - Writers get exclusive access.
    """
    def __init__(self) -> None:
        self._readers = 0
        self._rlock = threading.Lock()
        self._wlock = threading.Lock()
        self._cond  = threading.Condition(self._rlock)

    def acquire_read(self) -> None:
        with self._cond:
            self._readers += 1

    def release_read(self) -> None:
        with self._cond:
            self._readers -= 1
            if self._readers == 0:
                self._cond.notify_all()

    def acquire_write(self) -> None:
        self._wlock.acquire()
        self._rlock.acquire()
        # wait until no active readers
        while self._readers > 0:
            self._cond.wait()

    def release_write(self) -> None:
        self._rlock.release()
        self._wlock.release()


# ========= Config =========
@dataclass(frozen=True)
class VocabDeviceBankConfig:
    name: str
    dim: int
    store: str = "full"              # "full" | "pooled" | "both"
    vertex_count: int = 5
    normalize: str = "l1"            # "l1" | "none" (for similarity precompute)
    pinned_cpu: bool = True          # pin CPU memory for faster HtoD
    finalize_mode: str = "post_mean" # must match your PretrainedGeometricVocab usage
    cache_pooled: bool = True        # precompute pooled O(1) lookups


# ========= Core bank =========
class VocabDeviceBank:
    """
    A device-safe bank built from PretrainedGeometricVocab.
    Holds CPU "master" tensors; can share across processes and replicate/shard onto GPUs.
    """
    def __init__(self, cfg: VocabDeviceBankConfig) -> None:
        self.cfg = cfg
        self.lock = _RWLock()

        # CPU (master) tensors
        self._cpu_ids: Optional[torch.Tensor] = None            # [N] long
        self._cpu_pooled: Optional[torch.Tensor] = None         # [N, D] float
        self._cpu_crystals: Optional[torch.Tensor] = None       # [N, V, D] float (optional)
        self._cpu_pooled_l1: Optional[torch.Tensor] = None      # [N, D] float (normalized)
        self._token_to_id: Dict[str, int] = {}
        self._id_to_token: Dict[int, str] = {}
        self._id_to_row: Dict[int, int] = {}

        # Device replicas (per-device cache)
        self._dev_cache: Dict[str, Dict[str, torch.Tensor]] = {}  # key=device_str -> dict of tensors

    # ---------- Build from PretrainedGeometricVocab ----------
    @classmethod
    def from_pretrained(
        cls,
        vocab: "PretrainedGeometricVocab",
        *,
        name: str,
        store: str = "full",
        finalize_mode: str = "post_mean",
        normalize: str = "l1",
        pinned_cpu: bool = True,
        cache_pooled: bool = True,
    ) -> "VocabDeviceBank":
        cfg = VocabDeviceBankConfig(
            name=name,
            dim=int(vocab.dim),
            store=store,
            vertex_count=5,
            normalize=normalize,
            pinned_cpu=pinned_cpu,
            finalize_mode=finalize_mode,
            cache_pooled=cache_pooled,
        )
        bank = cls(cfg)
        bank._build_from_vocab(vocab)
        return bank

    def _build_from_vocab(self, vocab: "PretrainedGeometricVocab") -> None:
        """
        Pulls data from the CPU master (pretrained vocab) into contiguous torch tensors.
        """
        self.lock.acquire_write()
        try:
            # Freeze maps (deterministic ordering by token_id)
            tids = sorted(list(vocab._valid_token_ids))
            N = len(tids)
            D = self.cfg.dim
            V = self.cfg.vertex_count

            # Maps
            self._token_to_id = dict(vocab._token_to_id)    # shallow copy
            self._id_to_token = dict(vocab._id_to_token)
            self._id_to_row = {tid: i for i, tid in enumerate(tids)}

            # Allocate CPU tensors
            ids = torch.empty(N, dtype=torch.long)
            pooled = torch.empty(N, D, dtype=torch.float32)
            crystals = None
            if self.cfg.store in ("full", "both"):
                crystals = torch.empty(N, V, D, dtype=torch.float32)

            # Fill
            for i, tid in enumerate(tids):
                ids[i] = tid
                X = vocab.embedding(tid)
                if X is None:
                    raise RuntimeError(f"Missing embedding for id={tid}")
                X = np.asarray(X, np.float32)
                if X.ndim == 2 and X.shape == (V, D):
                    if crystals is not None:
                        crystals[i] = torch.from_numpy(X)
                    # pooled then cached
                    p = X.mean(axis=0).astype(np.float32, copy=False)
                elif X.ndim == 1 and X.shape[0] == D:
                    # pooled-only bank
                    if crystals is not None:
                        raise RuntimeError("store='full' requires full crystals; vocab returned pooled.")
                    p = X
                else:
                    raise RuntimeError(f"Unexpected embedding shape for id={tid}: {X.shape}")
                pooled[i] = torch.from_numpy(np.asarray(p, np.float32))

            # Pin if requested
            if self.cfg.pinned_cpu:
                ids = ids.pin_memory()
                pooled = pooled.pin_memory()
                if crystals is not None:
                    crystals = crystals.pin_memory()

            # Save CPU master tensors
            self._cpu_ids = ids
            self._cpu_pooled = pooled.contiguous()
            self._cpu_crystals = crystals.contiguous() if crystals is not None else None

            # Precompute L1-normalized pooled for rose similarity (CPU master)
            if self.cfg.normalize == "l1":
                self._cpu_pooled_l1 = self._l1_normalize(self._cpu_pooled)
            else:
                self._cpu_pooled_l1 = None

            # Clear device cache on rebuild
            self._dev_cache.clear()

        finally:
            self.lock.release_write()

    # ---------- Utilities ----------
    @staticmethod
    def _l1_normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        s = x.abs().sum(dim=-1, keepdim=True) + eps
        return x / s

    @staticmethod
    def _device_str(device: Union[str, torch.device]) -> str:
        d = torch.device(device)
        if d.type == "cuda":
            return f"cuda:{d.index if d.index is not None else torch.cuda.current_device()}"
        if d.type == "mps":
            return "mps"
        return "cpu"

    def share_memory_(self) -> "VocabDeviceBank":
        """Share CPU tensors for multi-process dataloading on CPU."""
        self.lock.acquire_write()
        try:
            if self._cpu_ids is not None: self._cpu_ids.share_memory_()
            if self._cpu_pooled is not None: self._cpu_pooled.share_memory_()
            if self._cpu_pooled_l1 is not None: self._cpu_pooled_l1.share_memory_()
            if self._cpu_crystals is not None: self._cpu_crystals.share_memory_()
        finally:
            self.lock.release_write()
        return self

    # ---------- Migration to device (replicated) ----------
    def to(self, device: Union[str, torch.device], non_blocking: bool = True) -> None:
        """
        Materialize a device replica (pooled [+norm], and crystals if present).
        No-op if already resident.
        """
        dkey = self._device_str(device)
        self.lock.acquire_write()
        try:
            if dkey in self._dev_cache:
                return
            if self._cpu_ids is None or self._cpu_pooled is None:
                raise RuntimeError("Bank not built yet.")

            dev: Dict[str, torch.Tensor] = {}
            # Move tensors
            dev["ids"] = self._cpu_ids.to(device, non_blocking=non_blocking)
            dev["pooled"] = self._cpu_pooled.to(device, non_blocking=non_blocking)
            if self._cpu_pooled_l1 is not None:
                dev["pooled_l1"] = self._cpu_pooled_l1.to(device, non_blocking=non_blocking)
            if self._cpu_crystals is not None:
                dev["crystals"] = self._cpu_crystals.to(device, non_blocking=non_blocking)
            self._dev_cache[dkey] = dev
        finally:
            self.lock.release_write()

    # ---------- Lookups ----------
    def vocab_size(self) -> int:
        self.lock.acquire_read()
        try:
            return int(self._cpu_ids.numel()) if self._cpu_ids is not None else 0
        finally:
            self.lock.release_read()

    def id_for(self, token: str) -> Optional[int]:
        self.lock.acquire_read()
        try:
            return self._token_to_id.get(token)
        finally:
            self.lock.release_read()

    def token_for(self, tid: int) -> Optional[str]:
        self.lock.acquire_read()
        try:
            return self._id_to_token.get(tid)
        finally:
            self.lock.release_read()

    def rows_for_ids(self, ids: Iterable[int]) -> torch.Tensor:
        self.lock.acquire_read()
        try:
            rows = [self._id_to_row[i] for i in ids if i in self._id_to_row]
            return torch.as_tensor(rows, dtype=torch.long)
        finally:
            self.lock.release_read()

    # ---------- Nearest (GPU-native) ----------
    @torch.inference_mode()
    def nearest(
        self,
        query: Union[np.ndarray, torch.Tensor],
        *,
        k: int = 10,
        device: Union[str, torch.device] = "cpu",
        normalize: Optional[str] = None,
        return_tokens: bool = True,
        use_pooled: bool = True,
        chunk_size: int = 4096,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[List[List[str]]]]:
        """
        Top-k neighbors by directional L1 ("rose") similarity (default), fully on device.
        Returns (indices, scores, tokens?) where indices index into bank rows (0..N-1).
        """
        dkey = self._device_str(device)
        normalize = normalize or self.cfg.normalize
        self.to(device)  # ensure replica

        self.lock.acquire_read()
        try:
            dev = self._dev_cache[dkey]
            base = dev["pooled"] if use_pooled else dev["crystals"].mean(dim=1)  # [N, D]
            # Normalized bank
            if normalize == "l1":
                bank = dev.get("pooled_l1")
                if bank is None:
                    bank = self._l1_normalize(base)
                    dev["pooled_l1"] = bank  # cache on device
            else:
                bank = base

            # Prepare queries
            if isinstance(query, np.ndarray):
                Q = torch.as_tensor(query, dtype=bank.dtype, device=bank.device)
            else:
                Q = query.to(device=bank.device, dtype=bank.dtype, non_blocking=True)

            if Q.ndim == 1:
                Q = Q.unsqueeze(0)
            if normalize == "l1":
                Q = self._l1_normalize(Q)

            N = bank.shape[0]
            B = Q.shape[0]
            k = min(k, N)

            # Batched matmul + topk
            all_idx = []
            all_val = []
            for s in range(0, B, chunk_size):
                e = min(B, s + chunk_size)
                sims = torch.matmul(Q[s:e], bank.t())  # [b, N]
                vals, idx = torch.topk(sims, k=k, dim=-1, largest=True, sorted=True)
                all_idx.append(idx)
                all_val.append(vals)
            idx = torch.cat(all_idx, dim=0)
            vals = torch.cat(all_val, dim=0)

            if not return_tokens:
                return idx, vals, None

            # Map row indices -> tokens (CPU side)
            # We keep tokens as CPU strings; gather ids then map -> tokens.
            rows_cpu = idx.detach().cpu()
            ids_cpu = self._cpu_ids.index_select(0, rows_cpu.reshape(-1)).view_as(rows_cpu)
            toks: List[List[str]] = []
            for b in range(rows_cpu.size(0)):
                toks.append([self._id_to_token[int(i)] for i in ids_cpu[b].tolist()])
            return idx, vals, toks
        finally:
            self.lock.release_read()

    # ---------- Multi-GPU sharded nearest ----------
    @torch.inference_mode()
    def nearest_sharded(
        self,
        query: Union[np.ndarray, torch.Tensor],
        *,
        devices: List[Union[str, torch.device]],
        k: int = 10,
        normalize: Optional[str] = None,
        use_pooled: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[List[List[str]]]]:
        """
        Splits the bank evenly across provided devices, computes per-shard topk,
        and merges on CPU. Suitable when the full bank doesn't fit on one GPU.
        """
        # Prepare per-device replicas and shard ranges
        self.lock.acquire_read()
        try:
            N = self._cpu_pooled.shape[0]
        finally:
            self.lock.release_read()

        n_dev = len(devices)
        shard_size = math.ceil(N / n_dev)
        shard_ranges = [(i*shard_size, min(N, (i+1)*shard_size)) for i in range(n_dev)]

        # Compute per-shard results
        per_idx: List[torch.Tensor] = []
        per_val: List[torch.Tensor] = []
        per_tok: List[Optional[List[List[str]]]] = []

        for d, (s, e) in zip(devices, shard_ranges):
            # build a view by slicing the device replica lazily
            dkey = self._device_str(d)
            self.to(d)
            self.lock.acquire_read()
            try:
                dev = self._dev_cache[dkey]
                base = dev["pooled"] if use_pooled else dev["crystals"].mean(dim=1)
                bank_slice = base[s:e]
                if (normalize or self.cfg.normalize) == "l1":
                    bank_slice = self._l1_normalize(bank_slice)
            finally:
                self.lock.release_read()

            # project query to device
            if isinstance(query, np.ndarray):
                Q = torch.as_tensor(query, dtype=bank_slice.dtype, device=bank_slice.device)
            else:
                Q = query.to(device=bank_slice.device, dtype=bank_slice.dtype, non_blocking=True)
            if Q.ndim == 1:
                Q = Q.unsqueeze(0)
            if (normalize or self.cfg.normalize) == "l1":
                Q = self._l1_normalize(Q)

            sims = torch.matmul(Q, bank_slice.t())  # [B, e-s]
            vals, loc = torch.topk(sims, k=min(k, e-s), dim=-1, largest=True, sorted=True)
            # convert local shard indices to global row indices
            glob = loc + s
            per_idx.append(glob.detach().cpu())
            per_val.append(vals.detach().cpu())
            per_tok.append(None)  # merged later

        # Merge per-shard topk on CPU
        idx_cat = torch.cat(per_idx, dim=-1)
        val_cat = torch.cat(per_val, dim=-1)
        vals, merge = torch.topk(val_cat, k=min(k, val_cat.size(-1)), dim=-1, largest=True, sorted=True)
        rows = torch.gather(idx_cat, -1, merge)

        # map tokens on CPU
        toks: Optional[List[List[str]]] = []
        self.lock.acquire_read()
        try:
            ids_cpu = self._cpu_ids.index_select(0, rows.reshape(-1)).view_as(rows)
            for b in range(rows.size(0)):
                toks.append([self._id_to_token[int(i)] for i in ids_cpu[b].tolist()])
        finally:
            self.lock.release_read()

        return rows, vals, toks

    # ---------- Export / Import (optional) ----------
    def save_cpu_bank(self, path_prefix: str) -> None:
        """
        Save CPU bank to disk:
          - {path_prefix}.safetensors  for tensors
          - {path_prefix}.json         for string maps
        """
        from safetensors.torch import save_file as save_safetensors
        self.lock.acquire_read()
        try:
            tensors = {
                "ids": self._cpu_ids,
                "pooled": self._cpu_pooled,
            }
            if self._cpu_pooled_l1 is not None:
                tensors["pooled_l1"] = self._cpu_pooled_l1
            if self._cpu_crystals is not None:
                tensors["crystals"] = self._cpu_crystals
            save_safetensors(tensors, f"{path_prefix}.safetensors")
            meta = {
                "name": self.cfg.name,
                "dim": self.cfg.dim,
                "store": self.cfg.store,
                "normalize": self.cfg.normalize,
                "token_to_id": self._token_to_id,
                "id_to_token": self._id_to_token,
            }
            with open(f"{path_prefix}.json", "w", encoding="utf-8") as f:
                json.dump(meta, f)
        finally:
            self.lock.release_read()

# ========= Global registry (avoid duplicating banks in multi-module apps) =========
class VocabBankRegistry:
    _banks: "weakref.WeakValueDictionary[str, VocabDeviceBank]" = weakref.WeakValueDictionary()
    _lock = threading.Lock()

    @classmethod
    def key(cls, name: str, dim: int) -> str:
        return f"{name}::dim={dim}"

    @classmethod
    def get_or_register(cls, bank: VocabDeviceBank) -> VocabDeviceBank:
        k = cls.key(bank.cfg.name, bank.cfg.dim)
        with cls._lock:
            if k in cls._banks:
                return cls._banks[k]
            cls._banks[k] = bank
            return bank

    @classmethod
    def get(cls, name: str, dim: int) -> Optional[VocabDeviceBank]:
        k = cls.key(name, dim)
        with cls._lock:
            return cls._banks.get(k)
