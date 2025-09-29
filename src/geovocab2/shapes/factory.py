# shape/factory.py
"""
ShapeFactory
------------

Stateless, functional shape factory.

For prototyping and rapid experimentation, registering simple shape builders is a highly efficient way to generate
shapes on either NumPy or PyTorch backends.

This provides a simple interface to allow larger structural synthesis systems to request shapes without needing to
know the details of how to build them. Simply register builders and call make().

The yielded shapes are best formatted in yield factory consumption formats rather than individual shapes.

The first prototype has a single shape synthesizer, so beware of deep lambda chains and slowdowns.

- register(name, numpy_builder, torch_builder=None, *, aliases=None, doc=None)
    Register a shape generator (numpy_builder returns a np.ndarray).
    Optionally provide a torch_builder that constructs the shape directly on-device.

- make(name, *args, backend="numpy", device="cpu", dtype=None, **kwargs)
    Pure functional call. Returns either a numpy.ndarray (backend="numpy")
    or a torch.Tensor (backend="torch").

Design decisions:
- No persistent caches. Each make() yields a fresh object.
- If `torch_builder` exists we call it (preferred: avoids host copies).
- If `torch_builder` is missing we generate numpy and move with pin_memory() + non_blocking=True.
- Minimal dtype coercion: accepts numpy dtypes or torch dtypes for torch backend.

It's currently expensive, so don't call make() in inner loops.
Expectation is to call once per shape and produce a yielding structure. This is a prototype so don't expect it to be done yet.

It needs logistics tests, benchmarking, and time profiling to be considered stable and efficient.

Lambda processes are inherently slow and can dive to deep depths, so be cautious when tinkering.

Inner structural synthesis will require a caching substructure and multiple conjoined forms of larger shapes for optimization.
This can be further sped up with profile tests and compilation, but the diminishing returns will hit fast.

License: Apache License 2.0
"""

from typing import Callable, Optional, Dict, Tuple, Any
import numpy as np

# Optional torch import
try:
    import torch
    HAS_TORCH = True
except Exception:
    torch = None  # type: ignore
    HAS_TORCH = False


NumpyBuilder = Callable[..., np.ndarray]
TorchBuilder = Optional[Callable[..., "torch.Tensor"]]
RegistryEntry = Tuple[NumpyBuilder, TorchBuilder]


class ShapeFactory:
    def __init__(self) -> None:
        # registry maps canonical name -> (numpy_builder, optional_torch_builder)
        self._registry: Dict[str, RegistryEntry] = {}
        # metadata store: doc, aliases
        self._meta: Dict[str, Dict[str, Any]] = {}

    # --------------------------
    # Registration / Introspection
    # --------------------------
    def register(
        self,
        name: str,
        numpy_builder: NumpyBuilder,
        torch_builder: TorchBuilder = None,
        *,
        aliases: Optional[list[str]] = None,
        doc: Optional[str] = None,
    ) -> None:
        """Register a shape generator and optional torch direct-builder.

        The numpy_builder MUST accept dtype kwarg (default np.float32) unless documented otherwise.
        The torch_builder SHOULD accept device and dtype kwargs (device like "cuda:0").
        """
        key = name.lower()
        self._registry[key] = (numpy_builder, torch_builder)
        self._meta[key] = {"doc": doc or "", "aliases": aliases or []}
        if aliases:
            for a in aliases:
                self._registry[a.lower()] = (numpy_builder, torch_builder)
                self._meta[a.lower()] = {"doc": f"alias for {name}", "aliases": [name]}

    def available(self) -> list[str]:
        return list(self._registry.keys())

    def metadata(self, name: str) -> Dict[str, Any]:
        return self._meta.get(name.lower(), {})

    # --------------------------
    # Utilities
    # --------------------------
    def _recommend_torch_dtype(self, device_idx: int = 0):
        """Heuristic: prefer bfloat16 on modern Ampere/Hopper, else float16."""
        if not HAS_TORCH or not torch.cuda.is_available():
            return torch.float32 if HAS_TORCH else np.float32  # fallback type objects
        props = torch.cuda.get_device_properties(device_idx)
        # Ampere/Hopper (compute capability >= 8) -> prefer bfloat16 for stability
        try:
            cc = getattr(props, "major", 0)
            if cc >= 8 and hasattr(torch, "bfloat16"):
                return torch.bfloat16
        except Exception:
            pass
        return torch.float16

    def _ensure_torch_dtype(self, dtype) -> "torch.dtype":
        """Map numpy dtype or torch.dtype to torch.dtype. If None, recommend."""
        if dtype is None:
            return self._recommend_torch_dtype(0)
        # if user passed a numpy dtype
        if isinstance(dtype, np.dtype) or dtype in (np.float32, np.float64, np.float16):
            mapping = {
                np.dtype(np.float32): torch.float32,
                np.dtype(np.float64): torch.float64,
                np.dtype(np.float16): torch.float16,
            }
            return mapping.get(np.dtype(dtype), torch.float32)
        # assume it's already a torch dtype
        return dtype  # type: ignore

    def _pinned_transfer_to_device(self, arr: np.ndarray, device: str, dtype: Optional["torch.dtype"] = None):
        """Fast path: numpy -> pinned CPU tensor -> async to device"""
        assert HAS_TORCH, "PyTorch is required for pinned transfer"
        # Ensure contiguous float32 (or input float)
        arrc = np.ascontiguousarray(arr.astype(np.float32))
        cpu_t = torch.from_numpy(arrc).pin_memory()
        dev = torch.device(device)
        target_dtype = dtype or torch.float32
        return cpu_t.to(device=dev, non_blocking=True, dtype=target_dtype)

    # --------------------------
    # Core functional factory
    # --------------------------
    def make(
        self,
        name: str,
        *args,
        backend: str = "numpy",
        device: str = "cpu",
        dtype: Optional[Any] = None,
        **kwargs,
    ):
        """
        Build shape in requested backend.

        - backend: "numpy" or "torch"
        - device: "cpu" or "cuda:0"-style (used only for torch backend)
        - dtype: numpy dtype for numpy backend (e.g., np.float32), or torch.dtype for torch backend
        """
        key = name.lower()
        if key not in self._registry:
            raise ValueError(f"Shape '{name}' not registered. available={self.available()}")

        numpy_builder, torch_builder = self._registry[key]

        # --- Numpy backend: pure functional, returns numpy.ndarray ---
        if backend.lower() == "numpy":
            ndtype = np.float32 if dtype is None else dtype
            return numpy_builder(*args, dtype=ndtype, **kwargs)

        # --- Torch backend ---
        if backend.lower() == "torch":
            if not HAS_TORCH:
                raise RuntimeError("PyTorch must be installed for backend='torch'")

            # Prefer direct torch_builder when provided
            if torch_builder is not None:
                # Allow torch_builder to accept device and dtype kwargs.
                # If dtype is a numpy dtype, convert it for torch_builder convenience.
                torch_dtype = None
                if dtype is not None:
                    # map numpy dtype -> torch dtype if needed
                    if isinstance(dtype, np.dtype) or dtype in (np.float32, np.float64, np.float16):
                        torch_dtype = self._ensure_torch_dtype(dtype)
                    else:
                        torch_dtype = dtype  # user likely passed torch dtype
                # Call torch_builder; expecting it to return a torch.Tensor
                try:
                    return torch_builder(*args, device=device, dtype=torch_dtype, **kwargs)
                except TypeError:
                    # fallback: builder might not accept device/dtype; call plain and then move
                    t = torch_builder(*args, **kwargs)
                    return t.to(device=device, dtype=torch_dtype or t.dtype)

            # No torch_builder -> fallback: numpy builder + pinned async transfer
            arr = numpy_builder(*args, dtype=np.float32, **kwargs)
            return self._pinned_transfer_to_device(arr, device=device, dtype=self._ensure_torch_dtype(dtype))

        raise ValueError(f"Unsupported backend '{backend}' (allowed: 'numpy', 'torch')")


# --------------------------
# Example builders and registration
# --------------------------
def _regular_simplex_numpy(dim: int, dtype=np.float32) -> np.ndarray:
    E = np.eye(dim, dtype=np.float64)
    centroid = E.mean(axis=0, keepdims=True)
    S = E - centroid
    S = S / np.linalg.norm(S[0] - S[1])
    return S.astype(dtype)


def _regular_simplex_torch(dim: int, device: str = "cuda:0", dtype: Optional["torch.dtype"] = None):
    # build directly on-device (no host copy). dtype should be torch.dtype.
    if not HAS_TORCH:
        raise RuntimeError("PyTorch not available")
    dev = torch.device(device)
    tdtype = dtype or torch.float32
    E = torch.eye(dim, device=dev, dtype=tdtype)
    centroid = E.mean(dim=0, keepdim=True)
    S = E - centroid
    return S / (S[0] - S[1]).norm()

# --------------------------
# Generic tensor fallback (NumPy + Torch)
# --------------------------
def _tensor_numpy(*, dim, batch: Optional[int] = None, seed: Optional[int] = None, dtype=np.float32) -> np.ndarray:
    """
    Generic NumPy tensor:
      - dim: int or tuple[int,...]
      - batch: optional int to prefix the shape
      - seed: if None -> zeros; else -> standard normal with that seed
    """
    print("Using numpy tensor builder")
    # resolve shape
    if isinstance(dim, int):
        shape = (batch, dim) if batch is not None else (dim,)
    else:
        shape = (batch, *tuple(dim)) if batch is not None else tuple(dim)

    if seed is None:
        return np.zeros(shape, dtype=dtype)

    rng = np.random.default_rng(int(seed))
    return rng.standard_normal(shape).astype(dtype, copy=False)


def _tensor_torch(
    *,
    dim,
    batch: Optional[int] = None,
    seed: Optional[int] = None,
    device: str = "cpu",
    dtype: Optional["torch.dtype"] = None,
):
    """
    Generic Torch tensor on the requested device/dtype:
      - dim: int or tuple[int,...]
      - batch: optional int to prefix the shape
      - seed: if None -> zeros; else -> standard normal with that seed (per-call generator)
    """
    if not HAS_TORCH:
        raise RuntimeError("PyTorch not available")

    # resolve shape
    print("Using torch tensor builder")
    if isinstance(dim, int):
        shape = (batch, dim) if batch is not None else (dim,)
    else:
        shape = (batch, *tuple(dim)) if batch is not None else tuple(dim)

    dev = torch.device(device)
    tdtype = dtype or torch.float32

    if seed is None:
        return torch.zeros(shape, device=dev, dtype=tdtype)

    # use a local generator to avoid contaminating global RNG state
    gen = torch.Generator()  # CPU generator is fine for CUDA sampling
    gen.manual_seed(int(seed))
    return torch.randn(shape, device=dev, dtype=tdtype, generator=gen)


if __name__ == "__main__":
    # create a default factory singleton and register common shapes
    default_factory = ShapeFactory()
    default_factory.register("simplex", _regular_simplex_numpy, _regular_simplex_torch, aliases=["regular_simplex"])
    default_factory.register(
        name="pentachoron",
        numpy_builder=lambda dtype=np.float64: _regular_simplex_numpy(5, dtype=dtype),
        torch_builder=lambda device="cuda:0", dtype=None: _regular_simplex_torch(5, device=device, dtype=dtype),
        aliases=["4-simplex"],
    )


    # Register the generic fallback (aliases optional)
    default_factory.register(
        "tensor",
        _tensor_numpy,
        # torch builder accepts device/dtype; factory will pass them
        torch_builder=lambda *, dim, batch=None, seed=None, device="cpu", dtype=None: _tensor_torch(
            dim=dim, batch=batch, seed=seed, device=device, dtype=dtype
        ),
        aliases=["generic", "any"],
        doc="Generic tensor creator: (dim[, batch], seed)-> zeros or randn on NumPy/Torch",
    )
    # keep your factory and registrations as-is

    # Build NumPy and Torch versions
    tensor_np = default_factory.make("tensor", dim=(3, 4), batch=2,
                                     seed=42, backend="numpy", dtype=np.float64)
    tensor_torch = default_factory.make("tensor", dim=(3, 4), batch=2,
                                        seed=42, backend="torch", device="cpu", dtype=torch.float64)

    print("tensor numpy:", type(tensor_np), tensor_np.shape)
    print("tensor torch:", type(tensor_torch), tensor_torch.shape)

    # ---- Option A: compare identical data -> cosine should be 1.0
    t1 = torch.from_numpy(tensor_np).to(dtype=torch.float64)  # same values as NumPy
    t2 = torch.from_numpy(tensor_np).to(dtype=torch.float64)  # identical copy

    cos = torch.nn.CosineSimilarity(dim=-1)
    sim_identical = cos(t1.reshape(-1, 12), t2.reshape(-1, 12))
    print("cosine (identical data) should be 1.0:", sim_identical)

    # ---- Option B: compare NumPy RNG vs Torch RNG -> not identical, cosine != 1.0
    # Convert NumPy to torch for the op (don't convert torch -> numpy)
    t_np = torch.from_numpy(tensor_np).to(dtype=torch.float64)
    t_th = tensor_torch  # already torch.float64

    sim_mismatch = cos(t_np.reshape(-1, 12), t_th.reshape(-1, 12))
    print("cosine (numpy RNG vs torch RNG) ~!= 1.0:", sim_mismatch)
