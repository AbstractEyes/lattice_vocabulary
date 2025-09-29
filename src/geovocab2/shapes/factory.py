# shape/factory.py
"""
ShapeFactory
------------

Stateless, functional shape factory.

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


# create a default factory singleton and register common shapes
default_factory = ShapeFactory()
default_factory.register("simplex", _regular_simplex_numpy, _regular_simplex_torch, aliases=["regular_simplex"])
default_factory.register(
    "pentachoron",
    lambda dtype=np.float32: _regular_simplex_numpy(5, dtype=dtype),
    lambda device="cuda:0", dtype=None: _regular_simplex_torch(5, device=device, dtype=dtype),
    aliases=["4-simplex"],
)
