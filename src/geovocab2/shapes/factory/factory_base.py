"""
FactoryBase
-----------
Abstract base class for stateless tensor/object factories.

Provides the structural spine for manufacturing various mathematical objects
(simplices, tensors, geometric structures) with backend flexibility and
metadata tracking.

Design Philosophy:
    Factories CREATE objects; Formulas COMPUTE properties.

    - Factories are stateless constructors (produce fresh objects each call)
    - Support both NumPy and PyTorch backends
    - Metadata for introspection and cataloging
    - Validation hooks for output checking

    Similar to FormulaBase but focused on construction rather than computation.

License: MIT
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union
import numpy as np

# Optional torch
try:
    import torch

    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False


class FactoryBase(ABC):
    """
    Abstract base for tensor/object factories.

    Subclasses implement:
        - build_numpy(): NumPy backend construction
        - build_torch(): PyTorch backend construction (optional)
        - validate(): Optional output validation
        - info(): Factory metadata

    Attributes:
        name: Human-readable factory name
        uid: Unique identifier for registry systems
    """

    def __init__(self, name: str, uid: str):
        """
        Initialize factory with identifying metadata.

        Args:
            name: Human-readable name (e.g., "simplex_factory")
            uid: Unique identifier (e.g., "factory.simplex.regular")
        """
        self.name = name
        self.uid = uid

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Abstract construction methods (subclasses MUST implement)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    @abstractmethod
    def build_numpy(self, *args, dtype=np.float32, **kwargs) -> np.ndarray:
        """
        Build object using NumPy backend.

        Args:
            *args: Factory-specific positional arguments
            dtype: NumPy dtype for output
            **kwargs: Factory-specific keyword arguments

        Returns:
            NumPy array representing the constructed object
        """
        pass

    def build_torch(
            self,
            *args,
            device: str = "cpu",
            dtype: Optional["torch.dtype"] = None,
            **kwargs
    ) -> "torch.Tensor":
        """
        Build object using PyTorch backend (optional override).

        Default implementation: build NumPy then transfer.
        Subclasses can override for direct on-device construction.

        Args:
            *args: Factory-specific positional arguments
            device: Target device ("cpu", "cuda:0", etc.)
            dtype: PyTorch dtype (if None, inferred from device)
            **kwargs: Factory-specific keyword arguments

        Returns:
            PyTorch tensor on specified device
        """
        if not HAS_TORCH:
            raise RuntimeError(f"PyTorch required for build_torch() in {self.name}")

        # Default: build numpy and transfer with pinned memory
        arr = self.build_numpy(*args, dtype=np.float32, **kwargs)
        arr = np.ascontiguousarray(arr)

        # Pinned transfer for async copy
        cpu_tensor = torch.from_numpy(arr).pin_memory()

        target_dtype = dtype or self._infer_torch_dtype(device)
        return cpu_tensor.to(device=device, dtype=target_dtype, non_blocking=True)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Unified build interface
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def build(
            self,
            *args,
            backend: str = "numpy",
            device: str = "cpu",
            dtype: Optional[Any] = None,
            validate: bool = True,
            **kwargs
    ) -> Union[np.ndarray, "torch.Tensor"]:
        """
        Unified build interface with backend selection.

        Args:
            *args: Factory-specific positional arguments
            backend: "numpy" or "torch"
            device: Device for torch backend
            dtype: Output dtype (backend-specific)
            validate: Run validation checks on output
            **kwargs: Factory-specific keyword arguments

        Returns:
            Constructed object (ndarray or Tensor)
        """
        if backend.lower() == "numpy":
            output = self.build_numpy(*args, dtype=dtype or np.float32, **kwargs)

        elif backend.lower() == "torch":
            if not HAS_TORCH:
                raise RuntimeError(f"PyTorch required for backend='torch' in {self.name}")
            output = self.build_torch(*args, device=device, dtype=dtype, **kwargs)

        else:
            raise ValueError(f"Invalid backend '{backend}' (allowed: 'numpy', 'torch')")

        # Optional validation
        if validate:
            is_valid, error_msg = self.validate(output)
            if not is_valid:
                raise ValueError(f"Factory {self.name} validation failed: {error_msg}")

        return output

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Optional validation and metadata (subclasses MAY override)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def validate(self, output: Union[np.ndarray, "torch.Tensor"]) -> tuple[bool, str]:
        """
        Validate factory output.

        Args:
            output: Constructed object from build()

        Returns:
            (is_valid, error_message)
        """
        # Default: accept anything
        return True, ""

    def info(self) -> Dict[str, Any]:
        """
        Factory metadata for introspection.

        Returns:
            Dictionary with factory information
        """
        return {
            "name": self.name,
            "uid": self.uid,
            "description": "No description provided",
            "backend_support": {
                "numpy": True,
                "torch": HAS_TORCH
            }
        }

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Utility methods
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _infer_torch_dtype(self, device: str) -> "torch.dtype":
        """Infer appropriate torch dtype based on device."""
        if not HAS_TORCH:
            raise RuntimeError("PyTorch not available")

        if "cuda" in device:
            # Try to use compute capability to choose dtype
            try:
                device_idx = int(device.split(":")[-1]) if ":" in device else 0
                props = torch.cuda.get_device_properties(device_idx)
                cc = props.major

                # Ampere/Hopper (CC >= 8.0) prefer bfloat16 for stability
                if cc >= 8 and hasattr(torch, "bfloat16"):
                    return torch.bfloat16
            except Exception:
                pass

            return torch.float16  # Default for CUDA

        return torch.float32  # Default for CPU

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # String representations
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', uid='{self.uid}')"

    def __str__(self) -> str:
        return f"Factory[{self.name}] ({self.__class__.__name__})"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Example: Generic tensor factory
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TensorFactory(FactoryBase):
    """Generic tensor factory with flexible shapes and initialization."""

    def __init__(self, init_mode: str = "zeros"):
        """
        Args:
            init_mode: "zeros", "randn", "ones"
        """
        super().__init__("tensor_factory", "factory.tensor.generic")
        self.init_mode = init_mode

    def build_numpy(
            self,
            shape: Union[int, tuple],
            *,
            dtype=np.float32,
            seed: Optional[int] = None
    ) -> np.ndarray:
        """Build generic tensor with NumPy."""
        shape = (shape,) if isinstance(shape, int) else tuple(shape)

        if self.init_mode == "zeros":
            return np.zeros(shape, dtype=dtype)

        elif self.init_mode == "ones":
            return np.ones(shape, dtype=dtype)

        elif self.init_mode == "randn":
            rng = np.random.default_rng(seed)
            return rng.standard_normal(shape).astype(dtype, copy=False)

        else:
            raise ValueError(f"Invalid init_mode: {self.init_mode}")

    def build_torch(
            self,
            shape: Union[int, tuple],
            *,
            device: str = "cpu",
            dtype: Optional["torch.dtype"] = None,
            seed: Optional[int] = None
    ) -> "torch.Tensor":
        """Build generic tensor directly on device (no host copy)."""
        if not HAS_TORCH:
            raise RuntimeError("PyTorch required")

        shape = (shape,) if isinstance(shape, int) else tuple(shape)
        target_dtype = dtype or self._infer_torch_dtype(device)
        dev = torch.device(device)

        if self.init_mode == "zeros":
            return torch.zeros(shape, device=dev, dtype=target_dtype)

        elif self.init_mode == "ones":
            return torch.ones(shape, device=dev, dtype=target_dtype)

        elif self.init_mode == "randn":
            if seed is not None:
                gen = torch.Generator(device="cpu")  # Generator is CPU-only
                gen.manual_seed(seed)
                return torch.randn(shape, generator=gen, dtype=target_dtype).to(dev)
            return torch.randn(shape, device=dev, dtype=target_dtype)

        else:
            raise ValueError(f"Invalid init_mode: {self.init_mode}")

    def validate(self, output: Union[np.ndarray, "torch.Tensor"]) -> tuple[bool, str]:
        """Validate tensor output."""
        # Check for NaN/Inf
        if isinstance(output, np.ndarray):
            if not np.all(np.isfinite(output)):
                return False, "Output contains NaN or Inf"
        else:  # torch.Tensor
            if not torch.all(torch.isfinite(output)):
                return False, "Output contains NaN or Inf"

        return True, ""

    def info(self) -> Dict[str, Any]:
        base_info = super().info()
        base_info.update({
            "description": f"Generic tensor factory with {self.init_mode} initialization",
            "init_mode": self.init_mode,
            "supported_modes": ["zeros", "ones", "randn"]
        })
        return base_info


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Example usage and testing
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    print("=" * 70)
    print("FACTORYBASE DEMONSTRATION")
    print("=" * 70)

    # Create factories
    zeros_factory = TensorFactory(init_mode="zeros")
    randn_factory = TensorFactory(init_mode="randn")

    print("\n[Example 1] NumPy backend - zeros")
    arr_zeros = zeros_factory.build((3, 4), backend="numpy")
    print(f"  Type: {type(arr_zeros)}")
    print(f"  Shape: {arr_zeros.shape}")
    print(f"  Sample values: {arr_zeros[0, :3]}")

    print("\n[Example 2] NumPy backend - randn with seed")
    arr_randn = randn_factory.build((3, 4), backend="numpy", seed=42)
    print(f"  Type: {type(arr_randn)}")
    print(f"  Shape: {arr_randn.shape}")
    print(f"  Sample values: {arr_randn[0, :3]}")

    if HAS_TORCH:
        print("\n[Example 3] PyTorch backend - CPU")
        tensor_cpu = randn_factory.build(
            (3, 4),
            backend="torch",
            device="cpu",
            seed=42
        )
        print(f"  Type: {type(tensor_cpu)}")
        print(f"  Shape: {tensor_cpu.shape}")
        print(f"  Device: {tensor_cpu.device}")
        print(f"  Sample values: {tensor_cpu[0, :3]}")

        if torch.cuda.is_available():
            print("\n[Example 4] PyTorch backend - CUDA")
            tensor_cuda = randn_factory.build(
                (3, 4),
                backend="torch",
                device="cuda:0"
            )
            print(f"  Type: {type(tensor_cuda)}")
            print(f"  Shape: {tensor_cuda.shape}")
            print(f"  Device: {tensor_cuda.device}")
            print(f"  Dtype: {tensor_cuda.dtype}")

    print("\n[Example 5] Factory metadata")
    info = randn_factory.info()
    for key, value in info.items():
        print(f"  {key}: {value}")

    print("\n[Example 6] Validation")
    # This will pass
    valid_tensor = zeros_factory.build((2, 2), backend="numpy", validate=True)
    print(f"  Valid tensor created: {valid_tensor.shape}")

    print("\n" + "=" * 70)
    print("FactoryBase ready for extension")
    print("=" * 70)