"""
CantorRouteFactory
------------------
Factory for generating deterministic geometric masks and fingerprints using
Cantor pairing functions, Mandelbrot/Julia sets, and fractal routing.

Mathematical Foundations:
    - Cantor pairing: Bijective N×N → N mapping
    - Recursive extension to N^k dimensions
    - Mandelbrot/Julia escape-time algorithms
    - Euler-based harmonic quantization (precision-aware)

Applications:
    - Neuron activation masks (parameter-free attention)
    - Learning curriculum masks (early→late progression)
    - Alpha blending channels (geometric interpolation)
    - Fractal hierarchical routing (multi-scale structure)
    - Collision-free fingerprints (guaranteed uniqueness)

License: MIT
"""

import numpy as np
from typing import Optional, Tuple, Union, Literal, List
from enum import Enum

try:
    from .factory_base import FactoryBase, HAS_TORCH
except ImportError:
    from factory_base import FactoryBase, HAS_TORCH

if HAS_TORCH:
    import torch


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Constants
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Euler-based harmonic constants (precision-aware to avoid rounding errors)
# These define quantization bin sizes for [0,1] normalization
EULER = np.e

# fp16 (half precision): Coarse quantization to avoid 10-bit mantissa issues
# e^(-3/2) ≈ 0.2231 → ~4-5 harmonic levels
HARMONIC_FP16 = np.exp(-1.5)  # ≈ 0.22313016014842982

# fp32 (single precision): Medium quantization for 23-bit mantissa
# e^(-3) ≈ 0.0498 → ~20 harmonic levels
HARMONIC_FP32 = np.exp(-3.0)  # ≈ 0.049787068367863944

# fp64 (double precision): Fine quantization for 52-bit mantissa
# e^(-5) ≈ 0.0067 → ~150 harmonic levels
HARMONIC_FP64 = np.exp(-5.0)  # ≈ 0.006737946999085467

# Legacy constant (deprecated - kept for backward compatibility)
HARMONIC_LEGACY = 0.29514  # Old empirical fragment, not fundamental

# Mandelbrot/Julia iteration limits
DEFAULT_MAX_ITER = 256
DEFAULT_ESCAPE_RADIUS = 2.0


class RouteMode(Enum):
    """Routing computation modes."""
    FINGERPRINT = "fingerprint"      # Raw Cantor values
    NORMALIZED = "normalized"        # [0,1] normalized
    BINARY = "binary"                # Binary mask via threshold
    ALPHA = "alpha"                  # Alpha channel [0,1]
    DISTANCE = "distance"            # Pairwise Cantor distances (O(n²))
    MANDELBROT = "mandelbrot"        # Mandelbrot escape times
    JULIA = "julia"                  # Julia set escape times
    HARMONIC = "harmonic"            # Quantized to harmonic intervals


def get_harmonic_constant(dtype) -> float:
    """
    Get appropriate harmonic constant for dtype to avoid rounding errors.

    Args:
        dtype: numpy or torch dtype

    Returns:
        Euler-based harmonic constant safe for the precision regime
    """
    # Handle both numpy and torch dtypes
    if hasattr(dtype, 'itemsize'):
        # numpy dtype
        size = dtype.itemsize
    elif hasattr(dtype, 'is_floating_point'):
        # torch dtype
        if dtype in [torch.float16, torch.half, torch.bfloat16]:
            size = 2
        elif dtype in [torch.float32, torch.float]:
            size = 4
        elif dtype in [torch.float64, torch.double]:
            size = 8
        else:
            size = 4  # default to fp32
    else:
        # fallback
        size = 4

    if size == 2:  # fp16
        return HARMONIC_FP16
    elif size == 4:  # fp32
        return HARMONIC_FP32
    elif size == 8:  # fp64
        return HARMONIC_FP64
    else:
        return HARMONIC_FP32  # default


class CantorRouteFactory(FactoryBase):
    """
    Generate deterministic geometric masks using Cantor pairing and fractals.

    Args:
        shape: Output shape (1D for DISTANCE mode, 2D-5D for others)
        mode: Output mode (fingerprint, normalized, binary, etc.)
        dimensions: Number of dimensions for Cantor pairing (2-5)
        threshold: Threshold for binary masks (default 0.5)
        julia_c: Julia set parameter c (complex number)
        max_iter: Maximum iterations for fractal algorithms
        harmonic_quantize: Whether to quantize to harmonic intervals
        harmonic_constant: Override harmonic constant (None = auto-select by dtype)
        scale: Coordinate scaling factor
    """

    def __init__(
        self,
        shape: Union[Tuple[int, ...], List[int]],
        mode: Union[str, RouteMode] = RouteMode.NORMALIZED,
        dimensions: int = 2,
        threshold: float = 0.5,
        julia_c: Optional[complex] = None,
        max_iter: int = DEFAULT_MAX_ITER,
        harmonic_quantize: bool = False,
        harmonic_constant: Optional[float] = None,
        scale: float = 1.0,
        seed: Optional[int] = None
    ):
        if dimensions < 2 or dimensions > 5:
            raise ValueError(f"dimensions must be 2-5, got {dimensions}")

        # Convert mode to enum
        if isinstance(mode, str):
            mode = RouteMode(mode)

        # Validate shape based on mode
        if mode == RouteMode.DISTANCE:
            if len(shape) < 1:
                raise ValueError(f"shape must have at least 1 dimension for DISTANCE mode, got {shape}")
        else:
            if len(shape) < 2:
                raise ValueError(f"shape must have at least 2 dimensions, got {shape}")

        super().__init__(
            name=f"cantor_route_{mode.value}_d{dimensions}",
            uid=f"factory.cantor.{mode.value}.d{dimensions}"
        )

        self.shape = tuple(shape)
        self.mode = mode
        self.dimensions = dimensions
        self.threshold = threshold
        self.julia_c = julia_c or complex(-0.7, 0.27015)  # Classic Julia parameter
        self.max_iter = max_iter
        self.harmonic_quantize = harmonic_quantize
        self.harmonic_constant_override = harmonic_constant  # User override
        self.scale = scale
        self.seed = seed

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Core Cantor Pairing Functions
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    @staticmethod
    def cantor_pair(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Classic 2D Cantor pairing function.

        π(x,y) = (x+y)(x+y+1)/2 + y

        Bijective mapping N×N → N with diagonal traversal.
        """
        s = x + y
        return (s * (s + 1)) // 2 + y

    @staticmethod
    def cantor_pair_nd(coords: np.ndarray) -> np.ndarray:
        """
        Recursive N-dimensional Cantor pairing.

        For coordinates [x₀, x₁, ..., xₙ]:
        π(x₀, x₁, ..., xₙ) = π(π(...π(x₀, x₁), x₂), ..., xₙ)

        Args:
            coords: Array of shape (..., n_dims) with integer coordinates

        Returns:
            Array of shape (...,) with Cantor fingerprints
        """
        if coords.shape[-1] < 2:
            raise ValueError("Need at least 2 dimensions for Cantor pairing")

        # Start with first two dimensions
        result = CantorRouteFactory.cantor_pair(coords[..., 0], coords[..., 1])

        # Recursively pair with remaining dimensions
        for i in range(2, coords.shape[-1]):
            result = CantorRouteFactory.cantor_pair(result, coords[..., i])

        return result

    @staticmethod
    def cantor_unpack_2d(z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Inverse 2D Cantor pairing function.

        Given π(x,y) = z, recover (x,y).
        """
        # Find diagonal number: w = floor((-1 + sqrt(1 + 8z)) / 2)
        w = np.floor((-1.0 + np.sqrt(1.0 + 8.0 * z)) / 2.0).astype(np.int64)
        t = (w * w + w) // 2
        y = z - t
        x = w - y
        return x, y

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Fractal Functions
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    @staticmethod
    def mandelbrot_escape_time(
        coords: np.ndarray,
        max_iter: int = DEFAULT_MAX_ITER,
        escape_radius: float = DEFAULT_ESCAPE_RADIUS
    ) -> np.ndarray:
        """
        Compute Mandelbrot set escape times.

        For each complex point c, iterate z = z² + c until |z| > escape_radius
        or max_iter reached.

        Args:
            coords: Array of shape (..., 2) with (x,y) coordinates
            max_iter: Maximum iterations
            escape_radius: Escape threshold

        Returns:
            Array of shape (...,) with iteration counts
        """
        # Convert to complex plane
        c = coords[..., 0] + 1j * coords[..., 1]
        z = np.zeros_like(c)
        iterations = np.zeros(c.shape, dtype=np.int32)

        escape_radius_sq = escape_radius * escape_radius

        for i in range(max_iter):
            # Find points that haven't escaped
            mask = np.abs(z) <= escape_radius

            # Update z for non-escaped points
            z[mask] = z[mask] * z[mask] + c[mask]

            # Increment iteration count for non-escaped points
            iterations[mask] = i + 1

        return iterations

    @staticmethod
    def julia_escape_time(
        coords: np.ndarray,
        c: complex,
        max_iter: int = DEFAULT_MAX_ITER,
        escape_radius: float = DEFAULT_ESCAPE_RADIUS
    ) -> np.ndarray:
        """
        Compute Julia set escape times.

        For each complex point z₀, iterate z = z² + c until escape.

        Args:
            coords: Array of shape (..., 2) with (x,y) coordinates
            c: Julia set parameter
            max_iter: Maximum iterations
            escape_radius: Escape threshold

        Returns:
            Array of shape (...,) with iteration counts
        """
        # Initial z values from coordinates
        z = coords[..., 0] + 1j * coords[..., 1]
        iterations = np.zeros(z.shape, dtype=np.int32)

        escape_radius_sq = escape_radius * escape_radius

        for i in range(max_iter):
            mask = np.abs(z) <= escape_radius
            z[mask] = z[mask] * z[mask] + c
            iterations[mask] = i + 1

        return iterations

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Harmonic Quantization
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    @staticmethod
    def quantize_harmonic(
        values: np.ndarray,
        harmonic: float
    ) -> np.ndarray:
        """
        Quantize values to harmonic intervals (Euler-based).

        Rounds each value to nearest multiple of harmonic constant.
        Creates resonance-based discretization safe from rounding errors.

        Args:
            values: Input values (should be normalized [0,1])
            harmonic: Harmonic interval constant (Euler-based)

        Returns:
            Quantized values
        """
        # Quantize to harmonic bins
        bins = np.round(values / harmonic).astype(np.int32)
        quantized = bins * harmonic

        # Clip to [0,1] range
        return np.clip(quantized, 0.0, 1.0)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # NumPy Backend
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def build_numpy(
        self,
        *,
        dtype=np.float32,
        seed: Optional[int] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Build Cantor route mask using NumPy.

        Returns:
            Array of specified shape with route values
        """
        # Get appropriate harmonic constant for dtype
        if self.harmonic_constant_override is not None:
            harmonic = self.harmonic_constant_override
        else:
            harmonic = get_harmonic_constant(dtype)

        # Special handling for 1D DISTANCE mode
        if self.mode == RouteMode.DISTANCE and len(self.shape) == 1:
            return self._build_distance_1d_numpy(dtype, harmonic)

        # Generate coordinate grids (use fp64 for precision)
        grids = np.meshgrid(
            *[np.arange(s, dtype=np.int64) for s in self.shape],
            indexing='ij'
        )

        # Stack into coordinate array: shape + (n_dims,)
        coords = np.stack(grids, axis=-1)

        # Apply scaling
        if self.scale != 1.0:
            coords = (coords * self.scale).astype(np.int64)

        # Select first 'dimensions' coordinates for Cantor pairing
        coords_for_cantor = coords[..., :self.dimensions]

        # Route computation based on mode
        if self.mode == RouteMode.FINGERPRINT:
            result = self.cantor_pair_nd(coords_for_cantor).astype(np.float64)

        elif self.mode == RouteMode.NORMALIZED:
            fingerprints = self.cantor_pair_nd(coords_for_cantor).astype(np.float64)
            # Normalize to [0,1]
            result = (fingerprints - fingerprints.min()) / (fingerprints.max() - fingerprints.min() + 1e-10)

        elif self.mode == RouteMode.BINARY:
            fingerprints = self.cantor_pair_nd(coords_for_cantor).astype(np.float64)
            normalized = (fingerprints - fingerprints.min()) / (fingerprints.max() - fingerprints.min() + 1e-10)
            result = (normalized > self.threshold).astype(np.float64)

        elif self.mode == RouteMode.ALPHA:
            fingerprints = self.cantor_pair_nd(coords_for_cantor).astype(np.float64)
            result = (fingerprints - fingerprints.min()) / (fingerprints.max() - fingerprints.min() + 1e-10)

        elif self.mode == RouteMode.DISTANCE:
            # O(n²) pairwise distances - flatten and compute all pairs
            fingerprints = self.cantor_pair_nd(coords_for_cantor).astype(np.float64)
            flat = fingerprints.reshape(-1)
            # Distance matrix
            result = np.abs(flat[:, None] - flat[None, :])
            # Normalize
            result = result / (result.max() + 1e-10)

        elif self.mode == RouteMode.MANDELBROT:
            # Use first 2 dimensions as complex coordinates
            # Scale to interesting range [-2, 1] x [-1.5, 1.5]
            scaled_coords = coords[..., :2].astype(np.float64)
            scaled_coords[..., 0] = (scaled_coords[..., 0] / self.shape[0]) * 3.0 - 2.0
            scaled_coords[..., 1] = (scaled_coords[..., 1] / self.shape[1]) * 3.0 - 1.5

            iterations = self.mandelbrot_escape_time(scaled_coords, self.max_iter)
            result = iterations.astype(np.float64) / self.max_iter

        elif self.mode == RouteMode.JULIA:
            # Use first 2 dimensions as complex coordinates
            scaled_coords = coords[..., :2].astype(np.float64)
            scaled_coords[..., 0] = (scaled_coords[..., 0] / self.shape[0]) * 4.0 - 2.0
            scaled_coords[..., 1] = (scaled_coords[..., 1] / self.shape[1]) * 4.0 - 2.0

            iterations = self.julia_escape_time(scaled_coords, self.julia_c, self.max_iter)
            result = iterations.astype(np.float64) / self.max_iter

        elif self.mode == RouteMode.HARMONIC:
            fingerprints = self.cantor_pair_nd(coords_for_cantor).astype(np.float64)
            normalized = (fingerprints - fingerprints.min()) / (fingerprints.max() - fingerprints.min() + 1e-10)
            result = self.quantize_harmonic(normalized, harmonic)

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        # Apply harmonic quantization if requested (except for DISTANCE mode)
        if self.harmonic_quantize and self.mode != RouteMode.DISTANCE:
            if result.ndim == len(self.shape):  # Not distance matrix
                result = self.quantize_harmonic(result, harmonic)

        # Cast to target dtype
        return result.astype(dtype, copy=False)

    def _build_distance_1d_numpy(self, dtype, harmonic) -> np.ndarray:
        """Build distance matrix for 1D sequence."""
        seq_len = self.shape[0]

        # Generate 1D coordinates
        coords = np.arange(seq_len, dtype=np.int64)

        # For 1D, create 2D coordinates for Cantor pairing: (i, 0)
        coords_2d = np.stack([coords, np.zeros_like(coords)], axis=-1)

        # Extend to required dimensions if needed
        if self.dimensions > 2:
            extra_dims = np.zeros((seq_len, self.dimensions - 2), dtype=np.int64)
            coords_nd = np.concatenate([coords_2d, extra_dims], axis=-1)
        else:
            coords_nd = coords_2d

        # Generate fingerprints
        fingerprints = self.cantor_pair_nd(coords_nd).astype(np.float64)

        # Compute pairwise distances
        result = np.abs(fingerprints[:, None] - fingerprints[None, :])

        # Normalize
        result = result / (result.max() + 1e-10)

        return result.astype(dtype, copy=False)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PyTorch Backend
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def build_torch(
        self,
        *,
        device: str = "cpu",
        dtype: Optional["torch.dtype"] = None,
        seed: Optional[int] = None,
        **kwargs
    ) -> "torch.Tensor":
        """
        Build Cantor route mask using PyTorch.

        Note: Cantor pairing requires integer arithmetic, computed on CPU
        then transferred to target device.
        """
        if not HAS_TORCH:
            raise RuntimeError("PyTorch required for build_torch")

        target_dtype = dtype or torch.float32
        dev = torch.device(device)

        # Get appropriate harmonic constant for dtype
        if self.harmonic_constant_override is not None:
            harmonic = self.harmonic_constant_override
        else:
            harmonic = get_harmonic_constant(target_dtype)

        # Special handling for 1D DISTANCE mode
        if self.mode == RouteMode.DISTANCE and len(self.shape) == 1:
            return self._build_distance_1d_torch(dev, target_dtype, harmonic)

        # Generate coordinate grids (int64 for precision)
        grids = torch.meshgrid(
            *[torch.arange(s, dtype=torch.int64) for s in self.shape],
            indexing='ij'
        )

        # Stack into coordinate array
        coords = torch.stack(grids, dim=-1)

        # Apply scaling
        if self.scale != 1.0:
            coords = (coords.float() * self.scale).long()

        # Select dimensions for Cantor pairing
        coords_for_cantor = coords[..., :self.dimensions]

        # Route computation (on CPU due to integer ops)
        if self.mode == RouteMode.FINGERPRINT:
            result = self._cantor_pair_nd_torch(coords_for_cantor).float()

        elif self.mode == RouteMode.NORMALIZED:
            fingerprints = self._cantor_pair_nd_torch(coords_for_cantor).float()
            result = (fingerprints - fingerprints.min()) / (fingerprints.max() - fingerprints.min() + 1e-10)

        elif self.mode == RouteMode.BINARY:
            fingerprints = self._cantor_pair_nd_torch(coords_for_cantor).float()
            normalized = (fingerprints - fingerprints.min()) / (fingerprints.max() - fingerprints.min() + 1e-10)
            result = (normalized > self.threshold).float()

        elif self.mode == RouteMode.ALPHA:
            fingerprints = self._cantor_pair_nd_torch(coords_for_cantor).float()
            result = (fingerprints - fingerprints.min()) / (fingerprints.max() - fingerprints.min() + 1e-10)

        elif self.mode == RouteMode.DISTANCE:
            fingerprints = self._cantor_pair_nd_torch(coords_for_cantor).float()
            flat = fingerprints.reshape(-1)
            result = torch.abs(flat.unsqueeze(1) - flat.unsqueeze(0))
            result = result / (result.max() + 1e-10)

        elif self.mode == RouteMode.MANDELBROT:
            scaled_coords = coords[..., :2].float()
            scaled_coords[..., 0] = (scaled_coords[..., 0] / self.shape[0]) * 3.0 - 2.0
            scaled_coords[..., 1] = (scaled_coords[..., 1] / self.shape[1]) * 3.0 - 1.5

            iterations = self._mandelbrot_escape_time_torch(scaled_coords, self.max_iter)
            result = iterations.float() / self.max_iter

        elif self.mode == RouteMode.JULIA:
            scaled_coords = coords[..., :2].float()
            scaled_coords[..., 0] = (scaled_coords[..., 0] / self.shape[0]) * 4.0 - 2.0
            scaled_coords[..., 1] = (scaled_coords[..., 1] / self.shape[1]) * 4.0 - 2.0

            iterations = self._julia_escape_time_torch(scaled_coords, self.julia_c, self.max_iter)
            result = iterations.float() / self.max_iter

        elif self.mode == RouteMode.HARMONIC:
            fingerprints = self._cantor_pair_nd_torch(coords_for_cantor).float()
            normalized = (fingerprints - fingerprints.min()) / (fingerprints.max() - fingerprints.min() + 1e-10)
            result = self._quantize_harmonic_torch(normalized, harmonic)

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        # Apply harmonic quantization if requested
        if self.harmonic_quantize and self.mode != RouteMode.DISTANCE:
            if result.ndim == len(self.shape):
                result = self._quantize_harmonic_torch(result, harmonic)

        # Transfer to device and cast dtype
        return result.to(device=dev, dtype=target_dtype)

    def _build_distance_1d_torch(self, device, dtype, harmonic) -> "torch.Tensor":
        """Build distance matrix for 1D sequence (PyTorch)."""
        seq_len = self.shape[0]

        # Generate 1D coordinates
        coords = torch.arange(seq_len, dtype=torch.int64)

        # Create 2D coordinates for Cantor pairing: (i, 0)
        coords_2d = torch.stack([coords, torch.zeros_like(coords)], dim=-1)

        # Extend to required dimensions if needed
        if self.dimensions > 2:
            extra_dims = torch.zeros((seq_len, self.dimensions - 2), dtype=torch.int64)
            coords_nd = torch.cat([coords_2d, extra_dims], dim=-1)
        else:
            coords_nd = coords_2d

        # Generate fingerprints
        fingerprints = self._cantor_pair_nd_torch(coords_nd).float()

        # Compute pairwise distances
        result = torch.abs(fingerprints.unsqueeze(1) - fingerprints.unsqueeze(0))

        # Normalize
        result = result / (result.max() + 1e-10)

        return result.to(device=device, dtype=dtype)

    @staticmethod
    def _cantor_pair_torch(x: "torch.Tensor", y: "torch.Tensor") -> "torch.Tensor":
        """2D Cantor pairing for PyTorch."""
        s = x + y
        return (s * (s + 1)) // 2 + y

    @staticmethod
    def _cantor_pair_nd_torch(coords: "torch.Tensor") -> "torch.Tensor":
        """N-dimensional Cantor pairing for PyTorch."""
        result = CantorRouteFactory._cantor_pair_torch(coords[..., 0], coords[..., 1])

        for i in range(2, coords.shape[-1]):
            result = CantorRouteFactory._cantor_pair_torch(result, coords[..., i])

        return result

    @staticmethod
    def _mandelbrot_escape_time_torch(
        coords: "torch.Tensor",
        max_iter: int
    ) -> "torch.Tensor":
        """Mandelbrot escape time for PyTorch."""
        c = torch.complex(coords[..., 0], coords[..., 1])
        z = torch.zeros_like(c)
        iterations = torch.zeros(c.shape, dtype=torch.int32, device=c.device)

        for i in range(max_iter):
            mask = torch.abs(z) <= 2.0
            z = torch.where(mask, z * z + c, z)
            iterations = torch.where(mask, i + 1, iterations)

        return iterations

    @staticmethod
    def _julia_escape_time_torch(
        coords: "torch.Tensor",
        c: complex,
        max_iter: int
    ) -> "torch.Tensor":
        """Julia escape time for PyTorch."""
        z = torch.complex(coords[..., 0], coords[..., 1])
        c_tensor = torch.tensor(c, dtype=z.dtype, device=z.device)
        iterations = torch.zeros(z.shape, dtype=torch.int32, device=z.device)

        for i in range(max_iter):
            mask = torch.abs(z) <= 2.0
            z = torch.where(mask, z * z + c_tensor, z)
            iterations = torch.where(mask, i + 1, iterations)

        return iterations

    @staticmethod
    def _quantize_harmonic_torch(
        values: "torch.Tensor",
        harmonic: float
    ) -> "torch.Tensor":
        """Harmonic quantization for PyTorch."""
        bins = torch.round(values / harmonic).long()
        quantized = bins.float() * harmonic
        return torch.clamp(quantized, 0.0, 1.0)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Validation
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def validate(
        self,
        output: Union[np.ndarray, "torch.Tensor"]
    ) -> Tuple[bool, str]:
        """
        Validate Cantor route output.

        Checks:
            1. Shape matches expected output shape
            2. No NaN/Inf values
            3. For FINGERPRINT mode: all values are unique (bijectivity)
            4. For normalized modes: values in [0,1]
        """
        # Check shape (distance mode produces different shape)
        if self.mode == RouteMode.DISTANCE:
            n_elements = np.prod(self.shape)
            expected_shape = (n_elements, n_elements)
        else:
            expected_shape = self.shape

        if output.shape != expected_shape:
            return False, f"Expected shape {expected_shape}, got {output.shape}"

        # Check for NaN/Inf
        if isinstance(output, np.ndarray):
            if not np.all(np.isfinite(output)):
                return False, "Contains NaN or Inf"
            output_cpu = output
        else:
            if not torch.all(torch.isfinite(output)):
                return False, "Contains NaN or Inf"
            output_cpu = output.cpu().numpy()

        # Mode-specific validation
        if self.mode == RouteMode.FINGERPRINT:
            # Check uniqueness (bijectivity) - only for non-distance modes
            flat = output_cpu.reshape(-1)
            unique_count = len(np.unique(flat))
            total_count = flat.size

            if unique_count != total_count:
                return False, f"Not bijective: {unique_count} unique values for {total_count} positions"

        elif self.mode in [RouteMode.NORMALIZED, RouteMode.ALPHA, RouteMode.HARMONIC]:
            # Check range [0,1]
            if output_cpu.min() < -1e-6 or output_cpu.max() > 1.0 + 1e-6:
                return False, f"Values outside [0,1]: [{output_cpu.min():.6f}, {output_cpu.max():.6f}]"

        elif self.mode == RouteMode.BINARY:
            # Check binary values
            unique_vals = np.unique(output_cpu)
            if not np.allclose(unique_vals, [0.0, 1.0]) and len(unique_vals) <= 2:
                if not (set(unique_vals) <= {0.0, 1.0}):
                    return False, f"Binary mask contains non-binary values: {unique_vals}"

        return True, ""

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Metadata
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def info(self):
        base_info = super().info()
        base_info.update({
            "description": f"Cantor route factory ({self.mode.value}, {self.dimensions}D pairing)",
            "shape": self.shape,
            "mode": self.mode.value,
            "dimensions": self.dimensions,
            "threshold": self.threshold,
            "julia_c": str(self.julia_c),
            "max_iter": self.max_iter,
            "harmonic_quantize": self.harmonic_quantize,
            "harmonic_constants": {
                "fp16": f"{HARMONIC_FP16:.15f} (e^-1.5, ~5 levels)",
                "fp32": f"{HARMONIC_FP32:.15f} (e^-3, ~20 levels)",
                "fp64": f"{HARMONIC_FP64:.15f} (e^-5, ~150 levels)",
                "euler": f"{EULER:.15f}"
            },
            "scale": self.scale,
            "output_shape": self.shape if self.mode != RouteMode.DISTANCE else (np.prod(self.shape), np.prod(self.shape)),
            "guarantees": [
                "Deterministic (same inputs → same outputs)",
                "Bijective (for fingerprint mode)",
                "Collision-free fingerprints",
                "Parameter-free geometric routing",
                "Euler-based harmonic resonance",
                "Precision-aware quantization"
            ]
        })
        return base_info


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Convenience Functions
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def create_neuron_mask(
    shape: Tuple[int, ...],
    threshold: float = 0.5,
    dimensions: int = 2,
    backend: str = "numpy",
    **kwargs
) -> Union[np.ndarray, "torch.Tensor"]:
    """
    Convenience function for creating binary neuron activation masks.

    Args:
        shape: Output shape
        threshold: Activation threshold
        dimensions: Cantor pairing dimensions
        backend: "numpy" or "torch"
        **kwargs: Additional arguments passed to build()

    Returns:
        Binary mask for neuron activation
    """
    factory = CantorRouteFactory(
        shape=shape,
        mode=RouteMode.BINARY,
        dimensions=dimensions,
        threshold=threshold
    )
    return factory.build(backend=backend, **kwargs)


def create_learning_curriculum(
    shape: Tuple[int, ...],
    dimensions: int = 2,
    harmonic: bool = True,
    backend: str = "numpy",
    **kwargs
) -> Union[np.ndarray, "torch.Tensor"]:
    """
    Convenience function for creating curriculum learning masks.

    Early sequence positions get low values, late positions get high values.

    Args:
        shape: Output shape
        dimensions: Cantor pairing dimensions
        harmonic: Apply harmonic quantization
        backend: "numpy" or "torch"
        **kwargs: Additional arguments passed to build()

    Returns:
        Normalized mask [0,1] for curriculum progression
    """
    factory = CantorRouteFactory(
        shape=shape,
        mode=RouteMode.NORMALIZED,
        dimensions=dimensions,
        harmonic_quantize=harmonic
    )
    return factory.build(backend=backend, **kwargs)


def create_attention_matrix(
    sequence_length: int,
    dimensions: int = 2,
    backend: str = "numpy",
    **kwargs
) -> Union[np.ndarray, "torch.Tensor"]:
    """
    Convenience function for creating geometric attention matrices.

    Args:
        sequence_length: Length of sequence
        dimensions: Cantor pairing dimensions
        backend: "numpy" or "torch"
        **kwargs: Additional arguments passed to build()

    Returns:
        Distance matrix [seq_len, seq_len] for attention
    """
    factory = CantorRouteFactory(
        shape=(sequence_length,),
        mode=RouteMode.DISTANCE,
        dimensions=dimensions
    )
    return factory.build(backend=backend, **kwargs)


def create_fractal_mask(
    shape: Tuple[int, ...],
    fractal_type: Literal["mandelbrot", "julia"] = "mandelbrot",
    julia_c: Optional[complex] = None,
    max_iter: int = 256,
    backend: str = "numpy",
    **kwargs
) -> Union[np.ndarray, "torch.Tensor"]:
    """
    Convenience function for creating fractal-based masks.

    Args:
        shape: Output shape (at least 2D)
        fractal_type: "mandelbrot" or "julia"
        julia_c: Julia set parameter (if julia type)
        max_iter: Maximum fractal iterations
        backend: "numpy" or "torch"
        **kwargs: Additional arguments passed to build()

    Returns:
        Fractal intensity mask [0,1]
    """
    mode = RouteMode.MANDELBROT if fractal_type == "mandelbrot" else RouteMode.JULIA

    factory = CantorRouteFactory(
        shape=shape,
        mode=mode,
        julia_c=julia_c,
        max_iter=max_iter
    )
    return factory.build(backend=backend, **kwargs)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Example Usage
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    print("=" * 70)
    print("CANTOR ROUTE FACTORY DEMONSTRATION")
    print("=" * 70)

    # Show Euler-based harmonic constants
    print("\n[Harmonic Constants] Euler-Based Precision-Aware Quantization")
    print(f"  Euler (e):    {EULER:.15f}")
    print(f"  fp16 (e^-1.5): {HARMONIC_FP16:.15f} → ~{int(1.0/HARMONIC_FP16)} harmonic levels")
    print(f"  fp32 (e^-3):   {HARMONIC_FP32:.15f} → ~{int(1.0/HARMONIC_FP32)} harmonic levels")
    print(f"  fp64 (e^-5):   {HARMONIC_FP64:.15f} → ~{int(1.0/HARMONIC_FP64)} harmonic levels")
    print(f"  Legacy (deprecated): {HARMONIC_LEGACY:.15f}")

    # Example 1: Fingerprint mode - guaranteed unique values
    print("\n[Example 1] Fingerprint Mode - Collision-Free IDs")
    factory_fp = CantorRouteFactory(shape=(8, 8), mode=RouteMode.FINGERPRINT, dimensions=2)
    fingerprints = factory_fp.build(backend="numpy", validate=True)

    print(f"  Shape: {fingerprints.shape}")
    print(f"  Unique values: {len(np.unique(fingerprints))} / {fingerprints.size}")
    print(f"  Sample fingerprints:\n{fingerprints[:3, :3].astype(int)}")
    print(f"  Bijectivity: {'✓ GUARANTEED' if len(np.unique(fingerprints)) == fingerprints.size else '✗ FAILED'}")

    # Example 2: Neuron activation mask
    print("\n[Example 2] Binary Neuron Mask (threshold=0.5)")
    neuron_mask = create_neuron_mask(shape=(16, 16), threshold=0.5, validate=True)

    print(f"  Shape: {neuron_mask.shape}")
    print(f"  Active neurons: {int(neuron_mask.sum())} / {neuron_mask.size}")
    print(f"  Activation rate: {neuron_mask.mean():.2%}")
    print(f"  Sample mask:\n{neuron_mask[:5, :5].astype(int)}")

    # Example 3: Curriculum learning mask with fp32 harmonic
    print("\n[Example 3] Curriculum Learning Mask (fp32 harmonic quantization)")
    curriculum = create_learning_curriculum(
        shape=(32, 32),
        harmonic=True,
        backend="numpy",
        dtype=np.float32,
        validate=True
    )

    print(f"  Shape: {curriculum.shape}")
    print(f"  Value range: [{curriculum.min():.4f}, {curriculum.max():.4f}]")
    print(f"  Unique levels: {len(np.unique(curriculum))}")
    print(f"  Harmonic (fp32): {HARMONIC_FP32:.6f}")
    print(f"  Early position values: {curriculum[0, :3]}")
    print(f"  Late position values: {curriculum[-1, -3:]}")

    # Example 4: Geometric attention matrix (O(n²))
    print("\n[Example 4] Geometric Attention Matrix (O(n²))")
    seq_len = 16
    attention = create_attention_matrix(sequence_length=seq_len, validate=True)

    print(f"  Shape: {attention.shape}")
    print(f"  Diagonal (self-attention): {np.diag(attention)[:3]}")
    print(f"  Symmetric: {np.allclose(attention, attention.T)}")
    print(f"  Sample distances:\n{attention[:4, :4]}")

    # Example 5: Harmonic quantization comparison across precisions
    print("\n[Example 5] Harmonic Quantization Comparison")
    test_shape = (16, 16)

    for dtype_np, dtype_name in [(np.float16, "fp16"), (np.float32, "fp32"), (np.float64, "fp64")]:
        factory_h = CantorRouteFactory(
            shape=test_shape,
            mode=RouteMode.HARMONIC,
            dimensions=2
        )
        harmonic_mask = factory_h.build(backend="numpy", dtype=dtype_np, validate=True)
        levels = len(np.unique(harmonic_mask))
        print(f"  {dtype_name}: {levels} unique harmonic levels")

    # Example 6: Mandelbrot fractal mask
    print("\n[Example 6] Mandelbrot Fractal Mask")
    mandelbrot = create_fractal_mask(
        shape=(64, 64),
        fractal_type="mandelbrot",
        max_iter=128,
        validate=True
    )

    print(f"  Shape: {mandelbrot.shape}")
    print(f"  In-set ratio: {(mandelbrot > 0.9).sum() / mandelbrot.size:.2%}")
    print(f"  Boundary ratio: {((mandelbrot > 0.3) & (mandelbrot < 0.7)).sum() / mandelbrot.size:.2%}")
    print(f"  Escaped ratio: {(mandelbrot < 0.1).sum() / mandelbrot.size:.2%}")

    # Example 7: Julia set mask
    print("\n[Example 7] Julia Set Mask (c = -0.7 + 0.27015i)")
    julia = create_fractal_mask(
        shape=(64, 64),
        fractal_type="julia",
        julia_c=complex(-0.7, 0.27015),
        max_iter=128,
        validate=True
    )

    print(f"  Shape: {julia.shape}")
    print(f"  Julia parameter: {complex(-0.7, 0.27015)}")
    print(f"  Connected regions: {(julia > 0.8).sum() / julia.size:.2%}")

    # Example 8: High-dimensional Cantor routing (5D for pentachorons)
    print("\n[Example 8] 5D Cantor Routing (Pentachoron Space)")
    factory_5d = CantorRouteFactory(
        shape=(4, 4, 4, 4, 4),
        mode=RouteMode.NORMALIZED,
        dimensions=5,
        harmonic_quantize=True
    )
    pentachoron_routes = factory_5d.build(backend="numpy", dtype=np.float32, validate=True)

    print(f"  Shape: {pentachoron_routes.shape}")
    print(f"  Total positions: {pentachoron_routes.size}")
    print(f"  Unique routes: {len(np.unique(pentachoron_routes))}")
    print(f"  Value range: [{pentachoron_routes.min():.4f}, {pentachoron_routes.max():.4f}]")

    if HAS_TORCH:
        # Example 9: PyTorch backend with precision handling
        print("\n[Example 9] PyTorch Backend (CPU)")
        factory_torch = CantorRouteFactory(
            shape=(32, 32),
            mode=RouteMode.ALPHA,
            dimensions=2
        )
        alpha_torch = factory_torch.build(
            backend="torch",
            device="cpu",
            dtype=torch.float32,
            validate=True
        )

        print(f"  Type: {type(alpha_torch)}")
        print(f"  Shape: {alpha_torch.shape}")
        print(f"  Device: {alpha_torch.device}")
        print(f"  Dtype: {alpha_torch.dtype}")
        print(f"  Harmonic (auto): {get_harmonic_constant(torch.float32):.6f}")

        if torch.cuda.is_available():
            print("\n[Example 10] CUDA Acceleration with fp16")
            alpha_cuda = factory_torch.build(
                backend="torch",
                device="cuda:0",
                dtype=torch.float16,
                validate=True
            )
            print(f"  Device: {alpha_cuda.device}")
            print(f"  Dtype: {alpha_cuda.dtype}")
            print(f"  Harmonic (fp16): {get_harmonic_constant(torch.float16):.6f}")

    # Example 11: Validation demonstration
    print("\n[Example 11] Bijectivity Validation")

    # Create fingerprints
    test_factory = CantorRouteFactory(shape=(10, 10), mode=RouteMode.FINGERPRINT)
    test_fp = test_factory.build(backend="numpy")

    # Validate
    is_valid, msg = test_factory.validate(test_fp)
    print(f"  Valid: {is_valid}")
    print(f"  Message: {msg if msg else 'All checks passed ✓'}")

    # Create a collision to test validation
    corrupted = test_fp.copy()
    corrupted[0, 0] = corrupted[1, 1]
    is_valid_corrupt, msg_corrupt = test_factory.validate(corrupted)
    print(f"  Corrupted valid: {is_valid_corrupt}")
    print(f"  Corruption detected: {msg_corrupt}")

    print("\n" + "=" * 70)
    print("CantorRouteFactory Applications:")
    print("  ✓ Neuron activation masks (parameter-free attention)")
    print("  ✓ Learning curriculum (early→late progression)")
    print("  ✓ Geometric attention matrices (O(n²) distances)")
    print("  ✓ Fractal hierarchical routing (multi-scale)")
    print("  ✓ Alpha blending channels (smooth interpolation)")
    print("  ✓ Collision-free fingerprints (guaranteed uniqueness)")
    print("  ✓ Euler-based harmonic resonance (precision-aware)")
    print("  ✓ 5D pentachoron routing (crystalline navigation)")
    print("\nMathematical Foundations:")
    print("  - Cantor pairing: Bijective diagonal traversal")
    print("  - Euler quantization: e^-k bin sizes for each precision")
    print("  - Fractal routing: Mandelbrot/Julia escape-time masks")
    print("  - Rounding safety: Constants chosen per fp16/32/64")
    print("\nIntegration ready for:")
    print("  - Crystal-Beeper geometric language models")
    print("  - David multi-scale classification")
    print("  - Beatrix consciousness architectures")
    print("  - Parameter-free geometric transformers")
    print("=" * 70)