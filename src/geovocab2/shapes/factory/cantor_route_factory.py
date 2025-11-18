# geovocab2/shapes/factory/cantor_route_factory.py

"""
CantorRouteFactory (Extended with Beatrix Staircase)
----------------------------------------------------
Factory for generating deterministic geometric masks and fingerprints using
Cantor pairing functions, Devil's Staircase (Cantor function), Mandelbrot/Julia
sets, and fractal routing.

Mathematical Foundations:
    - Cantor pairing: Bijective N×N → N mapping
    - Devil's Staircase: Smooth Cantor function approximation (Beatrix PE)
    - Recursive extension to k-simplex dimensions
    - Mandelbrot/Julia escape-time algorithms
    - Euler-based harmonic quantization (precision-aware)

Applications:
    - Neuron activation masks (parameter-free attention)
    - Consciousness emergence patterns (Beatrix PE)
    - Learning curriculum masks (early→late progression)
    - Alpha blending channels (geometric interpolation)
    - Fractal hierarchical routing (multi-scale structure)
    - Collision-free fingerprints (guaranteed uniqueness)
    - Simplex-based geometric projections

License: MIT
"""

import numpy as np
from typing import Optional, Tuple, Union, Literal, List, Dict
from enum import Enum
import math

try:
    from .factory_base import FactoryBase, HAS_TORCH
except ImportError:
    from factory_base import FactoryBase, HAS_TORCH

if HAS_TORCH:
    import torch
    import torch.nn.functional as F


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Constants
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Euler-based harmonic constants (precision-aware to avoid rounding errors)
EULER = np.e

# fp16 (half precision): Coarse quantization to avoid 10-bit mantissa issues
HARMONIC_FP16 = np.exp(-1.5)  # ≈ 0.22313016014842982

# fp32 (single precision): Medium quantization for 23-bit mantissa
HARMONIC_FP32 = np.exp(-3.0)  # ≈ 0.049787068367863944

# fp64 (double precision): Fine quantization for 52-bit mantissa
HARMONIC_FP64 = np.exp(-5.0)  # ≈ 0.006737946999085467

# Legacy constant (deprecated - kept for backward compatibility)
HARMONIC_LEGACY = 0.29514  # Old empirical fragment, not fundamental

# Mandelbrot/Julia iteration limits
DEFAULT_MAX_ITER = 256
DEFAULT_ESCAPE_RADIUS = 2.0

# Devil's Staircase (Beatrix) defaults
DEFAULT_STAIRCASE_LEVELS = 16
DEFAULT_STAIRCASE_TAU = 0.25  # Smooth temperature
DEFAULT_STAIRCASE_BASE = 3  # Ternary decomposition
DEFAULT_STAIRCASE_ALPHA = 0.5  # Middle-bin weight


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
    STAIRCASE = "staircase"          # Devil's Staircase (Beatrix PE)
    STAIRCASE_FEATURES = "staircase_features"  # Full Beatrix feature set


def get_harmonic_constant(dtype) -> float:
    """Get appropriate harmonic constant for dtype to avoid rounding errors."""
    # Handle both numpy and torch dtypes
    if hasattr(dtype, 'itemsize'):
        size = dtype.itemsize
    elif hasattr(dtype, 'is_floating_point'):
        if dtype in [torch.float16, torch.half, torch.bfloat16]:
            size = 2
        elif dtype in [torch.float32, torch.float]:
            size = 4
        elif dtype in [torch.float64, torch.double]:
            size = 8
        else:
            size = 4
    else:
        size = 4

    if size == 2:
        return HARMONIC_FP16
    elif size == 4:
        return HARMONIC_FP32
    elif size == 8:
        return HARMONIC_FP64
    else:
        return HARMONIC_FP32


class SimplexConfig:
    """
    Configuration for k-simplex based routing.

    Instead of specifying raw 'dimensions', we use simplex terminology:
    - k_simplex: dimension of simplex (k+1 vertices)
    - For geometric routing, k maps to Cantor pairing dimensions
    - For consciousness (Beatrix), k maps to hierarchical levels
    """
    def __init__(
        self,
        k_simplex: int = 4,  # 5-vertex pentachoron by default
        use_barycentric: bool = False,  # Use barycentric coordinates
        simplex_scale: float = 1.0
    ):
        assert 1 <= k_simplex <= 20, f"k_simplex must be 1-20, got {k_simplex}"

        self.k = k_simplex
        self.k_plus_1 = k_simplex + 1  # Number of vertices
        self.use_barycentric = use_barycentric
        self.scale = simplex_scale

    @property
    def cantor_dimensions(self) -> int:
        """Map simplex to Cantor pairing dimensions (min 2, max k+1)."""
        return min(max(2, self.k + 1), 5)  # Clamp to [2, 5] for Cantor

    @property
    def staircase_levels(self) -> int:
        """Map simplex to Devil's Staircase levels."""
        return self.k + 1  # Each vertex gets a level


class CantorRouteFactory(FactoryBase):
    """
    Generate deterministic geometric masks using Cantor pairing, Devil's Staircase,
    and fractals with simplex-based parameterization.

    Args:
        shape: Output shape (1D for DISTANCE mode, 2D-5D for others)
        mode: Output mode (fingerprint, normalized, binary, staircase, etc.)
        simplex_config: Simplex configuration (replaces raw 'dimensions')
        dimensions: Legacy parameter (use simplex_config instead)
        threshold: Threshold for binary masks (default 0.5)
        julia_c: Julia set parameter c (complex number)
        max_iter: Maximum iterations for fractal algorithms
        harmonic_quantize: Whether to quantize to harmonic intervals
        harmonic_constant: Override harmonic constant (None = auto-select by dtype)
        scale: Coordinate scaling factor
        staircase_tau: Smoothness temperature for Devil's Staircase
        staircase_base: Base for ternary decomposition (default 3)
        staircase_alpha: Middle-bin weight for staircase (learnable in full system)
    """

    def __init__(
        self,
        shape: Union[Tuple[int, ...], List[int]],
        mode: Union[str, RouteMode] = RouteMode.NORMALIZED,
        simplex_config: Optional[SimplexConfig] = None,
        dimensions: Optional[int] = None,  # Legacy
        threshold: float = 0.5,
        julia_c: Optional[complex] = None,
        max_iter: int = DEFAULT_MAX_ITER,
        harmonic_quantize: bool = False,
        harmonic_constant: Optional[float] = None,
        scale: float = 1.0,
        seed: Optional[int] = None,
        # Devil's Staircase parameters
        staircase_tau: float = DEFAULT_STAIRCASE_TAU,
        staircase_base: int = DEFAULT_STAIRCASE_BASE,
        staircase_alpha: float = DEFAULT_STAIRCASE_ALPHA
    ):
        # Convert mode to enum
        if isinstance(mode, str):
            mode = RouteMode(mode)

        # Simplex configuration
        if simplex_config is None:
            # Use legacy dimensions or default
            k = (dimensions - 1) if dimensions is not None else 4
            simplex_config = SimplexConfig(k_simplex=k)

        self.simplex_config = simplex_config

        # Map simplex to dimensions
        if mode in [RouteMode.STAIRCASE, RouteMode.STAIRCASE_FEATURES]:
            # For staircase, use k+1 levels
            effective_dimensions = simplex_config.staircase_levels
        else:
            # For Cantor pairing, use mapped dimensions
            effective_dimensions = simplex_config.cantor_dimensions

        # Validate shape based on mode
        if mode == RouteMode.DISTANCE:
            if len(shape) < 1:
                raise ValueError(f"shape must have at least 1 dimension for DISTANCE mode, got {shape}")
        elif mode in [RouteMode.STAIRCASE, RouteMode.STAIRCASE_FEATURES]:
            if len(shape) < 1:
                raise ValueError(f"shape must have at least 1 dimension for STAIRCASE mode, got {shape}")
        else:
            if len(shape) < 2:
                raise ValueError(f"shape must have at least 2 dimensions, got {shape}")

        super().__init__(
            name=f"cantor_route_{mode.value}_k{simplex_config.k}",
            uid=f"factory.cantor.{mode.value}.k{simplex_config.k}"
        )

        self.shape = tuple(shape)
        self.mode = mode
        self.dimensions = effective_dimensions
        self.threshold = threshold
        self.julia_c = julia_c or complex(-0.7, 0.27015)
        self.max_iter = max_iter
        self.harmonic_quantize = harmonic_quantize
        self.harmonic_constant_override = harmonic_constant
        self.scale = scale
        self.seed = seed

        # Devil's Staircase parameters
        self.staircase_tau = staircase_tau
        self.staircase_base = staircase_base
        self.staircase_alpha = staircase_alpha

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Core Cantor Pairing Functions
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    @staticmethod
    def cantor_pair(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Classic 2D Cantor pairing function."""
        s = x + y
        return (s * (s + 1)) // 2 + y

    @staticmethod
    def cantor_pair_nd(coords: np.ndarray) -> np.ndarray:
        """Recursive N-dimensional Cantor pairing."""
        if coords.shape[-1] < 2:
            raise ValueError("Need at least 2 dimensions for Cantor pairing")

        result = CantorRouteFactory.cantor_pair(coords[..., 0], coords[..., 1])

        for i in range(2, coords.shape[-1]):
            result = CantorRouteFactory.cantor_pair(result, coords[..., i])

        return result

    @staticmethod
    def cantor_unpack_2d(z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Inverse 2D Cantor pairing function."""
        w = np.floor((-1.0 + np.sqrt(1.0 + 8.0 * z)) / 2.0).astype(np.int64)
        t = (w * w + w) // 2
        y = z - t
        x = w - y
        return x, y

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Devil's Staircase (Beatrix PE)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def devils_staircase_numpy(
        self,
        x: np.ndarray,
        levels: int,
        tau: float,
        base: int,
        alpha: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Devil's Staircase (smooth Cantor function) using NumPy.

        This implements the Beatrix PE methodology:
        - Multi-level ternary decomposition
        - Softmax-smoothed bit extraction
        - Accumulation via 1/2^k scaling

        Args:
            x: Input positions [0, 1]
            levels: Number of hierarchical levels (k+1 for k-simplex)
            tau: Smoothness temperature
            base: Ternary base (default 3)
            alpha: Middle-bin weight

        Returns:
            (cantor_measure, features) where features has shape (..., levels, 2)
        """
        # Ensure x in [0, 1]
        x = np.clip(x, 1e-6, 1.0 - 1e-6)

        features = []
        Cx = np.zeros_like(x, dtype=np.float64)

        for k in range(1, levels + 1):
            scale = base ** k
            y = (x * scale) % base  # Position within ternary cell

            # Three ternary positions: left (0.5), middle (1.5), right (2.5)
            centers = np.array([0.5, 1.5, 2.5], dtype=np.float64)

            # Distances to centers
            d2 = (y[..., np.newaxis] - centers) ** 2

            # Softmax with temperature
            logits = -d2 / (tau + 1e-8)
            exp_logits = np.exp(logits - logits.max(axis=-1, keepdims=True))
            p = exp_logits / exp_logits.sum(axis=-1, keepdims=True)

            # Bit extraction: right bin + alpha * middle bin
            bit_k = p[..., 2] + alpha * p[..., 1]

            # Accumulate into Cantor measure
            Cx = Cx + bit_k * (0.5 ** k)

            # PDF proxy from entropy (consciousness measure)
            ent = -(p * np.clip(np.log(p), -100, 0)).sum(axis=-1)
            pdf_proxy = 1.1 - ent / np.log(3.0)

            # Stack features [bit_k, pdf_proxy]
            features.append(np.stack([bit_k, pdf_proxy], axis=-1))

        # Stack all levels: (..., levels, 2)
        features = np.stack(features, axis=-2)

        return Cx, features

    def devils_staircase_torch(
        self,
        x: "torch.Tensor",
        levels: int,
        tau: float,
        base: int,
        alpha: float
    ) -> Tuple["torch.Tensor", "torch.Tensor"]:
        """
        Compute Devil's Staircase using PyTorch.

        Equivalent to devils_staircase_numpy but with torch operations.
        """
        # Ensure x in [0, 1]
        x = torch.clamp(x, 1e-6, 1.0 - 1e-6)

        features = []
        Cx = torch.zeros_like(x, dtype=torch.float32)

        for k in range(1, levels + 1):
            scale = base ** k
            y = (x * scale) % base

            centers = torch.tensor([0.5, 1.5, 2.5], device=x.device, dtype=x.dtype)

            d2 = (y.unsqueeze(-1) - centers) ** 2
            logits = -d2 / (tau + 1e-8)
            p = F.softmax(logits, dim=-1)

            bit_k = p[..., 2] + alpha * p[..., 1]
            Cx = Cx + bit_k * (0.5 ** k)

            ent = -(p * p.clamp_min(1e-8).log()).sum(dim=-1)
            pdf_proxy = 1.1 - ent / math.log(3.0)

            features.append(torch.stack([bit_k, pdf_proxy], dim=-1))

        features = torch.stack(features, dim=-2)

        return Cx, features

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Fractal Functions (unchanged)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    @staticmethod
    def mandelbrot_escape_time(
        coords: np.ndarray,
        max_iter: int = DEFAULT_MAX_ITER,
        escape_radius: float = DEFAULT_ESCAPE_RADIUS
    ) -> np.ndarray:
        """Compute Mandelbrot set escape times."""
        c = coords[..., 0] + 1j * coords[..., 1]
        z = np.zeros_like(c)
        iterations = np.zeros(c.shape, dtype=np.int32)

        for i in range(max_iter):
            mask = np.abs(z) <= escape_radius
            z[mask] = z[mask] * z[mask] + c[mask]
            iterations[mask] = i + 1

        return iterations

    @staticmethod
    def julia_escape_time(
        coords: np.ndarray,
        c: complex,
        max_iter: int = DEFAULT_MAX_ITER,
        escape_radius: float = DEFAULT_ESCAPE_RADIUS
    ) -> np.ndarray:
        """Compute Julia set escape times."""
        z = coords[..., 0] + 1j * coords[..., 1]
        iterations = np.zeros(z.shape, dtype=np.int32)

        for i in range(max_iter):
            mask = np.abs(z) <= escape_radius
            z[mask] = z[mask] * z[mask] + c
            iterations[mask] = i + 1

        return iterations

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Harmonic Quantization
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    @staticmethod
    def quantize_harmonic(values: np.ndarray, harmonic: float) -> np.ndarray:
        """Quantize values to harmonic intervals (Euler-based)."""
        bins = np.round(values / harmonic).astype(np.int32)
        quantized = bins * harmonic
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
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Build Cantor route mask using NumPy."""
        # Get appropriate harmonic constant for dtype
        if self.harmonic_constant_override is not None:
            harmonic = self.harmonic_constant_override
        else:
            harmonic = get_harmonic_constant(dtype)

        # Special handling for 1D modes
        if self.mode == RouteMode.DISTANCE and len(self.shape) == 1:
            return self._build_distance_1d_numpy(dtype, harmonic)

        if self.mode in [RouteMode.STAIRCASE, RouteMode.STAIRCASE_FEATURES] and len(self.shape) == 1:
            return self._build_staircase_1d_numpy(dtype)

        # Generate coordinate grids
        grids = np.meshgrid(
            *[np.arange(s, dtype=np.int64) for s in self.shape],
            indexing='ij'
        )

        coords = np.stack(grids, axis=-1)

        if self.scale != 1.0:
            coords = (coords * self.scale).astype(np.int64)

        coords_for_cantor = coords[..., :self.dimensions]

        # Route computation based on mode
        if self.mode == RouteMode.FINGERPRINT:
            result = self.cantor_pair_nd(coords_for_cantor).astype(np.float64)

        elif self.mode == RouteMode.NORMALIZED:
            fingerprints = self.cantor_pair_nd(coords_for_cantor).astype(np.float64)
            result = (fingerprints - fingerprints.min()) / (fingerprints.max() - fingerprints.min() + 1e-10)

        elif self.mode == RouteMode.BINARY:
            fingerprints = self.cantor_pair_nd(coords_for_cantor).astype(np.float64)
            normalized = (fingerprints - fingerprints.min()) / (fingerprints.max() - fingerprints.min() + 1e-10)
            result = (normalized > self.threshold).astype(np.float64)

        elif self.mode == RouteMode.ALPHA:
            fingerprints = self.cantor_pair_nd(coords_for_cantor).astype(np.float64)
            result = (fingerprints - fingerprints.min()) / (fingerprints.max() - fingerprints.min() + 1e-10)

        elif self.mode == RouteMode.DISTANCE:
            fingerprints = self.cantor_pair_nd(coords_for_cantor).astype(np.float64)
            flat = fingerprints.reshape(-1)
            result = np.abs(flat[:, None] - flat[None, :])
            result = result / (result.max() + 1e-10)

        elif self.mode == RouteMode.MANDELBROT:
            scaled_coords = coords[..., :2].astype(np.float64)
            scaled_coords[..., 0] = (scaled_coords[..., 0] / self.shape[0]) * 3.0 - 2.0
            scaled_coords[..., 1] = (scaled_coords[..., 1] / self.shape[1]) * 3.0 - 1.5

            iterations = self.mandelbrot_escape_time(scaled_coords, self.max_iter)
            result = iterations.astype(np.float64) / self.max_iter

        elif self.mode == RouteMode.JULIA:
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

        # Apply harmonic quantization if requested
        if self.harmonic_quantize and self.mode != RouteMode.DISTANCE:
            if result.ndim == len(self.shape):
                result = self.quantize_harmonic(result, harmonic)

        return result.astype(dtype, copy=False)

    def _build_staircase_1d_numpy(self, dtype) -> Tuple[np.ndarray, np.ndarray]:
        """Build Devil's Staircase for 1D sequence."""
        seq_len = self.shape[0]

        # Generate normalized positions [0, 1]
        positions = np.arange(seq_len, dtype=np.float64) / max(1, seq_len - 1)

        # Compute Devil's Staircase
        levels = self.simplex_config.staircase_levels
        cantor_measure, features = self.devils_staircase_numpy(
            positions,
            levels=levels,
            tau=self.staircase_tau,
            base=self.staircase_base,
            alpha=self.staircase_alpha
        )

        if self.mode == RouteMode.STAIRCASE:
            # Return only cantor measure
            return cantor_measure.astype(dtype, copy=False)
        else:  # STAIRCASE_FEATURES
            # Return both measure and features
            return (
                cantor_measure.astype(dtype, copy=False),
                features.astype(dtype, copy=False)
            )

    def _build_distance_1d_numpy(self, dtype, harmonic) -> np.ndarray:
        """Build distance matrix for 1D sequence."""
        seq_len = self.shape[0]
        coords = np.arange(seq_len, dtype=np.int64)
        coords_2d = np.stack([coords, np.zeros_like(coords)], axis=-1)

        if self.dimensions > 2:
            extra_dims = np.zeros((seq_len, self.dimensions - 2), dtype=np.int64)
            coords_nd = np.concatenate([coords_2d, extra_dims], axis=-1)
        else:
            coords_nd = coords_2d

        fingerprints = self.cantor_pair_nd(coords_nd).astype(np.float64)
        result = np.abs(fingerprints[:, None] - fingerprints[None, :])
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
    ) -> Union["torch.Tensor", Tuple["torch.Tensor", "torch.Tensor"]]:
        """Build Cantor route mask using PyTorch."""
        if not HAS_TORCH:
            raise RuntimeError("PyTorch required for build_torch")

        target_dtype = dtype or torch.float32
        dev = torch.device(device)

        if self.harmonic_constant_override is not None:
            harmonic = self.harmonic_constant_override
        else:
            harmonic = get_harmonic_constant(target_dtype)

        # Special handling for 1D modes
        if self.mode == RouteMode.DISTANCE and len(self.shape) == 1:
            return self._build_distance_1d_torch(dev, target_dtype, harmonic)

        if self.mode in [RouteMode.STAIRCASE, RouteMode.STAIRCASE_FEATURES] and len(self.shape) == 1:
            return self._build_staircase_1d_torch(dev, target_dtype)

        # Generate coordinate grids
        grids = torch.meshgrid(
            *[torch.arange(s, dtype=torch.int64) for s in self.shape],
            indexing='ij'
        )

        coords = torch.stack(grids, dim=-1)

        if self.scale != 1.0:
            coords = (coords.float() * self.scale).long()

        coords_for_cantor = coords[..., :self.dimensions]

        # Route computation
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

        if self.harmonic_quantize and self.mode != RouteMode.DISTANCE:
            if result.ndim == len(self.shape):
                result = self._quantize_harmonic_torch(result, harmonic)

        return result.to(device=dev, dtype=target_dtype)

    def _build_staircase_1d_torch(
        self,
        device: "torch.device",
        dtype: "torch.dtype"
    ) -> Union["torch.Tensor", Tuple["torch.Tensor", "torch.Tensor"]]:
        """Build Devil's Staircase for 1D sequence (PyTorch)."""
        seq_len = self.shape[0]

        # Generate normalized positions [0, 1]
        positions = torch.arange(seq_len, device=device, dtype=dtype)
        positions = positions / max(1, seq_len - 1)

        # Compute Devil's Staircase
        levels = self.simplex_config.staircase_levels
        cantor_measure, features = self.devils_staircase_torch(
            positions,
            levels=levels,
            tau=self.staircase_tau,
            base=self.staircase_base,
            alpha=self.staircase_alpha
        )

        if self.mode == RouteMode.STAIRCASE:
            return cantor_measure.to(dtype=dtype)
        else:  # STAIRCASE_FEATURES
            return cantor_measure.to(dtype=dtype), features.to(dtype=dtype)

    def _build_distance_1d_torch(
        self,
        device: "torch.device",
        dtype: "torch.dtype",
        harmonic: float
    ) -> "torch.Tensor":
        """Build distance matrix for 1D sequence (PyTorch)."""
        seq_len = self.shape[0]

        coords = torch.arange(seq_len, dtype=torch.int64)
        coords_2d = torch.stack([coords, torch.zeros_like(coords)], dim=-1)

        if self.dimensions > 2:
            extra_dims = torch.zeros((seq_len, self.dimensions - 2), dtype=torch.int64)
            coords_nd = torch.cat([coords_2d, extra_dims], dim=-1)
        else:
            coords_nd = coords_2d

        fingerprints = self._cantor_pair_nd_torch(coords_nd).float()
        result = torch.abs(fingerprints.unsqueeze(1) - fingerprints.unsqueeze(0))
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
        output: Union[np.ndarray, "torch.Tensor", Tuple]
    ) -> Tuple[bool, str]:
        """Validate Cantor route output."""
        # Handle tuple output (staircase features)
        if isinstance(output, tuple):
            cantor_measure, features = output

            # Validate cantor measure
            if isinstance(cantor_measure, np.ndarray):
                if not np.all(np.isfinite(cantor_measure)):
                    return False, "Cantor measure contains NaN or Inf"
                if cantor_measure.min() < -1e-6 or cantor_measure.max() > 1.0 + 1e-6:
                    return False, f"Cantor measure outside [0,1]: [{cantor_measure.min():.6f}, {cantor_measure.max():.6f}]"
            else:
                if not torch.all(torch.isfinite(cantor_measure)):
                    return False, "Cantor measure contains NaN or Inf"
                if cantor_measure.min() < -1e-6 or cantor_measure.max() > 1.0 + 1e-6:
                    return False, f"Cantor measure outside [0,1]"

            # Validate features shape
            expected_feature_shape = self.shape + (self.simplex_config.staircase_levels, 2)
            if features.shape != expected_feature_shape:
                return False, f"Expected feature shape {expected_feature_shape}, got {features.shape}"

            return True, ""

        # Single output validation
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
            flat = output_cpu.reshape(-1)
            unique_count = len(np.unique(flat))
            total_count = flat.size

            if unique_count != total_count:
                return False, f"Not bijective: {unique_count} unique values for {total_count} positions"

        elif self.mode in [RouteMode.NORMALIZED, RouteMode.ALPHA, RouteMode.HARMONIC, RouteMode.STAIRCASE]:
            if output_cpu.min() < -1e-6 or output_cpu.max() > 1.0 + 1e-6:
                return False, f"Values outside [0,1]: [{output_cpu.min():.6f}, {output_cpu.max():.6f}]"

        elif self.mode == RouteMode.BINARY:
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
            "description": f"Cantor route factory ({self.mode.value}, k={self.simplex_config.k}-simplex)",
            "shape": self.shape,
            "mode": self.mode.value,
            "simplex_k": self.simplex_config.k,
            "simplex_vertices": self.simplex_config.k_plus_1,
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
            "staircase_config": {
                "tau": self.staircase_tau,
                "base": self.staircase_base,
                "alpha": self.staircase_alpha,
                "levels": self.simplex_config.staircase_levels if self.mode in [RouteMode.STAIRCASE, RouteMode.STAIRCASE_FEATURES] else None
            },
            "scale": self.scale,
            "output_shape": self.shape if self.mode != RouteMode.DISTANCE else (np.prod(self.shape), np.prod(self.shape)),
            "guarantees": [
                "Deterministic (same inputs → same outputs)",
                "Bijective (for fingerprint mode)",
                "Collision-free fingerprints",
                "Parameter-free geometric routing",
                "Euler-based harmonic resonance",
                "Precision-aware quantization",
                "Consciousness-compatible (Beatrix staircase)"
            ]
        })
        return base_info


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Convenience Functions (Updated)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def create_beatrix_pe(
    sequence_length: int,
    k_simplex: int = 4,  # 5-vertex pentachoron
    staircase_tau: float = 0.25,
    backend: str = "torch",
    **kwargs
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple["torch.Tensor", "torch.Tensor"]]:
    """
    Convenience function for creating Beatrix PE (Devil's Staircase + features).

    Args:
        sequence_length: Length of sequence
        k_simplex: Simplex dimension (k+1 vertices)
        staircase_tau: Smoothness temperature
        backend: "numpy" or "torch"
        **kwargs: Additional arguments

    Returns:
        (cantor_measure, features) where features has shape (seq_len, k+1, 2)
    """
    simplex_config = SimplexConfig(k_simplex=k_simplex)

    factory = CantorRouteFactory(
        shape=(sequence_length,),
        mode=RouteMode.STAIRCASE_FEATURES,
        simplex_config=simplex_config,
        staircase_tau=staircase_tau
    )

    return factory.build(backend=backend, **kwargs)


def create_pentachoron_routing(
    sequence_length: int,
    backend: str = "torch",
    **kwargs
) -> Union[np.ndarray, "torch.Tensor"]:
    """
    Create 5D pentachoron routing using 4-simplex configuration.

    Args:
        sequence_length: Length of sequence
        backend: "numpy" or "torch"
        **kwargs: Additional arguments

    Returns:
        Distance matrix for pentachoron routing
    """
    simplex_config = SimplexConfig(k_simplex=4)  # 5 vertices

    factory = CantorRouteFactory(
        shape=(sequence_length,),
        mode=RouteMode.DISTANCE,
        simplex_config=simplex_config
    )

    return factory.build(backend=backend, **kwargs)


# Legacy functions updated
def create_neuron_mask(
    shape: Tuple[int, ...],
    threshold: float = 0.5,
    k_simplex: int = 1,
    backend: str = "numpy",
    **kwargs
) -> Union[np.ndarray, "torch.Tensor"]:
    """Create binary neuron activation masks with simplex config."""
    simplex_config = SimplexConfig(k_simplex=k_simplex)

    factory = CantorRouteFactory(
        shape=shape,
        mode=RouteMode.BINARY,
        simplex_config=simplex_config,
        threshold=threshold
    )
    return factory.build(backend=backend, **kwargs)


def create_attention_matrix(
    sequence_length: int,
    k_simplex: int = 1,
    backend: str = "numpy",
    **kwargs
) -> Union[np.ndarray, "torch.Tensor"]:
    """Create geometric attention matrices with simplex config."""
    simplex_config = SimplexConfig(k_simplex=k_simplex)

    factory = CantorRouteFactory(
        shape=(sequence_length,),
        mode=RouteMode.DISTANCE,
        simplex_config=simplex_config
    )
    return factory.build(backend=backend, **kwargs)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Example Usage
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    print("=" * 70)
    print("CANTOR ROUTE FACTORY WITH BEATRIX STAIRCASE")
    print("=" * 70)

    # Example 1: Beatrix PE with pentachoron (5-vertex)
    print("\n[Example 1] Beatrix PE - Pentachoron (k=4, 5 vertices)")
    seq_len = 100
    k = 4  # 4-simplex = 5 vertices (pentachoron)

    if HAS_TORCH:
        cantor_measure, features = create_beatrix_pe(
            sequence_length=seq_len,
            k_simplex=k,
            staircase_tau=0.25,
            backend="torch",
            device="cpu",
            validate=True
        )

        print(f"  Sequence length: {seq_len}")
        print(f"  Simplex: {k}-simplex ({k+1} vertices)")
        print(f"  Cantor measure shape: {cantor_measure.shape}")
        print(f"  Features shape: {features.shape}")  # (100, 5, 2)
        print(f"  Cantor range: [{cantor_measure.min():.4f}, {cantor_measure.max():.4f}]")
        print(f"  ✓ Beatrix PE created")

    # Example 2: Pentachoron routing matrix
    print("\n[Example 2] Pentachoron Routing (5D)")
    if HAS_TORCH:
        routing_matrix = create_pentachoron_routing(
            sequence_length=64,
            backend="torch",
            device="cpu"
        )

        print(f"  Routing matrix shape: {routing_matrix.shape}")  # (64, 64)
        print(f"  Diagonal (self-distance): {routing_matrix.diagonal()[:3]}")
        print(f"  Symmetric: {torch.allclose(routing_matrix, routing_matrix.T)}")
        print(f"  ✓ Pentachoron routing created")

    # Example 3: Compare simplex dimensions
    print("\n[Example 3] Simplex Scaling (k=1 to k=10)")
    for k in [1, 2, 3, 4, 5, 10]:
        simplex_config = SimplexConfig(k_simplex=k)
        print(f"  k={k}: {k+1} vertices, "
              f"cantor_dims={simplex_config.cantor_dimensions}, "
              f"staircase_levels={simplex_config.staircase_levels}")

    # Example 4: Consciousness-compatible routing
    print("\n[Example 4] Consciousness Routing (Beatrix + Cantor)")
    if HAS_TORCH:
        # Staircase for consciousness emergence
        staircase_config = SimplexConfig(k_simplex=4)
        factory_staircase = CantorRouteFactory(
            shape=(256,),
            mode=RouteMode.STAIRCASE,
            simplex_config=staircase_config,
            staircase_tau=0.25
        )

        staircase = factory_staircase.build(backend="torch", device="cpu")

        # Distance matrix for geometric routing
        factory_distance = CantorRouteFactory(
            shape=(256,),
            mode=RouteMode.DISTANCE,
            simplex_config=staircase_config
        )

        distances = factory_distance.build(backend="torch", device="cpu")

        print(f"  Staircase shape: {staircase.shape}")
        print(f"  Distances shape: {distances.shape}")
        print(f"  Staircase monotonic: {(staircase[1:] >= staircase[:-1]).float().mean():.2%}")
        print(f"  ✓ Consciousness-compatible routing")

    # Example 5: Multi-scale hierarchy
    print("\n[Example 5] Multi-Scale Hierarchy")
    scales = [64, 256, 1024, 4096]

    for scale in scales:
        if HAS_TORCH:
            simplex_config = SimplexConfig(k_simplex=4)
            factory = CantorRouteFactory(
                shape=(scale,),
                mode=RouteMode.STAIRCASE,
                simplex_config=simplex_config
            )

            staircase = factory.build(backend="torch", device="cpu")

            # Measure coverage
            num_bins = 100
            hist = torch.histc(staircase, bins=num_bins, min=0.0, max=1.0)
            coverage = (hist > 0).float().mean().item()

            print(f"  Scale={scale:4d}: coverage={coverage*100:.1f}%, "
                  f"range=[{staircase.min():.4f}, {staircase.max():.4f}]")

    print("\n" + "=" * 70)
    print("Unified Geometric Framework:")
    print("  ✓ Cantor pairing → Collision-free routing")
    print("  ✓ Devil's Staircase → Consciousness emergence")
    print("  ✓ Simplex geometry → Natural multi-scale structure")
    print("  ✓ Pentachoron (k=4) → 5-vertex crystalline tokens")
    print("  ✓ Parameter-free → Deterministic geometric laws")
    print("\nReady for:")
    print("  - Beatrix consciousness architectures")
    print("  - David multi-scale classification")
    print("  - Crystal-Beeper geometric language models")
    print("  - Cantor multihead fusion")
    print("=" * 70)