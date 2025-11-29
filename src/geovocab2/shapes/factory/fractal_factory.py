"""
FractalFactory
--------------
Factory for generating Mandelbrot/Julia fractal training data.

This is NOT a route factory like the cantor route factory, but rather a data generator.

Generates fractal fingerprints for exploring expert specialization:
    - Julia set images from named regions
    - Orbit sequences (z_n → z_{n+1}) for transformer input
    - Escape grids for patch-based analysis
    - Multi-scale renders for zoom-invariant features

Named Regions:
    - cardioid: Main cardioid bulb (bounded, period-1)
    - period2: Period-2 bulb (bounded, period-2 cycle)
    - seahorse: Seahorse valley (intricate spirals)
    - elephant: Elephant valley (trunk-like filaments)
    - antenna: Main antenna (-2 to -1.75 on real axis)
    - spiral: Mini spirals near seahorse
    - cusp: Cardioid cusp (most intricate boundary)

Output Modes:
    - image: 2D escape-time image [H, W]
    - orbit: Sequence of (re, im, |z|, arg(z)) [T, 4]
    - orbit_complex: Raw complex orbit [T, 2] (re, im only)
    - escape_grid: Patchified escape times [P, P]
    - multi_scale: Stack of zoom levels [S, H, W]

License: MIT
"""

import numpy as np
import cmath
import math
from typing import Optional, Tuple, Union, Literal, Dict, Any, List
from enum import Enum
from dataclasses import dataclass

try:
    from .factory_base import FactoryBase, HAS_TORCH
except ImportError:
    from factory_base import FactoryBase, HAS_TORCH

if HAS_TORCH:
    import torch

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Constants and Enums
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DEFAULT_MAX_ITER = 256
DEFAULT_ESCAPE_RADIUS = 2.0


class FractalMode(Enum):
    """Output modes for fractal generation."""
    IMAGE = "image"  # 2D escape-time image
    ORBIT = "orbit"  # Full orbit features [T, 4]
    ORBIT_COMPLEX = "orbit_complex"  # Raw orbit [T, 2]
    ESCAPE_GRID = "escape_grid"  # Patchified escape times
    MULTI_SCALE = "multi_scale"  # Multi-zoom stack
    JULIA_IMAGE = "julia_image"  # Julia set image (fixed c)
    MANDELBROT_IMAGE = "mandelbrot_image"  # Mandelbrot region


class FractalType(Enum):
    """Type of fractal to generate."""
    MANDELBROT = "mandelbrot"
    JULIA = "julia"


@dataclass
class FractalRegion:
    """Named region in the Mandelbrot set."""
    name: str
    center: complex
    radius: float
    description: str
    expected_behavior: str  # "bounded", "slow_escape", "fast_escape", "chaotic"

    def sample_c(self, rng: np.random.Generator) -> complex:
        """Sample a c parameter from this region."""
        # Sample uniformly in disk
        r = self.radius * np.sqrt(rng.uniform(0, 1))
        theta = rng.uniform(0, 2 * np.pi)
        offset = r * (np.cos(theta) + 1j * np.sin(theta))
        return self.center + offset

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for metadata."""
        return {
            "name": self.name,
            "center_re": self.center.real,
            "center_im": self.center.imag,
            "radius": self.radius,
            "description": self.description,
            "expected_behavior": self.expected_behavior
        }


# Named regions in the Mandelbrot set
FRACTAL_REGIONS: Dict[str, FractalRegion] = {
    "cardioid": FractalRegion(
        name="cardioid",
        center=complex(-0.25, 0.0),
        radius=0.4,
        description="Main cardioid bulb (period-1 interior)",
        expected_behavior="bounded"
    ),
    "period2": FractalRegion(
        name="period2",
        center=complex(-1.0, 0.0),
        radius=0.2,
        description="Period-2 bulb (2-cycle interior)",
        expected_behavior="bounded"
    ),
    "period3": FractalRegion(
        name="period3",
        center=complex(-0.125, 0.744),
        radius=0.05,
        description="Period-3 bulb (3-cycle, smaller)",
        expected_behavior="bounded"
    ),
    "seahorse": FractalRegion(
        name="seahorse",
        center=complex(-0.75, 0.1),
        radius=0.05,
        description="Seahorse valley (intricate spirals)",
        expected_behavior="chaotic"
    ),
    "elephant": FractalRegion(
        name="elephant",
        center=complex(0.3, 0.5),
        radius=0.1,
        description="Elephant valley (trunk-like filaments)",
        expected_behavior="slow_escape"
    ),
    "antenna": FractalRegion(
        name="antenna",
        center=complex(-1.75, 0.0),
        radius=0.1,
        description="Main antenna (real axis spike)",
        expected_behavior="fast_escape"
    ),
    "spiral": FractalRegion(
        name="spiral",
        center=complex(-0.761574, 0.0847596),
        radius=0.02,
        description="Mini Mandelbrot spiral region",
        expected_behavior="chaotic"
    ),
    "cusp": FractalRegion(
        name="cusp",
        center=complex(0.25, 0.0),
        radius=0.05,
        description="Cardioid cusp (boundary region)",
        expected_behavior="slow_escape"
    ),
    "mini_brot": FractalRegion(
        name="mini_brot",
        center=complex(-1.769, 0.0),
        radius=0.015,
        description="Mini Mandelbrot on antenna",
        expected_behavior="bounded"
    ),
    "dendrite": FractalRegion(
        name="dendrite",
        center=complex(-0.1, 0.65),
        radius=0.03,
        description="Dendrite filament region",
        expected_behavior="chaotic"
    ),
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FractalFactory
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class FractalFactory(FactoryBase):
    """
    Generate fractal training data for expert fingerprinting experiments.

    Args:
        mode: Output mode (image, orbit, escape_grid, multi_scale)
        region: Named region to sample from (or None for full set)
        fractal_type: mandelbrot or julia
        size: Image size (H, W) or sequence length for orbits
        max_iter: Maximum iteration count
        escape_radius: Escape radius for iteration
        c_override: Fixed c parameter (overrides region sampling)
        z0: Initial z value for orbits (default 0)
        num_scales: Number of zoom levels for multi_scale mode
        zoom_factor: Zoom multiplier between scales
        seed: Random seed for reproducibility
    """

    def __init__(
            self,
            mode: Union[str, FractalMode] = FractalMode.ORBIT,
            region: Optional[str] = None,
            fractal_type: Union[str, FractalType] = FractalType.JULIA,
            size: Union[int, Tuple[int, int]] = 64,
            max_iter: int = DEFAULT_MAX_ITER,
            escape_radius: float = DEFAULT_ESCAPE_RADIUS,
            c_override: Optional[complex] = None,
            z0: complex = 0j,
            num_scales: int = 3,
            zoom_factor: float = 2.0,
            seed: Optional[int] = None
    ):
        # Convert enums
        if isinstance(mode, str):
            mode = FractalMode(mode)
        if isinstance(fractal_type, str):
            fractal_type = FractalType(fractal_type)

        # Validate region
        if region is not None and region not in FRACTAL_REGIONS:
            valid = list(FRACTAL_REGIONS.keys())
            raise ValueError(f"Unknown region '{region}'. Valid: {valid}")

        # Parse size
        if isinstance(size, int):
            if mode in [FractalMode.IMAGE, FractalMode.JULIA_IMAGE,
                        FractalMode.MANDELBROT_IMAGE, FractalMode.ESCAPE_GRID,
                        FractalMode.MULTI_SCALE]:
                size = (size, size)
            else:
                size = (size,)  # Sequence length for orbits

        super().__init__(
            name=f"fractal_{mode.value}_{region or 'full'}",
            uid=f"factory.fractal.{mode.value}.{region or 'full'}"
        )

        self.mode = mode
        self.region = region
        self.region_config = FRACTAL_REGIONS.get(region) if region else None
        self.fractal_type = fractal_type
        self.size = size
        self.max_iter = max_iter
        self.escape_radius = escape_radius
        self.c_override = c_override
        self.z0 = z0
        self.num_scales = num_scales
        self.zoom_factor = zoom_factor
        self.seed = seed

        # Track last generated c for metadata
        self._last_c: Optional[complex] = None
        self._last_region: Optional[str] = None

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Core Fractal Computation
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _sample_c(self, rng: np.random.Generator) -> Tuple[complex, str]:
        """Sample c parameter from region or uniformly."""
        if self.c_override is not None:
            return self.c_override, "override"

        if self.region_config is not None:
            return self.region_config.sample_c(rng), self.region

        # Random region
        region_name = rng.choice(list(FRACTAL_REGIONS.keys()))
        region = FRACTAL_REGIONS[region_name]
        return region.sample_c(rng), region_name

    def _compute_orbit(
            self,
            c: complex,
            z0: complex,
            max_steps: int
    ) -> Tuple[np.ndarray, int]:
        """
        Compute orbit sequence z_{n+1} = z_n² + c.

        Returns:
            orbit: Array of [re, im, |z|, arg(z)] for each step
            escape_step: Step at which orbit escaped (or max_steps if bounded)
        """
        orbit = []
        z = z0
        escape_step = max_steps

        for step in range(max_steps):
            # Record state: [real, imag, magnitude, phase]
            mag = abs(z)
            phase = cmath.phase(z)
            orbit.append([z.real, z.imag, mag, phase])

            # Check escape
            if mag > self.escape_radius:
                escape_step = step
                break

            # Iterate
            z = z * z + c

        # Pad to max_steps if escaped early
        while len(orbit) < max_steps:
            # Repeat last state (or zeros for numerical stability)
            orbit.append([0.0, 0.0, 0.0, 0.0])

        return np.array(orbit, dtype=np.float64), escape_step

    def _compute_orbit_complex(
            self,
            c: complex,
            z0: complex,
            max_steps: int
    ) -> Tuple[np.ndarray, int]:
        """
        Compute orbit as [re, im] pairs only.

        Returns:
            orbit: Array of [re, im] for each step
            escape_step: Step at which orbit escaped
        """
        orbit = []
        z = z0
        escape_step = max_steps

        for step in range(max_steps):
            orbit.append([z.real, z.imag])

            if abs(z) > self.escape_radius:
                escape_step = step
                break

            z = z * z + c

        # Pad
        while len(orbit) < max_steps:
            orbit.append([0.0, 0.0])

        return np.array(orbit, dtype=np.float64), escape_step

    def _render_julia(
            self,
            c: complex,
            size: Tuple[int, int],
            center: complex = 0j,
            radius: float = 2.0
    ) -> np.ndarray:
        """Render Julia set escape times for fixed c."""
        h, w = size

        # Create coordinate grid
        x = np.linspace(center.real - radius, center.real + radius, w)
        y = np.linspace(center.imag - radius, center.imag + radius, h)
        X, Y = np.meshgrid(x, y)
        Z = X + 1j * Y

        # Compute escape times
        iterations = np.zeros(Z.shape, dtype=np.int32)
        escaped = np.zeros(Z.shape, dtype=bool)

        for i in range(self.max_iter):
            mask = ~escaped
            Z[mask] = Z[mask] ** 2 + c

            new_escaped = np.abs(Z) > self.escape_radius
            just_escaped = new_escaped & ~escaped
            iterations[just_escaped] = i + 1
            escaped = new_escaped

        # Bounded points get max_iter
        iterations[~escaped] = self.max_iter

        return iterations

    def _render_mandelbrot(
            self,
            size: Tuple[int, int],
            center: complex = complex(-0.5, 0),
            radius: float = 1.5
    ) -> np.ndarray:
        """Render Mandelbrot set escape times."""
        h, w = size

        # Coordinate grid
        x = np.linspace(center.real - radius, center.real + radius, w)
        y = np.linspace(center.imag - radius, center.imag + radius, h)
        X, Y = np.meshgrid(x, y)
        C = X + 1j * Y

        Z = np.zeros_like(C)
        iterations = np.zeros(C.shape, dtype=np.int32)
        escaped = np.zeros(C.shape, dtype=bool)

        for i in range(self.max_iter):
            mask = ~escaped
            Z[mask] = Z[mask] ** 2 + C[mask]

            new_escaped = np.abs(Z) > self.escape_radius
            just_escaped = new_escaped & ~escaped
            iterations[just_escaped] = i + 1
            escaped = new_escaped

        iterations[~escaped] = self.max_iter

        return iterations

    def _render_multi_scale(
            self,
            c: complex,
            base_size: Tuple[int, int]
    ) -> np.ndarray:
        """Render Julia set at multiple zoom levels."""
        scales = []
        radius = 2.0

        for scale_idx in range(self.num_scales):
            img = self._render_julia(c, base_size, center=0j, radius=radius)
            scales.append(img)
            radius /= self.zoom_factor

        return np.stack(scales, axis=0)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # NumPy Backend
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def build_numpy(
            self,
            *,
            dtype=np.float32,
            seed: Optional[int] = None,
            return_metadata: bool = False,
            **kwargs
    ) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, Any]]]:
        """
        Build fractal data using NumPy.

        Args:
            dtype: Output dtype
            seed: Random seed
            return_metadata: If True, return (data, metadata) tuple

        Returns:
            Fractal data array, optionally with metadata dict
        """
        seed = seed if seed is not None else self.seed
        rng = np.random.default_rng(seed)

        # Sample c parameter
        c, region_name = self._sample_c(rng)
        self._last_c = c
        self._last_region = region_name

        # Generate based on mode
        if self.mode == FractalMode.ORBIT:
            seq_len = self.size[0]
            orbit, escape_step = self._compute_orbit(c, self.z0, seq_len)
            result = orbit.astype(dtype)

        elif self.mode == FractalMode.ORBIT_COMPLEX:
            seq_len = self.size[0]
            orbit, escape_step = self._compute_orbit_complex(c, self.z0, seq_len)
            result = orbit.astype(dtype)

        elif self.mode in [FractalMode.IMAGE, FractalMode.JULIA_IMAGE]:
            img = self._render_julia(c, self.size)
            result = (img.astype(np.float64) / self.max_iter).astype(dtype)
            escape_step = None

        elif self.mode == FractalMode.MANDELBROT_IMAGE:
            if self.region_config:
                img = self._render_mandelbrot(
                    self.size,
                    center=self.region_config.center,
                    radius=self.region_config.radius * 2
                )
            else:
                img = self._render_mandelbrot(self.size)
            result = (img.astype(np.float64) / self.max_iter).astype(dtype)
            c = None
            escape_step = None

        elif self.mode == FractalMode.ESCAPE_GRID:
            # Patchified escape times
            img = self._render_julia(c, self.size)
            result = (img.astype(np.float64) / self.max_iter).astype(dtype)
            escape_step = None

        elif self.mode == FractalMode.MULTI_SCALE:
            imgs = self._render_multi_scale(c, self.size)
            result = (imgs.astype(np.float64) / self.max_iter).astype(dtype)
            escape_step = None

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        if return_metadata:
            metadata = self._build_metadata(c, region_name, escape_step)
            return result, metadata

        return result

    def _build_metadata(
            self,
            c: Optional[complex],
            region_name: str,
            escape_step: Optional[int]
    ) -> Dict[str, Any]:
        """Build metadata dictionary for generated fractal."""
        meta = {
            "mode": self.mode.value,
            "fractal_type": self.fractal_type.value,
            "region": region_name,
            "max_iter": self.max_iter,
            "escape_radius": self.escape_radius,
        }

        if c is not None:
            meta["c_real"] = c.real
            meta["c_imag"] = c.imag
            meta["c_magnitude"] = abs(c)
            meta["c_phase"] = cmath.phase(c)

        if escape_step is not None:
            meta["escape_step"] = escape_step
            meta["is_bounded"] = escape_step >= self.max_iter - 1

        if self.region_config:
            meta["region_config"] = self.region_config.to_dict()

        return meta

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Custom Build (handles return_metadata)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def build(
            self,
            *args,
            backend: str = "numpy",
            device: str = "cpu",
            dtype=None,
            validate: bool = True,
            return_metadata: bool = False,
            **kwargs
    ) -> Union[np.ndarray, "torch.Tensor", Tuple[Any, Dict[str, Any]]]:
        """
        Build fractal with optional metadata return.

        Overrides base class to handle return_metadata properly.
        """
        # Pass return_metadata to the backend-specific method
        kwargs['return_metadata'] = return_metadata

        if backend.lower() == "numpy":
            output = self.build_numpy(*args, dtype=dtype or np.float32, **kwargs)
        elif backend.lower() == "torch":
            if not HAS_TORCH:
                raise RuntimeError("PyTorch required for backend='torch'")
            output = self.build_torch(*args, device=device, dtype=dtype, **kwargs)
        else:
            raise ValueError(f"Invalid backend '{backend}'")

        # Handle validation - extract data if tuple
        if validate:
            data_to_validate = output[0] if return_metadata else output
            is_valid, error_msg = self.validate(data_to_validate)
            if not is_valid:
                raise ValueError(f"Factory {self.name} validation failed: {error_msg}")

        return output

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PyTorch Backend
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def build_torch(
            self,
            *,
            device: str = "cpu",
            dtype: Optional["torch.dtype"] = None,
            seed: Optional[int] = None,
            return_metadata: bool = False,
            **kwargs
    ) -> Union["torch.Tensor", Tuple["torch.Tensor", Dict[str, Any]]]:
        """Build fractal data using PyTorch."""
        if not HAS_TORCH:
            raise RuntimeError("PyTorch required for build_torch")

        # Build with numpy first
        result = self.build_numpy(
            dtype=np.float32,
            seed=seed,
            return_metadata=return_metadata,
            **kwargs
        )

        target_dtype = dtype or torch.float32
        dev = torch.device(device)

        if return_metadata:
            data, metadata = result
            tensor = torch.from_numpy(data).to(device=dev, dtype=target_dtype)
            return tensor, metadata
        else:
            tensor = torch.from_numpy(result).to(device=dev, dtype=target_dtype)
            return tensor

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Batch Generation (for datasets)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def generate_batch(
            self,
            batch_size: int,
            *,
            backend: str = "numpy",
            device: str = "cpu",
            dtype=None,
            seed: Optional[int] = None,
            return_metadata: bool = True,
            **kwargs
    ) -> Union[
        Tuple[np.ndarray, List[Dict[str, Any]]],
        Tuple["torch.Tensor", List[Dict[str, Any]]]
    ]:
        """
        Generate a batch of fractal samples.

        Returns:
            (batch_data, metadata_list)
        """
        rng = np.random.default_rng(seed)

        samples = []
        metadata_list = []

        for i in range(batch_size):
            sample_seed = rng.integers(0, 2 ** 31)
            data, meta = self.build(
                backend=backend,
                device=device if backend == "torch" else None,
                dtype=dtype,
                seed=sample_seed,
                return_metadata=True,
                validate=False,
                **kwargs
            )
            samples.append(data)
            metadata_list.append(meta)

        if backend == "torch":
            batch = torch.stack(samples, dim=0)
        else:
            batch = np.stack(samples, axis=0)

        return batch, metadata_list

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Validation
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def validate(
            self,
            output: Union[np.ndarray, "torch.Tensor"]
    ) -> Tuple[bool, str]:
        """Validate fractal output."""
        # Check for NaN/Inf
        is_numpy = isinstance(output, np.ndarray)

        if is_numpy:
            if not np.all(np.isfinite(output)):
                return False, "Contains NaN or Inf"
            arr = output
        else:
            # PyTorch tensor
            if HAS_TORCH:
                if not torch.all(torch.isfinite(output)):
                    return False, "Contains NaN or Inf"
                arr = output.cpu().numpy()
            else:
                return False, "Got non-numpy array but torch not available"

        # Check expected shape based on mode
        if self.mode in [FractalMode.ORBIT, FractalMode.ORBIT_COMPLEX]:
            if arr.ndim != 2:
                return False, f"Expected 2D orbit, got shape {arr.shape}"

            expected_features = 4 if self.mode == FractalMode.ORBIT else 2
            if arr.shape[1] != expected_features:
                return False, f"Expected {expected_features} features, got {arr.shape[1]}"

        elif self.mode in [FractalMode.IMAGE, FractalMode.JULIA_IMAGE,
                           FractalMode.MANDELBROT_IMAGE, FractalMode.ESCAPE_GRID]:
            if arr.ndim != 2:
                return False, f"Expected 2D image, got shape {arr.shape}"

            # Values should be in [0, 1] (normalized escape times)
            if arr.min() < 0 or arr.max() > 1:
                return False, f"Values outside [0,1]: [{arr.min():.3f}, {arr.max():.3f}]"

        elif self.mode == FractalMode.MULTI_SCALE:
            if arr.ndim != 3:
                return False, f"Expected 3D (scales, h, w), got shape {arr.shape}"
            if arr.shape[0] != self.num_scales:
                return False, f"Expected {self.num_scales} scales, got {arr.shape[0]}"

        return True, ""

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Metadata
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def info(self) -> Dict[str, Any]:
        base_info = super().info()
        base_info.update({
            "description": f"Fractal factory ({self.mode.value}, region={self.region})",
            "mode": self.mode.value,
            "fractal_type": self.fractal_type.value,
            "region": self.region,
            "size": self.size,
            "max_iter": self.max_iter,
            "escape_radius": self.escape_radius,
            "num_regions": len(FRACTAL_REGIONS),
            "available_regions": list(FRACTAL_REGIONS.keys())
        })
        return base_info

    @property
    def last_c(self) -> Optional[complex]:
        """Return the last generated c parameter."""
        return self._last_c

    @property
    def last_region(self) -> Optional[str]:
        """Return the last sampled region."""
        return self._last_region


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Convenience Functions
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def create_orbit_dataset(
        num_samples: int,
        sequence_length: int = 64,
        regions: Optional[List[str]] = None,
        seed: Optional[int] = None,
        backend: str = "numpy"
) -> Tuple[Union[np.ndarray, "torch.Tensor"], List[Dict[str, Any]]]:
    """
    Create a dataset of orbit sequences for region classification.

    Args:
        num_samples: Number of samples to generate
        sequence_length: Length of each orbit
        regions: List of regions to sample from (None = all)
        seed: Random seed
        backend: "numpy" or "torch"

    Returns:
        (data, metadata) where data has shape (N, T, 4)
    """
    if regions is None:
        regions = list(FRACTAL_REGIONS.keys())

    rng = np.random.default_rng(seed)
    all_samples = []
    all_metadata = []

    samples_per_region = num_samples // len(regions)
    remainder = num_samples % len(regions)

    for i, region in enumerate(regions):
        n = samples_per_region + (1 if i < remainder else 0)

        factory = FractalFactory(
            mode=FractalMode.ORBIT,
            region=region,
            size=sequence_length,
            max_iter=sequence_length
        )

        for _ in range(n):
            sample_seed = rng.integers(0, 2 ** 31)
            data, meta = factory.build(
                backend=backend,
                seed=sample_seed,
                return_metadata=True,
                validate=False
            )
            meta["region_idx"] = i
            all_samples.append(data)
            all_metadata.append(meta)

    if backend == "torch":
        batch = torch.stack(all_samples, dim=0)
    else:
        batch = np.stack(all_samples, axis=0)

    return batch, all_metadata


def create_julia_image_dataset(
        num_samples: int,
        image_size: int = 32,
        regions: Optional[List[str]] = None,
        seed: Optional[int] = None,
        backend: str = "numpy"
) -> Tuple[Union[np.ndarray, "torch.Tensor"], List[Dict[str, Any]]]:
    """
    Create a dataset of Julia set images for region classification.

    Args:
        num_samples: Number of samples
        image_size: Size of each image (H=W)
        regions: Regions to sample from
        seed: Random seed
        backend: "numpy" or "torch"

    Returns:
        (data, metadata) where data has shape (N, H, W)
    """
    if regions is None:
        regions = list(FRACTAL_REGIONS.keys())

    rng = np.random.default_rng(seed)
    all_samples = []
    all_metadata = []

    samples_per_region = num_samples // len(regions)
    remainder = num_samples % len(regions)

    for i, region in enumerate(regions):
        n = samples_per_region + (1 if i < remainder else 0)

        factory = FractalFactory(
            mode=FractalMode.JULIA_IMAGE,
            region=region,
            size=(image_size, image_size)
        )

        for _ in range(n):
            sample_seed = rng.integers(0, 2 ** 31)
            data, meta = factory.build(
                backend=backend,
                seed=sample_seed,
                return_metadata=True,
                validate=False
            )
            meta["region_idx"] = i
            all_samples.append(data)
            all_metadata.append(meta)

    if backend == "torch":
        batch = torch.stack(all_samples, dim=0)
    else:
        batch = np.stack(all_samples, axis=0)

    return batch, all_metadata


def get_region_names() -> List[str]:
    """Return list of available region names."""
    return list(FRACTAL_REGIONS.keys())


def get_region_info(region: str) -> Dict[str, Any]:
    """Return info about a specific region."""
    if region not in FRACTAL_REGIONS:
        raise ValueError(f"Unknown region: {region}")
    return FRACTAL_REGIONS[region].to_dict()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Example Usage
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    print("=" * 70)
    print("FRACTAL FACTORY DEMONSTRATION")
    print("=" * 70)

    # Example 1: Orbit sequence from seahorse region
    print("\n[Example 1] Orbit from seahorse valley")
    factory = FractalFactory(
        mode=FractalMode.ORBIT,
        region="seahorse",
        size=64
    )

    orbit, meta = factory.build(backend="numpy", seed=42, return_metadata=True)
    print(f"  Orbit shape: {orbit.shape}")
    print(f"  c = {meta['c_real']:.4f} + {meta['c_imag']:.4f}i")
    print(f"  Escape step: {meta.get('escape_step', 'bounded')}")
    print(f"  First 3 points: {orbit[:3, :2]}")

    # Example 2: Julia image from cardioid
    print("\n[Example 2] Julia image from cardioid")
    factory = FractalFactory(
        mode=FractalMode.JULIA_IMAGE,
        region="cardioid",
        size=(32, 32)
    )

    img, meta = factory.build(backend="numpy", seed=123, return_metadata=True)
    print(f"  Image shape: {img.shape}")
    print(f"  c = {meta['c_real']:.4f} + {meta['c_imag']:.4f}i")
    print(f"  Value range: [{img.min():.3f}, {img.max():.3f}]")

    # Example 3: Multi-scale Julia
    print("\n[Example 3] Multi-scale Julia (3 zoom levels)")
    factory = FractalFactory(
        mode=FractalMode.MULTI_SCALE,
        region="spiral",
        size=(32, 32),
        num_scales=3,
        zoom_factor=4.0
    )

    scales, meta = factory.build(backend="numpy", seed=456, return_metadata=True)
    print(f"  Scales shape: {scales.shape}")
    for i, s in enumerate(scales):
        print(f"    Scale {i}: range [{s.min():.3f}, {s.max():.3f}]")

    # Example 4: Create orbit dataset
    print("\n[Example 4] Orbit dataset (100 samples, 6 regions)")
    data, metadata = create_orbit_dataset(
        num_samples=100,
        sequence_length=64,
        regions=["cardioid", "period2", "seahorse", "elephant", "antenna", "spiral"],
        seed=789
    )
    print(f"  Dataset shape: {data.shape}")
    print(f"  Region distribution: {[m['region'] for m in metadata[:6]]}")

    # Example 5: PyTorch backend
    if HAS_TORCH:
        print("\n[Example 5] PyTorch CUDA" if torch.cuda.is_available() else "\n[Example 5] PyTorch CPU")
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

        factory = FractalFactory(
            mode=FractalMode.ORBIT,
            region="cusp",
            size=128
        )

        orbit_torch, meta = factory.build(
            backend="torch",
            device=device,
            return_metadata=True,
            seed=999
        )
        print(f"  Device: {orbit_torch.device}")
        print(f"  Shape: {orbit_torch.shape}")
        print(f"  Dtype: {orbit_torch.dtype}")

    # Example 6: All regions overview
    print("\n[Example 6] Available regions")
    for name in get_region_names():
        info = get_region_info(name)
        print(f"  {name:12s}: {info['description'][:40]}... ({info['expected_behavior']})")

    print("\n" + "=" * 70)
    print("FractalFactory ready for:")
    print("  - Expert fingerprinting experiments")
    print("  - Orbit sequence classification")
    print("  - Julia set region prediction")
    print("  - Multi-scale fractal features")
    print("=" * 70)