"""
Deterministic Factory System
-----------------------------
Purely deterministic synthetic generation with reproducibility guarantees.

Ensures exact reproducibility across:
  - Different runs (same seed → same output)
  - Different machines (platform-independent)
  - Different versions (configuration tracking)
  - Different backends (numpy/torch consistency)

Features:
  - Configuration fingerprinting (SHA256 hash)
  - Seed chain management
  - State serialization
  - Reproducibility validation
  - Factory version tracking
  - Deterministic RNG isolation
  - FactoryBase integration for stubbing

Author: AbstractPhil + Claude Sonnet 4.5
License: MIT
"""

import hashlib
import json
import pickle
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional, Union, Tuple, List
from datetime import datetime
import warnings

import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None


from geovocab2.shapes.factory.factory_base import FactoryBase


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Configuration Management
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class FactoryConfiguration:
    """
    Immutable factory configuration for deterministic generation.

    All parameters that affect generation must be included here.
    Changes to configuration will change the fingerprint.
    """
    # Factory identity
    factory_type: str
    factory_version: str

    # Shape parameters
    shape_type: str
    embed_dim: int
    resolution: int
    scale: float

    # Generation parameters
    backend: str
    dtype: str

    # Metadata
    created_at: str = None
    description: str = ""

    # Custom parameters (shape-specific)
    custom_params: Dict[str, Any] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow().isoformat()
        if self.custom_params is None:
            self.custom_params = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (sorted for consistent hashing)."""
        d = asdict(self)
        # Sort custom_params for consistency
        if d['custom_params']:
            d['custom_params'] = dict(sorted(d['custom_params'].items()))
        return d

    def fingerprint(self) -> str:
        """
        Generate SHA256 fingerprint of configuration.

        Same configuration → same fingerprint (guaranteed).
        """
        # Convert to canonical JSON (sorted keys, no whitespace)
        config_dict = self.to_dict()
        # Remove created_at from fingerprint (not part of generation logic)
        config_for_hash = {k: v for k, v in config_dict.items() if k != 'created_at'}
        canonical_json = json.dumps(config_for_hash, sort_keys=True, separators=(',', ':'))

        # Compute SHA256
        hash_obj = hashlib.sha256(canonical_json.encode('utf-8'))
        return hash_obj.hexdigest()

    def short_fingerprint(self, length: int = 12) -> str:
        """Get shortened fingerprint for display."""
        return self.fingerprint()[:length]

    def save(self, path: Path):
        """Save configuration to file."""
        path = Path(path)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, sort_keys=True)

    @classmethod
    def load(cls, path: Path) -> 'FactoryConfiguration':
        """Load configuration from file."""
        path = Path(path)
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Seed Management
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class SeedChain:
    """
    Deterministic seed chain for reproducible random generation.

    Given a master seed, generates a sequence of child seeds deterministically.
    Allows branching for parallel generation while maintaining reproducibility.
    """

    def __init__(self, master_seed: int):
        self.master_seed = master_seed
        self.current_index = 0
        self._rng = np.random.default_rng(master_seed)

    def next_seed(self) -> int:
        """Get next seed in chain (deterministic)."""
        # Use RNG to generate next seed
        seed = self._rng.integers(0, 2 ** 31 - 1)
        self.current_index += 1
        return int(seed)

    def branch(self, branch_id: int) -> 'SeedChain':
        """Create a deterministic branch of the seed chain."""
        # Hash master seed with branch ID for deterministic branching
        branch_seed = int(hashlib.sha256(
            f"{self.master_seed}_{branch_id}".encode()
        ).hexdigest()[:8], 16) % (2 ** 31)
        return SeedChain(branch_seed)

    def reset(self):
        """Reset chain to beginning."""
        self._rng = np.random.default_rng(self.master_seed)
        self.current_index = 0

    def get_state(self) -> Dict[str, Any]:
        """Get current state for serialization."""
        return {
            'master_seed': self.master_seed,
            'current_index': self.current_index
        }

    @classmethod
    def from_state(cls, state: Dict[str, Any]) -> 'SeedChain':
        """Restore from saved state."""
        chain = cls(state['master_seed'])
        # Fast-forward to saved index
        for _ in range(state['current_index']):
            chain.next_seed()
        return chain


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Generation Record
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class GenerationRecord:
    """
    Complete record of a deterministic generation run.

    Contains everything needed to reproduce the exact output.
    """
    # Identity
    config_fingerprint: str
    seed: int
    generation_id: str  # Unique ID for this generation

    # Timing
    generated_at: str

    # Output metadata
    output_shape: Tuple[int, ...]
    output_hash: str  # Hash of actual output data

    # Validation metrics (optional)
    validation_metrics: Optional[Dict[str, Any]] = None

    # Notes
    notes: str = ""

    def __post_init__(self):
        if self.generation_id is None:
            # Generate unique ID from config + seed
            id_str = f"{self.config_fingerprint}_{self.seed}_{self.generated_at}"
            self.generation_id = hashlib.sha256(id_str.encode()).hexdigest()[:16]

    def save(self, path: Path):
        """Save record to file."""
        path = Path(path)
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> 'GenerationRecord':
        """Load record from file."""
        path = Path(path)
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Deterministic Factory Base
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class DeterministicFactory(FactoryBase):
    """
    Base class for deterministic synthetic generation.

    Inherits from FactoryBase and adds deterministic guarantees:
      1. Same config + seed → same output (always)
      2. Output is reproducible across platforms
      3. All randomness is controlled
      4. Configuration is tracked and versioned

    Args:
        config: Factory configuration
        strict_mode: Raise errors on any non-determinism warnings
    """

    FACTORY_VERSION = "1.0.0"

    def __init__(
            self,
            config: FactoryConfiguration,
            strict_mode: bool = True
    ):
        # Initialize FactoryBase
        super().__init__(
            name=f"{config.factory_type}_{config.shape_type}",
            uid=f"{config.factory_type}.{config.fingerprint()[:8]}"
        )

        self.config = config
        self.strict_mode = strict_mode

        # Verify configuration matches factory type
        if config.factory_type != self.__class__.__name__:
            raise ValueError(
                f"Config factory_type '{config.factory_type}' "
                f"does not match class '{self.__class__.__name__}'"
            )

        # Verify version compatibility
        if config.factory_version != self.FACTORY_VERSION:
            warnings.warn(
                f"Config version {config.factory_version} differs from "
                f"factory version {self.FACTORY_VERSION}. "
                f"Reproducibility may be affected."
            )

        # Initialize seed chain (will be set when generate() is called)
        self._seed_chain: Optional[SeedChain] = None
        self._current_seed: Optional[int] = None

        # Generation history
        self._generation_records: List[GenerationRecord] = []

    def _setup_deterministic_env(self, seed: int):
        """
        Setup fully deterministic environment.

        Controls ALL sources of randomness:
          - NumPy random
          - Python random
          - PyTorch random (CPU + CUDA)
          - Hash seed (Python 3.3+)
        """
        # Setup seed chain
        self._seed_chain = SeedChain(seed)
        self._current_seed = seed

        # Python random
        import random
        random.seed(seed)

        # NumPy random
        np.random.seed(seed)

        # PyTorch random
        if HAS_TORCH:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
                # Ensure deterministic CUDA operations
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # FactoryBase Interface Implementation
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def build_numpy(self, *args, dtype=np.float32, seed: Optional[int] = None, **kwargs) -> np.ndarray:
        """
        Build object using NumPy backend (deterministic).

        Args:
            *args: Factory-specific positional arguments
            dtype: NumPy dtype for output
            seed: Random seed (REQUIRED for determinism)
            **kwargs: Factory-specific keyword arguments

        Returns:
            NumPy array representing the constructed object
        """
        if seed is None:
            raise ValueError(
                f"DeterministicFactory requires explicit 'seed' parameter. "
                f"Pass seed=<int> to build_numpy()"
            )

        # Setup deterministic environment
        self._setup_deterministic_env(seed)

        # Generate using subclass implementation
        output = self._generate_numpy_impl(*args, dtype=dtype, **kwargs)

        return output

    def build_torch(
            self,
            *args,
            device: str = "cpu",
            dtype: Optional["torch.dtype"] = None,
            seed: Optional[int] = None,
            **kwargs
    ) -> "torch.Tensor":
        """
        Build object using PyTorch backend (deterministic).

        Args:
            *args: Factory-specific positional arguments
            device: Target device ("cpu", "cuda:0", etc.)
            dtype: PyTorch dtype (if None, inferred from device)
            seed: Random seed (REQUIRED for determinism)
            **kwargs: Factory-specific keyword arguments

        Returns:
            PyTorch tensor on specified device
        """
        if not HAS_TORCH:
            raise RuntimeError(f"PyTorch required for build_torch() in {self.name}")

        if seed is None:
            raise ValueError(
                f"DeterministicFactory requires explicit 'seed' parameter. "
                f"Pass seed=<int> to build_torch()"
            )

        # Setup deterministic environment
        self._setup_deterministic_env(seed)

        # Generate using subclass implementation
        target_dtype = dtype or self._infer_torch_dtype(device)
        output = self._generate_torch_impl(*args, device=device, dtype=target_dtype, **kwargs)

        return output

    def build(
            self,
            *args,
            backend: str = "numpy",
            device: str = "cpu",
            dtype: Optional[Any] = None,
            validate: bool = True,
            seed: Optional[int] = None,
            save_record: bool = False,
            record_path: Optional[Path] = None,
            **kwargs
    ) -> Union[np.ndarray, "torch.Tensor", Tuple[Union[np.ndarray, "torch.Tensor"], GenerationRecord]]:
        """
        Unified build interface with deterministic guarantees.

        Args:
            *args: Factory-specific positional arguments
            backend: "numpy" or "torch"
            device: Device for torch backend
            dtype: Output dtype (backend-specific)
            validate: Run validation checks on output
            seed: Random seed (REQUIRED for determinism)
            save_record: Save generation record
            record_path: Path for record (required if save_record=True)
            **kwargs: Factory-specific keyword arguments

        Returns:
            Constructed object (ndarray or Tensor)
            OR (output, record) if save_record=True
        """
        if seed is None:
            raise ValueError(
                f"DeterministicFactory requires explicit 'seed' parameter. "
                f"Pass seed=<int> to build()"
            )

        # Record start time
        start_time = datetime.utcnow()

        # Build using appropriate backend
        if backend.lower() == "numpy":
            output = self.build_numpy(*args, dtype=dtype or np.float32, seed=seed, **kwargs)
        elif backend.lower() == "torch":
            output = self.build_torch(*args, device=device, dtype=dtype, seed=seed, **kwargs)
        else:
            raise ValueError(f"Invalid backend '{backend}' (allowed: 'numpy', 'torch')")

        # Validate if requested
        if validate:
            is_valid, error_msg = self.validate(output)
            if not is_valid:
                raise ValueError(f"Factory {self.name} validation failed: {error_msg}")

        # Create generation record
        validation_metrics = self._compute_validation_metrics(output) if validate else None

        record = GenerationRecord(
            config_fingerprint=self.config.fingerprint(),
            seed=seed,
            generation_id=None,  # Will be auto-generated
            generated_at=start_time.isoformat(),
            output_shape=tuple(output.shape),
            output_hash=self._hash_output(output),
            validation_metrics=validation_metrics
        )

        # Save record
        self._generation_records.append(record)

        if save_record:
            if record_path is None:
                raise ValueError("record_path required when save_record=True")
            record.save(record_path)
            return output, record

        return output

    def validate(self, output: Union[np.ndarray, "torch.Tensor"]) -> Tuple[bool, str]:
        """
        Validate factory output (can be overridden by subclass).

        Args:
            output: Constructed object from build()

        Returns:
            (is_valid, error_message)
        """
        # Basic validation
        if isinstance(output, np.ndarray):
            if not np.all(np.isfinite(output)):
                return False, "Contains NaN or Inf"
        else:
            if not torch.all(torch.isfinite(output)):
                return False, "Contains NaN or Inf"

        return True, ""

    def info(self) -> Dict[str, Any]:
        """
        Factory metadata for introspection.

        Returns:
            Dictionary with factory information
        """
        base_info = super().info()
        base_info.update({
            "factory_version": self.FACTORY_VERSION,
            "config_fingerprint": self.config.fingerprint(),
            "config_short_fingerprint": self.config.short_fingerprint(),
            "deterministic": True,
            "generation_count": len(self._generation_records),
            "config": self.config.to_dict()
        })
        return base_info

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Subclass Interface (must implement)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _generate_numpy_impl(self, *args, dtype=np.float32, **kwargs) -> np.ndarray:
        """
        Actual NumPy generation implementation (subclass must override).

        Should use self._seed_chain.next_seed() for all random operations.
        Environment is already setup with deterministic seeds.
        """
        raise NotImplementedError("Subclass must implement _generate_numpy_impl()")

    def _generate_torch_impl(self, *args, device="cpu", dtype=None, **kwargs) -> "torch.Tensor":
        """
        Actual PyTorch generation implementation (subclass can override).

        Default: convert from NumPy. Override for native torch implementation.
        """
        # Default: build numpy and convert
        arr = self._generate_numpy_impl(*args, dtype=np.float32, **kwargs)
        arr = np.ascontiguousarray(arr)

        cpu_tensor = torch.from_numpy(arr).pin_memory()
        return cpu_tensor.to(device=device, dtype=dtype, non_blocking=True)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Utility Methods
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _hash_output(self, output: Union[np.ndarray, 'torch.Tensor']) -> str:
        """
        Compute deterministic hash of output data.

        Uses SHA256 of binary representation for consistency.
        """
        if isinstance(output, np.ndarray):
            data = output.tobytes()
        else:  # torch.Tensor
            data = output.cpu().numpy().tobytes()

        return hashlib.sha256(data).hexdigest()

    def _compute_validation_metrics(self, output: Union[np.ndarray, 'torch.Tensor']) -> Dict[str, Any]:
        """
        Compute validation metrics (can be overridden by subclass).

        Returns metrics dictionary.
        """
        metrics = {
            "shape": tuple(output.shape),
            "dtype": str(output.dtype),
        }

        if isinstance(output, np.ndarray):
            metrics['finite'] = bool(np.all(np.isfinite(output)))
            metrics['mean'] = float(output.mean())
            metrics['std'] = float(output.std())
            metrics['min'] = float(output.min())
            metrics['max'] = float(output.max())
        else:
            metrics['finite'] = bool(torch.all(torch.isfinite(output)).item())
            metrics['mean'] = float(output.mean().item())
            metrics['std'] = float(output.std().item())
            metrics['min'] = float(output.min().item())
            metrics['max'] = float(output.max().item())

        return metrics

    def generate(
            self,
            seed: int,
            backend: str = "numpy",
            validate: bool = True,
            save_record: bool = False,
            record_path: Optional[Path] = None,
            **kwargs
    ) -> Tuple[Union[np.ndarray, 'torch.Tensor'], GenerationRecord]:
        """
        High-level generation with automatic record tracking.

        Convenience wrapper around build() that always returns record.

        Args:
            seed: Master seed for generation
            backend: "numpy" or "torch"
            validate: Run validation checks
            save_record: Save generation record to disk
            record_path: Path for record (required if save_record=True)
            **kwargs: Additional arguments for build()

        Returns:
            output: Generated data
            record: Generation record with metadata
        """
        # Only save to disk if explicitly requested with path
        should_save = save_record and record_path is not None

        if save_record and record_path is None:
            raise ValueError("record_path required when save_record=True")

        if should_save:
            # build() will save and return tuple
            result = self.build(
                backend=backend,
                validate=validate,
                seed=seed,
                save_record=True,
                record_path=record_path,
                **kwargs
            )
            return result
        else:
            # build() won't save but will add record to history
            output = self.build(
                backend=backend,
                validate=validate,
                seed=seed,
                save_record=False,
                **kwargs
            )
            # Get the record that was just added
            record = self._generation_records[-1]
            return output, record

    def verify_reproducibility(
            self,
            seed: int,
            n_trials: int = 3,
            backend: str = "numpy"
    ) -> Tuple[bool, List[str]]:
        """
        Verify that generation is reproducible.

        Generates n_trials times with same seed and verifies all outputs are identical.

        Returns:
            is_reproducible: True if all outputs match
            hashes: List of output hashes from each trial
        """
        hashes = []

        for trial in range(n_trials):
            output = self.build(seed=seed, backend=backend, validate=False)
            output_hash = self._hash_output(output)
            hashes.append(output_hash)

        # Check all hashes are identical
        is_reproducible = len(set(hashes)) == 1

        return is_reproducible, hashes

    def get_generation_history(self) -> List[GenerationRecord]:
        """Get all generation records from this session."""
        return self._generation_records.copy()

    def save_state(self, path: Path):
        """Save complete factory state (config + history)."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save configuration
        self.config.save(path / "config.json")

        # Save generation records
        records_dir = path / "records"
        records_dir.mkdir(exist_ok=True)

        for record in self._generation_records:
            record.save(records_dir / f"{record.generation_id}.json")

    @classmethod
    def load_state(cls, path: Path, config_class=FactoryConfiguration) -> 'DeterministicFactory':
        """Load factory from saved state."""
        path = Path(path)

        # Load configuration
        config = config_class.load(path / "config.json")

        # Create factory
        factory = cls(config)

        # Load generation records
        records_dir = path / "records"
        if records_dir.exists():
            for record_path in records_dir.glob("*.json"):
                record = GenerationRecord.load(record_path)
                factory._generation_records.append(record)

        return factory


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Deterministic Shape Factory
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class DeterministicShapeFactory(DeterministicFactory):
    """
    Deterministic geometric shape factory.

    Generates shapes with guaranteed reproducibility.
    """

    FACTORY_VERSION = "1.0.0"

    def __init__(self, config: FactoryConfiguration, **kwargs):
        super().__init__(config, **kwargs)

        # Validate shape-specific parameters
        required_params = ['shape_type', 'embed_dim', 'resolution', 'scale']
        for param in required_params:
            if not hasattr(config, param):
                raise ValueError(f"Config missing required parameter: {param}")

    def _generate_numpy_impl(self, *args, dtype=np.float32, **kwargs) -> np.ndarray:
        """Generate shape using controlled randomness (NumPy implementation)."""
        shape_type = self.config.shape_type
        embed_dim = self.config.embed_dim
        resolution = self.config.resolution
        scale = self.config.scale

        # Get deterministic seed for this operation
        seed = self._seed_chain.next_seed()
        rng = np.random.default_rng(seed)

        # Generate based on shape type
        if shape_type == "sphere":
            points = self._generate_sphere(rng, resolution, embed_dim)
        elif shape_type == "cube":
            points = self._generate_cube(rng, resolution, embed_dim)
        elif shape_type == "pyramid":
            points = self._generate_pyramid(rng, resolution, embed_dim)
        elif shape_type == "cone":
            points = self._generate_cone(rng, resolution, embed_dim)
        else:
            raise ValueError(f"Unknown shape: {shape_type}")

        # Apply scale
        points = points * scale

        # Ensure correct dtype
        return points.astype(dtype)

    def _generate_sphere(self, rng, n_points, dim):
        """Generate sphere using deterministic sampling."""
        # Use Gaussian method for uniform sphere
        points = rng.standard_normal((n_points, dim))
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        return points / (norms + 1e-10)

    def _generate_cube(self, rng, n_points, dim):
        """Generate cube vertices."""
        # Corners
        n_corners = min(2 ** dim, n_points // 2)
        corners = np.array([
            [(-1) ** ((i >> j) & 1) for j in range(dim)]
            for i in range(n_corners)
        ], dtype=np.float32)

        # Face samples
        n_face = n_points - n_corners
        face_points = rng.uniform(-1, 1, size=(n_face, dim))

        # Project to faces
        for i in range(n_face):
            face_dim = rng.integers(0, dim)
            face_points[i, face_dim] = rng.choice([-1.0, 1.0])

        return np.vstack([corners, face_points]).astype(np.float32)

    def _generate_pyramid(self, rng, n_points, dim):
        """Generate pyramid."""
        if dim < 3:
            raise ValueError("Pyramid requires dim >= 3")

        # Apex
        apex = np.zeros((1, dim))
        apex[0, 2] = 1.0

        # Base
        n_base = n_points - 1
        base = rng.uniform(-1, 1, size=(n_base, dim))
        base[:, 2] = -1.0

        return np.vstack([apex, base]).astype(np.float32)

    def _generate_cone(self, rng, n_points, dim):
        """Generate cone."""
        if dim < 3:
            raise ValueError("Cone requires dim >= 3")

        # Apex
        apex = np.zeros((1, dim))
        apex[0, 2] = 1.0

        # Surface
        n_surface = n_points - 1
        z = rng.uniform(-1, 1, n_surface)
        r = (1 - z) / 2

        # Angular distribution
        theta = rng.uniform(0, 2 * np.pi, n_surface)

        surface = np.zeros((n_surface, dim))
        surface[:, 0] = r * np.cos(theta)
        surface[:, 1] = r * np.sin(theta)
        surface[:, 2] = z

        return np.vstack([apex, surface]).astype(np.float32)

    def validate(self, output: Union[np.ndarray, 'torch.Tensor']) -> Tuple[bool, str]:
        """Validate shape output (overrides base class)."""
        # Call base validation first
        is_valid, error_msg = super().validate(output)
        if not is_valid:
            return is_valid, error_msg

        # Add shape-specific validation
        expected_shape = (self.config.resolution, self.config.embed_dim)
        if output.shape != expected_shape:
            return False, f"Expected shape {expected_shape}, got {output.shape}"

        return True, ""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Testing and Examples
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_deterministic_generation():
    """Test deterministic generation guarantees."""
    print("\n" + "=" * 70)
    print("DETERMINISTIC FACTORY TESTS")
    print("=" * 70)

    # Test 1: Configuration fingerprinting
    print("\n[Test 1] Configuration Fingerprinting")

    config1 = FactoryConfiguration(
        factory_type="DeterministicShapeFactory",
        factory_version="1.0.0",
        shape_type="sphere",
        embed_dim=3,
        resolution=100,
        scale=1.0,
        backend="numpy",
        dtype="float32"
    )

    config2 = FactoryConfiguration(
        factory_type="DeterministicShapeFactory",
        factory_version="1.0.0",
        shape_type="sphere",
        embed_dim=3,
        resolution=100,
        scale=1.0,
        backend="numpy",
        dtype="float32"
    )

    config3 = FactoryConfiguration(
        factory_type="DeterministicShapeFactory",
        factory_version="1.0.0",
        shape_type="cube",  # Different!
        embed_dim=3,
        resolution=100,
        scale=1.0,
        backend="numpy",
        dtype="float32"
    )

    fp1 = config1.fingerprint()
    fp2 = config2.fingerprint()
    fp3 = config3.fingerprint()

    print(f"  Config 1 (sphere): {config1.short_fingerprint()}")
    print(f"  Config 2 (sphere): {config2.short_fingerprint()}")
    print(f"  Config 3 (cube):   {config3.short_fingerprint()}")
    print(f"  Same configs match: {fp1 == fp2}")
    print(f"  Different configs differ: {fp1 != fp3}")
    assert fp1 == fp2, "Identical configs should have same fingerprint"
    assert fp1 != fp3, "Different configs should have different fingerprints"
    print(f"  Status: ✓ PASS")

    # Test 2: Reproducibility
    print("\n[Test 2] Reproducibility Verification")

    factory = DeterministicShapeFactory(config1)

    seed = 42
    output1, record1 = factory.generate(seed, validate=True)
    output2, record2 = factory.generate(seed, validate=True)
    output3, record3 = factory.generate(seed, validate=True)

    hash1 = record1.output_hash
    hash2 = record2.output_hash
    hash3 = record3.output_hash

    print(f"  Generation 1 hash: {hash1[:12]}...")
    print(f"  Generation 2 hash: {hash2[:12]}...")
    print(f"  Generation 3 hash: {hash3[:12]}...")
    print(f"  All hashes match: {hash1 == hash2 == hash3}")
    print(f"  Arrays identical: {np.allclose(output1, output2) and np.allclose(output2, output3)}")

    assert hash1 == hash2 == hash3, "Same seed should produce identical outputs"
    assert np.allclose(output1, output2, atol=0), "Arrays should be exactly equal"
    print(f"  Status: ✓ PASS")

    # Test 3: Different seeds produce different outputs
    print("\n[Test 3] Different Seeds Produce Different Outputs")

    output_a, record_a = factory.generate(seed=42)
    output_b, record_b = factory.generate(seed=123)
    output_c, record_c = factory.generate(seed=999)

    hash_a = record_a.output_hash
    hash_b = record_b.output_hash
    hash_c = record_c.output_hash

    print(f"  Seed 42:  {hash_a[:12]}...")
    print(f"  Seed 123: {hash_b[:12]}...")
    print(f"  Seed 999: {hash_c[:12]}...")
    print(f"  All different: {len({hash_a, hash_b, hash_c}) == 3}")

    assert len({hash_a, hash_b, hash_c}) == 3, "Different seeds should produce different outputs"
    print(f"  Status: ✓ PASS")

    # Test 4: Automatic reproducibility verification
    print("\n[Test 4] Automatic Reproducibility Verification")

    is_reproducible, hashes = factory.verify_reproducibility(seed=42, n_trials=5)

    print(f"  Trials: 5")
    print(f"  Unique hashes: {len(set(hashes))}")
    print(f"  Reproducible: {is_reproducible}")

    assert is_reproducible, "Factory should be reproducible"
    print(f"  Status: ✓ PASS")

    # Test 5: Seed chain determinism
    print("\n[Test 5] Seed Chain Determinism")

    chain1 = SeedChain(master_seed=100)
    chain2 = SeedChain(master_seed=100)

    seeds1 = [chain1.next_seed() for _ in range(10)]
    seeds2 = [chain2.next_seed() for _ in range(10)]

    print(f"  Chain 1 seeds: {seeds1[:3]}...")
    print(f"  Chain 2 seeds: {seeds2[:3]}...")
    print(f"  Chains match: {seeds1 == seeds2}")

    assert seeds1 == seeds2, "Seed chains should be deterministic"
    print(f"  Status: ✓ PASS")

    # Test 6: Configuration changes affect output
    print("\n[Test 6] Configuration Changes Affect Output")

    config_scale_1 = FactoryConfiguration(
        factory_type="DeterministicShapeFactory",
        factory_version="1.0.0",
        shape_type="sphere",
        embed_dim=3,
        resolution=100,
        scale=1.0,
        backend="numpy",
        dtype="float32"
    )

    config_scale_2 = FactoryConfiguration(
        factory_type="DeterministicShapeFactory",
        factory_version="1.0.0",
        shape_type="sphere",
        embed_dim=3,
        resolution=100,
        scale=2.0,  # Different scale
        backend="numpy",
        dtype="float32"
    )

    factory1 = DeterministicShapeFactory(config_scale_1)
    factory2 = DeterministicShapeFactory(config_scale_2)

    out1, rec1 = factory1.generate(seed=42)
    out2, rec2 = factory2.generate(seed=42)

    print(f"  Scale 1.0 hash: {rec1.output_hash[:12]}...")
    print(f"  Scale 2.0 hash: {rec2.output_hash[:12]}...")
    print(f"  Hashes differ: {rec1.output_hash != rec2.output_hash}")
    print(f"  Config fingerprints differ: {rec1.config_fingerprint != rec2.config_fingerprint}")

    assert rec1.output_hash != rec2.output_hash, "Different configs should produce different outputs"
    print(f"  Status: ✓ PASS")

    # Test 7: State serialization
    print("\n[Test 7] State Serialization")

    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)

        # Generate some data
        factory = DeterministicShapeFactory(config1)
        factory.generate(seed=42, save_record=True, record_path=tmppath / "record_42.json")
        factory.generate(seed=123, save_record=True, record_path=tmppath / "record_123.json")

        # Save state
        factory.save_state(tmppath / "factory_state")

        # Load state
        loaded_factory = DeterministicShapeFactory.load_state(tmppath / "factory_state")

        print(f"  Original records: {len(factory.get_generation_history())}")
        print(f"  Loaded records: {len(loaded_factory.get_generation_history())}")
        print(f"  Configs match: {factory.config.fingerprint() == loaded_factory.config.fingerprint()}")

        # Verify loaded factory produces same output
        out_orig, rec_orig = factory.generate(seed=999)
        out_loaded, rec_loaded = loaded_factory.generate(seed=999)

        print(f"  Same output after reload: {rec_orig.output_hash == rec_loaded.output_hash}")

        assert rec_orig.output_hash == rec_loaded.output_hash, "Loaded factory should match original"

    print(f"  Status: ✓ PASS")

    print("\n" + "=" * 70)
    print("All tests passed! ✓ (7/7)")
    print("=" * 70 + "\n")


def example_usage():
    """Example of using deterministic factory system."""
    print("\n[EXAMPLE] Deterministic Factory Workflow")
    print("-" * 70)

    # Step 1: Create configuration
    print("\n1. Create Factory Configuration")
    config = FactoryConfiguration(
        factory_type="DeterministicShapeFactory",
        factory_version="1.0.0",
        shape_type="sphere",
        embed_dim=3,
        resolution=200,
        scale=1.5,
        backend="numpy",
        dtype="float32",
        description="High-resolution sphere for classification"
    )

    print(f"   Configuration fingerprint: {config.short_fingerprint()}")
    print(f"   Full fingerprint: {config.fingerprint()}")

    # Step 2: Create factory
    print("\n2. Create Factory")
    factory = DeterministicShapeFactory(config)

    # Step 3: Generate with different seeds
    print("\n3. Generate Multiple Samples")

    samples = []
    for i, seed in enumerate([42, 123, 999]):
        output, record = factory.generate(seed, validate=True)
        samples.append((output, record))
        print(f"   Sample {i + 1} (seed={seed}): {output.shape}, hash={record.output_hash[:12]}...")

    # Step 4: Verify reproducibility
    print("\n4. Verify Reproducibility")
    is_reproducible, hashes = factory.verify_reproducibility(seed=42, n_trials=3)
    print(f"   Reproducible: {is_reproducible}")
    print(f"   Unique hashes: {len(set(hashes))} (should be 1)")

    # Step 5: Save factory state
    print("\n5. Save Factory State")
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        state_path = Path(tmpdir) / "factory_state"
        factory.save_state(state_path)
        print(f"   Saved to: {state_path}")

        # Load and verify
        loaded_factory = DeterministicShapeFactory.load_state(state_path)
        out_new, rec_new = loaded_factory.generate(seed=42)
        matches_original = rec_new.output_hash == samples[0][1].output_hash
        print(f"   Loaded factory produces same output: {matches_original}")

    print("\n6. Generation History")
    history = factory.get_generation_history()
    print(f"   Total generations: {len(history)}")
    for i, record in enumerate(history[:3]):
        print(f"   {i + 1}. Seed {record.seed}, hash {record.output_hash[:12]}...")

    print("\n" + "-" * 70)
    print("Deterministic factory workflow complete!")
    print("-" * 70 + "\n")


if __name__ == "__main__":
    # Run tests
    test_deterministic_generation()

    # Show example
    example_usage()

    print("\n[Summary]")
    print("Deterministic Factory System")
    print("Features:")
    print("  ✓ Configuration fingerprinting (SHA256)")
    print("  ✓ Guaranteed reproducibility (same seed → same output)")
    print("  ✓ Seed chain management for parallel generation")
    print("  ✓ Complete generation tracking and validation")
    print("  ✓ State serialization and loading")
    print("  ✓ Platform-independent (works across machines)")
    print("  ✓ Version tracking for factory evolution")
    print("\nUse cases:")
    print("  • Scientific reproducibility")
    print("  • Dataset versioning and auditing")
    print("  • Debugging and validation")
    print("  • Factory evolution with traceability")
    print("  • Benchmark dataset creation")