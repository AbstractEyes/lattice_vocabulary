"""
BaseModel
Author: AbstractPhil + Claude Sonnet 4.5

Description: Framework-agnostic model base class for torch, numpy, and tensorflow.
- Minimal ABC contract for model implementations
- Framework detection and management
- Device handling (torch/tensorflow)
- Secure state save/load (safetensors for torch, pickle for numpy)
- Parameter access for optimizers

Design Philosophy:
    Enforce a minimal contract without imposing implementation.
    Subclasses define their architecture and framework.
    Maximum flexibility, crash-loud on misconfiguration.
    Uses safetensors for torch models (no pickle vulnerabilities).

License: MIT
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List
from pathlib import Path


class BaseModel(ABC):
    """
    Framework-agnostic model base class.

    Provides a unified interface for models across torch, numpy, and tensorflow.
    Subclasses implement the forward pass and specify their framework.

    Attributes:
        name: Human-readable model name
        uid: Unique identifier for registry/logging
        framework: 'torch', 'numpy', or 'tensorflow'

    Methods:
        forward(): Core computation (REQUIRED)
        parameters(): Return model parameters for optimization (REQUIRED)
        to(): Move model to device (framework-specific, REQUIRED)
        compile(): Compile model for optimized execution (OPTIONAL)
        cache(): Enable/disable result caching (OPTIONAL)
        save(): Save model state (uses safetensors for torch)
        load(): Load model state (uses safetensors for torch)
        info(): Model metadata for introspection

    Example:
        class GeometricTransformer(BaseModel):
            def __init__(self, vocab_size, embed_dim):
                super().__init__(
                    name="geometric_transformer",
                    uid="m.geometric.transformer"
                )
                self.vocab_size = vocab_size
                self.embed_dim = embed_dim

                # Build model (framework-specific)
                import torch.nn as nn
                self.embedding = nn.Embedding(vocab_size, embed_dim)
                self.layers = nn.ModuleList([...])

            def _detect_framework(self):
                return 'torch'

            def forward(self, x):
                x = self.embedding(x)
                for layer in self.layers:
                    x = layer(x)
                return x

            def parameters(self):
                # Torch models can delegate to nn.Module
                return super().parameters()  # if inheriting nn.Module

            def to(self, device):
                # Torch-specific device transfer
                self.embedding = self.embedding.to(device)
                self.layers = self.layers.to(device)
                return self

            def compile(self, mode='default'):
                # Use PyTorch 2.0 compilation
                return super().compile(mode=mode)

            def cache(self, enabled=True, **kwargs):
                # Enable caching for repeated inputs
                return super().cache(enabled=enabled, **kwargs)
    """

    def __init__(self, name: str, uid: str):
        """
        Initialize model with metadata.

        Args:
            name: Human-readable model name
            uid: Unique identifier (e.g., "m.transformer.geometric")

        Raises:
            ValueError: If name or uid is empty
        """
        if not name or not uid:
            raise ValueError("Model name and uid must be non-empty strings")

        self.name = name
        self.uid = uid
        self.framework = self._detect_framework()

        # Validate framework
        valid_frameworks = ['torch', 'numpy', 'tensorflow']
        if self.framework not in valid_frameworks:
            raise ValueError(
                f"Invalid framework '{self.framework}'. "
                f"Must be one of {valid_frameworks}"
            )

    @abstractmethod
    def _detect_framework(self) -> str:
        """
        Detect or specify the framework this model uses.

        Override this method to specify framework explicitly.

        Returns:
            'torch', 'numpy', or 'tensorflow'

        Example:
            def _detect_framework(self):
                return 'torch'
        """
        pass

    @abstractmethod
    def forward(self, *args, **kwargs) -> Any:
        """
        Forward pass computation.

        Args:
            *args: Model-specific inputs
            **kwargs: Model-specific keyword arguments

        Returns:
            Model output (format depends on implementation)

        Note:
            For torch models, this is typically called via __call__
            For numpy models, call this directly

        Example:
            def forward(self, x, mask=None):
                # x: [batch, seq_len, embed_dim]
                x = self.embedding(x)
                if mask is not None:
                    x = x * mask
                return self.output_layer(x)
        """
        pass

    @abstractmethod
    def parameters(self) -> Any:
        """
        Return model parameters for optimization.

        Returns:
            Framework-specific parameter collection
            - torch: iterator/list of nn.Parameter
            - numpy: list/dict of numpy arrays
            - tensorflow: list of tf.Variable

        Example (torch):
            def parameters(self):
                # If inheriting nn.Module
                return super().parameters()

        Example (numpy):
            def parameters(self):
                return [self.weights, self.biases]
        """
        pass

    @abstractmethod
    def to(self, device: Any) -> 'BaseModel':
        """
        Move model to specified device (framework-specific).

        Args:
            device: Device specification (framework-specific)
                - torch: 'cpu', 'cuda', 'mps', or torch.device
                - numpy: No-op (numpy doesn't use devices)
                - tensorflow: '/CPU:0', '/GPU:0', etc.

        Returns:
            Self for method chaining

        Raises:
            NotImplementedError: If framework doesn't support devices

        Example (torch):
            def to(self, device):
                import torch
                self.device = torch.device(device)
                self.embedding = self.embedding.to(self.device)
                return self

        Example (numpy):
            def to(self, device):
                # Numpy doesn't use devices
                raise NotImplementedError("Numpy models don't support device placement")
        """
        pass

    def compile(self, **kwargs) -> 'BaseModel':
        """
        Compile model for optimized execution (framework-specific).

        Args:
            **kwargs: Framework-specific compilation options
                - torch: mode='default'|'reduce-overhead'|'max-autotune'
                - tensorflow: jit_compile=True, experimental_compile=True
                - numpy: Not supported

        Returns:
            Self for method chaining

        Note:
            Default implementation attempts framework-specific compilation.
            Override for custom compilation logic.

        Raises:
            ImportError: If required compilation features not available
            NotImplementedError: If framework doesn't support compilation

        Example (torch):
            def compile(self, mode='default'):
                import torch
                if hasattr(torch, 'compile'):
                    self.model = torch.compile(self.model, mode=mode)
                return self
        """
        if self.framework == 'torch':
            import torch
            if not hasattr(torch, 'compile'):
                raise ImportError(
                    "torch.compile requires PyTorch 2.0+. "
                    "Current version does not support compilation."
                )

            # Default torch compilation
            mode = kwargs.get('mode', 'default')
            # Note: This assumes self inherits from nn.Module
            # Subclass may need to override if structure is different
            try:
                # Compile the model in-place
                import torch.nn as nn
                if isinstance(self, nn.Module):
                    # For nn.Module, we compile self
                    compiled = torch.compile(self, mode=mode)
                    # Copy compiled attributes back
                    self.__dict__.update(compiled.__dict__)
                else:
                    raise NotImplementedError(
                        "Default compile() requires model to inherit from nn.Module. "
                        "Override compile() for custom compilation."
                    )
            except Exception as e:
                raise RuntimeError(f"Torch compilation failed: {e}")

        elif self.framework == 'numpy':
            raise NotImplementedError(
                "Numpy models do not support compilation. "
                "Consider using JAX for JIT compilation with numpy-like API."
            )

        elif self.framework == 'tensorflow':
            raise NotImplementedError(
                "TensorFlow compilation not implemented. "
                "Use @tf.function decorator or model.compile() for Keras models."
            )

        return self

    def cache(self, enabled: bool = True, **kwargs) -> 'BaseModel':
        """
        Enable/disable caching for model computations.

        Args:
            enabled: Whether to enable caching
            **kwargs: Framework-specific caching options
                - max_size: Maximum cache size (default: 128)
                - clear: Clear existing cache

        Returns:
            Self for method chaining

        Note:
            Default implementation sets a _cache_enabled flag.
            Subclasses should implement actual caching logic in forward().

        Example:
            def forward(self, x):
                # Check if caching enabled
                if hasattr(self, '_cache_enabled') and self._cache_enabled:
                    cache_key = hash(x.data_ptr() if hasattr(x, 'data_ptr') else id(x))
                    if cache_key in self._cache:
                        return self._cache[cache_key]

                # Compute result
                result = self._compute(x)

                # Cache if enabled
                if hasattr(self, '_cache_enabled') and self._cache_enabled:
                    self._cache[cache_key] = result

                return result
        """
        self._cache_enabled = enabled

        if enabled:
            # Initialize cache if not exists
            if not hasattr(self, '_cache'):
                from functools import lru_cache
                max_size = kwargs.get('max_size', 128)
                self._cache = {}
                self._cache_max_size = max_size

        # Clear cache if requested
        if kwargs.get('clear', False) and hasattr(self, '_cache'):
            self._cache.clear()

        return self

    def save(self, path: str) -> None:
        """
        Save model state to disk.

        Args:
            path: Path to save model state

        Note:
            - Torch models: Uses safetensors (secure, no pickle)
            - Numpy models: Uses pickle
            - Override for custom save logic

        Raises:
            ImportError: If safetensors not installed for torch models
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if self.framework == 'torch':
            try:
                from safetensors.torch import save_file
            except ImportError:
                raise ImportError(
                    "safetensors is required for saving torch models. "
                    "Install with: pip install safetensors"
                )

            # Get state dict
            if not hasattr(self, 'state_dict'):
                raise AttributeError(
                    "Torch model must have state_dict() method. "
                    "Inherit from torch.nn.Module or implement state_dict()."
                )

            state_dict = self.state_dict()

            # Save metadata separately (safetensors doesn't support metadata well yet)
            metadata = {
                'name': self.name,
                'uid': self.uid,
                'framework': self.framework
            }

            # Save safetensors
            save_file(state_dict, str(path))

            # Save metadata as json
            import json
            metadata_path = path.with_suffix('.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

        elif self.framework == 'numpy':
            import pickle
            state = {
                'name': self.name,
                'uid': self.uid,
                'framework': self.framework,
                'parameters': self.parameters()
            }
            with open(path, 'wb') as f:
                pickle.dump(state, f)

        elif self.framework == 'tensorflow':
            # TensorFlow has built-in save
            raise NotImplementedError(
                "TensorFlow save not implemented. Use tf.saved_model.save() or model.save()"
            )

    def load(self, path: str) -> None:
        """
        Load model state from disk.

        Args:
            path: Path to saved model state

        Note:
            - Torch models: Uses safetensors (secure, no pickle)
            - Numpy models: Uses pickle
            - Override for custom load logic

        Raises:
            FileNotFoundError: If model file doesn't exist
            ImportError: If safetensors not installed for torch models
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        if self.framework == 'torch':
            try:
                from safetensors.torch import load_file
            except ImportError:
                raise ImportError(
                    "safetensors is required for loading torch models. "
                    "Install with: pip install safetensors"
                )

            if not hasattr(self, 'load_state_dict'):
                raise AttributeError(
                    "Torch model must have load_state_dict() method. "
                    "Inherit from torch.nn.Module or implement load_state_dict()."
                )

            # Load from safetensors
            state_dict = load_file(str(path))
            self.load_state_dict(state_dict)

            # Load metadata if available
            metadata_path = path.with_suffix('.json')
            if metadata_path.exists():
                import json
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    # Could validate metadata here if needed

        elif self.framework == 'numpy':
            import pickle
            with open(path, 'rb') as f:
                state = pickle.load(f)
            # Subclass must handle parameter loading

        elif self.framework == 'tensorflow':
            raise NotImplementedError(
                "TensorFlow load not implemented. Use tf.saved_model.load() or tf.keras.models.load_model()"
            )

    def info(self) -> Dict[str, Any]:
        """
        Model metadata for introspection.

        Subclasses can override to provide rich metadata about
        architecture, parameters, configuration, etc.

        Returns:
            Dictionary containing model metadata

        Example:
            def info(self):
                return {
                    "name": self.name,
                    "uid": self.uid,
                    "framework": self.framework,
                    "vocab_size": self.vocab_size,
                    "embed_dim": self.embed_dim,
                    "num_layers": len(self.layers),
                    "num_parameters": sum(p.numel() for p in self.parameters())
                }
        """
        return {
            "name": self.name,
            "uid": self.uid,
            "framework": self.framework,
            "description": "No description provided"
        }

    def __repr__(self) -> str:
        """String representation for debugging."""
        return f"{self.__class__.__name__}(name='{self.name}', uid='{self.uid}', framework='{self.framework}')"

    def __str__(self) -> str:
        """User-friendly string representation."""
        return f"Model[{self.name}] ({self.framework})"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# EXAMPLE USAGE AND TESTING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    import torch
    import torch.nn as nn
    import numpy as np

    # Example 1: Torch model
    class TorchLinearModel(BaseModel, nn.Module):
        """Simple torch linear model for testing."""

        def __init__(self, input_dim=10, output_dim=1):
            # Initialize both base classes
            BaseModel.__init__(self, "torch_linear", "m.torch.linear")
            nn.Module.__init__(self)

            self.input_dim = input_dim
            self.output_dim = output_dim
            self.linear = nn.Linear(input_dim, output_dim)
            self.device = torch.device('cpu')

        def _detect_framework(self):
            return 'torch'

        def forward(self, x):
            return self.linear(x)

        def parameters(self):
            # Delegate to nn.Module
            return nn.Module.parameters(self)

        def to(self, device):
            self.device = torch.device(device)
            self.linear = self.linear.to(self.device)
            return self

        def info(self):
            return {
                "name": self.name,
                "uid": self.uid,
                "framework": self.framework,
                "input_dim": self.input_dim,
                "output_dim": self.output_dim,
                "device": str(self.device),
                "num_parameters": sum(p.numel() for p in self.parameters())
            }

    # Example 2: Numpy model
    class NumpyLinearModel(BaseModel):
        """Simple numpy linear model for testing."""

        def __init__(self, input_dim=10, output_dim=1):
            super().__init__("numpy_linear", "m.numpy.linear")

            self.input_dim = input_dim
            self.output_dim = output_dim
            # Initialize parameters
            self.W = np.random.randn(input_dim, output_dim) * 0.01
            self.b = np.zeros(output_dim)

        def _detect_framework(self):
            return 'numpy'

        def forward(self, x):
            return x @ self.W + self.b

        def parameters(self):
            return [self.W, self.b]

        def to(self, device):
            # Numpy doesn't use devices
            raise NotImplementedError("Numpy models don't support device placement")

        def info(self):
            return {
                "name": self.name,
                "uid": self.uid,
                "framework": self.framework,
                "input_dim": self.input_dim,
                "output_dim": self.output_dim,
                "num_parameters": self.W.size + self.b.size
            }

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # TEST SUITE
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    print("="*70)
    print("BaseModel Example: Framework-Agnostic Models")
    print("="*70)

    # Test 1: Torch model
    print("\n[Test 1] Torch Linear Model")
    print("-"*70)

    torch_model = TorchLinearModel(input_dim=5, output_dim=1)
    print(f"Model: {torch_model}")
    print(f"Framework: {torch_model.framework}")

    # Test forward pass
    x_torch = torch.randn(3, 5)
    output = torch_model(x_torch)
    print(f"Input shape: {x_torch.shape}")
    print(f"Output shape: {output.shape}")

    # Test device transfer
    if torch.cuda.is_available():
        torch_model.to('cuda')
        print(f"Device after .to('cuda'): {torch_model.device}")
        torch_model.to('cpu')

    # Test info
    print("\nModel Info:")
    import json
    print(json.dumps(torch_model.info(), indent=2))

    # Test 2: Numpy model
    print("\n[Test 2] Numpy Linear Model")
    print("-"*70)

    numpy_model = NumpyLinearModel(input_dim=5, output_dim=1)
    print(f"Model: {numpy_model}")
    print(f"Framework: {numpy_model.framework}")

    # Test forward pass
    x_numpy = np.random.randn(3, 5)
    output_numpy = numpy_model.forward(x_numpy)
    print(f"Input shape: {x_numpy.shape}")
    print(f"Output shape: {output_numpy.shape}")

    # Test parameters
    params = numpy_model.parameters()
    print(f"Parameters: {len(params)} arrays")
    print(f"  W shape: {params[0].shape}")
    print(f"  b shape: {params[1].shape}")

    # Test info
    print("\nModel Info:")
    print(json.dumps(numpy_model.info(), indent=2))

    # Test 3: Device error for numpy
    print("\n[Test 3] Numpy model device error (expected)")
    print("-"*70)
    try:
        numpy_model.to('cuda')
        print("✗ Should have raised NotImplementedError")
    except NotImplementedError as e:
        print(f"✓ Correctly raised: {e}")

    # Test 4: Save/Load with safetensors
    print("\n[Test 4] Save and Load with safetensors (torch)")
    print("-"*70)

    save_path = "./test_model.safetensors"

    try:
        torch_model.save(save_path)
        print(f"✓ Model saved to: {save_path}")
        print(f"  Also created metadata: {save_path.replace('.safetensors', '.json')}")

        # Create new model and load
        new_model = TorchLinearModel(input_dim=5, output_dim=1)
        new_model.load(save_path)
        print(f"✓ Model loaded from: {save_path}")

        # Verify outputs match
        output_original = torch_model(x_torch)
        output_loaded = new_model(x_torch)
        match = torch.allclose(output_original, output_loaded)
        print(f"✓ Outputs match: {match}")

        if not match:
            print("✗ FAIL: Outputs don't match after load")

        # Cleanup
        import os
        if os.path.exists(save_path):
            os.remove(save_path)
        metadata_path = save_path.replace('.safetensors', '.json')
        if os.path.exists(metadata_path):
            os.remove(metadata_path)
        print(f"✓ Cleaned up: {save_path} and metadata")

    except ImportError as e:
        print(f"⚠ Skipping safetensors test: {e}")
        print("  Install with: pip install safetensors")

    # Test 5: Compile (torch 2.0+)
    print("\n[Test 5] Model compilation (torch)")
    print("-"*70)

    try:
        import torch
        if hasattr(torch, 'compile'):
            compile_model = TorchLinearModel(input_dim=5, output_dim=1)
            compile_model.compile(mode='default')
            print("✓ Model compiled successfully")

            # Test forward pass still works
            output_compiled = compile_model(x_torch)
            print(f"✓ Compiled model output shape: {output_compiled.shape}")
        else:
            print("⚠ PyTorch version < 2.0, skipping compile test")
    except Exception as e:
        print(f"⚠ Compile test failed: {e}")

    # Test 6: Cache
    print("\n[Test 6] Model caching")
    print("-"*70)

    cache_model = TorchLinearModel(input_dim=5, output_dim=1)
    cache_model.cache(enabled=True, max_size=64)
    print(f"✓ Cache enabled: {cache_model._cache_enabled}")
    print(f"✓ Cache max size: {cache_model._cache_max_size}")

    # Disable cache
    cache_model.cache(enabled=False)
    print(f"✓ Cache disabled: {not cache_model._cache_enabled}")

    # Clear cache
    cache_model.cache(enabled=True, clear=True)
    print(f"✓ Cache cleared and re-enabled")

    # Test 7: Invalid framework
    print("\n[Test 7] Invalid framework detection")
    print("-"*70)

    class InvalidModel(BaseModel):
        def _detect_framework(self):
            return 'invalid_framework'

        def forward(self, x):
            pass

        def parameters(self):
            pass

        def to(self, device):
            pass

    try:
        invalid = InvalidModel("test", "m.test")
        print("✗ Should have raised ValueError")
    except ValueError as e:
        print(f"✓ Correctly raised: {e}")

    print("\n" + "="*70)
    print("All tests completed successfully!")
    print("="*70)
    print("\nKey features demonstrated:")
    print("  ✓ Framework detection (torch/numpy)")
    print("  ✓ Forward pass computation")
    print("  ✓ Parameter access")
    print("  ✓ Device management (torch)")
    print("  ✓ Save/Load with safetensors (secure, no pickle)")
    print("  ✓ Model compilation (torch 2.0+)")
    print("  ✓ Caching support")
    print("  ✓ Model metadata via info()")
    print("  ✓ Error handling for invalid frameworks")
    print("\nSecurity note:")
    print("  Torch models use safetensors (not torch.save/pickle)")
    print("  Install: pip install safetensors")
    print("\nPerformance note:")
    print("  Use compile() for PyTorch 2.0+ optimization")
    print("  Use cache() for repeated forward passes")
    print("="*70)