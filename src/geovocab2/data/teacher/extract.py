# ModelExtractionBase
# License: MIT
# Date: 10/23/2025
# Author: AbstractPhil
# Description: Base class for model extraction methodologies
from datasets import Dataset
from abc import ABC, abstractmethod
from typing import Optional, Dict, List, Any, Callable, Union, Literal
from dataclasses import dataclass, field
from enum import Enum
import torch
import torch.nn as nn


class CaptureMode(Enum):
    """Defines what data to capture from hooks"""
    OUTPUTS_ONLY = "outputs_only"
    INPUTS_ONLY = "inputs_only"
    BOTH = "both"
    NONE = "none"


@dataclass
class HookConfig:
    """Configuration for individual layer hook behavior"""
    capture_mode: CaptureMode = CaptureMode.OUTPUTS_ONLY
    detach: bool = True  # Detach from computation graph
    clone: bool = False  # Clone tensors (memory vs safety trade-off)
    to_cpu: bool = False  # Move to CPU (memory management)
    capture_gradients: bool = False  # Store gradient info
    max_captures: Optional[int] = None  # Limit captures per layer

    # Fine-grained control for complex outputs
    tuple_index: Optional[int] = None  # Extract specific index from tuple
    dict_keys: Optional[List[str]] = None  # Extract specific keys from dict


@dataclass
class CacheSchema:
    """Schema for cached data structure"""
    version: str = "1.0.0"
    extraction_type: str = "features"  # features, activations, gradients, etc.
    layer_configs: Dict[str, HookConfig] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize cache schema"""
        return {
            'version': self.version,
            'extraction_type': self.extraction_type,
            'layer_configs': {
                k: {
                    'capture_mode': v.capture_mode.value,
                    'detach': v.detach,
                    'clone': v.clone,
                    'to_cpu': v.to_cpu,
                    'capture_gradients': v.capture_gradients,
                    'max_captures': v.max_captures,
                    'tuple_index': v.tuple_index,
                    'dict_keys': v.dict_keys
                } for k, v in self.layer_configs.items()
            },
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CacheSchema':
        """Deserialize cache schema"""
        layer_configs = {}
        for k, v in data.get('layer_configs', {}).items():
            layer_configs[k] = HookConfig(
                capture_mode=CaptureMode(v['capture_mode']),
                detach=v['detach'],
                clone=v['clone'],
                to_cpu=v['to_cpu'],
                capture_gradients=v['capture_gradients'],
                max_captures=v.get('max_captures'),
                tuple_index=v.get('tuple_index'),
                dict_keys=v.get('dict_keys')
            )

        return cls(
            version=data.get('version', '1.0.0'),
            extraction_type=data.get('extraction_type', 'features'),
            layer_configs=layer_configs,
            metadata=data.get('metadata', {})
        )

from geovocab2.train.config.config_base import BaseConfig


class ExtractionSchema(BaseConfig):
    """Configuration schema for model extraction"""
    name: str = "extraction_schema"
    uid: str = "c.extraction.schema"
    model_type: str = "features"
    model_name: str = "default_model"
    batch_size: int = 32
    num_samples: int = 1000
    use_cache: bool = True
    target_layers: List[str] = field(default_factory=list)  # Layer names to hook

    # Hook configuration
    default_hook_config: HookConfig = field(default_factory=HookConfig)
    layer_hook_configs: Dict[str, HookConfig] = field(default_factory=dict)  # Per-layer overrides

    # Cache configuration
    cache_schema: CacheSchema = field(default_factory=CacheSchema)


class ModelExtractionBase(ABC):
    """Base class for feature extraction from neural network models"""

    def __init__(
            self,
            name: str,
            u_id: str,
            model: Optional[nn.Module] = None,
            cache_dataset: Optional[Dataset] = None,
            config: Optional[ExtractionSchema] = None
    ):
        self.name = name
        self.id = u_id
        self.model = model
        self.cache_dataset = cache_dataset
        self.config = config or ExtractionSchema()

        # Hook management
        self._hook_handles: List[Any] = []
        self._hook_data: Dict[str, List[Dict[str, Any]]] = {}  # stores {input, output, metadata}

    @abstractmethod
    def forward(self, *args, **kwargs) -> Any:
        """Main extraction logic - must be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement forward()")

    @abstractmethod
    def save_cache(self, data: Any) -> None:
        """Save extracted data to cache - must be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement save_cache()")

    @abstractmethod
    def load_cache(self) -> Any:
        """Load cached data - must be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement load_cache()")

    def register_hooks(self, layer_names: Optional[List[str]] = None) -> None:
        """
        Attach forward hooks to specified model layers

        Args:
            layer_names: List of layer names to hook. Uses config if not provided.

        Raises:
            ValueError: If model is None, no layers specified, or layers don't exist
        """
        if self.model is None:
            raise ValueError("Model must be set before registering hooks")

        target_layers = layer_names or self.config.target_layers
        if not target_layers:
            raise ValueError("No target layers specified for hooking")

        # Validate layer names exist
        available_layers = {name for name, _ in self.model.named_modules()}
        invalid_layers = set(target_layers) - available_layers
        if invalid_layers:
            raise ValueError(
                f"Layer(s) not found in model: {invalid_layers}. "
                f"Available layers: {sorted(available_layers)}"
            )

        # Clear existing hooks
        self.remove_hooks()

        # Register new hooks and store configs
        for name, module in self.model.named_modules():
            if name in target_layers:
                # Get config for this layer
                hook_config = self.config.layer_hook_configs.get(
                    name,
                    self.config.default_hook_config
                )

                # Store in cache schema for serialization
                self.config.cache_schema.layer_configs[name] = hook_config

                # Register hook
                handle = module.register_forward_hook(
                    self._create_hook_fn(name)
                )
                self._hook_handles.append(handle)
                self._hook_data[name] = []

    def _create_hook_fn(self, layer_name: str) -> Callable:
        """
        Create a configurable hook function for a specific layer.
        Respects layer-specific config or falls back to default config.
        """
        # Get config for this layer (layer-specific or default)
        hook_config = self.config.layer_hook_configs.get(
            layer_name,
            self.config.default_hook_config
        )

        def hook_fn(module: nn.Module, input: Any, output: Any) -> None:
            # Check if we've hit max captures for this layer
            if hook_config.max_captures is not None:
                if len(self._hook_data[layer_name]) >= hook_config.max_captures:
                    return

            capture_data = {}

            # Capture based on mode
            if hook_config.capture_mode in (CaptureMode.INPUTS_ONLY, CaptureMode.BOTH):
                capture_data['input'] = self._process_hook_data(
                    input, hook_config
                )

            if hook_config.capture_mode in (CaptureMode.OUTPUTS_ONLY, CaptureMode.BOTH):
                capture_data['output'] = self._process_hook_data(
                    output, hook_config
                )

            if hook_config.capture_mode != CaptureMode.NONE:
                capture_data['module_name'] = layer_name
                self._hook_data[layer_name].append(capture_data)

        return hook_fn

    def _process_hook_data(self, data: Any, config: HookConfig) -> Any:
        """
        Process captured data according to HookConfig settings.
        Handles tensors, tuples, dicts, and applies transformations.
        """
        if data is None:
            return None

        # Handle tuple outputs with indexing
        if isinstance(data, tuple):
            if config.tuple_index is not None:
                if config.tuple_index < len(data):
                    data = data[config.tuple_index]
                else:
                    return None  # Index out of range
            # Otherwise keep full tuple

        # Handle dict outputs with key filtering
        if isinstance(data, dict):
            if config.dict_keys is not None:
                data = {k: v for k, v in data.items() if k in config.dict_keys}
            # Otherwise keep full dict

        # Process tensors (recursively handle nested structures)
        return self._process_tensor_data(data, config)

    def _process_tensor_data(self, data: Any, config: HookConfig) -> Any:
        """
        Apply tensor transformations (detach, clone, cpu) recursively.
        """
        if isinstance(data, torch.Tensor):
            tensor = data

            if config.detach and tensor.requires_grad:
                tensor = tensor.detach()

            if config.clone:
                tensor = tensor.clone()

            if config.to_cpu and tensor.device.type != 'cpu':
                tensor = tensor.cpu()

            return tensor

        elif isinstance(data, tuple):
            return tuple(self._process_tensor_data(item, config) for item in data)

        elif isinstance(data, list):
            return [self._process_tensor_data(item, config) for item in data]

        elif isinstance(data, dict):
            return {k: self._process_tensor_data(v, config) for k, v in data.items()}

        else:
            # Non-tensor data, return as-is
            return data

    def remove_hooks(self) -> None:
        """Remove all registered hooks"""
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()

    def clear_hook_data(self) -> None:
        """Clear all stored hook data"""
        for key in self._hook_data:
            self._hook_data[key].clear()

    def get_hook_data(self, layer_name: Optional[str] = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get stored hook data

        Args:
            layer_name: Specific layer to retrieve. Returns all if None.

        Returns:
            Dictionary mapping layer names to list of hook captures
            Each capture contains: {'input': ..., 'output': ..., 'module': ...}
        """
        if layer_name:
            return {layer_name: self._hook_data.get(layer_name, [])}
        return self._hook_data.copy()

    def list_available_layers(self) -> List[str]:
        """Return list of all available layer names in the model"""
        if self.model is None:
            raise ValueError("Model must be set before listing layers")
        return [name for name, _ in self.model.named_modules() if name]

    def get_cache_schema(self) -> CacheSchema:
        """Get the current cache schema with all hook configurations"""
        return self.config.cache_schema

    def set_layer_hook_config(self, layer_name: str, hook_config: HookConfig) -> None:
        """
        Set hook configuration for a specific layer.
        Must be called before register_hooks() to take effect.
        """
        self.config.layer_hook_configs[layer_name] = hook_config

    def get_hook_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics about captured hook data"""
        stats = {}
        for layer_name, captures in self._hook_data.items():
            stats[layer_name] = {
                'num_captures': len(captures),
                'has_inputs': any('input' in c for c in captures),
                'has_outputs': any('output' in c for c in captures),
                'memory_est_mb': self._estimate_memory(captures)
            }
        return stats

    def _estimate_memory(self, captures: List[Dict[str, Any]]) -> float:
        """Rough memory estimate for captured data in MB"""
        total_bytes = 0
        for capture in captures:
            for key, value in capture.items():
                if isinstance(value, torch.Tensor):
                    total_bytes += value.element_size() * value.nelement()
        return total_bytes / (1024 * 1024)

    @staticmethod
    def create_first_layer_config() -> HookConfig:
        """
        Preset config for first layers - capture inputs since they're meaningful.
        Useful for input preprocessing analysis.
        """
        return HookConfig(
            capture_mode=CaptureMode.BOTH,
            detach=True,
            clone=False,
            to_cpu=False
        )

    @staticmethod
    def create_intermediate_config() -> HookConfig:
        """
        Preset config for intermediate layers - outputs only, no unnecessary data.
        Most common use case for feature extraction.
        """
        return HookConfig(
            capture_mode=CaptureMode.OUTPUTS_ONLY,
            detach=True,
            clone=False,
            to_cpu=False
        )

    @staticmethod
    def create_memory_efficient_config() -> HookConfig:
        """
        Preset config for memory-constrained scenarios.
        Moves to CPU, limits captures.
        """
        return HookConfig(
            capture_mode=CaptureMode.OUTPUTS_ONLY,
            detach=True,
            clone=False,
            to_cpu=True,
            max_captures=100
        )

    def __enter__(self):
        """Context manager support for automatic hook cleanup"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Ensure hooks are removed on exit"""
        self.remove_hooks()