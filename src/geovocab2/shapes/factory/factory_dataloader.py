"""
FactoryDataset + ProcessingPipeline
-----------------------------------
Modular data generation system with pluggable processing stages.

Architecture:
    FactoryDataset → DataLoader → PipelineStage(s) → Output

- FactoryDataset: Shape generation using any FactoryBase instance
- PipelineStage: Formula application (pluggable processors)
- Collate: Batch formatting with error handling

Author: AbstractPhil + Claude Sonnet 4.5
License: MIT
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from torch.utils.data import Dataset, DataLoader
import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Factory Dataset
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class FactoryDataset(Dataset):
    """
    Dataset generating samples on-demand using FactoryBase instances.

    Compatible with any FactoryBase subclass. Generates samples using
    the unified build() interface with error handling for robust
    multiprocessing.

    Args:
        factory: FactoryBase instance
        count: Number of samples in dataset
        factory_args: Positional arguments for factory.build()
        factory_kwargs: Keyword arguments for factory.build()
        seed_offset: Starting seed for deterministic generation
        backend: 'numpy' or 'torch'
        device: Device for torch backend (workers are CPU-only)

    Example:
        from geovocab2.shapes.factory import SimplexFactory
        factory = SimplexFactory()
        dataset = FactoryDataset(
            factory=factory,
            count=1000,
            factory_kwargs={'n_points': 4, 'dimension': 3},
            backend="numpy"
        )
    """

    def __init__(
        self,
        factory,  # FactoryBase instance
        count: int,
        factory_args: tuple = (),
        factory_kwargs: Optional[Dict[str, Any]] = None,
        seed_offset: int = 0,
        backend: str = "numpy",
        device: str = "cpu"
    ):
        self.factory = factory
        self.count = count
        self.factory_args = factory_args
        self.factory_kwargs = factory_kwargs or {}
        self.seed_offset = seed_offset
        self.backend = backend
        # DataLoader workers are CPU-only, device transfer happens in collate
        self.device = "cpu" if backend == "torch" else None

    def __len__(self):
        return self.count

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Generate single sample with metadata.

        Returns dict with:
            - data: Generated sample (or None if failed)
            - idx: Sample index
            - seed: Random seed used
            - factory_uid: Factory identifier
            - success: Whether generation succeeded
            - error: Error message if failed
        """
        # Inject seed if not already provided
        kwargs = self.factory_kwargs.copy()
        if 'seed' not in kwargs:
            kwargs['seed'] = self.seed_offset + idx

        try:
            # Use unified build() interface from FactoryBase
            data = self.factory.build(
                *self.factory_args,
                backend=self.backend,
                device=self.device,
                validate=False,  # Defer validation to pipeline stage
                **kwargs
            )

            return {
                'data': data,
                'idx': idx,
                'seed': kwargs.get('seed'),
                'factory_uid': self.factory.uid,
                'success': True,
                'error': None
            }

        except Exception as e:
            # Return error marker instead of crashing worker
            return {
                'data': None,
                'idx': idx,
                'seed': kwargs.get('seed'),
                'factory_uid': self.factory.uid,
                'success': False,
                'error': str(e)
            }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Collate Functions
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def default_collate(
    batch: List[Dict[str, Any]],
    device: Optional[str] = None
) -> Dict[str, Any]:
    """
    Default collate function with error handling.

    Filters failed samples and stacks successful ones into batches.
    Transfers to target device if specified.

    Args:
        batch: List of sample dicts from FactoryDataset
        device: Target device for torch tensors (None = no transfer)

    Returns:
        Batch dict with:
            - data: Stacked samples [batch_size, ...]
            - indices: Sample indices
            - seeds: Random seeds used
            - factory_uid: Factory identifier
            - batch_size: Number of successful samples
            - failures: Number of failed samples
            - error_messages: List of error messages (if any)

    Raises:
        RuntimeError: If all samples failed generation
    """
    # Separate successes from failures
    successes = [item for item in batch if item['success']]
    failures = [item for item in batch if not item['success']]

    if not successes:
        error_msgs = [item['error'] for item in failures]
        raise RuntimeError(
            f"All {len(batch)} samples failed generation. "
            f"Errors: {error_msgs}"
        )

    # Stack successful samples
    data_items = [item['data'] for item in successes]

    if isinstance(data_items[0], np.ndarray):
        stacked_data = np.stack(data_items, axis=0)

        # Convert to torch if device specified
        if device is not None and HAS_TORCH:
            stacked_data = torch.from_numpy(stacked_data).to(device)

    elif HAS_TORCH and isinstance(data_items[0], torch.Tensor):
        stacked_data = torch.stack(data_items, dim=0)

        # Transfer to device if specified
        if device is not None:
            stacked_data = stacked_data.to(device)
    else:
        # Keep as list if unknown type
        stacked_data = data_items

    return {
        'data': stacked_data,
        'indices': np.array([item['idx'] for item in successes]),
        'seeds': np.array([item['seed'] for item in successes]),
        'factory_uid': successes[0]['factory_uid'],
        'batch_size': len(successes),
        'failures': len(failures),
        'error_messages': [item['error'] for item in failures] if failures else []
    }


def device_collate(device: str):
    """
    Factory function for device-specific collate.

    Args:
        device: Target device ('cuda:0', 'cpu', etc.)

    Returns:
        Collate function that transfers to device

    Example:
        loader = DataLoader(
            dataset,
            batch_size=32,
            collate_fn=device_collate('cuda:0')
        )
    """
    return lambda batch: default_collate(batch, device=device)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Pipeline Stages (Pluggable Processors)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class PipelineStage(ABC):
    """
    Abstract processing stage for batch transformation.

    Subclasses implement process() to transform batches after
    collation. Examples: formula application, validation,
    augmentation, caching.

    Args:
        name: Stage identifier
    """

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def process(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process batch from DataLoader.

        Args:
            batch: Dictionary with 'data' key and metadata

        Returns:
            Transformed batch dictionary
        """
        pass

    def __call__(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        return self.process(batch)

    def __repr__(self):
        return f"{self.__class__.__name__}[{self.name}]"


class IdentityStage(PipelineStage):
    """Pass-through stage (no transformation)."""

    def __init__(self):
        super().__init__("identity")

    def process(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        return batch


class LoggingStage(PipelineStage):
    """Log batch statistics."""

    def __init__(self, verbose: bool = True):
        super().__init__("logging")
        self.verbose = verbose
        self.batch_count = 0

    def process(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        self.batch_count += 1

        if self.verbose:
            print(f"Batch {self.batch_count}:")
            print(f"  Size: {batch['batch_size']}")
            print(f"  Failures: {batch['failures']}")
            if batch['failures'] > 0:
                print(f"  Errors: {batch['error_messages'][:3]}")

        return batch


class ProcessingPipeline:
    """
    Chain of PipelineStage processors.

    Applies stages sequentially to batches from DataLoader.

    Example:
        pipeline = ProcessingPipeline([
            LoggingStage(),
            ValidationStage(),
            FormulaStage()
        ])
        for batch in dataloader:
            result = pipeline(batch)
    """

    def __init__(self, stages: Optional[List[PipelineStage]] = None):
        self.stages = stages or []

    def add_stage(self, stage: PipelineStage):
        """Add stage to end of pipeline."""
        self.stages.append(stage)
        return self

    def __call__(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Process batch through all stages."""
        for stage in self.stages:
            batch = stage(batch)
        return batch

    def __repr__(self):
        names = [s.name for s in self.stages]
        return f"Pipeline({' → '.join(names)})"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Example Usage
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    print("=" * 70)
    print("FACTORY DATALOADER DEMONSTRATION")
    print("=" * 70)

    # Mock factory for demonstration
    from geovocab2.shapes.factory.factory_base import TensorFactory

    # 1. Create factory
    factory = TensorFactory(init_mode="randn")

    # 2. Create dataset
    dataset = FactoryDataset(
        factory=factory,
        count=100,
        factory_kwargs={'shape': (3, 4)},
        backend="numpy",
        seed_offset=42
    )

    print(f"\nDataset created: {len(dataset)} samples")
    print(f"Factory: {factory.name}")

    # 3. Create dataloader
    loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
        num_workers=2,
        collate_fn=default_collate
    )

    # 4. Create processing pipeline
    pipeline = ProcessingPipeline([
        LoggingStage(verbose=True),
        IdentityStage()
    ])

    print(f"\nPipeline: {pipeline}")

    # 5. Process batches
    print("\n" + "-" * 70)
    print("Processing batches:")
    print("-" * 70)

    for i, batch in enumerate(loader):
        processed = pipeline(batch)

        if i >= 2:  # Show only first 3 batches
            print(f"\n... (showing first 3 of {len(loader)} batches)")
            break

    print("\n" + "=" * 70)
    print("FactoryDataset ready for use with any FactoryBase subclass")
    print("=" * 70)