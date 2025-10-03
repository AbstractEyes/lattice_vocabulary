"""
FactoryDataset + ProcessingPipeline
-----------------------------------
Modular data generation system with pluggable processing stages.

Architecture:
    FactoryDataset → DataLoader → PipelineStage(s) → Output

- FactoryDataset: Shape generation (pluggable Factory)
- PipelineStage: Formula application (pluggable processors)
- Collate/Transform: Batch formatting (pluggable collators)
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Callable
from torch.utils.data import Dataset, DataLoader
import numpy as np

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Abstract Pipeline Stage (Pluggable Processors)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class PipelineStage(ABC):
    """
    Abstract processing stage for factory outputs.

    Subclasses implement process() to transform batches.
    Examples: FormulaStage, ValidationStage, AugmentationStage
    """

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def process(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a batch from DataLoader.

        Args:
            batch: Dictionary with 'data' key (and optionally metadata)

        Returns:
            Transformed batch dictionary
        """
        pass

    def __call__(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        return self.process(batch)

    def __repr__(self):
        return f"PipelineStage[{self.name}]"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Factory Dataset (Pluggable Factory Backend)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class FactoryDataset(Dataset):
    """
    Dataset that generates samples on-demand using a FactoryBase instance.

    Args:
        factory: FactoryBase subclass instance
        count: Number of samples in dataset
        seed_offset: Starting seed for deterministic generation
        backend: 'numpy' or 'torch' (workers are CPU-only)
        **factory_kwargs: Passed to factory.build()
    """

    def __init__(
            self,
            factory,  # FactoryBase instance
            count: int,
            seed_offset: int = 0,
            backend: str = "numpy",
            **factory_kwargs
    ):
        self.factory = factory
        self.count = count
        self.seed_offset = seed_offset
        self.backend = backend
        self.factory_kwargs = factory_kwargs

    def __len__(self):
        return self.count

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Generate single sample with metadata."""
        seed = self.seed_offset + idx

        # Build using factory (CPU-only for multiprocessing compatibility)
        if self.backend == "numpy":
            data = self.factory.build_numpy(
                seed=seed,
                **self.factory_kwargs
            )
        elif self.backend == "torch":
            data = self.factory.build_torch(
                device="cpu",
                seed=seed,
                **self.factory_kwargs
            )
        else:
            raise ValueError(f"Invalid backend: {self.backend}")

        return {
            'data': data,
            'idx': idx,
            'seed': seed,
            'factory_name': self.factory.name
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Pluggable Collate Functions
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class CollateBase(ABC):
    """Abstract collate function for DataLoader."""

    @abstractmethod
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        pass


class StackCollate(CollateBase):
    """Stack samples into batched arrays/tensors."""

    def __init__(self, to_torch: bool = False, device: str = "cpu"):
        self.to_torch = to_torch
        self.device = device

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Stack data
        data_items = [item['data'] for item in batch]

        if isinstance(data_items[0], np.ndarray):
            stacked_data = np.stack(data_items, axis=0)

            if self.to_torch and HAS_TORCH:
                stacked_data = torch.from_numpy(stacked_data).to(self.device)

        elif HAS_TORCH and isinstance(data_items[0], torch.Tensor):
            stacked_data = torch.stack(data_items, dim=0).to(self.device)

        else:
            # Fallback: keep as list
            stacked_data = data_items

        # Stack metadata
        return {
            'data': stacked_data,
            'indices': np.array([item['idx'] for item in batch]),
            'seeds': np.array([item['seed'] for item in batch]),
            'factory_name': batch[0]['factory_name']
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Processing Pipeline (Chain of Stages)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ProcessingPipeline:
    """
    Chain of PipelineStage processors.

    Usage:
        pipeline = ProcessingPipeline([
            FormulaStage("cayley_menger"),
            ValidationStage("capacity_check"),
            ArrowWriteStage("output.parquet")
        ])

        for batch in dataloader:
            result = pipeline(batch)
    """

    def __init__(self, stages: Optional[List[PipelineStage]] = None):
        self.stages = stages or []

    def add_stage(self, stage: PipelineStage):
        """Add stage to pipeline."""
        self.stages.append(stage)
        return self

    def __call__(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Process batch through all stages."""
        for stage in self.stages:
            batch = stage(batch)
        return batch

    def __repr__(self):
        stage_names = [s.name for s in self.stages]
        return f"ProcessingPipeline({' → '.join(stage_names)})"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Empty Pipeline Stages (Templates for Specialization)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class IdentityStage(PipelineStage):
    """Pass-through stage (no transformation)."""

    def __init__(self):
        super().__init__("identity")

    def process(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        return batch


class FormulaStage(PipelineStage):
    """
    Apply formula(s) to batch data.

    TEMPLATE: Subclass and override process() with specific formulas.
    """

    def __init__(self, name: str = "formula"):
        super().__init__(name)

    def process(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        OVERRIDE THIS METHOD.

        Example:
            data = batch['data']  # Shape: (B, N, D)
            volumes = compute_simplex_volume(data)
            batch['volumes'] = volumes
            return batch
        """
        # Default: pass-through
        return batch


class ValidationStage(PipelineStage):
    """
    Validate batch data.

    TEMPLATE: Subclass and override process() with validation logic.
    """

    def __init__(self, name: str = "validation"):
        super().__init__(name)

    def process(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        OVERRIDE THIS METHOD.

        Example:
            data = batch['data']
            if not is_valid(data):
                raise ValueError("Invalid batch")
            return batch
        """
        # Default: accept everything
        return batch


class CacheStage(PipelineStage):
    """
    Cache batch results.

    TEMPLATE: Subclass and override with cache backend (Arrow, HDF5, etc).
    """

    def __init__(self, name: str = "cache"):
        super().__init__(name)
        self.cache = {}  # In-memory placeholder

    def process(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        OVERRIDE THIS METHOD.

        Example:
            for idx in batch['indices']:
                self.cache[idx] = batch['data'][idx]
            return batch
        """
        # Default: no-op
        return batch


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Example Usage
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    from shapes.factory.factory_base import TensorFactory  # Assuming FactoryBase is imported

    print("=" * 70)
    print("FACTORY DATALOADER PIPELINE DEMONSTRATION")
    print("=" * 70)

    # 1. Create factory
    factory = TensorFactory(init_mode="randn")

    # 2. Create dataset
    dataset = FactoryDataset(
        factory=factory,
        count=100,
        shape=(3, 4),
        backend="numpy"
    )

    # 3. Create dataloader
    collate = StackCollate(to_torch=False)
    loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
        num_workers=2,
        collate_fn=collate
    )

    # 4. Create processing pipeline (empty stages for now)
    pipeline = ProcessingPipeline([
        IdentityStage(),
        FormulaStage("placeholder_formula"),
        ValidationStage("placeholder_validation")
    ])

    # 5. Process batches
    print("\n[Processing batches through pipeline]")
    for i, batch in enumerate(loader):
        processed = pipeline(batch)

        print(f"\nBatch {i}:")
        print(f"  Data shape: {processed['data'].shape}")
        print(f"  Indices: {processed['indices'][:3]}...")
        print(f"  Factory: {processed['factory_name']}")

        if i >= 2:  # Show only first 3 batches
            break

    print("\n" + "=" * 70)
    print("Pipeline ready for specialization:")
    print("  - Subclass FormulaStage for Cayley-Menger/capacity")
    print("  - Subclass CacheStage for Arrow/Parquet writes")
    print("  - Subclass ValidationStage for shape validation")
    print("=" * 70)