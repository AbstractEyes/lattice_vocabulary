"""
SD1.5 Feature Extraction + Symbolic Caption Synthesis (Colab-Ready)
====================================================================
Connects SymbolicCaptionSynthesizer to SD15FeatureExtractor.

File: geovocab2/data/teacher/sd15_caption_integration.py
"""

import torch
from typing import Optional, Dict, List, Iterator
import numpy as np

from geovocab2.data.prompt.synthesis_tree import SynthesisSystem
from geovocab2.data.teacher.extract_sd15 import SD15FeatureExtractor, SD15ExtractionConfig


class BatchedCaptionGenerator:
    """Generates caption batches for extraction pipeline."""

    def __init__(
        self,
        synthesizer: SynthesisSystem,
        batch_size: int = 32,
        num_batches: Optional[int] = None,
        complexity_distribution: Optional[Dict[int, float]] = None,
        seed: Optional[int] = None
    ):
        self.synthesizer = synthesizer
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.complexity_dist = complexity_distribution
        self.rng = np.random.RandomState(seed) if seed else np.random.RandomState()
        self.batch_count = 0

    def __iter__(self) -> Iterator[List[str]]:
        while True:
            if self.num_batches and self.batch_count >= self.num_batches:
                break

            batch = []
            for _ in range(self.batch_size):
                if self.complexity_dist:
                    complexities = list(self.complexity_dist.keys())
                    probs = list(self.complexity_dist.values())
                    complexity = self.rng.choice(complexities, p=probs)
                else:
                    complexity = 3  # Default complexity

                result = self.synthesizer.synthesize(complexity=complexity)
                caption = result["text"]
                batch.append(caption)

            self.batch_count += 1
            yield batch

    def __len__(self):
        return self.num_batches if self.num_batches else float('inf')


class SimpleDataLoader:
    """DataLoader wrapper for BatchedCaptionGenerator."""
    def __init__(self, generator):
        self.generator = generator
    def __iter__(self):
        return iter(self.generator)
    def __len__(self):
        return len(self.generator)


def extract_sd15_with_symbolic_captions(
    num_samples: int = 100_000,
    batch_size: int = 32,
    hf_repo_id: Optional[str] = None,
    upload_interval: int = 10_000,
    checkpoint_interval: int = 50_000,
    device: str = "cuda",
    bias_weights_path: Optional[str] = None,
    complexity_distribution: Optional[Dict[int, float]] = None,
    seed: Optional[int] = None,
    extract_clip: bool = True,
    clip_pooled: bool = True,
    active_blocks: Optional[List[str]] = None
):
    """
    Extract SD1.5 features with symbolic captions.

    Args:
        num_samples: Total samples
        batch_size: Batch size
        hf_repo_id: HuggingFace repo (None = no upload)
        upload_interval: Upload every N samples
        checkpoint_interval: Checkpoint every N samples
        device: cuda/cpu
        bias_weights_path: Path to bias_weights.json
        complexity_distribution: {1: 0.1, 2: 0.3, 3: 0.4, 4: 0.2}
        seed: Random seed
        extract_clip: Extract CLIP embeddings
        clip_pooled: Extract CLIP pooled
        active_blocks: UNet blocks (None = defaults)

    Returns:
        extractor: SD15FeatureExtractor with results
    """

    # Load bias weights
    bias_weights = None
    if bias_weights_path:
        import json
        with open(bias_weights_path, 'r') as f:
            bias_weights = json.load(f)
        print(f"✓ Loaded {len(bias_weights)} bias weights")

    # Create synthesizer
    print("Initializing synthesizer...")
    synthesizer = SynthesisSystem()
    if bias_weights:
        # Note: apply_bias_weights may need implementation in SynthesisSystem
        if hasattr(synthesizer, 'apply_bias_weights'):
            synthesizer.apply_bias_weights(bias_weights)
        else:
            print("⚠ Bias weights provided but apply_bias_weights() not implemented")
    print("✓ Synthesizer ready")

    # Create generator
    caption_generator = BatchedCaptionGenerator(
        synthesizer=synthesizer,
        batch_size=batch_size,
        num_batches=num_samples // batch_size,
        complexity_distribution=complexity_distribution,
        seed=seed
    )
    dataloader = SimpleDataLoader(caption_generator)

    # Extraction config
    extraction_config = SD15ExtractionConfig(
        active_blocks=active_blocks or [
            'down_blocks.0.resnets.1',
            'down_blocks.1.resnets.1',
            'mid_block.resnets.1',
            'up_blocks.0.resnets.2',
            'up_blocks.1.resnets.2',
        ],
        extract_clip_embeddings=extract_clip,
        clip_pooled=clip_pooled,
        hf_repo_id=hf_repo_id,
        upload_interval=upload_interval,
        checkpoint_interval=checkpoint_interval,
        max_samples=num_samples
    )

    # Create extractor
    print(f"Initializing SD15 on {device}...")
    extractor = SD15FeatureExtractor(config=extraction_config, device=device)
    print("✓ Extractor ready\n")

    print(f"Config: {num_samples:,} samples, batch_size={batch_size}")
    if hf_repo_id:
        print(f"Upload: {hf_repo_id} (every {upload_interval:,})")
    print()

    # Extract
    extractor.extract_from_dataloader(dataloader, max_samples=num_samples)

    print("\n✓ Complete!")
    if hf_repo_id:
        print(f"https://huggingface.co/datasets/{hf_repo_id}")

    return extractor


if __name__ == "__main__":
    print("SD1.5 + Symbolic Captions Integration")
    print("Usage: extract_sd15_with_symbolic_captions(num_samples=100_000, hf_repo_id='user/dataset')")