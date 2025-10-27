"""
Concise seed generator for deterministic random operations using numpy's modern RNG.

Author: Phi + Claude Sonnet 4.5
Date: 2025-10-27
"""

import numpy as np
from typing import Optional


class SeedGen:
    """Lightweight deterministic seed generator using numpy's modern RNG."""

    def __init__(self, seed: Optional[int] = None):
        """Initialize with optional seed. If None, uses system entropy."""
        self.rng = np.random.default_rng(seed)
        self._initial_seed = seed

    def __call__(self) -> int:
        """Generate next seed in sequence."""
        return int(self.rng.integers(0, 2 ** 31))

    def fork(self) -> 'SeedGen':
        """Create independent child generator."""
        return SeedGen(self())

    @property
    def initial_seed(self) -> Optional[int]:
        """Return the initial seed used (None if non-deterministic)."""
        return self._initial_seed

    def choice(self, seq):
        """Convenience: random choice from sequence."""
        return self.rng.choice(seq)

    def sample(self, seq, k):
        """Convenience: random sample from sequence."""
        return self.rng.choice(seq, size=k, replace=False).tolist()

    def random(self) -> float:
        """Convenience: random float [0, 1)."""
        return float(self.rng.random())

    def randint(self, low, high) -> int:
        """Convenience: random integer [low, high]."""
        return int(self.rng.integers(low, high + 1))

    def shuffle(self, seq):
        """Convenience: in-place shuffle."""
        self.rng.shuffle(seq)
        return seq


# Example usage patterns:
if __name__ == "__main__":
    # Deterministic mode
    sg = SeedGen(42)
    print(f"Initial seed: {sg.initial_seed}")
    print(f"Generated seeds: {[sg() for _ in range(5)]}")
    print(f"Random choices: {[sg.choice(['a', 'b', 'c']) for _ in range(5)]}")

    # Fork for independent operations
    child_sg = sg.fork()
    print(f"\nForked generator:")
    print(f"Parent: {[sg() for _ in range(3)]}")
    print(f"Child:  {[child_sg() for _ in range(3)]}")

    # Non-deterministic mode
    sg_random = SeedGen()
    print(f"\nNon-deterministic seed: {sg_random.initial_seed}")
    print(f"Random values: {[sg_random() for _ in range(3)]}")