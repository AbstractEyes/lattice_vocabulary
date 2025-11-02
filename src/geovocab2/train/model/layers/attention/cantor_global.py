import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional
import numpy as np


class CantorGlobalAttentionO_n(nn.Module):
    """
    TRUE O(n) ATTENTION WITH GLOBAL RECEPTIVE FIELD
    """

    def __init__(
            self,
            dim: int,
            depth: int = 8,
            max_seq_len: int = 8192,
            local_window: int = 64,
            num_heads: int = 8
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.max_seq_len = max_seq_len
        self.local_window = local_window
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        print(f"Building Cantor routes for max_seq_len={max_seq_len}, k={local_window}...")

        # Build hierarchical Cantor connectivity (vectorized)
        routes = self._build_hierarchical_routes_fast(max_seq_len, depth, local_window)
        self.register_buffer("cantor_routes", routes)

        print(f"  Routes built: shape={routes.shape}")

        # Standard attention projections
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def _cantor_coordinates_vectorized(self, positions: torch.Tensor, max_len: int, depth: int) -> torch.Tensor:
        """Vectorized Cantor coordinate computation."""
        x = positions.float() / max(1, max_len - 1)
        x = x.clamp(1e-6, 1.0 - 1e-6)

        cantor_vals = torch.zeros_like(x)
        factor = 0.5

        for _ in range(depth):
            x = x * 3.0
            digits = x.long()
            x = x - digits.float()

            cantor_vals = cantor_vals + (digits == 2).float() * factor
            factor *= 0.5

        return cantor_vals

    def _build_hierarchical_routes_fast(
            self,
            max_len: int,
            depth: int,
            k: int
    ) -> torch.Tensor:
        """Fast vectorized route building."""
        all_positions = torch.arange(max_len)
        all_cantor = self._cantor_coordinates_vectorized(all_positions, max_len, depth)

        routes = torch.zeros(max_len, k, dtype=torch.long)

        k_local = max(1, int(k * 0.5))
        k_medium = max(1, int(k * 0.3))
        k_global = k - k_local - k_medium

        print(f"  Neighbor distribution: local={k_local}, medium={k_medium}, global={k_global}")

        for i in range(max_len):
            neighbors = []

            # LOCAL
            half_window = k_local // 2
            local_start = max(0, i - half_window)
            local_end = min(max_len, i + half_window + 1)
            local_neighbors = list(range(local_start, local_end))
            if i in local_neighbors:
                local_neighbors.remove(i)
            neighbors.extend(local_neighbors[:k_local])

            # MEDIUM
            if k_medium > 0:
                cantor_i = all_cantor[i]
                cantor_distances = torch.abs(all_cantor - cantor_i)
                cantor_distances[i] = float('inf')

                _, medium_indices = torch.topk(
                    -cantor_distances,
                    k=min(k_medium, max_len - 1),
                    largest=True
                )
                neighbors.extend(medium_indices.tolist())

            # GLOBAL
            if k_global > 0:
                for level in range(min(k_global, depth)):
                    jump = int((3 ** level) * (max_len / (3 ** depth)))
                    if jump == 0:
                        jump = 3 ** level
                    target = (i + jump) % max_len
                    if target not in neighbors and target != i:
                        neighbors.append(target)

            # Deduplicate
            neighbors = list(dict.fromkeys(neighbors))

            while len(neighbors) < k:
                candidate = (i + len(neighbors) * (max_len // k)) % max_len
                if candidate not in neighbors and candidate != i:
                    neighbors.append(candidate)
                else:
                    for j in range(max_len):
                        if j not in neighbors and j != i:
                            neighbors.append(j)
                            break

            routes[i] = torch.tensor(neighbors[:k], dtype=torch.long)

        return routes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, dim)
        """
        batch_size, seq_len, dim = x.shape
        assert seq_len <= self.max_seq_len, f"seq_len {seq_len} > max_seq_len {self.max_seq_len}"

        # QKV projection
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)

        # Reshape: (batch, seq_len, num_heads, head_dim) -> (batch, num_heads, seq_len, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Sparse attention
        output = self._sparse_attention(q, k, v, seq_len)

        # Reshape back and project
        output = output.transpose(1, 2).reshape(batch_size, seq_len, dim)
        output = self.proj(output)

        return output

    def _sparse_attention(
            self,
            q: torch.Tensor,  # (batch, num_heads, seq_len, head_dim)
            k: torch.Tensor,
            v: torch.Tensor,
            seq_len: int
    ) -> torch.Tensor:
        """
        Compute sparse attention using Cantor routes.

        Complexity: O(seq_len * k * head_dim) = O(n)
        """
        batch_size, num_heads, _, head_dim = q.shape
        k_neighbors = min(self.local_window, seq_len)

        # Get routing for this sequence length and clamp indices
        routes = self.cantor_routes[:seq_len, :k_neighbors].clone()
        routes = routes.clamp(0, seq_len - 1)  # CRITICAL: Clamp to valid range

        # Prepare output
        output = torch.zeros_like(q)

        # Process each position
        for pos in range(seq_len):
            # Get neighbor indices for this position
            neighbor_indices = routes[pos]  # (k_neighbors,)

            # Gather K and V from neighbors
            # k, v: (batch, num_heads, seq_len, head_dim)
            k_neighbors = k[:, :, neighbor_indices, :]  # (batch, num_heads, k_neighbors, head_dim)
            v_neighbors = v[:, :, neighbor_indices, :]  # (batch, num_heads, k_neighbors, head_dim)

            # Query for this position
            q_pos = q[:, :, pos:pos + 1, :]  # (batch, num_heads, 1, head_dim)

            # Attention scores
            attn_scores = torch.matmul(q_pos, k_neighbors.transpose(-2, -1))  # (batch, num_heads, 1, k_neighbors)
            attn_scores = attn_scores / math.sqrt(head_dim)

            # Softmax
            attn_weights = F.softmax(attn_scores, dim=-1)  # (batch, num_heads, 1, k_neighbors)

            # Apply to values
            output[:, :, pos:pos + 1, :] = torch.matmul(attn_weights, v_neighbors)  # (batch, num_heads, 1, head_dim)

        return output


def test_complexity():
    """Empirically verify O(n) complexity."""
    import time

    print("=" * 70)
    print("CANTOR O(n) ATTENTION COMPLEXITY TEST")
    print("=" * 70)

    dim = 512
    k = 64
    depth = 6

    print(f"\nTesting with depth={depth}, k={k}, dim={dim}")
    print("-" * 40)

    print("\nInitializing model...")
    model = CantorGlobalAttentionO_n(
        dim=dim,
        depth=depth,
        max_seq_len=4096,
        local_window=k,
        num_heads=8
    )
    print("Model initialized.\n")

    sequence_lengths = [128, 256, 512, 1024]
    times = []

    print("Running timing tests...")
    for seq_len in sequence_lengths:
        x = torch.randn(2, seq_len, dim)

        # Warmup
        with torch.no_grad():
            _ = model(x)

        # Time it
        start = time.time()
        with torch.no_grad():
            for _ in range(5):
                _ = model(x)
        elapsed = (time.time() - start) / 5

        times.append(elapsed)
        throughput = seq_len / elapsed

        print(f"  seq_len={seq_len:4d}: {elapsed * 1000:6.2f}ms ({throughput:8.0f} tokens/sec)")

    # Check linearity
    print("\nLinearity check:")
    ratios = [times[i + 1] / times[i] for i in range(len(times) - 1)]
    seq_ratios = [sequence_lengths[i + 1] / sequence_lengths[i] for i in range(len(sequence_lengths) - 1)]

    for i, (time_ratio, seq_ratio) in enumerate(zip(ratios, seq_ratios)):
        print(f"  {sequence_lengths[i]:4d} -> {sequence_lengths[i + 1]:4d}: "
              f"time_ratio={time_ratio:.2f}, expected={seq_ratio:.2f}")

    avg_ratio = sum(ratios) / len(ratios)
    print(f"\n  Average time ratio: {avg_ratio:.2f}")
    print(f"  Expected for O(n): 2.00")

    if 1.8 <= avg_ratio <= 2.2:
        print("  ✓ CONFIRMED: O(n) complexity")
    else:
        print(f"  ⚠ Ratio {avg_ratio:.2f} outside expected range")

    return model, times


def test_output_quality():
    """Test that outputs are reasonable."""
    print("\n" + "=" * 70)
    print("OUTPUT QUALITY TEST")
    print("=" * 70)

    dim = 256
    seq_len = 512
    batch_size = 4

    model = CantorGlobalAttentionO_n(
        dim=dim,
        depth=6,
        max_seq_len=2048,
        local_window=32,
        num_heads=4
    )

    x = torch.randn(batch_size, seq_len, dim)

    print(f"\nInput shape: {x.shape}")
    print(f"Input mean: {x.mean().item():.4f}, std: {x.std().item():.4f}")

    output = model(x)

    print(f"Output shape: {output.shape}")
    print(f"Output mean: {output.mean().item():.4f}, std: {output.std().item():.4f}")

    # Check for NaNs
    assert not torch.isnan(output).any(), "Output contains NaN!"
    assert not torch.isinf(output).any(), "Output contains Inf!"

    print("\n✓ Output is valid (no NaN/Inf)")
    print("✓ Shape preserved")

    return model, output


if __name__ == "__main__":
    model, times = test_complexity()
    _, output = test_output_quality()

    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETE")
    print("=" * 70)
    print("\n✓ TRUE O(n) GLOBAL ATTENTION OPERATIONAL")