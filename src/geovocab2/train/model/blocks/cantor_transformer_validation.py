# scripts/verify_complexity.py

"""
Empirically verify O(n) complexity of Cantor attention vs O(n²) standard attention.
"""

import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List

from geovocab2.train.model.layers.attention.cantor_global import (
    CantorAttention, CantorAttentionConfig
)


def profile_attention_complexity():
    """
    Profile both Cantor and Standard attention across different sequence lengths.
    Verify that Cantor is truly O(n) and Standard is O(n²).
    """

    print("=" * 70)
    print("ATTENTION COMPLEXITY VERIFICATION")
    print("=" * 70)

    # Configuration
    dim = 512
    num_heads = 8
    head_dim = 64
    k_neighbors = 64
    batch_size = 4

    # Test with increasing sequence lengths
    seq_lengths = [64, 128, 256, 512, 1024, 2048]

    # Results storage
    cantor_times_forward = []
    cantor_times_backward = []
    cantor_memory = []

    standard_times_forward = []
    standard_times_backward = []
    standard_memory = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\nDevice: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Model dim: {dim}, Heads: {num_heads}")
    print(f"Cantor window: {k_neighbors}")

    # Test Cantor Attention
    print("\n" + "-" * 70)
    print("Testing CANTOR O(n) Attention")
    print("-" * 70)

    for seq_len in seq_lengths:
        print(f"\nSequence length: {seq_len}")

        # Create Cantor attention
        cantor_config = CantorAttentionConfig(
            dim=dim,
            num_heads=num_heads,
            head_dim=head_dim,
            depth=8,
            max_seq_len=max(seq_lengths),
            local_window=k_neighbors,
            dropout=0.0
        )
        cantor_attn = CantorAttention(cantor_config).to(device)

        # Input
        x = torch.randn(batch_size, seq_len, dim, device=device, requires_grad=True)

        # Warmup
        for _ in range(3):
            _ = cantor_attn(x)

        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        # Forward pass timing
        start = time.time()
        for _ in range(10):
            output = cantor_attn(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
        forward_time = (time.time() - start) / 10

        # Backward pass timing
        start = time.time()
        for _ in range(10):
            output = cantor_attn(x)
            loss = output.sum()
            loss.backward()
            if device.type == "cuda":
                torch.cuda.synchronize()
            x.grad = None
        backward_time = (time.time() - start) / 10 - forward_time

        # Memory
        if device.type == "cuda":
            memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
        else:
            memory_mb = 0

        cantor_times_forward.append(forward_time)
        cantor_times_backward.append(backward_time)
        cantor_memory.append(memory_mb)

        print(f"  Forward: {forward_time * 1000:.2f}ms")
        print(f"  Backward: {backward_time * 1000:.2f}ms")
        print(f"  Total: {(forward_time + backward_time) * 1000:.2f}ms")
        print(f"  Memory: {memory_mb:.2f}MB")

        # Clean up
        del cantor_attn, x, output
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Test Standard Attention
    print("\n" + "-" * 70)
    print("Testing STANDARD O(n²) Attention")
    print("-" * 70)

    for seq_len in seq_lengths:
        print(f"\nSequence length: {seq_len}")

        # Create standard attention
        standard_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=0.0,
            batch_first=True
        ).to(device)

        # Input
        x = torch.randn(batch_size, seq_len, dim, device=device, requires_grad=True)

        # Warmup
        for _ in range(3):
            _, _ = standard_attn(x, x, x)

        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        # Forward pass timing
        start = time.time()
        for _ in range(10):
            output, _ = standard_attn(x, x, x)
            if device.type == "cuda":
                torch.cuda.synchronize()
        forward_time = (time.time() - start) / 10

        # Backward pass timing
        start = time.time()
        for _ in range(10):
            output, _ = standard_attn(x, x, x)
            loss = output.sum()
            loss.backward()
            if device.type == "cuda":
                torch.cuda.synchronize()
            x.grad = None
        backward_time = (time.time() - start) / 10 - forward_time

        # Memory
        if device.type == "cuda":
            memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
        else:
            memory_mb = 0

        standard_times_forward.append(forward_time)
        standard_times_backward.append(backward_time)
        standard_memory.append(memory_mb)

        print(f"  Forward: {forward_time * 1000:.2f}ms")
        print(f"  Backward: {backward_time * 1000:.2f}ms")
        print(f"  Total: {(forward_time + backward_time) * 1000:.2f}ms")
        print(f"  Memory: {memory_mb:.2f}MB")

        # Clean up
        del standard_attn, x, output
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Analysis
    print("\n" + "=" * 70)
    print("COMPLEXITY ANALYSIS")
    print("=" * 70)

    # Compute time ratios (should be 2x for O(n), 4x for O(n²))
    print("\nTime ratios when sequence length doubles:")
    print("\nCANTOR (should be ~2.0 for O(n)):")
    for i in range(len(seq_lengths) - 1):
        ratio = (cantor_times_forward[i + 1] + cantor_times_backward[i + 1]) / \
                (cantor_times_forward[i] + cantor_times_backward[i])
        print(f"  {seq_lengths[i]:4d} -> {seq_lengths[i + 1]:4d}: {ratio:.2f}x")

    print("\nSTANDARD (should be ~4.0 for O(n²)):")
    for i in range(len(seq_lengths) - 1):
        ratio = (standard_times_forward[i + 1] + standard_times_backward[i + 1]) / \
                (standard_times_forward[i] + standard_times_backward[i])
        print(f"  {seq_lengths[i]:4d} -> {seq_lengths[i + 1]:4d}: {ratio:.2f}x")

    # Speedup
    print("\nSpeedup (Standard / Cantor):")
    for i, seq_len in enumerate(seq_lengths):
        total_cantor = cantor_times_forward[i] + cantor_times_backward[i]
        total_standard = standard_times_forward[i] + standard_times_backward[i]
        speedup = total_standard / total_cantor
        print(f"  seq_len={seq_len:4d}: {speedup:.2f}x faster")

    # Memory comparison
    print("\nMemory usage:")
    print("\nCANTOR:")
    for i, seq_len in enumerate(seq_lengths):
        print(f"  seq_len={seq_len:4d}: {cantor_memory[i]:.2f}MB")

    print("\nSTANDARD:")
    for i, seq_len in enumerate(seq_lengths):
        print(f"  seq_len={seq_len:4d}: {standard_memory[i]:.2f}MB")

    # Plot results
    plot_complexity_results(
        seq_lengths,
        cantor_times_forward, cantor_times_backward,
        standard_times_forward, standard_times_backward,
        cantor_memory, standard_memory
    )

    # Verify O(n) vs O(n²)
    print("\n" + "=" * 70)
    print("VERIFICATION")
    print("=" * 70)

    # Fit complexity curves
    cantor_total = [f + b for f, b in zip(cantor_times_forward, cantor_times_backward)]
    standard_total = [f + b for f, b in zip(standard_times_forward, standard_times_backward)]

    # Check if Cantor is linear
    cantor_ratios = [cantor_total[i + 1] / cantor_total[i] for i in range(len(cantor_total) - 1)]
    cantor_avg_ratio = np.mean(cantor_ratios)

    # Check if Standard is quadratic
    standard_ratios = [standard_total[i + 1] / standard_total[i] for i in range(len(standard_total) - 1)]
    standard_avg_ratio = np.mean(standard_ratios)

    print(f"\nCantor average doubling ratio: {cantor_avg_ratio:.2f}")
    print(f"  Expected for O(n): 2.0")
    print(f"  Status: {'✓ PASS' if 1.8 <= cantor_avg_ratio <= 2.2 else '✗ FAIL'}")

    print(f"\nStandard average doubling ratio: {standard_avg_ratio:.2f}")
    print(f"  Expected for O(n²): 4.0")
    print(f"  Status: {'✓ PASS' if 3.5 <= standard_avg_ratio <= 4.5 else '✗ FAIL'}")

    print("\n" + "=" * 70)


def plot_complexity_results(
        seq_lengths,
        cantor_forward, cantor_backward,
        standard_forward, standard_backward,
        cantor_memory, standard_memory
):
    """Plot complexity results."""

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    seq_lengths = np.array(seq_lengths)

    # Plot 1: Forward time
    ax = axes[0, 0]
    ax.plot(seq_lengths, np.array(cantor_forward) * 1000, 'o-', label='Cantor O(n)', linewidth=2)
    ax.plot(seq_lengths, np.array(standard_forward) * 1000, 's-', label='Standard O(n²)', linewidth=2)
    ax.set_xlabel('Sequence Length', fontsize=12)
    ax.set_ylabel('Forward Time (ms)', fontsize=12)
    ax.set_title('Forward Pass Time Complexity', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')

    # Plot 2: Total time
    ax = axes[0, 1]
    cantor_total = (np.array(cantor_forward) + np.array(cantor_backward)) * 1000
    standard_total = (np.array(standard_forward) + np.array(standard_backward)) * 1000
    ax.plot(seq_lengths, cantor_total, 'o-', label='Cantor O(n)', linewidth=2)
    ax.plot(seq_lengths, standard_total, 's-', label='Standard O(n²)', linewidth=2)
    ax.set_xlabel('Sequence Length', fontsize=12)
    ax.set_ylabel('Total Time (ms)', fontsize=12)
    ax.set_title('Forward + Backward Time Complexity', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')

    # Plot 3: Speedup
    ax = axes[1, 0]
    speedup = standard_total / cantor_total
    ax.plot(seq_lengths, speedup, 'go-', linewidth=2, markersize=8)
    ax.axhline(y=1, color='r', linestyle='--', label='No speedup', alpha=0.5)
    ax.set_xlabel('Sequence Length', fontsize=12)
    ax.set_ylabel('Speedup (Standard / Cantor)', fontsize=12)
    ax.set_title('Cantor Speedup Over Standard', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log', base=2)

    # Plot 4: Memory
    ax = axes[1, 1]
    ax.plot(seq_lengths, cantor_memory, 'o-', label='Cantor O(n)', linewidth=2)
    ax.plot(seq_lengths, standard_memory, 's-', label='Standard O(n²)', linewidth=2)
    ax.set_xlabel('Sequence Length', fontsize=12)
    ax.set_ylabel('Peak Memory (MB)', fontsize=12)
    ax.set_title('Memory Usage', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig('complexity_verification.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Plot saved: complexity_verification.png")
    plt.close()


def check_attention_patterns():
    """
    Verify that Cantor attention only computes k neighbors, not all pairs.
    """
    print("\n" + "=" * 70)
    print("ATTENTION PATTERN VERIFICATION")
    print("=" * 70)

    dim = 128
    num_heads = 4
    seq_len = 256
    k = 32
    batch_size = 2

    config = CantorAttentionConfig(
        dim=dim,
        num_heads=num_heads,
        depth=6,
        max_seq_len=512,
        local_window=k,
        dropout=0.0
    )

    cantor_attn = CantorAttention(config)

    # Check routes
    routes = cantor_attn.cantor_routes[:seq_len, :k]

    print(f"\nSequence length: {seq_len}")
    print(f"Neighbors per token: {k}")
    print(f"Routes shape: {routes.shape}")

    # Count unique connections
    total_possible = seq_len * seq_len
    total_actual = seq_len * k
    sparsity = total_actual / total_possible

    print(f"\nConnectivity:")
    print(f"  Possible (O(n²)): {total_possible:,} connections")
    print(f"  Actual (O(n)): {total_actual:,} connections")
    print(f"  Sparsity: {sparsity:.4f} ({sparsity * 100:.2f}%)")
    print(f"  Reduction: {total_possible / total_actual:.1f}x fewer connections")

    # Verify each token connects to exactly k others
    connections_per_token = [len(set(routes[i].tolist())) for i in range(min(10, seq_len))]
    print(f"\nConnections per token (first 10): {connections_per_token}")
    print(f"  Expected: {k} for each")
    print(f"  Status: {'✓ PASS' if all(c == k for c in connections_per_token) else '✗ FAIL'}")

    # Check routing diversity
    all_routes = routes.flatten().tolist()
    unique_targets = len(set(all_routes))
    print(f"\nRouting coverage:")
    print(f"  Unique target positions: {unique_targets}/{seq_len}")
    print(f"  Coverage: {unique_targets / seq_len * 100:.1f}%")
    print(f"  Status: {'✓ GOOD' if unique_targets / seq_len > 0.95 else '⚠ LIMITED'}")


if __name__ == "__main__":
    profile_attention_complexity()
    check_attention_patterns()

    print("\n" + "=" * 70)
    print("VERIFICATION COMPLETE")
    print("=" * 70)