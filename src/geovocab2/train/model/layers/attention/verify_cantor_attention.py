# ============================================================================
# COMPLETE CANTOR ATTENTION BENCHMARK - SINGLE COLAB CELL
# Copy and paste this entire cell into Colab and run!
# ============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict
from dataclasses import dataclass
import math
import time
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime

print("🚀 Initializing Cantor Attention Benchmark...")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class CantorGlobalAttentionConfig:
    dim: int = 512
    num_heads: int = 8
    head_dim: Optional[int] = None
    depth: int = 8
    max_seq_len: int = 65536
    local_window: int = 64
    dropout: float = 0.0
    causal: bool = False
    qkv_bias: bool = True
    out_bias: bool = True

    def __post_init__(self):
        if self.head_dim is None:
            assert self.dim % self.num_heads == 0
            self.head_dim = self.dim // self.num_heads


# ============================================================================
# DYNAMIC CANTOR ATTENTION
# ============================================================================

class CantorGlobalAttentionDynamic(nn.Module):
    """
    Dynamic Cantor Attention with O(n) complexity.
    Builds routes on-demand for each sequence length.
    """

    def __init__(self, config: CantorGlobalAttentionConfig):
        super().__init__()
        self.config = config
        self.dim = config.dim
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.num_neighbors = config.local_window

        # Projections
        self.qkv = nn.Linear(config.dim, 3 * config.dim, bias=config.qkv_bias)
        self.out_proj = nn.Linear(config.dim, config.dim, bias=config.out_bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Routes cache
        self.routes_cache: Dict[int, torch.Tensor] = {}
        self.max_cache_entries = 20

        # Pre-build common sizes
        common_sizes = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
        for size in common_sizes:
            if size <= config.max_seq_len:
                routes = self._build_cantor_routes(size, self.num_neighbors, config.depth)
                self.routes_cache[size] = routes

    def _cantor_coordinate(self, position: int, max_len: int, depth: int) -> float:
        """Compute Cantor coordinate for a position."""
        x = position / max(1, max_len - 1)
        x = max(1e-6, min(x, 1.0 - 1e-6))

        cantor_val = 0.0
        factor = 0.5

        for _ in range(depth):
            x *= 3.0
            digit = int(x)
            x -= digit
            if digit == 2:
                cantor_val += factor
            factor *= 0.5

        return cantor_val

    def _build_cantor_routes(self, max_len: int, k: int, depth: int) -> torch.Tensor:
        """Build routing table based on Cantor distance."""
        cantor_coords = torch.tensor([
            self._cantor_coordinate(pos, max_len, depth)
            for pos in range(max_len)
        ], dtype=torch.float32)

        routes = torch.zeros(max_len, k, dtype=torch.long)

        for i in range(max_len):
            distances = torch.abs(cantor_coords - cantor_coords[i])
            _, nearest = torch.topk(distances, k, largest=False)
            routes[i] = nearest

        return routes

    def _get_routes_for_seq_len(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Get or build routes for specific sequence length."""
        if seq_len in self.routes_cache:
            return self.routes_cache[seq_len].to(device)

        # Find next larger cached size
        cached_sizes = sorted([s for s in self.routes_cache.keys() if s >= seq_len])
        if cached_sizes:
            larger_size = cached_sizes[0]
            routes = self.routes_cache[larger_size][:seq_len, :].to(device)
            routes = torch.clamp(routes, 0, seq_len - 1)
            return routes

        # Build on-demand
        routes = self._build_cantor_routes(seq_len, self.num_neighbors, self.config.depth)
        if len(self.routes_cache) < self.max_cache_entries:
            self.routes_cache[seq_len] = routes
        return routes.to(device)

    def _sparse_attention_dynamic(
            self,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            seq_len: int
    ) -> torch.Tensor:
        """Sparse attention using dynamic routes."""
        batch_size, num_heads, _, head_dim = q.shape
        device = q.device
        k_neighbors = self.num_neighbors

        # Get routes for exact seq_len
        routes = self._get_routes_for_seq_len(seq_len, device)

        # Create broadcast indices
        batch_idx = torch.arange(batch_size, device=device).view(-1, 1, 1, 1)
        head_idx = torch.arange(num_heads, device=device).view(1, -1, 1, 1)
        routes_bc = routes.view(1, 1, seq_len, k_neighbors)

        # Expand
        batch_idx = batch_idx.expand(batch_size, num_heads, seq_len, k_neighbors)
        head_idx = head_idx.expand(batch_size, num_heads, seq_len, k_neighbors)
        routes_bc = routes_bc.expand(batch_size, num_heads, seq_len, k_neighbors)

        # Gather K and V
        k_gathered = k[batch_idx, head_idx, routes_bc, :]
        v_gathered = v[batch_idx, head_idx, routes_bc, :]

        # Compute attention
        scores = torch.einsum('bhqd,bhqkd->bhqk', q, k_gathered) * self.scale
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        output = torch.einsum('bhqk,bhqkd->bhqd', attn_weights, v_gathered)

        return output

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass."""
        batch_size, seq_len, _ = x.shape

        # QKV projection
        qkv = self.qkv(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Sparse attention
        attn_output = self._sparse_attention_dynamic(q, k, v, seq_len)

        # Reshape and project
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.dim)
        output = self.out_proj(attn_output)
        output = self.resid_dropout(output)

        return output


# ============================================================================
# BENCHMARK RUNNER
# ============================================================================

class BenchmarkRunner:
    """Comprehensive benchmark runner."""

    def __init__(self, device: str = "cuda"):
        self.device = torch.device(device)
        self.results = {
            'device_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
            'device_memory_gb': torch.cuda.get_device_properties(
                0).total_memory / 1024 ** 3 if torch.cuda.is_available() else 0,
            'timestamp': datetime.now().isoformat(),
            'benchmarks': []
        }

    def benchmark_config(
            self,
            model: nn.Module,
            seq_len: int,
            batch_size: int,
            dim: int,
            num_iterations: int = 50,
            warmup: int = 10,
            model_name: str = "model"
    ) -> Dict:
        """Benchmark a specific configuration."""
        x = torch.randn(batch_size, seq_len, dim, device=self.device, requires_grad=True)

        # Warmup
        for _ in range(warmup):
            x.grad = None
            if isinstance(model, nn.MultiheadAttention):
                output, _ = model(x, x, x)
            else:
                output = model(x)
            loss = output.sum()
            loss.backward()

        if self.device.type == "cuda":
            torch.cuda.synchronize()

        # Benchmark
        times = []
        for _ in range(num_iterations):
            x.grad = None

            if self.device.type == "cuda":
                torch.cuda.synchronize()
            start = time.time()

            if isinstance(model, nn.MultiheadAttention):
                output, _ = model(x, x, x)
            else:
                output = model(x)
            loss = output.sum()
            loss.backward()

            if self.device.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.time() - start)

        # Memory
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
            x.grad = None
            if isinstance(model, nn.MultiheadAttention):
                output, _ = model(x, x, x)
            else:
                output = model(x)
            loss = output.sum()
            loss.backward()
            peak_memory_mb = torch.cuda.max_memory_allocated() / 1024 ** 2
        else:
            peak_memory_mb = 0

        # Statistics
        times = np.array(times) * 1000  # Convert to ms

        result = {
            'model_name': model_name,
            'seq_len': seq_len,
            'batch_size': batch_size,
            'time_mean_ms': float(np.mean(times)),
            'time_std_ms': float(np.std(times)),
            'memory_mb': float(peak_memory_mb),
            'throughput_samples_per_sec': batch_size / (np.mean(times) / 1000),
        }

        # Cleanup
        del x, output, loss
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        return result

    def run_comprehensive_benchmark(self):
        """Run comprehensive comparison."""

        print("\n" + "=" * 80)
        print("COMPREHENSIVE BENCHMARK - CANTOR vs STANDARD ATTENTION")
        print("=" * 80)
        print(f"Device: {self.results['device_name']}")
        print(f"VRAM: {self.results['device_memory_gb']:.1f} GB")

        # Configuration
        dim = 512
        num_heads = 8

        # Adaptive test configs based on available memory
        vram_gb = self.results['device_memory_gb']

        if vram_gb >= 40:  # A100
            print("Detected A100-class GPU - testing up to 32K sequence length!")
            test_configs = [
                (64, 64), (128, 64), (256, 64), (512, 64),
                (1024, 32), (2048, 32), (4096, 16), (8192, 8),
                (16384, 4), (32768, 2)
            ]
        elif vram_gb >= 24:  # A10, RTX 3090
            print("Detected high-memory GPU - testing up to 16K sequence length!")
            test_configs = [
                (64, 64), (128, 64), (256, 64), (512, 64),
                (1024, 32), (2048, 32), (4096, 16), (8192, 8),
                (16384, 4)
            ]
        else:  # T4, V100
            print("Detected standard GPU - testing up to 8K sequence length!")
            test_configs = [
                (64, 32), (128, 32), (256, 32), (512, 32),
                (1024, 32), (2048, 16), (4096, 8), (8192, 4)
            ]

        config = CantorGlobalAttentionConfig(
            dim=dim,
            num_heads=num_heads,
            depth=8,
            max_seq_len=65536,
            local_window=64,
            dropout=0.0
        )

        # Test each configuration
        for seq_len, batch_size in test_configs:
            print(f"\n{'=' * 80}")
            print(f"seq_len={seq_len:>6}, batch={batch_size:>3}")
            print(f"{'=' * 80}")

            comparison = {
                'seq_len': seq_len,
                'batch_size': batch_size,
                'results': {}
            }

            # Test Cantor
            print(f"[1/2] Cantor O(n)...", end=" ", flush=True)
            try:
                cantor_model = CantorGlobalAttentionDynamic(config).to(self.device)
                result = self.benchmark_config(
                    cantor_model, seq_len, batch_size, dim,
                    num_iterations=50, warmup=10, model_name="Cantor"
                )
                comparison['results']['cantor'] = result
                print(f"✓ {result['time_mean_ms']:.2f}ms, {result['memory_mb']:.0f}MB")

                del cantor_model
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
            except Exception as e:
                print(f"✗ {e}")
                comparison['results']['cantor'] = {'error': str(e)}
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()

            # Test Standard
            print(f"[2/2] Standard O(n²)...", end=" ", flush=True)
            try:
                standard_model = nn.MultiheadAttention(
                    dim, num_heads, dropout=0.0, batch_first=True
                ).to(self.device)
                result = self.benchmark_config(
                    standard_model, seq_len, batch_size, dim,
                    num_iterations=50, warmup=10, model_name="Standard"
                )
                comparison['results']['standard'] = result
                print(f"✓ {result['time_mean_ms']:.2f}ms, {result['memory_mb']:.0f}MB")

                del standard_model
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
            except Exception as e:
                print(f"✗ {e}")
                comparison['results']['standard'] = {'error': str(e)}
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()

            # Comparison
            if ('cantor' in comparison['results'] and 'standard' in comparison['results'] and
                    'error' not in comparison['results']['cantor'] and
                    'error' not in comparison['results']['standard']):
                cantor_time = comparison['results']['cantor']['time_mean_ms']
                standard_time = comparison['results']['standard']['time_mean_ms']
                speedup = standard_time / cantor_time

                cantor_mem = comparison['results']['cantor']['memory_mb']
                standard_mem = comparison['results']['standard']['memory_mb']
                mem_ratio = cantor_mem / standard_mem

                winner = "🚀 CANTOR WINS!" if speedup > 1.0 else "Standard faster"
                print(f"      → Speedup: {speedup:.3f}x, Mem: {mem_ratio:.2f}x {winner}")

                comparison['speedup'] = speedup
                comparison['memory_ratio'] = mem_ratio

            self.results['benchmarks'].append(comparison)

        # Analysis
        self.print_analysis()

    def print_analysis(self):
        """Print comprehensive analysis."""

        print("\n" + "=" * 80)
        print("ANALYSIS")
        print("=" * 80)

        # Extract successful comparisons
        comparisons = [b for b in self.results['benchmarks'] if 'speedup' in b]

        if not comparisons:
            print("\nNo successful comparisons!")
            return

        # Summary table
        print("\n" + "─" * 80)
        print(f"{'Seq Len':<10} {'Batch':<8} {'Cantor (ms)':<15} {'Standard (ms)':<15} {'Speedup':<12}")
        print("─" * 80)

        for comp in comparisons:
            cantor_time = comp['results']['cantor']['time_mean_ms']
            standard_time = comp['results']['standard']['time_mean_ms']
            speedup = comp['speedup']
            winner = "🚀" if speedup > 1.0 else "  "

            print(f"{comp['seq_len']:<10} {comp['batch_size']:<8} "
                  f"{cantor_time:>10.2f}      {standard_time:>10.2f}        "
                  f"{speedup:>7.3f}x {winner}")

        print("─" * 80)

        # Find crossover
        print("\n[Crossover Point]")
        crossover_seq = None
        for comp in comparisons:
            if comp['speedup'] > 1.0:
                crossover_seq = comp['seq_len']
                print(f"✓ Cantor faster at seq_len={crossover_seq}: {comp['speedup']:.3f}x speedup")
                break

        if not crossover_seq:
            last = comparisons[-1]
            if last['speedup'] > 0.7:
                extrapolated = int(last['seq_len'] * (1.0 / last['speedup']))
                print(f"  Approaching crossover. Current: {last['speedup']:.3f}x at seq={last['seq_len']}")
                print(f"  Extrapolated crossover: ~{extrapolated:,}")
            else:
                print(f"  Not yet reached. Best: {last['speedup']:.3f}x at seq={last['seq_len']}")

        # Complexity verification
        print("\n[Complexity Verification]")
        print("Cantor scaling (O(n) → 2.0x when doubling):")
        for i in range(len(comparisons) - 1):
            if comparisons[i]['batch_size'] == comparisons[i + 1]['batch_size']:
                ratio = (comparisons[i + 1]['results']['cantor']['time_mean_ms'] /
                         comparisons[i]['results']['cantor']['time_mean_ms'])
                print(f"  {comparisons[i]['seq_len']:5d} → {comparisons[i + 1]['seq_len']:5d}: {ratio:.2f}x")

        print("\nStandard scaling (O(n²) → 4.0x when doubling):")
        for i in range(len(comparisons) - 1):
            if comparisons[i]['batch_size'] == comparisons[i + 1]['batch_size']:
                ratio = (comparisons[i + 1]['results']['standard']['time_mean_ms'] /
                         comparisons[i]['results']['standard']['time_mean_ms'])
                print(f"  {comparisons[i]['seq_len']:5d} → {comparisons[i + 1]['seq_len']:5d}: {ratio:.2f}x")

        # Memory analysis
        print("\n[Memory Efficiency]")
        for comp in comparisons[-3:]:  # Last 3 for long sequences
            cantor_mem = comp['results']['cantor']['memory_mb']
            standard_mem = comp['results']['standard']['memory_mb']
            ratio = cantor_mem / standard_mem
            print(f"  seq={comp['seq_len']:5d}: Cantor {cantor_mem:6.0f}MB, "
                  f"Standard {standard_mem:6.0f}MB (ratio: {ratio:.2f}x)")

        # Summary
        cantor_wins = sum(1 for c in comparisons if c['speedup'] > 1.0)
        print(f"\n[Summary]")
        print(f"  Cantor wins: {cantor_wins}/{len(comparisons)} configurations")
        print(
            f"  Best speedup: {max(c['speedup'] for c in comparisons):.3f}x at seq={max(comparisons, key=lambda c: c['speedup'])['seq_len']}")


# ============================================================================
# RUN BENCHMARK
# ============================================================================

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("❌ CUDA not available! This benchmark requires a GPU.")
    else:
        runner = BenchmarkRunner()
        runner.run_comprehensive_benchmark()

        print("\n" + "=" * 80)
        print("✓ BENCHMARK COMPLETE")
        print("=" * 80)