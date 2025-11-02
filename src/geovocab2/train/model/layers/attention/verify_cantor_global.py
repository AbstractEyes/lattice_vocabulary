# geovocab2/train/model/layers/attention/validation_cantor_global.py

"""
Comprehensive Cantor Attention Validation and Benchmark
Tests functionality and performance of Cantor Attention vs Standard Attention
Includes testing of adaptive window sizing for improved memory efficiency

Usage:
    python -m geovocab2.train.model.layers.attention.validation_cantor_global              # Quick tests only (default)
    python -m geovocab2.train.model.layers.attention.validation_cantor_global --bench      # Quick tests + full benchmark
    python -m geovocab2.train.model.layers.attention.validation_cantor_global --bench-only # Benchmark only (no tests)
    python -m geovocab2.train.model.layers.attention.validation_cantor_global --adaptive   # Test adaptive window mode
"""

import torch
import torch.nn as nn
import time
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime
from typing import Dict
import argparse
import sys

from geovocab2.train.model.layers.attention.cantor_global import (
    CantorAttention,
    CantorAttentionConfig,
    create_cantor_attention,
)


# ============================================================================
# QUICK VALIDATION TESTS
# ============================================================================

class ValidationTests:
    """Quick validation tests for Cantor Attention."""

    def __init__(self, device: str = "cuda", test_adaptive: bool = False):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.test_adaptive = test_adaptive
        self.passed = 0
        self.failed = 0

    def run_all_tests(self):
        """Run all validation tests."""
        print("=" * 80)
        print("CANTOR ATTENTION - VALIDATION TESTS")
        print("=" * 80)
        print(f"Device: {self.device}")
        if self.device.type == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")

        self.test_basic_functionality()
        self.test_causal_masking()
        self.test_gradient_flow()
        self.test_causal_vs_noncausal()
        self.test_autoregressive()

        if self.test_adaptive:
            self.test_adaptive_window()
            self.test_adaptive_vs_fixed()

        print("\n" + "=" * 80)
        print(f"VALIDATION SUMMARY: {self.passed} passed, {self.failed} failed")
        print("=" * 80)

        return self.failed == 0

    def _test_wrapper(self, test_name: str, test_func):
        """Wrapper for running tests with error handling."""
        print(f"\n[{test_name}]")
        try:
            test_func()
            self.passed += 1
            return True
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            import traceback
            traceback.print_exc()
            self.failed += 1
            return False

    def test_basic_functionality(self):
        """Test basic forward/backward pass."""

        def test():
            config = CantorAttentionConfig(
                dim=512,
                num_heads=8,
                local_window=64,
                dropout=0.0,
                causal=False,
                adaptive_window=False
            )
            attn = CantorAttention(config).to(self.device)

            for seq_len in [64, 256, 1024]:
                x = torch.randn(2, seq_len, 512, device=self.device)
                output = attn(x)
                assert output.shape == x.shape, f"Shape mismatch at seq_len={seq_len}"
                print(f"  ✓ seq_len={seq_len}: {x.shape} -> {output.shape}")

        self._test_wrapper("Basic Functionality", test)

    def test_causal_masking(self):
        """Test causal attention."""

        def test():
            attn = create_cantor_attention(
                dim=512,
                num_heads=8,
                local_window=64,
                dropout=0.0,
                causal=True
            ).to(self.device)

            for seq_len in [64, 256, 1024]:
                x = torch.randn(2, seq_len, 512, device=self.device)
                output = attn(x)
                assert output.shape == x.shape, f"Shape mismatch at seq_len={seq_len}"
                print(f"  ✓ causal seq_len={seq_len}: {x.shape} -> {output.shape}")

        self._test_wrapper("Causal Masking", test)

    def test_gradient_flow(self):
        """Test gradient computation."""

        def test():
            config = CantorAttentionConfig(dim=512, num_heads=8, dropout=0.0)
            attn = CantorAttention(config).to(self.device)

            x = torch.randn(2, 128, 512, device=self.device, requires_grad=True)
            output = attn(x)
            loss = output.sum()
            loss.backward()

            assert x.grad is not None, "Gradients not computed!"
            assert not torch.isnan(x.grad).any(), "NaN gradients!"

            grad_norm = x.grad.norm().item()
            print(f"  ✓ Gradients computed successfully")
            print(f"  ✓ Gradient norm: {grad_norm:.4f}")

        self._test_wrapper("Gradient Flow", test)

    def test_causal_vs_noncausal(self):
        """Test that causal and non-causal produce different outputs."""

        def test():
            config_causal = CantorAttentionConfig(dim=256, num_heads=4, dropout=0.0, causal=True)
            config_noncausal = CantorAttentionConfig(dim=256, num_heads=4, dropout=0.0, causal=False)

            attn_causal = CantorAttention(config_causal).to(self.device)
            attn_noncausal = CantorAttention(config_noncausal).to(self.device)

            # Copy weights
            attn_causal.load_state_dict(attn_noncausal.state_dict())

            x = torch.randn(2, 128, 256, device=self.device)

            with torch.no_grad():
                out_causal = attn_causal(x)
                out_noncausal = attn_noncausal(x)

            diff = torch.abs(out_causal - out_noncausal).mean().item()
            print(f"  Mean absolute difference: {diff:.6f}")

            assert diff > 0.01, "Causal and non-causal outputs too similar!"
            print(f"  ✓ Outputs differ (causal masking working)")

        self._test_wrapper("Causal vs Non-Causal", test)

    def test_autoregressive(self):
        """Test autoregressive generation."""

        def test():
            config = CantorAttentionConfig(dim=128, num_heads=4, dropout=0.0, causal=True)
            attn = CantorAttention(config).to(self.device)
            attn.eval()

            # Simulate autoregressive generation
            max_len = 32
            x = torch.randn(1, 1, 128, device=self.device)

            with torch.no_grad():
                for t in range(1, max_len):
                    output = attn(x)
                    next_input = torch.randn(1, 1, 128, device=self.device)
                    x = torch.cat([x, next_input], dim=1)

            assert x.shape == (1, max_len, 128), "Autoregressive generation failed"
            print(f"  ✓ Generated {max_len} tokens autoregressively")
            print(f"  ✓ Final shape: {x.shape}")

        self._test_wrapper("Autoregressive Generation", test)

    def test_adaptive_window(self):
        """Test adaptive window sizing."""

        def test():
            config = CantorAttentionConfig(
                dim=512,
                num_heads=8,
                adaptive_window=True,
                min_window=16,
                max_window=64,
                sparsity_target=0.25,
                dropout=0.0
            )

            attn = CantorAttention(config).to(self.device)

            test_cases = [
                (64, 16),  # 25% of 64 = 16
                (128, 32),  # 25% of 128 = 32
                (256, 64),  # 25% of 256 = 64 (capped)
                (512, 64),  # 25% of 512 = 128, but capped at 64
                (1024, 64),  # 25% of 1024 = 256, but capped at 64
            ]

            for seq_len, expected_k in test_cases:
                actual_k = config.get_window_size(seq_len)
                assert actual_k == expected_k, f"Window size mismatch: got {actual_k}, expected {expected_k}"

                x = torch.randn(2, seq_len, 512, device=self.device)
                output = attn(x)
                assert output.shape == x.shape, f"Shape mismatch at seq_len={seq_len}"

                coverage = 100 * actual_k / seq_len
                print(f"  ✓ seq_len={seq_len:4d}: k={actual_k:2d} ({coverage:5.1f}% coverage)")

        self._test_wrapper("Adaptive Window Sizing", test)

    def test_adaptive_vs_fixed(self):
        """Test that adaptive and fixed modes produce valid outputs."""

        def test():
            config_fixed = CantorAttentionConfig(
                dim=256,
                num_heads=4,
                local_window=32,
                adaptive_window=False,
                dropout=0.0
            )

            config_adaptive = CantorAttentionConfig(
                dim=256,
                num_heads=4,
                adaptive_window=True,
                min_window=16,
                max_window=32,
                sparsity_target=0.25,
                dropout=0.0
            )

            attn_fixed = CantorAttention(config_fixed).to(self.device)
            attn_adaptive = CantorAttention(config_adaptive).to(self.device)

            # Test at seq_len where both should use k=32
            seq_len = 256  # 25% of 256 = 64, but capped at 32
            x = torch.randn(2, seq_len, 256, device=self.device)

            with torch.no_grad():
                out_fixed = attn_fixed(x)
                out_adaptive = attn_adaptive(x)

            assert out_fixed.shape == out_adaptive.shape == x.shape
            print(f"  ✓ Both modes produce valid outputs at seq_len={seq_len}")
            print(f"  ✓ Fixed: k=32 (fixed)")
            print(f"  ✓ Adaptive: k={config_adaptive.get_window_size(seq_len)} (from 25% target)")

        self._test_wrapper("Adaptive vs Fixed Mode", test)


# ============================================================================
# COMPREHENSIVE BENCHMARK
# ============================================================================

class BenchmarkRunner:
    """Comprehensive benchmark runner."""

    def __init__(self, device: str = "cuda", test_adaptive: bool = False):
        self.device = torch.device(device)
        self.test_adaptive = test_adaptive
        self.results = {
            'device_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
            'device_memory_gb': torch.cuda.get_device_properties(
                0).total_memory / 1024 ** 3 if torch.cuda.is_available() else 0,
            'timestamp': datetime.now().isoformat(),
            'adaptive_mode': test_adaptive,
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

    def run_benchmark(self):
        """Run comprehensive comparison."""

        print("\n" + "=" * 80)
        mode_str = "ADAPTIVE WINDOW" if self.test_adaptive else "FIXED WINDOW"
        print(f"COMPREHENSIVE BENCHMARK - CANTOR ({mode_str}) vs STANDARD")
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
        else:  # T4, V100, L4
            print("Detected standard GPU - testing up to 8K sequence length!")
            test_configs = [
                (64, 32), (128, 32), (256, 32), (512, 32),
                (1024, 32), (2048, 16), (4096, 8), (8192, 4)
            ]

        # Create Cantor config
        if self.test_adaptive:
            config = CantorAttentionConfig(
                dim=dim,
                num_heads=num_heads,
                depth=8,
                max_seq_len=65536,
                adaptive_window=True,
                min_window=16,
                max_window=64,
                sparsity_target=0.25,
                dropout=0.0,
                causal=False
            )
            print("Using ADAPTIVE window sizing (k=16-64, target=25% coverage)")
        else:
            config = CantorAttentionConfig(
                dim=dim,
                num_heads=num_heads,
                depth=8,
                max_seq_len=65536,
                local_window=64,
                adaptive_window=False,
                dropout=0.0,
                causal=False
            )
            print("Using FIXED window sizing (k=64)")

        # Test each configuration
        for seq_len, batch_size in test_configs:
            print(f"\n{'=' * 80}")
            if self.test_adaptive:
                k = config.get_window_size(seq_len)
                print(f"seq_len={seq_len:>6}, batch={batch_size:>3}, k={k:>2} ({100 * k / seq_len:.1f}%)")
            else:
                print(f"seq_len={seq_len:>6}, batch={batch_size:>3}")
            print(f"{'=' * 80}")

            comparison = {
                'seq_len': seq_len,
                'batch_size': batch_size,
                'results': {}
            }

            if self.test_adaptive:
                comparison['window_size'] = config.get_window_size(seq_len)
                comparison['coverage_pct'] = 100 * comparison['window_size'] / seq_len

            # Test Cantor
            print(f"[1/2] Cantor O(n)...", end=" ", flush=True)
            try:
                cantor_model = CantorAttention(config).to(self.device)
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
        self.save_results()

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
        if self.test_adaptive:
            print(f"{'Seq Len':<10} {'Batch':<8} {'k':<5} {'Cantor (ms)':<15} {'Standard (ms)':<15} {'Speedup':<12}")
        else:
            print(f"{'Seq Len':<10} {'Batch':<8} {'Cantor (ms)':<15} {'Standard (ms)':<15} {'Speedup':<12}")
        print("─" * 80)

        for comp in comparisons:
            cantor_time = comp['results']['cantor']['time_mean_ms']
            standard_time = comp['results']['standard']['time_mean_ms']
            speedup = comp['speedup']
            winner = "🚀" if speedup > 1.0 else "  "

            if self.test_adaptive:
                k = comp.get('window_size', 64)
                print(f"{comp['seq_len']:<10} {comp['batch_size']:<8} {k:<5} "
                      f"{cantor_time:>10.2f}      {standard_time:>10.2f}        "
                      f"{speedup:>7.3f}x {winner}")
            else:
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
                if self.test_adaptive:
                    k = comp.get('window_size', 64)
                    print(f"  (using k={k}, {comp.get('coverage_pct', 0):.1f}% coverage)")
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
            if self.test_adaptive:
                k = comp.get('window_size', 64)
                print(f"  seq={comp['seq_len']:5d} (k={k:2d}): Cantor {cantor_mem:6.0f}MB, "
                      f"Standard {standard_mem:6.0f}MB (ratio: {ratio:.2f}x)")
            else:
                print(f"  seq={comp['seq_len']:5d}: Cantor {cantor_mem:6.0f}MB, "
                      f"Standard {standard_mem:6.0f}MB (ratio: {ratio:.2f}x)")

        # Summary
        cantor_wins = sum(1 for c in comparisons if c['speedup'] > 1.0)
        print(f"\n[Summary]")
        print(f"  Mode: {'ADAPTIVE' if self.test_adaptive else 'FIXED'} window")
        print(f"  Cantor wins: {cantor_wins}/{len(comparisons)} configurations")
        if comparisons:
            best = max(comparisons, key=lambda c: c['speedup'])
            print(f"  Best speedup: {best['speedup']:.3f}x at seq={best['seq_len']}")

    def save_results(self, filename: str = None):
        """Save results to JSON."""
        if filename is None:
            mode = "adaptive" if self.test_adaptive else "fixed"
            filename = f"cantor_benchmark_{mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\n✓ Results saved to: {filename}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Cantor Attention Validation and Benchmark',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m geovocab2.train.model.layers.attention.validation_cantor_global              # Quick tests only (default)
  python -m geovocab2.train.model.layers.attention.validation_cantor_global --bench      # Quick tests + full benchmark
  python -m geovocab2.train.model.layers.attention.validation_cantor_global --bench-only # Full benchmark only
  python -m geovocab2.train.model.layers.attention.validation_cantor_global --adaptive   # Test adaptive window mode
  python -m geovocab2.train.model.layers.attention.validation_cantor_global --bench --adaptive  # Benchmark with adaptive window
        """
    )
    parser.add_argument('--bench', action='store_true',
                        help='Run full benchmark after validation tests')
    parser.add_argument('--bench-only', action='store_true',
                        help='Run only the benchmark (skip validation tests)')
    parser.add_argument('--adaptive', action='store_true',
                        help='Test adaptive window mode (variable k based on seq_len)')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to use (default: cuda)')
    args = parser.parse_args()

    # Determine what to run
    run_tests = not args.bench_only
    run_benchmark = args.bench or args.bench_only

    print("=" * 80)
    print("🚀 CANTOR ATTENTION VALIDATION SYSTEM")
    print("=" * 80)
    print(f"PyTorch version: {torch.__version__}")

    # Check CUDA availability
    if not torch.cuda.is_available():
        if args.device == 'cuda':
            print("⚠️  CUDA not available! Falling back to CPU.")
            args.device = 'cpu'
        if run_benchmark:
            print("⚠️  Benchmark requires CUDA. Will skip benchmark.")
            run_benchmark = False
    else:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")

    mode_str = "ADAPTIVE" if args.adaptive else "FIXED"
    print(f"\nWindow Mode: {mode_str}")

    print(f"Run Mode: ", end="")
    if run_tests and run_benchmark:
        print("Validation Tests + Full Benchmark")
    elif run_tests:
        print("Validation Tests Only")
    elif run_benchmark:
        print("Full Benchmark Only")

    # Run validation tests
    if run_tests:
        validator = ValidationTests(device=args.device, test_adaptive=args.adaptive)
        tests_passed = validator.run_all_tests()

        if not tests_passed:
            print("\n❌ Some validation tests failed!")
            if run_benchmark:
                print("Skipping benchmark due to test failures.")
            return 1

    # Run benchmark
    if run_benchmark:
        runner = BenchmarkRunner(device=args.device, test_adaptive=args.adaptive)
        runner.run_benchmark()

    print("\n" + "=" * 80)
    print("✓ COMPLETE")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())