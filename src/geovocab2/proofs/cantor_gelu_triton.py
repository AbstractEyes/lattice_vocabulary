# geofractal.function.cantor_gelu_triton.py
# Experimental cantor_gelu_triton module with Triton kernels and benchmarking suite.
# These are proven ineffective in current experimentation, stored for utilization later.
# =============================================================================
# Authors: AbstractPhil + Claude Opus 4.5
# =============================================================================
# License: Apache-2.0
# =============================================================================
# This version temporarily requires triton, there will be a standard pytorch version later.
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import time
from dataclasses import dataclass
from typing import List, Tuple
import gc


# =============================================================================
# PRODUCTION MODULES
# =============================================================================

@triton.jit
def cantor_gelu_fwd_kernel(x_ptr, out_ptr, n_elements, step, strength, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements

    x = tl.load(x_ptr + offs, mask=mask)

    x3 = x * x * x
    inner = 0.7978845608028654 * (x + 0.044715 * x3)
    e2 = tl.exp(2.0 * inner)
    tanh_val = (e2 - 1.0) / (e2 + 1.0)
    gelu = 0.5 * x * (1.0 + tanh_val)

    snapped = (x / step)
    snapped = snapped - (snapped % 1.0)
    snapped = snapped * step

    out = strength * snapped + (1.0 - strength) * gelu
    tl.store(out_ptr + offs, out, mask=mask)


@triton.jit
def cantor_gelu_bwd_kernel(grad_out_ptr, x_ptr, grad_x_ptr, n_elements, step, strength, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements

    grad_out = tl.load(grad_out_ptr + offs, mask=mask)
    x = tl.load(x_ptr + offs, mask=mask)

    x3 = x * x * x
    inner = 0.7978845608028654 * (x + 0.044715 * x3)
    e2 = tl.exp(2.0 * inner)
    tanh_inner = (e2 - 1.0) / (e2 + 1.0)

    sech2 = 1.0 - tanh_inner * tanh_inner
    f_prime = 0.7978845608028654 * (1.0 + 0.134145 * x * x)
    gelu_grad = 0.5 * (1.0 + tanh_inner) + 0.5 * x * sech2 * f_prime

    grad_x = grad_out * (strength + (1.0 - strength) * gelu_grad)
    tl.store(grad_x_ptr + offs, grad_x, mask=mask)


class _CantorGELUFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, step, strength):
        out = torch.empty_like(x)
        n = x.numel()
        cantor_gelu_fwd_kernel[(triton.cdiv(n, 1024),)](x, out, n, step, strength, BLOCK=1024)
        ctx.save_for_backward(x)
        ctx.step, ctx.strength = step, strength
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, = ctx.saved_tensors
        grad_x = torch.empty_like(x)
        n = x.numel()
        cantor_gelu_bwd_kernel[(triton.cdiv(n, 1024),)](
            grad_out.contiguous(), x, grad_x, n, ctx.step, ctx.strength, BLOCK=1024
        )
        return grad_x, None, None


class CantorGELU(nn.Module):
    def __init__(self, num_stairs: int = 16, value_range: float = 4.0, init_strength: float = 0.5):
        super().__init__()
        self.step = 2 * value_range / num_stairs
        self.strength = nn.Parameter(torch.tensor(init_strength))
        self.num_stairs = num_stairs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return F.gelu(x)
        s = torch.sigmoid(self.strength).item()
        return _CantorGELUFunc.apply(x.contiguous(), self.step, s)


class StochasticCantorGELU(nn.Module):
    def __init__(self, num_stairs=16, apply_prob=0.2):
        super().__init__()
        self.cantor = CantorGELU(num_stairs=num_stairs)
        self.apply_prob = apply_prob

    def forward(self, x):
        if not self.training:
            return F.gelu(x)
        if torch.rand(1).item() < self.apply_prob:
            return self.cantor(x)
        return F.gelu(x)


class RouteDropout(nn.Module):
    def __init__(self, drop_prob=0.1):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x, route_dim=-2):
        if not self.training or self.drop_prob == 0:
            return x
        num_routes = x.shape[route_dim]
        mask = (torch.rand(num_routes, device=x.device) > self.drop_prob).float()
        mask = mask / (1 - self.drop_prob)
        shape = [1] * x.dim()
        shape[route_dim] = num_routes
        return x * mask.view(shape)


# =============================================================================
# BENCHMARK UTILITIES
# =============================================================================

def sync():
    torch.cuda.synchronize()


def clear_cache():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def get_memory_mb():
    return torch.cuda.max_memory_allocated() / 1024 / 1024


def bench_forward(fn, x, warmup=20, iters=100):
    """Benchmark forward pass only."""
    for _ in range(warmup):
        _ = fn(x)
    sync()

    start = time.perf_counter()
    for _ in range(iters):
        _ = fn(x)
    sync()

    return (time.perf_counter() - start) / iters * 1000


def bench_forward_backward(fn, x, warmup=20, iters=100):
    """Benchmark forward + backward pass."""
    for _ in range(warmup):
        x_in = x.clone().requires_grad_(True)
        out = fn(x_in)
        out.sum().backward()
    sync()

    start = time.perf_counter()
    for _ in range(iters):
        x_in = x.clone().requires_grad_(True)
        out = fn(x_in)
        out.sum().backward()
    sync()

    return (time.perf_counter() - start) / iters * 1000


def bench_memory(fn, x):
    """Measure peak memory during forward+backward."""
    clear_cache()

    x_in = x.clone().requires_grad_(True)
    out = fn(x_in)
    out.sum().backward()
    sync()

    return get_memory_mb()


# =============================================================================
# BENCHMARK SCENARIOS
# =============================================================================

@dataclass
class BenchResult:
    name: str
    fwd_ms: float
    fwd_bwd_ms: float
    memory_mb: float
    vs_gelu_fwd: float
    vs_gelu_fwd_bwd: float


def run_scenario(name: str, shape: Tuple[int, ...], device: torch.device) -> List[BenchResult]:
    """Run all activations on a given shape."""
    results = []
    x = torch.randn(shape, device=device)

    # Baseline: F.gelu
    gelu_fwd = bench_forward(F.gelu, x)
    gelu_fwd_bwd = bench_forward_backward(F.gelu, x)
    gelu_mem = bench_memory(F.gelu, x)
    results.append(BenchResult("F.gelu", gelu_fwd, gelu_fwd_bwd, gelu_mem, 1.0, 1.0))

    # CantorGELU (train)
    cantor = CantorGELU().to(device).train()
    cantor_fwd = bench_forward(cantor, x)
    cantor_fwd_bwd = bench_forward_backward(cantor, x)
    cantor_mem = bench_memory(cantor, x)
    results.append(BenchResult(
        "CantorGELU", cantor_fwd, cantor_fwd_bwd, cantor_mem,
        cantor_fwd / gelu_fwd, cantor_fwd_bwd / gelu_fwd_bwd
    ))

    # CantorGELU (eval)
    cantor.eval()
    cantor_eval_fwd = bench_forward(cantor, x)
    cantor.train()
    results.append(BenchResult(
        "CantorGELU(eval)", cantor_eval_fwd, cantor_eval_fwd, 0,
        cantor_eval_fwd / gelu_fwd, cantor_eval_fwd / gelu_fwd
    ))

    # StochasticCantorGELU (20%)
    stoch20 = StochasticCantorGELU(apply_prob=0.2).to(device).train()
    stoch20_fwd = bench_forward(stoch20, x)
    stoch20_fwd_bwd = bench_forward_backward(stoch20, x)
    stoch20_mem = bench_memory(stoch20, x)
    results.append(BenchResult(
        "Stochastic(20%)", stoch20_fwd, stoch20_fwd_bwd, stoch20_mem,
        stoch20_fwd / gelu_fwd, stoch20_fwd_bwd / gelu_fwd_bwd
    ))

    # StochasticCantorGELU (50%)
    stoch50 = StochasticCantorGELU(apply_prob=0.5).to(device).train()
    stoch50_fwd = bench_forward(stoch50, x)
    stoch50_fwd_bwd = bench_forward_backward(stoch50, x)
    stoch50_mem = bench_memory(stoch50, x)
    results.append(BenchResult(
        "Stochastic(50%)", stoch50_fwd, stoch50_fwd_bwd, stoch50_mem,
        stoch50_fwd / gelu_fwd, stoch50_fwd_bwd / gelu_fwd_bwd
    ))

    # RouteDropout (needs 4D input)
    if len(shape) == 4:
        route_drop = RouteDropout(drop_prob=0.1).to(device).train()
        rd_fwd = bench_forward(route_drop, x)
        rd_fwd_bwd = bench_forward_backward(route_drop, x)
        rd_mem = bench_memory(route_drop, x)
        results.append(BenchResult(
            "RouteDropout", rd_fwd, rd_fwd_bwd, rd_mem,
            rd_fwd / gelu_fwd, rd_fwd_bwd / gelu_fwd_bwd
        ))

        # GELU + RouteDropout combo
        def gelu_route(x):
            return route_drop(F.gelu(x))

        gr_fwd = bench_forward(gelu_route, x)
        gr_fwd_bwd = bench_forward_backward(gelu_route, x)
        gr_mem = bench_memory(gelu_route, x)
        results.append(BenchResult(
            "GELU+RouteDrop", gr_fwd, gr_fwd_bwd, gr_mem,
            gr_fwd / gelu_fwd, gr_fwd_bwd / gelu_fwd_bwd
        ))

    return results


def print_results(scenario_name: str, shape: Tuple[int, ...], results: List[BenchResult]):
    """Pretty print benchmark results."""
    numel = 1
    for s in shape:
        numel *= s

    print(f"\n{'=' * 80}")
    print(f"SCENARIO: {scenario_name}")
    print(f"Shape: {list(shape)} = {numel:,} elements")
    print(f"{'=' * 80}")
    print(f"{'Module':<20} {'Fwd(ms)':<10} {'Fwd+Bwd(ms)':<12} {'Mem(MB)':<10} {'vs GELU(F)':<12} {'vs GELU(F+B)':<12}")
    print("-" * 80)

    for r in results:
        mem_str = f"{r.memory_mb:.1f}" if r.memory_mb > 0 else "-"
        print(
            f"{r.name:<20} {r.fwd_ms:<10.3f} {r.fwd_bwd_ms:<12.3f} {mem_str:<10} {r.vs_gelu_fwd:<12.2f}x {r.vs_gelu_fwd_bwd:<12.2f}x")


# =============================================================================
# SIMULATED DAVIDBEANS LAYER
# =============================================================================

class SimulatedWormholeBlock(nn.Module):
    """Simplified wormhole attention block for benchmarking."""

    def __init__(self, dim=512, num_heads=8, num_routes=8, mlp_ratio=4, activation='gelu'):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.num_routes = num_routes

        # Attention
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

        # MLP
        mlp_dim = dim * mlp_ratio
        self.mlp_fc1 = nn.Linear(dim, mlp_dim)
        self.mlp_fc2 = nn.Linear(mlp_dim, dim)

        # Activation
        if activation == 'gelu':
            self.act = nn.GELU()
        elif activation == 'cantor':
            self.act = CantorGELU()
        elif activation == 'stochastic':
            self.act = StochasticCantorGELU(apply_prob=0.2)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        # x: [B, N, D]
        B, N, D = x.shape

        # Attention (simplified - no actual routing)
        x = x + self._attn(self.norm1(x))
        x = x + self._mlp(self.norm2(x))

        return x

    def _attn(self, x):
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, D // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * (D // self.num_heads) ** -0.5
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, D)
        return self.proj(x)

    def _mlp(self, x):
        return self.mlp_fc2(self.act(self.mlp_fc1(x)))


def bench_davidbeans_layer():
    """Benchmark activation impact in realistic layer."""
    print("\n" + "=" * 80)
    print("DAVIDBEANS LAYER SIMULATION")
    print("=" * 80)

    device = torch.device('cuda')

    configs = [
        ("Small (B=8, N=256, D=512)", 8, 256, 512),
        ("Medium (B=16, N=256, D=512)", 16, 256, 512),
        ("Large (B=32, N=256, D=512)", 32, 256, 512),
        ("XL (B=64, N=256, D=512)", 64, 256, 512),
    ]

    activations = ['gelu', 'cantor', 'stochastic']

    for config_name, B, N, D in configs:
        print(f"\n--- {config_name} ---")
        print(f"{'Activation':<15} {'Fwd(ms)':<10} {'Fwd+Bwd(ms)':<12} {'vs GELU':<10}")
        print("-" * 50)

        x = torch.randn(B, N, D, device=device)
        gelu_time = None

        for act_name in activations:
            model = SimulatedWormholeBlock(dim=D, activation=act_name).to(device).train()

            fwd_ms = bench_forward(model, x, warmup=10, iters=50)
            fwd_bwd_ms = bench_forward_backward(model, x, warmup=10, iters=50)

            if act_name == 'gelu':
                gelu_time = fwd_bwd_ms
                ratio = 1.0
            else:
                ratio = fwd_bwd_ms / gelu_time

            print(f"{act_name:<15} {fwd_ms:<10.3f} {fwd_bwd_ms:<12.3f} {ratio:<10.2f}x")

            del model
            clear_cache()


# =============================================================================
# SCALING ANALYSIS
# =============================================================================

def bench_scaling():
    """How do activations scale with tensor size?"""
    print("\n" + "=" * 80)
    print("SCALING ANALYSIS")
    print("=" * 80)

    device = torch.device('cuda')

    sizes = [
        (8, 64, 8, 64),  # 262K
        (16, 128, 8, 64),  # 1M
        (32, 256, 8, 64),  # 4M (baseline)
        (64, 256, 8, 64),  # 8M
        (128, 256, 8, 64),  # 16M
        (256, 256, 8, 64),  # 32M
    ]

    print(f"\n{'Elements':<12} {'GELU(ms)':<10} {'Cantor(ms)':<12} {'Ratio':<8} {'Stoch20(ms)':<12} {'Ratio':<8}")
    print("-" * 70)

    cantor = CantorGELU().to(device).train()
    stoch = StochasticCantorGELU(apply_prob=0.2).to(device).train()

    for shape in sizes:
        numel = 1
        for s in shape:
            numel *= s

        x = torch.randn(shape, device=device)

        gelu_ms = bench_forward(F.gelu, x, warmup=10, iters=50)
        cantor_ms = bench_forward(cantor, x, warmup=10, iters=50)
        stoch_ms = bench_forward(stoch, x, warmup=10, iters=50)

        print(
            f"{numel:<12,} {gelu_ms:<10.3f} {cantor_ms:<12.3f} {cantor_ms / gelu_ms:<8.2f}x {stoch_ms:<12.3f} {stoch_ms / gelu_ms:<8.2f}x")

        del x
        clear_cache()


# =============================================================================
# THROUGHPUT TEST
# =============================================================================

def bench_throughput():
    """Sustained throughput over many iterations."""
    print("\n" + "=" * 80)
    print("THROUGHPUT TEST (1000 iterations)")
    print("=" * 80)

    device = torch.device('cuda')
    shape = (32, 256, 8, 64)
    x = torch.randn(shape, device=device)

    modules = [
        ("F.gelu", lambda x: F.gelu(x)),
        ("CantorGELU", CantorGELU().to(device).train()),
        ("Stochastic(20%)", StochasticCantorGELU(apply_prob=0.2).to(device).train()),
        ("RouteDropout", RouteDropout().to(device).train()),
    ]

    print(f"\nShape: {list(shape)}, 1000 forward passes")
    print(f"{'Module':<20} {'Total(ms)':<12} {'Per-iter(ms)':<12} {'Throughput(it/s)':<15}")
    print("-" * 60)

    for name, fn in modules:
        # Warmup
        for _ in range(50):
            _ = fn(x)
        sync()

        # Timed run
        start = time.perf_counter()
        for _ in range(1000):
            _ = fn(x)
        sync()
        total_ms = (time.perf_counter() - start) * 1000

        per_iter = total_ms / 1000
        throughput = 1000 / (total_ms / 1000)

        print(f"{name:<20} {total_ms:<12.1f} {per_iter:<12.3f} {throughput:<15.1f}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("CANTOR REGULARIZATION BENCHMARK SUITE")
    print("=" * 80)
    print(f"PyTorch: {torch.__version__}")
    print(f"Triton:  {triton.__version__}")
    print(f"CUDA:    {torch.version.cuda}")
    print(f"Device:  {torch.cuda.get_device_name(0)}")
    print("=" * 80)

    device = torch.device('cuda')

    # Scenario 1: Standard wormhole attention shape
    results = run_scenario("Wormhole Attention", (32, 256, 8, 64), device)
    print_results("Wormhole Attention", (32, 256, 8, 64), results)

    # Scenario 2: Larger batch
    results = run_scenario("Large Batch", (128, 256, 8, 64), device)
    print_results("Large Batch", (128, 256, 8, 64), results)

    # Scenario 3: MLP shape (no routes)
    results = run_scenario("MLP Activation", (32, 256, 2048), device)
    print_results("MLP Activation", (32, 256, 2048), results)

    # Scenario 4: Small/quick inference
    results = run_scenario("Small Inference", (4, 64, 8, 64), device)
    print_results("Small Inference", (4, 64, 8, 64), results)

    # Additional analyses
    bench_scaling()
    bench_throughput()
    bench_davidbeans_layer()

    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETE")
    print("=" * 80)