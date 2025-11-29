# Cantor Routing Experiments: Findings and Proofs

**Date:** November 28, 2025  
**Researchers:** AbstractPhil, Claude  
**Focus:** Stress-testing Cantor routing assumptions, ViT-Beans architecture validation

---

## Executive Summary

This session systematically tested assumptions about Cantor-based sparse attention routing. We discovered fundamental limitations of scalar Cantor distance for uniform coverage tasks, identified and fixed critical bugs in the ViT-Beans architecture, and validated that the corrected model learns on CIFAR-10.

### Key Findings

| Finding | Implication |
|---------|-------------|
| Scalar Cantor distance creates hub topology, not uniform coverage | Use Cantor for wormhole tasks, not grid coverage |
| Learnable parameters (centers, tau, weights, alpha) cannot fix topology | The distance metric itself determines connectivity |
| ViT-Beans v1 had disconnected CLS token | Zero gradients in 4/7 components |
| Hybrid routing (Cantor + positional) improves coverage | Balance shortcuts with local connectivity |
| Fixed ViT-Beans v2 learns on CIFAR-10 | 51%+ accuracy by epoch 3 |

---

## Experiment 1: Staircase Depth Analysis

### Question
Why was level L=5 chosen for sequence length 8192? Is it mathematically justified?

### Method
Computed required depth for proper resolution:
```
For sequence length N, need L ≥ log₃(N)
  seq_len=8192: log₃(N) = 8.2 → need L ≥ 9

Current L=5: 3^5 = 243 buckets for 8192 positions (33x undersampled!)
```

### Results
| L | 3^L | Unique Values | Hub σ | Linear Connectivity |
|---|-----|---------------|-------|---------------------|
| 5 | 243 | 756 | 5.5 | 0.00% |
| 9 | 19,683 | 763 | 6.1 | 0.00% |
| 13 | 1,594,323 | 766 | 6.1 | 0.00% |

### Finding
**Depth L does not affect linear patchwork connectivity.** With tau=0.25, softmax temperature saturates at ~760 unique Cantor values regardless of depth. The "hubs" at L=5 are artifacts of insufficient resolution that disappear at proper resolution, but the fundamental connectivity problem persists.

### File
`/mnt/user-data/outputs/fractalbert_v2_depth_analysis.py`

---

## Experiment 2: Hardcoded Parameters Investigation

### Question
Why are staircase centers hardcoded to [0, 0.5, 1]? Could learning them help?

### Configurations Tested
| Config | Centers | Description |
|--------|---------|-------------|
| Current | [0, 0.5, 1] | Arbitrary |
| Raw ternary | [0, 1, 2] | Digit values |
| Interval midpoints | [1/6, 1/2, 5/6] | Ternary interval centers |
| Ternary boundaries | [0, 1/3, 2/3] | Division points |
| Learned | nn.Parameter | Optimized |

### Learnable Configuration
```python
class LearnableBeatrixStaircase:
    centers: nn.Parameter([c₀, c₁, c₂])  # Ternary digit embedding
    tau: nn.Parameter(τ)                  # Softmax temperature
    weights: nn.Parameter([w₀...w_L])     # Level contributions
```

### Results (Uniform Distribution Target)
```
Initial: centers=[0, 1, 2], tau=0.25, weights=uniform
Final:   centers=[0.127, 1.425, 2.760], tau=0.747, weights=[0.062, 0.890, 0.019...]

Unique values: 711 → 910 (+28%)
Loss vs uniform: 0.004 → 0.0004 (10x better)
```

### Finding
**Weight collapse occurs:** Level 1 dominates at 89%. The model learned to nearly ignore fractal structure, using primarily one level for uniform coverage. This suggests the multi-scale structure isn't helpful for uniform distribution.

### File
`/mnt/user-data/outputs/learnable_beatrix_staircase.py`

---

## Experiment 3: Wormhole Task with Learnable Staircase

### Question
Can learnable parameters improve wormhole task performance?

### Configurations
1. Fixed L=5 (original)
2. Fixed L=9 (proper resolution)
3. Learned (centers, tau, weights)
4. Per-level (different centers per level)

### Results
| Config | Accuracy | Tau | Centers |
|--------|----------|-----|---------|
| Fixed L=5 | 100.00% | 0.250 | [0.000, 0.500, 1.000] |
| Fixed L=9 | 100.00% | 0.250 | [0.000, 0.500, 1.000] |
| Learned | 100.00% | 0.249 | [-0.005, 0.999, 2.006] |
| Per-level | 100.00% | 0.250 | [-0.004, 0.996, 1.995] |

### Finding
**Wormhole task is too easy** - provides no gradient signal for parameter optimization. Parameters barely moved from initialization. All configurations achieve 100% accuracy because Cantor routing naturally creates the highways needed for wormhole navigation.

### File
`/mnt/user-data/outputs/fractalbert_learnable_staircase.py`

---

## Experiment 4: Linear Patchwork with Learnable Staircase

### Question
Can learning fix the 0% connectivity problem for evenly-spaced positions?

### Setup
- 8 patches at positions [0, 1024, 2048, 3072, 4096, 5120, 6144, 7168]
- Sequence length: 8192
- k = 64 neighbors
- 100 epochs, aggressive LR=0.1 for staircase parameters

### Results
| Config | Accuracy | Connectivity | Centers |
|--------|----------|--------------|---------|
| Fixed L=5 | 14.29% | 0.00% | [0.000, 0.500, 1.000] |
| Learned (aggressive) | 14.29% | 0.00% | [0.035, 0.938, 1.978] |
| Per-level (full flex) | 14.29% | 0.00% | [0.105, 1.000, 2.152] |

### Hop Matrix (All Configs)
```
    0  1  2  3  4  5  6  7
0   ·  ✗  ✗  ✗  ✗  ✗  ✗  ✗
1   ✗  ·  ✗  ✗  ✗  ✗  ✗  ✗
2   ✗  ✗  ·  ✗  ✗  ✗  ✗  ✗
...
```

### Finding
**Learning cannot fix the fundamental distance metric issue.** The Cantor distance creates a topology where evenly-spaced positions are inherently far apart. No amount of parameter tuning can make sequential distance equal Cantor distance. The 14.29% accuracy is random chance (1/7 for 7 possible source patches).

### File
`/mnt/user-data/outputs/fractalbert_learnable_patchwork.py`

---

## Experiment 5: Alpha Saturation Hypothesis

### Hypothesis
The stress test proof uses learnable alpha controlling middle-third saturation:
```python
bit_k = p[..., 2] + self.alpha * p[..., 1]
```

As alpha → 1, Cantor gaps fill in, potentially enabling uniform coverage.

### Configurations
1. Fixed alpha=0.5 (frozen)
2. Learnable alpha (shared, LR=0.1)
3. Learnable alpha (per-level, LR=0.1)

### Results (150 epochs)
| Config | Accuracy | Connectivity | Alpha |
|--------|----------|--------------|-------|
| Fixed alpha=0.5 | 12.50% | 0.00% | 0.622 (frozen) |
| Learnable (shared) | 12.50% | 0.00% | 0.622 → 0.885 |
| Learnable (per-level) | 12.50% | 3.57% | [0.159, 0.955] |

### Alpha Evolution (Shared)
```
Epoch  30: α=0.795
Epoch  60: α=0.825
Epoch  90: α=0.832
Epoch 120: α=0.858
Epoch 150: α=0.885  ← Saturating toward 1.0
```

### Finding
**Alpha does saturate toward 1.0** as hypothesized, but this still uses scalar Cantor distance. Per-level alpha achieved slight improvement (0% → 3.57% connectivity) but fundamentally cannot solve the topology problem.

### File
`/mnt/user-data/outputs/fractalbert_alpha_saturation_v2.py`

---

## Experiment 6: Scalar vs Fingerprint Distance (Theoretical)

### The Fundamental Issue Identified

**Current Routing (Scalar):**
```python
cantor = sum(bit_k * 2^(-k-1))  # Single number in [0,1]
D = |cantor_i - cantor_j|       # Scalar distance
```

**What We're Throwing Away:**
```python
fingerprint = [
    [bit_0, entropy_0],   # 1/3 scale (coarse)
    [bit_1, entropy_1],   # 1/9 scale
    ...
    [bit_15, entropy_15]  # 1/3^16 scale (fine)
]
```

### Insight
- **Sequentially close positions:** Similar fingerprints at FINE levels (high k), different at coarse
- **Cantor-close positions (hubs):** Similar fingerprints at COARSE levels (low k), different at fine
- **Current routing:** Uses collapsed scalar, loses multi-scale structure

### Potential Solution (Not Fully Tested)
```python
d(i, j) = Σ_k weight_k * |features[i, k] - features[j, k]|

# Weight schemes:
# Coarse-weighted: w_k = 2^(-k) → current behavior, creates hubs
# Fine-weighted: w_k = 2^(k-L) → prioritizes sequential proximity
# Learned: w_k = softmax(θ_k) → task-adaptive
```

### Status
Identified as potential avenue for future work. Not fully implemented/tested this session.

---

## Experiment 7: ViT-Beans v1 Architecture Analysis

### Bug Discovery

**Bug 1: CLS Token Disconnected**
```python
# Original code:
cls_tok, patches = x[:, :1], x[:, 1:]
for block in self.blocks:
    patches = block(patches)  # CLS not included!
return self.head(self.norm(x[:, 0]))  # Classifying from disconnected CLS
```

**Bug 2: Cantor Used for Assignment, Not Routing**
```python
# Each expert only saw ~4 patches:
mask = (fingerprints >= fp_min) & (fingerprints < fp_max)
my_tokens = tokens[:, mask]  # Isolated patches!
scores = torch.bmm(Q, K.transpose(1, 2))  # Dense 4x4 attention
```

### Gradient Analysis (v1)
| Component | Gradient Norm |
|-----------|---------------|
| patch_embed | 0.000 ✗ |
| pos_embed | 119.599 ✓ |
| cls_token | 119.599 ✓ |
| qkv (block 0) | 0.000 ✗ |
| pentachoron | 0.000 ✗ |
| mlp (block 0) | 0.000 ✗ |
| head | 71.180 ✓ |

### Finding
**4 of 7 critical components had zero gradients.** The CLS token received gradient from the classification head backward through pos_embed, but the attention mechanism and experts were completely disconnected from the loss.

### Files
- Original: `vit_beans.py` (broken)
- Analysis: `vit_beans_v2_fixed.py`

---

## Experiment 8: ViT-Beans v2 Fixes

### Fixes Applied

1. **CLS Participates in Attention**
```python
# CLS gets dense attention to ALL patches
scores_cls = torch.einsum('bhqd,bhkd->bhqk', Q_cls, K) * self.scale  # [B, H, 1, S]
```

2. **Hybrid Routing**
```python
# Balance Cantor shortcuts with positional locality
D_hybrid = α * D_cantor + (1 - α) * D_positional
# Default α=0.3
```

3. **Experts Partition Features, Not Patches**
```python
# All experts see ALL patches, process different feature slices
x_slice = x[..., self.slice_start:self.slice_end]
```

### Connectivity Improvement
| α | Description | 3-hop Coverage |
|---|-------------|----------------|
| 1.0 | Pure Cantor | 31/64 (48%) |
| 0.7 | Hybrid | 44/64 (69%) |
| 0.5 | Hybrid | 49/64 (77%) |
| 0.3 | Hybrid | 53/64 (83%) |
| 0.0 | Pure positional | 58/64 (91%) |

### Gradient Analysis (v2 Fixed)
| Component | Gradient Norm |
|-----------|---------------|
| patch_embed | 11.989 ✓ |
| pos_embed | 36.151 ✓ |
| cls_token | 36.101 ✓ |
| qkv (block 0) | 31.906 ✓ |
| pentachoron | 1.608 ✓ |
| mlp (block 0) | 40.822 ✓ |
| head | 65.016 ✓ |

### Files
- `vit_beans_v2_debug.py` (fixed implementation)
- `vit_beans_migration_guide.py` (documentation)

---

## Experiment 9: CIFAR-10 Training Validation

### Configuration
```python
ViTBeansConfigV2(
    image_size=32,
    patch_size=4,
    dim=256,
    num_layers=4,
    num_heads=4,
    num_experts=4,
    k_neighbors=16,
    cantor_weight=0.3,
    num_classes=10,
)
# Parameters: 3,362,586
```

### Training Results
| Epoch | Train Loss | Train Acc | Test Loss | Test Acc |
|-------|------------|-----------|-----------|----------|
| 1 | 1.8221 | 31.71% | 1.5802 | 41.95% |
| 2 | 1.5819 | 41.60% | 1.4374 | 47.19% |
| 3 | 1.4327 | 47.45% | 1.3488 | 51.19% |

### Observations
- **Test > Train early:** Dropout and augmentation working properly
- **Geometric loss stable:** ~0.102 throughout (pentachora not collapsing)
- **Consistent improvement:** +5.2%, +4.0% per epoch
- **Throughput:** ~87s/epoch on CUDA

### Finding
**The fixed model learns.** Projected to reach ~55-60% by epoch 10, which is reasonable for a small ViT (256 dim, 4 layers, 3.3M params).

### File
`/mnt/user-data/outputs/train_cifar10.py`

---

## Summary: What Cantor Routing Is Good For

### ✅ Use Cantor Routing For:
- **Wormhole/teleportation tasks:** Natural hub structure enables efficient long-range information flow
- **Sparse attention with shortcuts:** O(n*k) complexity with global connectivity through hubs
- **Tasks with natural hierarchical structure:** Cantor geometry matches hierarchical data

### ❌ Do Not Use Cantor Routing For:
- **Uniform grid coverage:** Evenly-spaced positions are Cantor-isolated
- **Tasks requiring all-to-all potential connectivity:** Use hybrid routing instead
- **Replacing local attention entirely:** Combine with positional for local patterns

### Parameters That Don't Matter (for connectivity):
- Staircase depth L (5 vs 9 vs 13)
- Center values [0, 0.5, 1] vs [0.5, 1.5, 2.5]
- Tau (0.25 vs learned)
- Level weights (geometric vs learned)

### Parameters That Might Help:
- **Alpha saturation:** Fills Cantor gaps but doesn't fix scalar metric
- **Hybrid routing:** α * Cantor + (1-α) * positional
- **Fingerprint-based distance:** Use full [S, L, 2] structure (untested)

---

## Files Created This Session

| File | Purpose |
|------|---------|
| `fractalbert_v2_depth_analysis.py` | Staircase depth analysis |
| `learnable_beatrix_staircase.py` | Learnable centers, tau, weights |
| `fractalbert_learnable_staircase.py` | Wormhole task with learnable params |
| `fractalbert_learnable_patchwork.py` | Linear patchwork with learnable params |
| `fractalbert_alpha_saturation_v2.py` | Alpha saturation experiments |
| `fingerprint_routing.py` | Fingerprint-based distance analysis |
| `vit_beans_v2_fixed.py` | Corrected ViT-Beans implementation |
| `vit_beans_v2_debug.py` | Debug version with hybrid routing |
| `vit_beans_migration_guide.py` | Migration documentation |
| `train_cifar10.py` | CIFAR-10 training script |

---

## Open Questions for Future Work

1. **Fingerprint distance metric:** Does using full [S, L, 2] fingerprint structure enable both long-range and local routing?

2. **Multi-scale routing:** Different k per level? Coarse levels for long-range, fine levels for local?

3. **Hybrid architecture:** Cantor attention (coarse) + sliding window (fine)?

4. **Task-dependent alpha:** Learn α per-head or per-layer?

5. **Hub utilization:** Can we explicitly route through known hub positions?

---

*Document generated: November 28, 2025*