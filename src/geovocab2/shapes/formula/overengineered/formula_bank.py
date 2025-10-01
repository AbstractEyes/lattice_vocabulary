# geo_formula_enhanced.py - Extended library with 50+ formulas
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum, auto
import torch
import torch.nn.functional as F
from torch import Tensor
import time

from shapes.tensor.geo_tensor import Tier, GeoTensor


# ========= FORMULA CATEGORIES =========

class FormulaCategory(Enum):
    """Categories for organizing 50+ formulas."""
    # Core geometric operations
    MIXING = auto()
    PROJECTION = auto()
    NORMALIZATION = auto()

    # Attention mechanisms
    ATTENTION_BASIC = auto()
    ATTENTION_FLASH = auto()
    ATTENTION_SPARSE = auto()
    ATTENTION_RELATIVE = auto()

    # Similarity metrics
    SIMILARITY_COSINE = auto()
    SIMILARITY_EUCLIDEAN = auto()
    SIMILARITY_LEARNED = auto()

    # Transformations
    TRANSFORM_LINEAR = auto()
    TRANSFORM_NONLINEAR = auto()
    TRANSFORM_ADAPTIVE = auto()

    # Aggregations
    AGGREGATION_MEAN = auto()
    AGGREGATION_WEIGHTED = auto()
    AGGREGATION_HIERARCHICAL = auto()
    AGGREGATION_POOLING = auto()

    # Gates and routing
    GATING = auto()
    ROUTING = auto()

    # Specialized
    SPECIALIZED_NLP = auto()
    SPECIALIZED_VISION = auto()
    SPECIALIZED_GRAPH = auto()

    # Experimental
    EXPERIMENTAL = auto()




# ========= FORMULA BASE CLASSES =========

class BaseFormula:
    """Base class for all formulas."""

    def __init__(self, eps: float = 1e-12, **config):
        self.eps = eps
        self.config = config
        self._profiling_data = []

    def forward(self, *args: Tensor, **kwargs) -> Dict[str, Tensor]:
        raise NotImplementedError

    def __call__(self, *args: Tensor, profile: bool = False, **kwargs) -> Dict[str, Tensor]:
        if profile and torch.cuda.is_available():
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            result = self.forward(*args, **kwargs)
            end.record()
            torch.cuda.synchronize()
            self._profiling_data.append(start.elapsed_time(end))
            return result
        return self.forward(*args, **kwargs)


# ========= EXTENDED FORMULA IMPLEMENTATIONS (50+) =========

# === MIXING FORMULAS (10) ===

class LinearMix(BaseFormula):
    def forward(self, x: Tensor, y: Tensor, alpha: Tensor, **kwargs) -> Dict[str, Tensor]:
        out = torch.lerp(x, y, alpha.clamp(0, 1))
        gate = (out - x).norm(dim=-1, keepdim=True)
        return {"out": out, "gate": gate}


class GatedMix(BaseFormula):
    def forward(self, x: Tensor, y: Tensor, gate: Tensor, **kwargs) -> Dict[str, Tensor]:
        gate_sigmoid = torch.sigmoid(gate)
        out = x * (1 - gate_sigmoid) + y * gate_sigmoid
        return {"out": out, "gate": gate_sigmoid}


class AdaptiveMix(BaseFormula):
    def forward(self, x: Tensor, y: Tensor, context: Tensor, **kwargs) -> Dict[str, Tensor]:
        weights = F.softmax(context.mean(dim=-1, keepdim=True), dim=-2)
        out = x * (1 - weights) + y * weights
        return {"out": out, "weights": weights}


class SmoothMix(BaseFormula):
    def forward(self, x: Tensor, y: Tensor, t: Tensor, **kwargs) -> Dict[str, Tensor]:
        # Smooth interpolation using cosine
        t_smooth = 0.5 * (1 - torch.cos(torch.pi * t.clamp(0, 1)))
        out = torch.lerp(x, y, t_smooth)
        return {"out": out, "smooth_factor": t_smooth}


class HierarchicalMix(BaseFormula):
    def forward(self, x: Tensor, y: Tensor, z: Tensor, w1: Tensor, w2: Tensor, **kwargs) -> Dict[str, Tensor]:
        # Two-level mixing
        mix1 = torch.lerp(x, y, w1.clamp(0, 1))
        out = torch.lerp(mix1, z, w2.clamp(0, 1))
        return {"out": out}


class ConditionalMix(BaseFormula):
    def forward(self, x: Tensor, y: Tensor, condition: Tensor, **kwargs) -> Dict[str, Tensor]:
        # Mix based on condition threshold
        mask = (condition > 0.5).float()
        out = x * (1 - mask) + y * mask
        return {"out": out, "mask": mask}


class WeightedTriMix(BaseFormula):
    def forward(self, x: Tensor, y: Tensor, z: Tensor, **kwargs) -> Dict[str, Tensor]:
        # Equal weighted three-way mix
        out = (x + y + z) / 3.0
        return {"out": out}


class ExpMix(BaseFormula):
    def forward(self, x: Tensor, y: Tensor, alpha: Tensor, **kwargs) -> Dict[str, Tensor]:
        # Exponential weighting
        weight = torch.exp(-alpha.clamp(0, 10))
        out = x * weight + y * (1 - weight)
        return {"out": out, "weight": weight}


class MomentumMix(BaseFormula):
    def forward(self, x: Tensor, y: Tensor, momentum: Tensor, **kwargs) -> Dict[str, Tensor]:
        # Momentum-based mixing
        mom = momentum.clamp(0, 0.999)
        out = x * mom + y * (1 - mom)
        return {"out": out}


class StochasticMix(BaseFormula):
    def forward(self, x: Tensor, y: Tensor, **kwargs) -> Dict[str, Tensor]:
        # Random mixing weights
        alpha = torch.rand_like(x[..., :1])
        out = torch.lerp(x, y, alpha)
        return {"out": out, "alpha": alpha}


# === ATTENTION FORMULAS (10) ===

class ScaledDotProductAttention(BaseFormula):
    def forward(self, q: Tensor, k: Tensor, v: Tensor, **kwargs) -> Dict[str, Tensor]:
        d_k = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (d_k ** 0.5)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        return {"out": out, "attention": attn}


class FlashAttention(BaseFormula):
    def forward(self, q: Tensor, k: Tensor, v: Tensor, **kwargs) -> Dict[str, Tensor]:
        if hasattr(F, 'scaled_dot_product_attention'):
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
            gate = torch.ones(q.shape[:-1] + (1,), device=q.device)
            return {"out": out, "gate": gate}
        else:
            return ScaledDotProductAttention().forward(q, k, v, **kwargs)


class LocalAttention(BaseFormula):
    def __init__(self, window_size: int = 256, **config):
        super().__init__(**config)
        self.window_size = window_size

    def forward(self, q: Tensor, k: Tensor, v: Tensor, **kwargs) -> Dict[str, Tensor]:
        B, L, D = q.shape
        out = torch.zeros_like(v)
        for i in range(0, L, self.window_size):
            end = min(i + self.window_size, L)
            q_window = q[:, i:end]
            k_window = k[:, i:end]
            v_window = v[:, i:end]
            scores = torch.matmul(q_window, k_window.transpose(-2, -1)) / (D ** 0.5)
            attn = F.softmax(scores, dim=-1)
            out[:, i:end] = torch.matmul(attn, v_window)
        return {"out": out}


class MultiQueryAttention(BaseFormula):
    def forward(self, q: Tensor, k: Tensor, v: Tensor, **kwargs) -> Dict[str, Tensor]:
        # Multi-query attention (single K,V for multiple Q)
        B, H, L, D = q.shape[0], 1, q.shape[1], q.shape[2]
        scores = torch.matmul(q, k.transpose(-2, -1)) / (D ** 0.5)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        return {"out": out}


class CrossAttention(BaseFormula):
    def forward(self, q: Tensor, kv: Tensor, **kwargs) -> Dict[str, Tensor]:
        # Optimized cross attention with fused operations
        d_k = q.size(-1) ** 0.5

        # Single matmul with scaling
        scores = torch.baddbmm(
            torch.empty(q.size(0), q.size(1), kv.size(1), device=q.device, dtype=q.dtype).zero_(),
            q, kv.transpose(-2, -1),
            beta=0, alpha=1.0 / d_k
        )

        # Fused softmax
        attn = F.softmax(scores, dim=-1)

        # Output with single matmul
        out = torch.matmul(attn, kv)

        return {"out": out, "cross_attention": attn}


class LinearAttention(BaseFormula):
    def forward(self, q: Tensor, k: Tensor, v: Tensor, **kwargs) -> Dict[str, Tensor]:
        # Optimized linear attention using efficient associative property
        B, L, D = q.shape

        # Feature map with ReLU instead of ELU for better performance
        q = F.relu(q)
        k = F.relu(k)

        # Compute KV first (D x D matrix) - this is the key optimization
        # This reduces complexity from O(L²D) to O(LD²)
        kv = torch.einsum('bld,ble->bde', k, v)

        # Apply to queries
        out = torch.einsum('bld,bde->ble', q, kv)

        # Normalization
        k_sum = k.sum(dim=-2, keepdim=True)
        denominator = torch.einsum('bld,bnd->bln', q, k_sum.transpose(-2, -1)) + self.eps
        out = out / denominator

        return {"out": out}


class GatedAttention(BaseFormula):
    def forward(self, q: Tensor, k: Tensor, v: Tensor, gate: Tensor, **kwargs) -> Dict[str, Tensor]:
        d_k = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (d_k ** 0.5)
        scores = scores * gate.sigmoid()
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        return {"out": out}


class RelativeAttention(BaseFormula):
    def forward(self, q: Tensor, k: Tensor, v: Tensor, **kwargs) -> Dict[str, Tensor]:
        # Simplified relative position attention
        B, L, D = q.shape
        pos = torch.arange(L, device=q.device).unsqueeze(0) - torch.arange(L, device=q.device).unsqueeze(1)
        pos_enc = torch.sin(pos.float().unsqueeze(-1) / 10000)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (D ** 0.5)
        scores = scores + pos_enc.unsqueeze(0)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        return {"out": out}


class SparseAttention(BaseFormula):
    def forward(self, q: Tensor, k: Tensor, v: Tensor, **kwargs) -> Dict[str, Tensor]:
        # Optimized top-k sparse attention
        d_k = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (d_k ** 0.5)

        # More efficient sparse attention using masking
        top_k = min(32, scores.size(-1))
        topk_scores, topk_indices = scores.topk(top_k, dim=-1)

        # Create sparse mask instead of manual loops
        mask = torch.zeros_like(scores)
        mask.scatter_(-1, topk_indices, 1.0)

        # Apply mask and softmax
        masked_scores = scores * mask + (1 - mask) * float('-inf')
        attn = F.softmax(masked_scores, dim=-1)
        out = torch.matmul(attn, v)

        return {"out": out}


class CausalAttention(BaseFormula):
    def forward(self, q: Tensor, k: Tensor, v: Tensor, **kwargs) -> Dict[str, Tensor]:
        d_k = q.size(-1)
        L = q.size(-2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (d_k ** 0.5)
        mask = torch.triu(torch.ones(L, L, device=q.device), diagonal=1).bool()
        scores.masked_fill_(mask, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        return {"out": out}


# === SIMILARITY FORMULAS (8) ===

class CosineSimilarity(BaseFormula):
    def forward(self, x: Tensor, y: Tensor, **kwargs) -> Dict[str, Tensor]:
        # Optimized cosine similarity with fused normalization
        # Use F.cosine_similarity which is highly optimized
        similarity = F.cosine_similarity(x, y, dim=-1, eps=self.eps).unsqueeze(-1)
        return {"similarity": similarity}


class EuclideanSimilarity(BaseFormula):
    def forward(self, x: Tensor, y: Tensor, **kwargs) -> Dict[str, Tensor]:
        # Optimized Euclidean distance using squared L2 norm identity
        # ||x - y||² = ||x||² + ||y||² - 2⟨x,y⟩
        x_norm_sq = (x ** 2).sum(dim=-1, keepdim=True)
        y_norm_sq = (y ** 2).sum(dim=-1, keepdim=True)
        xy = (x * y).sum(dim=-1, keepdim=True)

        # Compute distance efficiently
        distance_sq = x_norm_sq + y_norm_sq - 2 * xy
        distance = (distance_sq + self.eps).sqrt()

        # Convert to similarity
        similarity = 1.0 / (1.0 + distance)

        return {"similarity": similarity, "distance": distance}


class DotProductSimilarity(BaseFormula):
    def forward(self, x: Tensor, y: Tensor, **kwargs) -> Dict[str, Tensor]:
        similarity = (x * y).sum(dim=-1, keepdim=True)
        return {"similarity": similarity}


class ManhattanSimilarity(BaseFormula):
    def forward(self, x: Tensor, y: Tensor, **kwargs) -> Dict[str, Tensor]:
        distance = (x - y).abs().sum(dim=-1, keepdim=True)
        similarity = 1.0 / (1.0 + distance)
        return {"similarity": similarity, "distance": distance}


class MinkowskiSimilarity(BaseFormula):
    def __init__(self, p: float = 3.0, **config):
        super().__init__(**config)
        self.p = p

    def forward(self, x: Tensor, y: Tensor, **kwargs) -> Dict[str, Tensor]:
        distance = (x - y).abs().pow(self.p).sum(dim=-1, keepdim=True).pow(1 / self.p)
        similarity = 1.0 / (1.0 + distance)
        return {"similarity": similarity, "distance": distance}


class PearsonSimilarity(BaseFormula):
    def forward(self, x: Tensor, y: Tensor, **kwargs) -> Dict[str, Tensor]:
        # Optimized Pearson correlation using vectorized operations
        # Compute means efficiently
        x_mean = x.mean(dim=-1, keepdim=True)
        y_mean = y.mean(dim=-1, keepdim=True)

        # Center the data
        x_c = x - x_mean
        y_c = y - y_mean

        # Compute correlation using fused operations
        # Numerator: sum of products
        num = (x_c * y_c).sum(dim=-1, keepdim=True)

        # Denominator: product of standard deviations
        # Use fused operations to avoid redundant memory access
        x_var = (x_c ** 2).sum(dim=-1, keepdim=True)
        y_var = (y_c ** 2).sum(dim=-1, keepdim=True)
        denom = (x_var * y_var).sqrt() + self.eps

        similarity = num / denom

        return {"similarity": similarity}


class JaccardSimilarity(BaseFormula):
    def forward(self, x: Tensor, y: Tensor, **kwargs) -> Dict[str, Tensor]:
        # For continuous vectors, use soft Jaccard
        intersection = torch.min(x, y).sum(dim=-1, keepdim=True)
        union = torch.max(x, y).sum(dim=-1, keepdim=True)
        similarity = intersection / (union + self.eps)
        return {"similarity": similarity}


class BilinearSimilarity(BaseFormula):
    def forward(self, x: Tensor, y: Tensor, weight: Tensor, **kwargs) -> Dict[str, Tensor]:
        # x^T W y similarity
        xw = torch.matmul(x.unsqueeze(-2), weight)
        similarity = torch.matmul(xw, y.unsqueeze(-1)).squeeze(-1)
        return {"similarity": similarity}


# === PROJECTION FORMULAS (7) ===

class UnitSphereProjection(BaseFormula):
    def forward(self, x: Tensor, **kwargs) -> Dict[str, Tensor]:
        out = F.normalize(x, dim=-1, eps=self.eps)
        scale = x.norm(dim=-1, keepdim=True)
        return {"out": out, "scale": scale}


class OrthogonalProjection(BaseFormula):
    def forward(self, x: Tensor, basis: Tensor, **kwargs) -> Dict[str, Tensor]:
        # Project x onto the space spanned by basis
        basis_norm = F.normalize(basis, dim=-1, eps=self.eps)
        projection = (x * basis_norm).sum(dim=-1, keepdim=True) * basis_norm
        return {"out": projection}


class RandomProjection(BaseFormula):
    def forward(self, x: Tensor, target_dim: int = None, **kwargs) -> Dict[str, Tensor]:
        D = x.size(-1)
        target_dim = target_dim or D // 2
        projection_matrix = torch.randn(D, target_dim, device=x.device) / (D ** 0.5)
        out = torch.matmul(x, projection_matrix)
        return {"out": out}


class PCAProjection(BaseFormula):
    def forward(self, x: Tensor, components: int = None, **kwargs) -> Dict[str, Tensor]:
        # Fast approximate PCA using torch.pca_lowrank
        components = components or min(x.size(-1) // 2, 6)  # Limit components for speed
        B, L, D = x.shape

        # Flatten batch dimension for PCA
        x_flat = x.view(-1, D)
        mean = x_flat.mean(dim=0, keepdim=True)
        x_centered = x_flat - mean

        # Use fast low-rank approximation instead of full eigendecomposition
        try:
            U, S, V = torch.pca_lowrank(x_centered, q=components, niter=2)
            # V contains the principal components
            out = torch.matmul(x_centered, V)
            out = out.view(B, L, components)
            eigenvalues = S[:components] ** 2 / (x_flat.size(0) - 1)
        except:
            # Fallback to simple random projection if PCA fails
            projection = torch.randn(D, components, device=x.device) / (D ** 0.5)
            out = torch.matmul(x, projection)
            eigenvalues = torch.ones(components, device=x.device)

        return {"out": out, "eigenvalues": eigenvalues}


class HyperbolicProjection(BaseFormula):
    def forward(self, x: Tensor, **kwargs) -> Dict[str, Tensor]:
        # Project to hyperbolic space (Poincaré ball)
        norm = x.norm(dim=-1, keepdim=True)
        scale = torch.tanh(norm) / (norm + self.eps)
        out = x * scale
        return {"out": out}


class StereographicProjection(BaseFormula):
    def forward(self, x: Tensor, **kwargs) -> Dict[str, Tensor]:
        # Stereographic projection
        norm_sq = (x ** 2).sum(dim=-1, keepdim=True)
        out = 2 * x / (1 + norm_sq + self.eps)
        return {"out": out}


class ConvexProjection(BaseFormula):
    def forward(self, x: Tensor, **kwargs) -> Dict[str, Tensor]:
        # Project to probability simplex
        out = F.softmax(x, dim=-1)
        return {"out": out}


# === AGGREGATION FORMULAS (8) ===

class WeightedMeanAggregation(BaseFormula):
    def forward(self, x: Tensor, weights: Tensor, **kwargs) -> Dict[str, Tensor]:
        weights_norm = F.softmax(weights, dim=-2)
        out = (x * weights_norm).sum(dim=-2, keepdim=True)
        return {"out": out, "weights": weights_norm}


class HierarchicalAggregation(BaseFormula):
    def forward(self, x: Tensor, levels: int = 2, **kwargs) -> Dict[str, Tensor]:
        out = x
        intermediates = []
        for level in range(levels):
            if out.shape[-2] > 1:
                out = (out[..., 0::2, :] + out[..., 1::2, :]) / 2
                intermediates.append(out)
        return {"out": out, "intermediates": intermediates}


class MaxPoolAggregation(BaseFormula):
    def forward(self, x: Tensor, **kwargs) -> Dict[str, Tensor]:
        out, indices = x.max(dim=-2, keepdim=True)
        return {"out": out, "indices": indices}


class MinPoolAggregation(BaseFormula):
    def forward(self, x: Tensor, **kwargs) -> Dict[str, Tensor]:
        out, indices = x.min(dim=-2, keepdim=True)
        return {"out": out, "indices": indices}


class GeometricMeanAggregation(BaseFormula):
    def forward(self, x: Tensor, **kwargs) -> Dict[str, Tensor]:
        # Compute geometric mean (avoiding negative values)
        x_positive = x.abs() + self.eps
        log_mean = x_positive.log().mean(dim=-2, keepdim=True)
        out = log_mean.exp()
        return {"out": out}


class HarmonicMeanAggregation(BaseFormula):
    def forward(self, x: Tensor, **kwargs) -> Dict[str, Tensor]:
        # Harmonic mean
        reciprocal = 1.0 / (x.abs() + self.eps)
        mean_reciprocal = reciprocal.mean(dim=-2, keepdim=True)
        out = 1.0 / (mean_reciprocal + self.eps)
        return {"out": out}


class MedianAggregation(BaseFormula):
    def forward(self, x: Tensor, **kwargs) -> Dict[str, Tensor]:
        out = x.median(dim=-2, keepdim=True)[0]
        return {"out": out}


class QuantileAggregation(BaseFormula):
    def __init__(self, quantile: float = 0.75, **config):
        super().__init__(**config)
        self.quantile = quantile

    def forward(self, x: Tensor, **kwargs) -> Dict[str, Tensor]:
        out = torch.quantile(x, self.quantile, dim=-2, keepdim=True)
        return {"out": out}


# === GATING FORMULAS (5) ===

class SigmoidGate(BaseFormula):
    def forward(self, x: Tensor, **kwargs) -> Dict[str, Tensor]:
        gate = torch.sigmoid(x)
        return {"gate": gate}


class TanhGate(BaseFormula):
    def forward(self, x: Tensor, **kwargs) -> Dict[str, Tensor]:
        gate = torch.tanh(x)
        return {"gate": gate}


class SoftmaxGate(BaseFormula):
    def forward(self, x: Tensor, **kwargs) -> Dict[str, Tensor]:
        gate = F.softmax(x, dim=-1)
        return {"gate": gate}


class GumbelGate(BaseFormula):
    def forward(self, x: Tensor, temperature: float = 1.0, **kwargs) -> Dict[str, Tensor]:
        # Gumbel-softmax gate
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(x) + self.eps) + self.eps)
        gate = F.softmax((x + gumbel_noise) / temperature, dim=-1)
        return {"gate": gate}


class SparseGate(BaseFormula):
    def forward(self, x: Tensor, sparsity: float = 0.1, **kwargs) -> Dict[str, Tensor]:
        # Top-k sparse gate
        k = max(1, int(x.size(-1) * sparsity))
        values, indices = x.topk(k, dim=-1)
        gate = torch.zeros_like(x)
        gate.scatter_(-1, indices, torch.ones_like(values))
        return {"gate": gate, "indices": indices}


# === NORMALIZATION FORMULAS (5) ===

class LayerNorm(BaseFormula):
    def forward(self, x: Tensor, **kwargs) -> Dict[str, Tensor]:
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        out = (x - mean) / (std + self.eps)
        return {"out": out, "mean": mean, "std": std}


class BatchNorm(BaseFormula):
    def forward(self, x: Tensor, **kwargs) -> Dict[str, Tensor]:
        mean = x.mean(dim=0, keepdim=True)
        std = x.std(dim=0, keepdim=True)
        out = (x - mean) / (std + self.eps)
        return {"out": out}


class RMSNorm(BaseFormula):
    def forward(self, x: Tensor, **kwargs) -> Dict[str, Tensor]:
        rms = (x ** 2).mean(dim=-1, keepdim=True).sqrt()
        out = x / (rms + self.eps)
        return {"out": out, "rms": rms}


class PowerNorm(BaseFormula):
    def __init__(self, p: float = 2.0, **config):
        super().__init__(**config)
        self.p = p

    def forward(self, x: Tensor, **kwargs) -> Dict[str, Tensor]:
        norm = (x.abs() ** self.p).mean(dim=-1, keepdim=True) ** (1 / self.p)
        out = x / (norm + self.eps)
        return {"out": out, "norm": norm}


class GroupNorm(BaseFormula):
    def __init__(self, num_groups: int = 8, **config):
        super().__init__(**config)
        self.num_groups = num_groups

    def forward(self, x: Tensor, **kwargs) -> Dict[str, Tensor]:
        B, L, D = x.shape
        x_grouped = x.view(B, L, self.num_groups, D // self.num_groups)
        mean = x_grouped.mean(dim=-1, keepdim=True)
        std = x_grouped.std(dim=-1, keepdim=True)
        out = (x_grouped - mean) / (std + self.eps)
        out = out.view(B, L, D)
        return {"out": out}


# ========= FORMULA REGISTRY =========

@dataclass(frozen=True, slots=True)
class Constraint:
    tier: Tier
    required: Tuple[str, ...]
    optional: Tuple[str, ...] = ()
    shape_bind: Optional[Dict[str, str]] = None
    allow_dtypes: Tuple[torch.dtype, ...] = (torch.float16, torch.bfloat16, torch.float32)


@dataclass(frozen=True, slots=True)
class FormulaSpec:
    fid: str
    category: FormulaCategory
    inputs: Tuple[str, ...]
    outputs: Tuple[str, ...]
    constraint: Constraint
    implications: Tuple[str, ...]
    fn: BaseFormula
    description: str = ""
    version: str = "1.0"
    experimental: bool = False


class FormulaLibrary:
    """Manages 50+ formulas efficiently."""

    def __init__(self):
        self._formulas: Dict[str, FormulaSpec] = {}
        self._categories: Dict[FormulaCategory, List[str]] = {cat: [] for cat in FormulaCategory}
        self._aliases: Dict[str, str] = {}
        self._performance_cache: Dict[str, Dict] = {}

    def register(self, spec: FormulaSpec, aliases: Optional[List[str]] = None):
        if spec.fid in self._formulas:
            raise ValueError(f"Formula {spec.fid} already registered")
        self._formulas[spec.fid] = spec
        self._categories[spec.category].append(spec.fid)
        if aliases:
            for alias in aliases:
                self._aliases[alias] = spec.fid

    def add_aliases(self, fid: str, aliases: List[str]):
        if fid not in self._formulas:
            raise ValueError(f"Formula {fid} not found")
        for alias in aliases:
            self._aliases[alias] = fid

    def get(self, fid: str) -> FormulaSpec:
        if fid in self._aliases:
            fid = self._aliases[fid]
        return self._formulas[fid]

    def list_by_category(self, category: FormulaCategory) -> List[str]:
        return self._categories[category]

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_formulas": len(self._formulas),
            "categories": {cat.name: len(fids) for cat, fids in self._categories.items()},
            "experimental": sum(1 for s in self._formulas.values() if s.experimental),
        }

    def bind(self, fid: str, gt: GeoTensor, mapping: Dict[str, str]) -> "Plan":
        spec = self.get(fid)
        return Plan(spec=spec, gt=gt.ensure(), bind=BindMap(mapping))

    def bulk_register(self, formulas: List[FormulaSpec]):
        for spec in formulas:
            self.register(spec)


# ========= EXECUTION PLANNING =========

@dataclass(frozen=True, slots=True)
class BindMap:
    mapping: Dict[str, str]


@dataclass(frozen=True, slots=True)
class Plan:
    spec: FormulaSpec
    gt: GeoTensor
    bind: BindMap

    def run(self, profile: bool = False, **kw) -> Dict[str, Tensor]:
        for req in self.spec.constraint.required:
            if not self.gt.has(self.bind.mapping.get(req, "")):
                raise KeyError(f"Missing required field: {req}")
        args = [self.gt.get(self.bind.mapping[name]) for name in self.spec.inputs]
        return self.spec.fn(*args, profile=profile, eps=self.gt.eps, **kw)


# ========= FORMULA LOADER =========

def create_formula_library() -> FormulaLibrary:
    """Create and populate formula library with 50+ formulas."""
    print("Warning: Many of these formulas aren't production use and are only meant to be templates for better implementations.")
    lib = FormulaLibrary()

    # All formula specifications (50+)
    all_formulas = []

    # MIXING (10)
    all_formulas.extend([
        FormulaSpec(fid="mix:linear:v1", category=FormulaCategory.MIXING, inputs=("x", "y", "alpha"),
                    outputs=("out", "gate"), constraint=Constraint(tier="windowed", required=("x", "y", "alpha")),
                    implications=("differentiable",), fn=LinearMix(), description="Linear interpolation"),
        FormulaSpec(fid="mix:gated:v1", category=FormulaCategory.MIXING, inputs=("x", "y", "gate"),
                    outputs=("out", "gate"), constraint=Constraint(tier="windowed", required=("x", "y", "gate")),
                    implications=("differentiable",), fn=GatedMix()),
        FormulaSpec(fid="mix:adaptive:v1", category=FormulaCategory.MIXING, inputs=("x", "y", "context"),
                    outputs=("out", "weights"), constraint=Constraint(tier="windowed", required=("x", "y", "context")),
                    implications=("differentiable",), fn=AdaptiveMix()),
        FormulaSpec(fid="mix:smooth:v1", category=FormulaCategory.MIXING, inputs=("x", "y", "t"),
                    outputs=("out", "smooth_factor"), constraint=Constraint(tier="windowed", required=("x", "y", "t")),
                    implications=("differentiable",), fn=SmoothMix()),
        FormulaSpec(fid="mix:hierarchical:v1", category=FormulaCategory.MIXING, inputs=("x", "y", "z", "w1", "w2"),
                    outputs=("out",), constraint=Constraint(tier="windowed", required=("x", "y", "z", "w1", "w2")),
                    implications=("differentiable",), fn=HierarchicalMix()),
        FormulaSpec(fid="mix:conditional:v1", category=FormulaCategory.MIXING, inputs=("x", "y", "condition"),
                    outputs=("out", "mask"), constraint=Constraint(tier="windowed", required=("x", "y", "condition")),
                    implications=("differentiable",), fn=ConditionalMix()),
        FormulaSpec(fid="mix:trimix:v1", category=FormulaCategory.MIXING, inputs=("x", "y", "z"), outputs=("out",),
                    constraint=Constraint(tier="windowed", required=("x", "y", "z")), implications=("differentiable",),
                    fn=WeightedTriMix()),
        FormulaSpec(fid="mix:exp:v1", category=FormulaCategory.MIXING, inputs=("x", "y", "alpha"),
                    outputs=("out", "weight"), constraint=Constraint(tier="windowed", required=("x", "y", "alpha")),
                    implications=("differentiable",), fn=ExpMix()),
        FormulaSpec(fid="mix:momentum:v1", category=FormulaCategory.MIXING, inputs=("x", "y", "momentum"),
                    outputs=("out",), constraint=Constraint(tier="windowed", required=("x", "y", "momentum")),
                    implications=("differentiable",), fn=MomentumMix()),
        FormulaSpec(fid="mix:stochastic:v1", category=FormulaCategory.MIXING, inputs=("x", "y"),
                    outputs=("out", "alpha"), constraint=Constraint(tier="windowed", required=("x", "y")),
                    implications=("stochastic",), fn=StochasticMix(), experimental=True),
    ])

    # ATTENTION (10)
    all_formulas.extend([
        FormulaSpec(fid="attn:scaled_dot:v1", category=FormulaCategory.ATTENTION_BASIC, inputs=("q", "k", "v"),
                    outputs=("out", "attention"), constraint=Constraint(tier="windowed", required=("q", "k", "v")),
                    implications=("differentiable",), fn=ScaledDotProductAttention()),
        FormulaSpec(fid="attn:flash:v1", category=FormulaCategory.ATTENTION_FLASH, inputs=("q", "k", "v"),
                    outputs=("out", "gate"), constraint=Constraint(tier="windowed", required=("q", "k", "v")),
                    implications=("memory-efficient",), fn=FlashAttention()),
        FormulaSpec(fid="attn:local:v1", category=FormulaCategory.ATTENTION_SPARSE, inputs=("q", "k", "v"),
                    outputs=("out",), constraint=Constraint(tier="windowed", required=("q", "k", "v")),
                    implications=("linear-complexity",), fn=LocalAttention()),
        FormulaSpec(fid="attn:multiquery:v1", category=FormulaCategory.ATTENTION_BASIC, inputs=("q", "k", "v"),
                    outputs=("out",), constraint=Constraint(tier="windowed", required=("q", "k", "v")),
                    implications=("differentiable",), fn=MultiQueryAttention()),
        FormulaSpec(fid="attn:cross:v1", category=FormulaCategory.ATTENTION_BASIC, inputs=("q", "kv"),
                    outputs=("out", "cross_attention"), constraint=Constraint(tier="windowed", required=("q", "kv")),
                    implications=("differentiable",), fn=CrossAttention()),
        FormulaSpec(fid="attn:linear:v1", category=FormulaCategory.ATTENTION_BASIC, inputs=("q", "k", "v"),
                    outputs=("out",), constraint=Constraint(tier="windowed", required=("q", "k", "v")),
                    implications=("linear-complexity",), fn=LinearAttention()),
        FormulaSpec(fid="attn:gated:v1", category=FormulaCategory.ATTENTION_BASIC, inputs=("q", "k", "v", "gate"),
                    outputs=("out",), constraint=Constraint(tier="windowed", required=("q", "k", "v", "gate")),
                    implications=("differentiable",), fn=GatedAttention()),
        FormulaSpec(fid="attn:relative:v1", category=FormulaCategory.ATTENTION_RELATIVE, inputs=("q", "k", "v"),
                    outputs=("out",), constraint=Constraint(tier="windowed", required=("q", "k", "v")),
                    implications=("position-aware",), fn=RelativeAttention()),
        FormulaSpec(fid="attn:sparse:v1", category=FormulaCategory.ATTENTION_SPARSE, inputs=("q", "k", "v"),
                    outputs=("out",), constraint=Constraint(tier="windowed", required=("q", "k", "v")),
                    implications=("sparse",), fn=SparseAttention()),
        FormulaSpec(fid="attn:causal:v1", category=FormulaCategory.ATTENTION_BASIC, inputs=("q", "k", "v"),
                    outputs=("out",), constraint=Constraint(tier="windowed", required=("q", "k", "v")),
                    implications=("causal",), fn=CausalAttention()),
    ])

    # SIMILARITY (8)
    all_formulas.extend([
        FormulaSpec(fid="sim:cosine:v1", category=FormulaCategory.SIMILARITY_COSINE, inputs=("x", "y"),
                    outputs=("similarity",), constraint=Constraint(tier="simple", required=("x", "y")),
                    implications=("normalized",), fn=CosineSimilarity()),
        FormulaSpec(fid="sim:euclidean:v1", category=FormulaCategory.SIMILARITY_EUCLIDEAN, inputs=("x", "y"),
                    outputs=("similarity", "distance"), constraint=Constraint(tier="simple", required=("x", "y")),
                    implications=("differentiable",), fn=EuclideanSimilarity()),
        FormulaSpec(fid="sim:dot:v1", category=FormulaCategory.SIMILARITY_COSINE, inputs=("x", "y"),
                    outputs=("similarity",), constraint=Constraint(tier="simple", required=("x", "y")),
                    implications=("differentiable",), fn=DotProductSimilarity()),
        FormulaSpec(fid="sim:manhattan:v1", category=FormulaCategory.SIMILARITY_EUCLIDEAN, inputs=("x", "y"),
                    outputs=("similarity", "distance"), constraint=Constraint(tier="simple", required=("x", "y")),
                    implications=("differentiable",), fn=ManhattanSimilarity()),
        FormulaSpec(fid="sim:minkowski:v1", category=FormulaCategory.SIMILARITY_EUCLIDEAN, inputs=("x", "y"),
                    outputs=("similarity", "distance"), constraint=Constraint(tier="simple", required=("x", "y")),
                    implications=("differentiable",), fn=MinkowskiSimilarity()),
        FormulaSpec(fid="sim:pearson:v1", category=FormulaCategory.SIMILARITY_COSINE, inputs=("x", "y"),
                    outputs=("similarity",), constraint=Constraint(tier="simple", required=("x", "y")),
                    implications=("correlation",), fn=PearsonSimilarity()),
        FormulaSpec(fid="sim:jaccard:v1", category=FormulaCategory.SIMILARITY_COSINE, inputs=("x", "y"),
                    outputs=("similarity",), constraint=Constraint(tier="simple", required=("x", "y")),
                    implications=("set-based",), fn=JaccardSimilarity()),
        FormulaSpec(fid="sim:bilinear:v1", category=FormulaCategory.SIMILARITY_LEARNED, inputs=("x", "y", "weight"),
                    outputs=("similarity",), constraint=Constraint(tier="simple", required=("x", "y", "weight")),
                    implications=("learned",), fn=BilinearSimilarity()),
    ])

    # PROJECTION (7)
    all_formulas.extend([
        FormulaSpec(fid="proj:unit_sphere:v1", category=FormulaCategory.PROJECTION, inputs=("x",),
                    outputs=("out", "scale"), constraint=Constraint(tier="simple", required=("x",)),
                    implications=("normalized",), fn=UnitSphereProjection()),
        FormulaSpec(fid="proj:orthogonal:v1", category=FormulaCategory.PROJECTION, inputs=("x", "basis"),
                    outputs=("out",), constraint=Constraint(tier="simple", required=("x", "basis")),
                    implications=("orthogonal",), fn=OrthogonalProjection()),
        FormulaSpec(fid="proj:random:v1", category=FormulaCategory.PROJECTION, inputs=("x",), outputs=("out",),
                    constraint=Constraint(tier="simple", required=("x",)), implications=("stochastic",),
                    fn=RandomProjection(), experimental=True),
        FormulaSpec(fid="proj:pca:v1", category=FormulaCategory.PROJECTION, inputs=("x",),
                    outputs=("out", "eigenvalues"), constraint=Constraint(tier="simple", required=("x",)),
                    implications=("linear",), fn=PCAProjection(), experimental=True),
        FormulaSpec(fid="proj:hyperbolic:v1", category=FormulaCategory.PROJECTION, inputs=("x",), outputs=("out",),
                    constraint=Constraint(tier="simple", required=("x",)), implications=("non-euclidean",),
                    fn=HyperbolicProjection()),
        FormulaSpec(fid="proj:stereographic:v1", category=FormulaCategory.PROJECTION, inputs=("x",), outputs=("out",),
                    constraint=Constraint(tier="simple", required=("x",)), implications=("geometric",),
                    fn=StereographicProjection()),
        FormulaSpec(fid="proj:convex:v1", category=FormulaCategory.PROJECTION, inputs=("x",), outputs=("out",),
                    constraint=Constraint(tier="simple", required=("x",)), implications=("probability",),
                    fn=ConvexProjection()),
    ])

    # AGGREGATION (8)
    all_formulas.extend([
        FormulaSpec(fid="agg:weighted_mean:v1", category=FormulaCategory.AGGREGATION_WEIGHTED, inputs=("x", "weights"),
                    outputs=("out", "weights"), constraint=Constraint(tier="windowed", required=("x", "weights")),
                    implications=("differentiable",), fn=WeightedMeanAggregation()),
        FormulaSpec(fid="agg:hierarchical:v1", category=FormulaCategory.AGGREGATION_HIERARCHICAL, inputs=("x",),
                    outputs=("out", "intermediates"), constraint=Constraint(tier="hierarchical", required=("x",)),
                    implications=("multi-scale",), fn=HierarchicalAggregation(), experimental=True),
        FormulaSpec(fid="agg:max_pool:v1", category=FormulaCategory.AGGREGATION_POOLING, inputs=("x",),
                    outputs=("out", "indices"), constraint=Constraint(tier="windowed", required=("x",)),
                    implications=("max",), fn=MaxPoolAggregation()),
        FormulaSpec(fid="agg:min_pool:v1", category=FormulaCategory.AGGREGATION_POOLING, inputs=("x",),
                    outputs=("out", "indices"), constraint=Constraint(tier="windowed", required=("x",)),
                    implications=("min",), fn=MinPoolAggregation()),
        FormulaSpec(fid="agg:geometric_mean:v1", category=FormulaCategory.AGGREGATION_MEAN, inputs=("x",),
                    outputs=("out",), constraint=Constraint(tier="windowed", required=("x",)),
                    implications=("multiplicative",), fn=GeometricMeanAggregation()),
        FormulaSpec(fid="agg:harmonic_mean:v1", category=FormulaCategory.AGGREGATION_MEAN, inputs=("x",),
                    outputs=("out",), constraint=Constraint(tier="windowed", required=("x",)),
                    implications=("reciprocal",), fn=HarmonicMeanAggregation()),
        FormulaSpec(fid="agg:median:v1", category=FormulaCategory.AGGREGATION_MEAN, inputs=("x",), outputs=("out",),
                    constraint=Constraint(tier="windowed", required=("x",)), implications=("robust",),
                    fn=MedianAggregation()),
        FormulaSpec(fid="agg:quantile:v1", category=FormulaCategory.AGGREGATION_MEAN, inputs=("x",), outputs=("out",),
                    constraint=Constraint(tier="windowed", required=("x",)), implications=("percentile",),
                    fn=QuantileAggregation()),
    ])

    # GATING (5)
    all_formulas.extend([
        FormulaSpec(fid="gate:sigmoid:v1", category=FormulaCategory.GATING, inputs=("x",), outputs=("gate",),
                    constraint=Constraint(tier="simple", required=("x",)), implications=("bounded",), fn=SigmoidGate()),
        FormulaSpec(fid="gate:tanh:v1", category=FormulaCategory.GATING, inputs=("x",), outputs=("gate",),
                    constraint=Constraint(tier="simple", required=("x",)), implications=("bounded",), fn=TanhGate()),
        FormulaSpec(fid="gate:softmax:v1", category=FormulaCategory.GATING, inputs=("x",), outputs=("gate",),
                    constraint=Constraint(tier="simple", required=("x",)), implications=("probability",),
                    fn=SoftmaxGate()),
        FormulaSpec(fid="gate:gumbel:v1", category=FormulaCategory.GATING, inputs=("x",), outputs=("gate",),
                    constraint=Constraint(tier="simple", required=("x",)), implications=("stochastic",),
                    fn=GumbelGate(), experimental=True),
        FormulaSpec(fid="gate:sparse:v1", category=FormulaCategory.GATING, inputs=("x",), outputs=("gate", "indices"),
                    constraint=Constraint(tier="simple", required=("x",)), implications=("sparse",), fn=SparseGate()),
    ])

    # NORMALIZATION (5)
    all_formulas.extend([
        FormulaSpec(fid="norm:layer:v1", category=FormulaCategory.NORMALIZATION, inputs=("x",),
                    outputs=("out", "mean", "std"), constraint=Constraint(tier="simple", required=("x",)),
                    implications=("standardized",), fn=LayerNorm()),
        FormulaSpec(fid="norm:batch:v1", category=FormulaCategory.NORMALIZATION, inputs=("x",), outputs=("out",),
                    constraint=Constraint(tier="simple", required=("x",)), implications=("batch-wise",),
                    fn=BatchNorm()),
        FormulaSpec(fid="norm:rms:v1", category=FormulaCategory.NORMALIZATION, inputs=("x",), outputs=("out", "rms"),
                    constraint=Constraint(tier="simple", required=("x",)), implications=("scale-invariant",),
                    fn=RMSNorm()),
        FormulaSpec(fid="norm:power:v1", category=FormulaCategory.NORMALIZATION, inputs=("x",), outputs=("out", "norm"),
                    constraint=Constraint(tier="simple", required=("x",)), implications=("p-norm",), fn=PowerNorm()),
        FormulaSpec(fid="norm:group:v1", category=FormulaCategory.NORMALIZATION, inputs=("x",), outputs=("out",),
                    constraint=Constraint(tier="simple", required=("x",)), implications=("group-wise",),
                    fn=GroupNorm()),
    ])

    # Register all formulas
    lib.bulk_register(all_formulas)

    # Add common aliases
    lib.add_aliases("mix:linear:v1", ["lerp", "blend"])
    lib.add_aliases("attn:scaled_dot:v1", ["attention", "sdpa"])
    lib.add_aliases("sim:cosine:v1", ["cosine"])
    lib.add_aliases("proj:unit_sphere:v1", ["normalize"])

    return lib


# ========= MAIN =========
print("Loading Extended Formula Library (50+ formulas)...")
LIB = create_formula_library()

if __name__ == "__main__":
    try:
        import IPython
        in_notebook = IPython.get_ipython() is not None
    except:
        in_notebook = False



    stats = LIB.get_stats()
    print(f"Formula Library: {stats['total_formulas']} formulas loaded")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {torch.cuda.get_device_name(0)} (SMs: {props.multi_processor_count})")

    print("\n" + "=" * 60)
    print("Running Comprehensive Benchmarks")
    print("=" * 60)

    # Test configurations
    test_sizes = [
        (2, 16, 64, "Small"),
        (4, 64, 256, "Medium"),
        (8, 128, 512, "Large")
    ]

    all_results = {}
    category_times = {}

    # Benchmark all formulas
    for fid, spec in LIB._formulas.items():
        results = {}

        for B, W, D, label in test_sizes:
            try:
                # Create test tensors based on formula requirements
                tensors = {}
                for input_name in spec.inputs:
                    if input_name in ["x", "y", "z", "q", "k", "v", "kv", "context", "basis"]:
                        tensors[input_name] = torch.randn(B, W, D, device=device, dtype=torch.float32)
                    elif input_name in ["alpha", "gate", "mask", "weights", "scale", "t", "w1", "w2", "momentum",
                                        "condition"]:
                        tensors[input_name] = torch.rand(B, W, 1, device=device, dtype=torch.float32)
                    elif input_name == "weight":
                        tensors[input_name] = torch.randn(D, D, device=device, dtype=torch.float32)
                    elif input_name == "temperature":
                        tensors[input_name] = torch.tensor(1.0, device=device)
                    elif input_name == "sparsity":
                        tensors[input_name] = torch.tensor(0.1, device=device)
                    elif input_name == "levels" or input_name == "components" or input_name == "target_dim":
                        tensors[input_name] = 2  # Simple integer

                gt = GeoTensor(tensors, tier=spec.constraint.tier)
                mapping = {name: name for name in spec.inputs}
                plan = LIB.bind(fid, gt, mapping)

                # Warmup
                for _ in range(3):
                    _ = plan.run()

                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                # Benchmark
                iterations = 20
                start = time.perf_counter()
                for _ in range(iterations):
                    _ = plan.run()

                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                elapsed = time.perf_counter() - start
                ms_per_iter = (elapsed / iterations) * 1000
                results[label] = ms_per_iter

            except Exception as e:
                results[label] = None  # Mark as failed

        all_results[fid] = results

        # Track category averages
        cat = spec.category.name
        if cat not in category_times:
            category_times[cat] = []
        if results.get("Medium") is not None:
            category_times[cat].append((fid, results["Medium"]))

    # Display results by category
    print("\n" + "=" * 60)
    print("Performance by Category (Medium size: 4x64x256)")
    print("=" * 60)

    for cat in sorted(category_times.keys()):
        if category_times[cat]:
            formulas = sorted(category_times[cat], key=lambda x: x[1])
            avg_time = sum(t for _, t in formulas) / len(formulas)

            print(f"\n{cat} ({len(formulas)} formulas, avg: {avg_time:.3f}ms)")
            print("-" * 40)

            # Show best and worst in category
            for fid, time_ms in formulas[:2]:  # Best 2
                print(f"  ✓ {fid:30s} {time_ms:8.3f}ms")
            if len(formulas) > 4:
                print(f"  ...")
            for fid, time_ms in formulas[-2:]:  # Worst 2
                if formulas.index((fid, time_ms)) > 1:
                    print(f"  ⚠ {fid:30s} {time_ms:8.3f}ms")

    # Overall summary
    print("\n" + "=" * 60)
    print("Overall Performance Summary")
    print("=" * 60)

    # Find global best and worst
    all_times = [(fid, res["Medium"]) for fid, res in all_results.items()
                 if res.get("Medium") is not None]
    all_times_sorted = sorted(all_times, key=lambda x: x[1])

    print("\nTop 5 Fastest Formulas:")
    for fid, time_ms in all_times_sorted[:5]:
        print(f"  {fid:30s} {time_ms:8.3f}ms")

    print("\nTop 5 Slowest Formulas:")
    for fid, time_ms in all_times_sorted[-5:]:
        print(f"  {fid:30s} {time_ms:8.3f}ms")

    # Size scaling analysis
    print("\n" + "=" * 60)
    print("Size Scaling Analysis (sample formulas)")
    print("=" * 60)

    sample_formulas = ["mix:linear:v1", "attn:flash:v1", "sim:cosine:v1",
                       "proj:unit_sphere:v1", "norm:rms:v1"]

    for fid in sample_formulas:
        if fid in all_results:
            res = all_results[fid]
            if all(res.get(label) for _, _, _, label in test_sizes):
                small = res["Small"]
                medium = res["Medium"]
                large = res["Large"]
                scale_factor = large / small if small > 0 else 0
                print(f"{fid:20s}: {small:6.2f} → {medium:6.2f} → {large:6.2f}ms (scale: {scale_factor:.1f}x)")

    # Memory usage
    if torch.cuda.is_available():
        mem_allocated = torch.cuda.memory_allocated() / 1024 ** 2
        mem_reserved = torch.cuda.memory_reserved() / 1024 ** 2
        print(f"\nGPU Memory: {mem_allocated:.1f}MB allocated, {mem_reserved:.1f}MB reserved")

    print("\n" + "=" * 60)
    print(f"Benchmarking Complete! Tested {stats['total_formulas']} formulas")
    print("=" * 60)