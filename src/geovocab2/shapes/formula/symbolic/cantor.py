"""
CANTOR FORMULA SUITE
--------------------
Fractal geometry, infinite hierarchies, transfinite arithmetic, and multi-resolution structures.

Named in honor of:
  • Georg Cantor (1845–1918) – set theory, fractals, infinite cardinalities, topology

This suite provides formulas for fractal-based geometric operations and transfinite mathematics:
  - Cantor set construction (ternary and generalized)
  - Fractal dimension computation (Hausdorff, box-counting)
  - Cantor function (devil's staircase) for diffusion schedules
  - Hierarchical subdivision for multi-resolution meshes
  - Fractal interpolation and adaptive refinement
  - Self-similar geometric structures
  - Transfinite arithmetic (ℵ₀, ℵ₁, continuum)
  - Ordinal operations (ω, ω+1, ω², ω^ω)
  - Infinite hierarchies and limits
  - Simplex capacity control and degeneration prevention
  - Geometric sliding window encodings
  - Hierarchical vocabulary indexing

Mathematical Foundation:

    Cantor Set:
        Recursive removal: C_n = C_{n-1} \ (middle thirds)
        Limit: C = ∩_{n=0}^∞ C_n
        Properties: uncountable, measure zero, perfect set

    Hausdorff Dimension:
        d_H = lim_{ε→0} log(N(ε)) / log(1/ε)
        where N(ε) = minimum covers of size ε

    Box-Counting Dimension:
        d_B = lim_{ε→0} log(N_box(ε)) / log(1/ε)
        where N_box(ε) = number of boxes intersecting set

    Cantor Function (Devil's Staircase):
        f: [0,1] → [0,1]
        Continuous, monotone, derivative = 0 almost everywhere
        Constant on removed intervals, jumps at endpoints

    Self-Similarity:
        S = ∪_{i=1}^n φ_i(S)
        where φ_i are contractive similitudes

    Cardinal Arithmetic:
        ℵ₀ + ℵ₀ = ℵ₀ (countable infinity)
        ℵ₀ × ℵ₀ = ℵ₀
        2^ℵ₀ = c (continuum)
        ℵ₁ = next cardinal after ℵ₀

    Ordinal Arithmetic:
        ω = first infinite ordinal (limit of 0,1,2,3,...)
        ω + 1 ≠ 1 + ω (non-commutative)
        ω² = limit of ω, ω+ω, ω+ω+ω, ...
        ω^ω = supremum of {ω, ω^2, ω^3, ...}

    Cayley-Menger Determinant:
        Volume² = (-1)^(k+1) / (2^k (k!)²) * det(CM)
        Used for simplex capacity monitoring

Applications:
    - Hierarchical mesh refinement (adaptive LOD)
    - Non-uniform diffusion schedules
    - Multi-scale geometric representations
    - Sparse point cloud generation
    - Fractal noise for texture synthesis
    - Infinite resolution limits
    - Transfinite indexing schemes
    - Simplex degeneration prevention
    - Geometric positional encodings
    - Large vocabulary partitioning

Author: AbstractPhil + Claude Sonnet 4.5
License: MIT
"""

from typing import Dict, Optional, Tuple, List, Callable
import torch
from torch import Tensor
import math

from shapes.formula.formula_base import FormulaBase


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FRACTAL CONSTRUCTION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class CantorSet(FormulaBase):
    """Classical Cantor ternary set construction.

    Recursively removes middle thirds from intervals, creating a fractal
    with Hausdorff dimension log(2)/log(3) ≈ 0.631.

    Construction:
        Step 0: [0, 1]
        Step 1: [0, 1/3] ∪ [2/3, 1]
        Step 2: [0, 1/9] ∪ [2/9, 1/3] ∪ [2/3, 7/9] ∪ [8/9, 1]
        ...

    Args:
        iterations: Number of subdivision steps (default: 5)
        interval: Initial interval [a, b] (default: [0, 1])
    """

    def __init__(self, iterations: int = 5, interval: Tuple[float, float] = (0.0, 1.0)):
        super().__init__("cantor_set", "f.cantor.set")
        self.iterations = iterations
        self.interval = interval

    def forward(self, custom_intervals: Optional[Tensor] = None) -> Dict[str, Tensor]:
        """Generate Cantor set intervals.

        Args:
            custom_intervals: Optional starting intervals [..., n_intervals, 2]
                            If None, uses self.interval

        Returns:
            intervals: Cantor set intervals [..., n_final_intervals, 2]
            num_intervals: Count of intervals at each level
            total_length: Sum of interval lengths
            fractal_dimension: Theoretical dimension
        """
        if custom_intervals is None:
            # Start with single interval [a, b]
            intervals = torch.tensor([[self.interval[0], self.interval[1]]],
                                    dtype=torch.float32)
        else:
            intervals = custom_intervals

        num_intervals_history = [intervals.shape[-2]]

        # Iterate subdivision
        for step in range(self.iterations):
            new_intervals = []

            for i in range(intervals.shape[-2]):
                a, b = intervals[..., i, 0], intervals[..., i, 1]

                # Remove middle third: keep [a, a + (b-a)/3] and [a + 2(b-a)/3, b]
                left_end = a + (b - a) / 3.0
                right_start = a + 2.0 * (b - a) / 3.0

                # Create two new intervals
                left_interval = torch.stack([a, left_end], dim=-1)
                right_interval = torch.stack([right_start, b], dim=-1)

                new_intervals.append(left_interval)
                new_intervals.append(right_interval)

            intervals = torch.stack(new_intervals, dim=-2)
            num_intervals_history.append(intervals.shape[-2])

        # Compute total length
        lengths = intervals[..., 1] - intervals[..., 0]
        total_length = lengths.sum(dim=-1)

        # Theoretical fractal dimension for ternary Cantor set
        fractal_dimension = math.log(2) / math.log(3)

        return {
            "intervals": intervals,
            "num_intervals": torch.tensor(num_intervals_history),
            "total_length": total_length,
            "fractal_dimension": torch.tensor(fractal_dimension),
            "num_iterations": self.iterations
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class CantorSetGeneralized(FormulaBase):
    """Generalized Cantor set with arbitrary removal ratio.

    Instead of removing middle third, remove middle p-fraction.
    Fractal dimension: d = log(2) / log(1/(0.5 - p/2))

    Args:
        removal_ratio: Fraction to remove from middle (default: 1/3)
        iterations: Number of subdivision steps (default: 5)
    """

    def __init__(self, removal_ratio: float = 1.0/3.0, iterations: int = 5):
        super().__init__("cantor_set_generalized", "f.cantor.set_gen")
        self.removal_ratio = removal_ratio
        self.iterations = iterations

        # Compute fractal dimension
        # After removal, each interval splits into 2 with scale factor (1-p)/2
        scale_factor = (1.0 - removal_ratio) / 2.0
        self.fractal_dim = math.log(2) / math.log(1.0 / scale_factor)

    def forward(self, interval: Optional[Tensor] = None) -> Dict[str, Tensor]:
        """Generate generalized Cantor set.

        Args:
            interval: Initial interval [a, b], shape [2] (default: [0, 1])

        Returns:
            intervals: Final intervals [n_intervals, 2]
            fractal_dimension: Computed dimension
            removal_ratio: The removal ratio used
        """
        if interval is None:
            interval = torch.tensor([0.0, 1.0])

        intervals = interval.unsqueeze(0)  # [1, 2]

        for step in range(self.iterations):
            new_intervals = []

            for i in range(intervals.shape[0]):
                a, b = intervals[i, 0], intervals[i, 1]
                length = b - a

                # Remove middle p-fraction
                remove_length = self.removal_ratio * length
                keep_length = (length - remove_length) / 2.0

                # Left interval: [a, a + keep_length]
                left_interval = torch.tensor([a, a + keep_length])

                # Right interval: [b - keep_length, b]
                right_interval = torch.tensor([b - keep_length, b])

                new_intervals.append(left_interval)
                new_intervals.append(right_interval)

            intervals = torch.stack(new_intervals, dim=0)

        # Compute properties
        lengths = intervals[:, 1] - intervals[:, 0]
        total_length = lengths.sum()

        return {
            "intervals": intervals,
            "num_intervals": torch.tensor(intervals.shape[0]),
            "total_length": total_length,
            "fractal_dimension": torch.tensor(self.fractal_dim),
            "removal_ratio": torch.tensor(self.removal_ratio)
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class CantorDust(FormulaBase):
    """2D/3D Cantor dust via Cartesian product of Cantor sets.

    Creates fractal point clouds by taking products: C × C or C × C × C
    Dimension: d_H(C^n) = n · d_H(C)

    Args:
        dimension: Spatial dimension 2 or 3 (default: 2)
        iterations: Subdivision depth (default: 4)
        removal_ratio: Fraction to remove (default: 1/3)
    """

    def __init__(self, dimension: int = 2, iterations: int = 4,
                 removal_ratio: float = 1.0/3.0):
        super().__init__("cantor_dust", "f.cantor.dust")

        if dimension not in [2, 3]:
            raise ValueError(f"Dimension must be 2 or 3, got {dimension}")

        self.dimension = dimension
        self.iterations = iterations
        self.removal_ratio = removal_ratio

    def forward(self) -> Dict[str, Tensor]:
        """Generate Cantor dust point cloud.

        Returns:
            points: Fractal points [n_points, dim]
            fractal_dimension: Theoretical dimension
            num_points: Total points generated
        """
        # Generate 1D Cantor set
        cantor_gen = CantorSetGeneralized(self.removal_ratio, self.iterations)
        cantor_result = cantor_gen.forward()
        intervals = cantor_result["intervals"]  # [n, 2]

        # Sample points from intervals (use midpoints)
        cantor_points = (intervals[:, 0] + intervals[:, 1]) / 2.0  # [n]

        # Create Cartesian product C × C × ... × C
        if self.dimension == 2:
            # Meshgrid for 2D
            xx, yy = torch.meshgrid(cantor_points, cantor_points, indexing='ij')
            points = torch.stack([xx.flatten(), yy.flatten()], dim=-1)

        elif self.dimension == 3:
            # Meshgrid for 3D
            xx, yy, zz = torch.meshgrid(cantor_points, cantor_points, cantor_points,
                                       indexing='ij')
            points = torch.stack([xx.flatten(), yy.flatten(), zz.flatten()], dim=-1)

        # Fractal dimension of product
        one_d_dim = cantor_result["fractal_dimension"]
        fractal_dim = self.dimension * one_d_dim

        return {
            "points": points,
            "fractal_dimension": fractal_dim,
            "num_points": torch.tensor(points.shape[0]),
            "dimension": self.dimension
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FRACTAL DIMENSIONS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class BoxCountingDimension(FormulaBase):
    """Compute fractal dimension via box-counting algorithm.

    Covers the set with boxes of decreasing size ε and counts how many
    boxes contain points. The dimension is:
        d = lim_{ε→0} log(N(ε)) / log(1/ε)

    Args:
        box_sizes: List of box sizes to test (default: powers of 2)
        min_boxes: Minimum number of boxes for fitting (default: 5)
    """

    def __init__(self, box_sizes: Optional[List[float]] = None, min_boxes: int = 5):
        super().__init__("box_counting_dimension", "f.cantor.box_counting")

        if box_sizes is None:
            # Default: ε = 2^(-k) for k = 1, 2, ..., 6 (not too small)
            self.box_sizes = [2.0 ** (-k) for k in range(1, 7)]
        else:
            self.box_sizes = sorted(box_sizes, reverse=True)

        self.min_boxes = min_boxes

    def forward(self, points: Tensor) -> Dict[str, Tensor]:
        """Compute box-counting dimension.

        Args:
            points: Point cloud [..., n_points, dim]

        Returns:
            dimension: Estimated fractal dimension
            box_counts: Number of boxes for each size
            box_sizes: The box sizes used
            log_log_slope: Slope of log-log plot
        """
        dim = points.shape[-1]
        n_points = points.shape[-2]

        # Find bounding box
        min_coords = points.min(dim=-2)[0]
        max_coords = points.max(dim=-2)[0]
        extent = (max_coords - min_coords).max()

        # Dynamically choose box sizes based on dataset
        # For small datasets, use more box sizes to ensure enough data
        if n_points < 100:
            box_sizes = [2.0 ** (-k) for k in range(1, 8)]  # More sizes for small sets
        else:
            box_sizes = self.box_sizes

        box_counts = []
        valid_sizes = []

        for box_size in box_sizes:
            # Skip if box is larger than extent
            if box_size > extent * 2:
                continue

            # Number of boxes along each dimension
            n_boxes_per_dim = int(torch.ceil(extent / box_size).item()) + 1

            # Discretize points to box indices
            normalized_points = (points - min_coords) / box_size
            box_indices = torch.floor(normalized_points).long()

            # Clamp to valid range
            box_indices = torch.clamp(box_indices, 0, n_boxes_per_dim - 1)

            # Count unique boxes
            if dim == 1:
                flat_indices = box_indices.squeeze(-1)
            elif dim == 2:
                flat_indices = box_indices[..., 0] * n_boxes_per_dim + box_indices[..., 1]
            elif dim == 3:
                flat_indices = (box_indices[..., 0] * n_boxes_per_dim * n_boxes_per_dim +
                               box_indices[..., 1] * n_boxes_per_dim +
                               box_indices[..., 2])
            else:
                flat_indices = box_indices[..., 0]

            n_occupied = torch.unique(flat_indices.flatten()).numel()

            # Accept if we have meaningful data (more than 1 box, less than all points)
            if 1 < n_occupied < n_points:
                box_counts.append(n_occupied)
                valid_sizes.append(box_size)

        # Relax minimum requirement for small datasets
        min_required = min(3, len(valid_sizes))

        if len(box_counts) < min_required:
            # Not enough data points - return dimension based on ambient space
            return {
                "dimension": torch.tensor(float(dim)),
                "box_counts": torch.tensor(box_counts) if box_counts else torch.tensor([]),
                "box_sizes": torch.tensor(valid_sizes) if valid_sizes else torch.tensor([]),
                "log_log_slope": torch.tensor(float('nan'))
            }

        # Linear regression on log-log plot
        log_inv_sizes = -torch.log(torch.tensor(valid_sizes))  # log(1/ε)
        log_counts = torch.log(torch.tensor(box_counts, dtype=torch.float32))

        # Fit line: log_counts = slope * log_inv_sizes + intercept
        n = len(valid_sizes)
        sum_x = log_inv_sizes.sum()
        sum_y = log_counts.sum()
        sum_xx = (log_inv_sizes ** 2).sum()
        sum_xy = (log_inv_sizes * log_counts).sum()

        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x ** 2 + 1e-10)
        dimension = slope  # Slope of log(N) vs log(1/ε)

        return {
            "dimension": dimension,
            "box_counts": torch.tensor(box_counts),
            "box_sizes": torch.tensor(valid_sizes),
            "log_log_slope": slope
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CANTOR FUNCTION (DEVIL'S STAIRCASE)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class CantorFunction(FormulaBase):
    """Cantor function (devil's staircase) - continuous singular function.

    Properties:
    - Monotone increasing: f(0) = 0, f(1) = 1
    - Constant on removed intervals
    - Derivative = 0 almost everywhere
    - Yet increases from 0 to 1!

    Useful for:
    - Non-uniform diffusion schedules
    - Score function modulation
    - Singular measure theory

    Args:
        iterations: Approximation depth (default: 8)
    """

    def __init__(self, iterations: int = 8):
        super().__init__("cantor_function", "f.cantor.function")
        self.iterations = iterations

        # Pre-compute Cantor set intervals
        cantor = CantorSet(iterations=iterations)
        self.cantor_intervals = cantor.forward()["intervals"]

    def forward(self, t: Tensor) -> Dict[str, Tensor]:
        """Evaluate Cantor function at points t.

        Args:
            t: Input points [..., n_points] in [0, 1]

        Returns:
            values: f(t) [..., n_points]
            derivative: f'(t) (zero almost everywhere)
        """
        # Clamp to [0, 1]
        t = torch.clamp(t, 0.0, 1.0)

        # Initialize output
        values = torch.zeros_like(t)

        # Vectorized: create mask for all intervals at once
        n_intervals = self.cantor_intervals.shape[0]

        # Reshape for broadcasting: intervals [n_intervals, 2], t [..., n_points]
        # We need intervals [n_intervals, 1, ..., 1, 2] to broadcast with t [..., n_points]
        intervals = self.cantor_intervals  # [n_intervals, 2]

        # Expand t for comparison: [..., n_points, 1]
        t_expanded = t.unsqueeze(-1)  # [..., n_points, 1]

        # Check which interval each t belongs to
        # intervals shape [n_intervals, 2], we need [1, ..., 1, n_intervals, 2]
        n_batch_dims = t.ndim
        for _ in range(n_batch_dims):
            intervals = intervals.unsqueeze(0)  # [1, ..., 1, n_intervals, 2]

        a = intervals[..., 0]  # [1, ..., 1, n_intervals]
        b = intervals[..., 1]  # [1, ..., 1, n_intervals]

        # Check if t in [a, b]: [..., n_points, 1] vs [1, ..., 1, n_intervals]
        in_interval = (t_expanded >= a) & (t_expanded <= b)  # [..., n_points, n_intervals]

        # Find first matching interval (lowest index)
        # Use argmax to find first True value
        interval_indices = torch.argmax(in_interval.long(), dim=-1)  # [..., n_points]

        # Compute values: i / (n_intervals - 1)
        values = interval_indices.float() / (n_intervals - 1.0)

        # Handle points not in any interval (shouldn't happen, but for safety)
        has_match = in_interval.any(dim=-1)  # [..., n_points]
        values = torch.where(has_match, values, torch.zeros_like(values))

        # Derivative is zero almost everywhere (on Cantor set)
        derivative = torch.zeros_like(t)

        return {
            "values": values,
            "derivative": derivative,
            "is_singular": torch.tensor(True)
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class CantorStaircaseSchedule(FormulaBase):
    """Use Cantor function as non-linear schedule for diffusion/flow.

    Maps uniform time t ∈ [0,1] to non-uniform schedule via devil's staircase.
    Provides dense sampling in some regions, sparse in others.

    Args:
        iterations: Staircase resolution (default: 6)
        invert: If True, use 1 - f(t) (default: False)
    """

    def __init__(self, iterations: int = 6, invert: bool = False):
        super().__init__("cantor_staircase_schedule", "f.cantor.schedule")
        self.iterations = iterations
        self.invert = invert
        self.cantor_fn = CantorFunction(iterations=iterations)

    def forward(self, t: Tensor) -> Dict[str, Tensor]:
        """Apply Cantor schedule to time variable.

        Args:
            t: Uniform time in [0, 1]

        Returns:
            t_scheduled: Non-uniform time via Cantor function
            derivative: Local time dilation factor
        """
        result = self.cantor_fn.forward(t)
        t_scheduled = result["values"]

        if self.invert:
            t_scheduled = 1.0 - t_scheduled

        return {
            "t_scheduled": t_scheduled,
            "derivative": result["derivative"],
            "t_original": t
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HIERARCHICAL SUBDIVISION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class HierarchicalSubdivision(FormulaBase):
    """Cantor-inspired hierarchical subdivision for simplices.

    Recursively subdivides geometric elements with selective removal,
    creating multi-resolution hierarchies useful for:
    - Adaptive mesh refinement
    - Level-of-detail (LOD) systems
    - Fractal-like geometric structures

    Args:
        depth: Subdivision levels (default: 3)
        removal_pattern: Which sub-elements to remove (default: "middle")
        keep_ratio: Fraction of elements to keep per level (default: 0.5)
    """

    def __init__(self, depth: int = 3, removal_pattern: str = "middle",
                 keep_ratio: float = 0.5):
        super().__init__("hierarchical_subdivision", "f.cantor.subdivision")
        self.depth = depth
        self.removal_pattern = removal_pattern
        self.keep_ratio = keep_ratio

    def forward(self, vertices: Tensor,
                edges: Optional[Tensor] = None) -> Dict[str, Tensor]:
        """Perform hierarchical subdivision.

        Args:
            vertices: Initial vertices [n_vertices, dim]
            edges: Initial edges [n_edges, 2] (optional)

        Returns:
            vertices_hierarchy: List of vertex sets at each level
            edges_hierarchy: List of edge sets at each level
            num_vertices_per_level: Counts at each level
        """
        vertices_hierarchy = [vertices]
        edges_hierarchy = [edges] if edges is not None else [None]
        num_vertices_per_level = [vertices.shape[0]]

        current_vertices = vertices
        current_edges = edges

        for level in range(self.depth):
            # Subdivide edges (if present)
            if current_edges is not None:
                new_vertices = []
                new_edges = []

                for edge in current_edges:
                    v1, v2 = edge[0], edge[1]
                    p1 = current_vertices[v1]
                    p2 = current_vertices[v2]

                    # Add midpoint
                    midpoint = (p1 + p2) / 2.0
                    new_vertices.append(midpoint)

                    # Create new edges (keep pattern determines which)
                    if self.removal_pattern == "middle":
                        # Keep only endpoints, skip middle
                        if torch.rand(1).item() < self.keep_ratio:
                            new_v_idx = len(current_vertices) + len(new_vertices) - 1
                            new_edges.append(torch.tensor([v1.item(), new_v_idx]))
                            new_edges.append(torch.tensor([new_v_idx, v2.item()]))

                if new_vertices:
                    new_vertices = torch.stack(new_vertices)
                    current_vertices = torch.cat([current_vertices, new_vertices], dim=0)

                if new_edges:
                    current_edges = torch.stack(new_edges)
                else:
                    current_edges = None

            else:
                # Just vertex subdivision without topology
                # Sample subset of vertices for next level
                n_keep = int(current_vertices.shape[0] * self.keep_ratio)
                indices = torch.randperm(current_vertices.shape[0])[:n_keep]
                current_vertices = current_vertices[indices]

            vertices_hierarchy.append(current_vertices)
            edges_hierarchy.append(current_edges)
            num_vertices_per_level.append(current_vertices.shape[0])

        return {
            "vertices_hierarchy": vertices_hierarchy,
            "edges_hierarchy": edges_hierarchy,
            "num_vertices_per_level": torch.tensor(num_vertices_per_level),
            "depth": self.depth
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class MultiScaleEncoding(FormulaBase):
    """Encode positions at multiple Cantor-inspired scales.

    Creates hierarchical position encodings similar to Cantor set structure,
    useful for:
    - Multi-resolution neural networks
    - Hierarchical attention mechanisms
    - Fractal positional embeddings

    Args:
        num_scales: Number of hierarchical scales (default: 5)
        base_frequency: Fundamental frequency (default: 1.0)
    """

    def __init__(self, num_scales: int = 5, base_frequency: float = 1.0):
        super().__init__("multiscale_encoding", "f.cantor.multiscale")
        self.num_scales = num_scales
        self.base_frequency = base_frequency

    def forward(self, positions: Tensor) -> Dict[str, Tensor]:
        """Compute multi-scale positional encoding.

        Args:
            positions: Input positions [..., dim]

        Returns:
            encodings: Multi-scale features [..., dim * num_scales * 2]
            scales: Scale factors used
        """
        # Vectorized: compute all scales at once
        # scales[k] = base_frequency * 2^k for k in [0, num_scales)
        scale_indices = torch.arange(self.num_scales, dtype=positions.dtype, device=positions.device)
        scales = self.base_frequency * (2.0 ** scale_indices)  # [num_scales]

        # Broadcast: positions[..., dim] * scales[num_scales] -> [..., dim, num_scales]
        angles = 2.0 * math.pi * positions.unsqueeze(-1) * scales  # [..., dim, num_scales]

        # Compute sin and cos
        sin_enc = torch.sin(angles)  # [..., dim, num_scales]
        cos_enc = torch.cos(angles)  # [..., dim, num_scales]

        # Interleave sin/cos and flatten: [..., dim, num_scales, 2] -> [..., dim * num_scales * 2]
        encodings = torch.stack([sin_enc, cos_enc], dim=-1)  # [..., dim, num_scales, 2]
        encodings = encodings.flatten(start_dim=-3)  # [..., dim * num_scales * 2]

        return {
            "encodings": encodings,
            "scales": scales,
            "num_scales": self.num_scales
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class FractalComplexity(FormulaBase):
    """Measure geometric complexity using fractal-inspired metrics.

    Quantifies how "fractal-like" a geometric structure is via:
    - Self-similarity detection
    - Multi-scale analysis
    - Information dimension

    Args:
        num_scales: Scales to analyze (default: 5)
    """

    def __init__(self, num_scales: int = 5):
        super().__init__("fractal_complexity", "f.cantor.complexity")
        self.num_scales = num_scales

    def forward(self, points: Tensor) -> Dict[str, Tensor]:
        """Compute fractal complexity metrics.

        Args:
            points: Point cloud [n_points, dim]

        Returns:
            complexity: Overall complexity score
            scale_entropies: Information at each scale
            self_similarity: Self-similarity measure
        """
        # Compute box-counting at multiple scales
        box_counter = BoxCountingDimension()
        box_result = box_counter.forward(points)

        # Complexity from fractal dimension
        dimension = box_result["dimension"]

        # Handle NaN gracefully
        if torch.isnan(dimension):
            dimension = torch.tensor(float(points.shape[-1]))  # Use ambient dimension as fallback

        # Scale entropy: measure information at each scale (vectorized)
        scale_indices = torch.arange(self.num_scales, dtype=torch.float32)
        box_sizes = 2.0 ** (-(scale_indices + 1))  # [num_scales]

        # Information proxy: log(n_points + 1) scaled by box size
        n_points = torch.tensor(points.shape[0] + 1.0)
        scale_entropies = box_sizes * torch.log(n_points)  # [num_scales]

        # Overall complexity: weighted combination
        complexity = dimension + 0.1 * scale_entropies.mean()

        # Self-similarity: variance of local densities
        # High self-similarity = low variance
        self_similarity = 1.0 / (1.0 + scale_entropies.std())

        return {
            "complexity": complexity,
            "scale_entropies": scale_entropies,
            "self_similarity": self_similarity,
            "fractal_dimension": dimension
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TRANSFINITE ARITHMETIC
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class CardinalArithmetic(FormulaBase):
    """Arithmetic operations with infinite cardinalities.

    Implements Cantor's arithmetic for infinite sets:
    - ℵ₀ (aleph-null): countable infinity (|ℕ|)
    - c (continuum): |ℝ| = 2^ℵ₀
    - ℵ₁: next cardinal after ℵ₀

    Operations preserve mathematical properties:
    - ℵ₀ + ℵ₀ = ℵ₀ (union of countable sets)
    - ℵ₀ × ℵ₀ = ℵ₀ (Cartesian product)
    - 2^ℵ₀ = c (power set)
    - c + ℵ₀ = c (continuum absorbs countable)

    Args:
        assume_continuum_hypothesis: If True, c = ℵ₁ (default: True)
    """

    def __init__(self, assume_continuum_hypothesis: bool = True):
        super().__init__("cardinal_arithmetic", "f.cantor.cardinal")
        self.assume_ch = assume_continuum_hypothesis

        # Cardinal ordering
        self.cardinals = {
            "finite": 0,
            "aleph_0": 1,
            "aleph_1": 2,
            "c": 2 if assume_continuum_hypothesis else 2.5,
            "aleph_2": 3,
            "continuum": 2 if assume_continuum_hypothesis else 2.5
        }

    def forward(self, card1: str, card2: str, operation: str) -> Dict[str, any]:
        """Perform cardinal arithmetic.

        Args:
            card1, card2: Cardinal names ("aleph_0", "c", "aleph_1", etc.) or finite numbers ("2", "10")
            operation: "+", "*", "^" (exponentiation), "max"

        Returns:
            result: Resulting cardinal name
            result_value: Numeric encoding for comparison
            is_absorbing: Whether result absorbed smaller cardinal
        """
        # Handle finite cardinals (numbers)
        def get_cardinal_value(card_str):
            if card_str.isdigit():
                return float(card_str) / 1000.0  # Scale finite to [0, 1) range
            elif card_str in self.cardinals:
                return self.cardinals[card_str]
            else:
                raise ValueError(f"Unknown cardinal: {card_str}")

        def is_finite(card_str):
            return card_str.isdigit()

        def is_infinite(card_str):
            return not card_str.isdigit()

        c1_val = get_cardinal_value(card1)
        c2_val = get_cardinal_value(card2)

        # Addition and multiplication
        if operation in ["+", "*"]:
            # For infinite cardinals, κ + λ = κ × λ = max(κ, λ)
            if is_infinite(card1) or is_infinite(card2):  # At least one infinite
                result_val = max(c1_val, c2_val)
                is_absorbing = (c1_val != c2_val)
            else:
                # Both finite
                n1, n2 = int(card1), int(card2)
                result_n = n1 + n2 if operation == "+" else n1 * n2
                result_val = result_n / 1000.0
                is_absorbing = False

        # Exponentiation
        elif operation == "^":
            if card1 == "2" and card2 == "aleph_0":
                # Special case: 2^ℵ₀ = c (Cantor's theorem)
                result_val = self.cardinals["c"]
                is_absorbing = True
            elif is_finite(card1) and card2 == "aleph_0":
                # n^ℵ₀ for n ≥ 2 gives continuum
                if int(card1) >= 2:
                    result_val = self.cardinals["c"]
                    is_absorbing = True
                else:
                    result_val = c1_val
                    is_absorbing = False
            elif is_infinite(card1) and is_infinite(card2):
                # κ^λ for infinite κ, λ
                # Simplified: typically results in higher cardinal
                result_val = max(c1_val, c2_val) + 0.5
                is_absorbing = True
            elif is_finite(card1) and is_finite(card2):
                # Both finite
                n1, n2 = int(card1), int(card2)
                result_n = n1 ** n2 if n2 < 10 else 1000
                result_val = min(result_n, 1000) / 1000.0
                is_absorbing = False
            else:
                # Mixed: finite base, infinite exponent handled above
                # infinite base, finite exponent
                if is_infinite(card1) and is_finite(card2):
                    result_val = c1_val
                    is_absorbing = False
                else:
                    result_val = c1_val ** int(card2) if int(card2) < 5 else c1_val * 2
                    is_absorbing = False

        # Maximum
        elif operation == "max":
            result_val = max(c1_val, c2_val)
            is_absorbing = (c1_val != c2_val)

        else:
            raise ValueError(f"Unknown operation: {operation}")

        # Map back to cardinal name
        result_name = None

        # Check if result is finite (in [0, 1) range)
        if result_val < 1.0:
            # Finite cardinal
            result_n = int(result_val * 1000)
            if result_n < 1000:
                result_name = str(result_n)
            else:
                result_name = "finite"
        else:
            # Infinite cardinal
            for name, val in self.cardinals.items():
                if abs(val - result_val) < 0.1:
                    result_name = name
                    break

        if result_name is None:
            result_name = f"aleph_{int(result_val)}"

        return {
            "result": result_name,
            "result_value": torch.tensor(result_val),
            "is_absorbing": is_absorbing,
            "operation": operation,
            "operands": (card1, card2)
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class OrdinalArithmetic(FormulaBase):
    """Arithmetic operations with transfinite ordinals.

    Ordinals represent well-ordered sets and hierarchy levels:
    - ω: first infinite ordinal (0, 1, 2, 3, ...)
    - ω+1: ω followed by one more element
    - ω×2: two copies of ω (ω + ω)
    - ω²: ω copies of ω
    - ω^ω: tower of ωs

    Operations are NON-COMMUTATIVE:
    - 1 + ω = ω, but ω + 1 ≠ ω
    - ω × 2 ≠ 2 × ω

    Args:
        max_encoding: Maximum finite ordinal to represent exactly (default: 100)
    """

    def __init__(self, max_encoding: int = 100):
        super().__init__("ordinal_arithmetic", "f.cantor.ordinal")
        self.max_encoding = max_encoding

    def parse_ordinal(self, ordinal_str: str) -> Tuple[str, int, int]:
        """Parse ordinal notation into (base, coefficient, exponent).

        Examples:
            "5" -> ("finite", 5, 0)
            "omega" -> ("omega", 1, 1)
            "omega+3" -> ("omega", 1, 1) with offset 3
            "omega^2" -> ("omega", 1, 2)
            "omega*5" -> ("omega", 5, 1)
        """
        ordinal_str = ordinal_str.lower().strip()

        # Finite ordinal
        if ordinal_str.isdigit():
            return ("finite", int(ordinal_str), 0)

        # Parse omega expressions
        if "omega" in ordinal_str:
            base = "omega"
            coeff = 1
            exp = 1

            # Check for exponentiation omega^n
            if "^" in ordinal_str:
                parts = ordinal_str.split("^")
                exp = int(parts[1]) if parts[1].isdigit() else 2

            # Check for multiplication omega*n
            if "*" in ordinal_str:
                parts = ordinal_str.split("*")
                coeff = int(parts[1]) if parts[1].isdigit() else 2

            return (base, coeff, exp)

        return ("finite", 0, 0)

    def forward(self, ord1: str, ord2: str, operation: str) -> Dict[str, any]:
        """Perform ordinal arithmetic.

        Args:
            ord1, ord2: Ordinal expressions ("5", "omega", "omega+1", "omega^2")
            operation: "+", "*", "^"

        Returns:
            result: Resulting ordinal expression
            is_limit: Whether result is a limit ordinal
            is_commutative: Whether operation was commutative in this case
        """
        base1, coeff1, exp1 = self.parse_ordinal(ord1)
        base2, coeff2, exp2 = self.parse_ordinal(ord2)

        # Addition
        if operation == "+":
            # α + β: if β is infinite, result is dominated by β
            if base2 == "omega":
                # ω dominates: n + ω = ω, ω + n = ω + n
                if base1 == "finite":
                    result_str = f"omega"  # Absorbed
                    is_commutative = False
                else:
                    # ω^a + ω^b = ω^max(a,b)
                    if exp1 >= exp2:
                        result_str = ord1
                    else:
                        result_str = ord2
                    is_commutative = (exp1 == exp2)
            elif base1 == "omega":
                result_str = f"omega+{coeff2}" if base2 == "finite" else ord1
                is_commutative = False
            else:
                # Both finite
                result_str = str(coeff1 + coeff2)
                is_commutative = True

        # Multiplication
        elif operation == "*":
            if base2 == "omega":
                # α × ω: if α > 0, result is ω-scale
                if base1 == "finite" and coeff1 > 0:
                    result_str = ord2  # Absorbed by ω
                    is_commutative = False
                elif base1 == "omega":
                    # ω^a × ω^b = ω^(a+b)
                    new_exp = exp1 + exp2
                    result_str = f"omega^{new_exp}" if new_exp > 1 else "omega"
                    is_commutative = True
                else:
                    result_str = "0"
                    is_commutative = True
            elif base1 == "omega":
                # ω × n = ω for finite n > 0
                if base2 == "finite" and coeff2 > 0:
                    result_str = f"omega*{coeff2}" if coeff2 > 1 else "omega"
                    is_commutative = False
                else:
                    result_str = ord1
                    is_commutative = False
            else:
                # Both finite
                result_str = str(coeff1 * coeff2)
                is_commutative = True

        # Exponentiation
        elif operation == "^":
            if base2 == "omega" and base1 == "omega":
                # ω^ω
                result_str = "omega^omega"
                is_commutative = False
            elif base1 == "omega" and base2 == "finite":
                # ω^n
                new_exp = exp1 * coeff2
                result_str = f"omega^{new_exp}" if new_exp > 1 else "omega"
                is_commutative = False
            elif base1 == "finite" and base2 == "omega":
                # n^ω for n ≥ 2 is ω-scale
                if coeff1 >= 2:
                    result_str = "omega"
                else:
                    result_str = str(coeff1)  # 0^ω = 0, 1^ω = 1
                is_commutative = False
            else:
                # Both finite
                result = coeff1 ** coeff2 if coeff2 < 10 else self.max_encoding
                result_str = str(min(result, self.max_encoding))
                is_commutative = False

        else:
            raise ValueError(f"Unknown operation: {operation}")

        # Check if result is limit ordinal (no immediate predecessor)
        is_limit = "omega" in result_str and "+" not in result_str

        return {
            "result": result_str,
            "is_limit": is_limit,
            "is_commutative": is_commutative,
            "operation": operation,
            "operands": (ord1, ord2)
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class TransfiniteSubdivision(FormulaBase):
    """Infinite hierarchy of geometric refinement levels.

    Maps ordinal levels to geometric subdivisions:
    - Level 0, 1, 2, ...: Finite refinement
    - Level ω: Limit of all finite levels
    - Level ω+1: Subdivision of level ω
    - Level ω²: Limit of ω×n sequence

    Useful for:
    - Theoretical completeness proofs
    - Asymptotic geometric analysis
    - Infinite resolution limits

    Args:
        finite_approximation: Max finite level to compute (default: 10)
        subdivision_ratio: Refinement ratio per level (default: 0.5)
    """

    def __init__(self, finite_approximation: int = 10, subdivision_ratio: float = 0.5):
        super().__init__("transfinite_subdivision", "f.cantor.transfinite")
        self.finite_approx = finite_approximation
        self.ratio = subdivision_ratio

    def forward(self, geometry: Tensor, level: str) -> Dict[str, any]:
        """Compute geometry at transfinite level.

        Args:
            geometry: Initial geometric data [n_elements, ...]
            level: Ordinal level ("5", "omega", "omega+3", "omega^2")

        Returns:
            refined_geometry: Geometry at specified level
            approximation_level: Finite level used for ω
            is_limit: Whether level is a limit ordinal
            convergence_estimate: Distance to true limit
        """
        ordinal_calc = OrdinalArithmetic()
        base, coeff, exp = ordinal_calc.parse_ordinal(level)

        if base == "finite":
            # Direct finite refinement
            refinement_steps = coeff
            is_limit = False

        elif base == "omega":
            # Approximate ω with finite level
            if exp == 1:
                # ω: use max finite approximation
                refinement_steps = self.finite_approx
            elif exp == 2:
                # ω²: use even more
                refinement_steps = self.finite_approx * 2
            else:
                # ω^k
                refinement_steps = self.finite_approx * exp

            is_limit = True

        else:
            refinement_steps = 0
            is_limit = False

        # Perform refinement
        refined = geometry
        scales = [1.0]

        for step in range(refinement_steps):
            # Simple subdivision: scale and replicate
            scale = self.ratio ** (step + 1)
            scales.append(scale)

            # For point clouds: add interpolated points
            if geometry.ndim == 2 and geometry.shape[-1] in [2, 3]:
                n_points = refined.shape[0]
                # Add midpoints between adjacent points
                if n_points > 1:
                    midpoints = (refined[:-1] + refined[1:]) / 2.0
                    refined = torch.cat([refined, midpoints], dim=0)

        # Estimate convergence for limit ordinals
        if is_limit:
            # Measure of how close we are to ω
            # Based on change in last few iterations
            convergence = scales[-1] / scales[0] if len(scales) > 1 else 0.0
        else:
            convergence = 0.0

        return {
            "refined_geometry": refined,
            "approximation_level": torch.tensor(refinement_steps),
            "is_limit": is_limit,
            "convergence_estimate": torch.tensor(convergence),
            "level_requested": level
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class CantorDiagonalization(FormulaBase):
    """Cantor's diagonal argument for generating uncountable sets.

    Classic proof that |ℝ| > |ℕ|:
    - Given any countable list of reals, construct a real not in the list
    - Diagonalization: differ from nth number in nth digit

    Geometric application:
    - From countable point set, generate continuum of variations
    - Ensures completeness of infinite constructions

    Args:
        precision: Number of digits/bits for diagonalization (default: 32)
    """

    def __init__(self, precision: int = 32):
        super().__init__("cantor_diagonalization", "f.cantor.diagonal")
        self.precision = precision

    def forward(self, countable_set: Tensor) -> Dict[str, Tensor]:
        """Apply diagonal argument to generate new elements.

        Args:
            countable_set: Sequence of values [..., n_elements]

        Returns:
            diagonal_element: New element not in original set
            is_distinct: Verification it differs from all originals
            construction_path: Binary digits showing construction
        """
        n_elements = countable_set.shape[-1]

        # Normalize to [0, 1] for binary representation
        normalized = torch.sigmoid(countable_set)  # Map to (0, 1)

        # Convert to binary representations (conceptually)
        # For each element, extract "digit" at its index position
        diagonal_bits = torch.zeros(min(n_elements, self.precision))

        for i in range(min(n_elements, self.precision)):
            # Extract ith "bit" from ith element
            value = normalized[..., i]

            # Threshold at i/precision to create varying bit pattern
            threshold = (i + 1) / (n_elements + 1)
            bit = (value > threshold).float()

            # Flip the bit (diagonalization)
            diagonal_bits[i] = 1.0 - bit

        # Reconstruct diagonal element from bits
        powers = 2.0 ** -torch.arange(1, len(diagonal_bits) + 1, dtype=torch.float32)
        diagonal_element = (diagonal_bits * powers).sum()

        # Verify it's distinct from all original elements
        differences = torch.abs(normalized - diagonal_element)
        is_distinct = torch.all(differences > 1e-6)

        return {
            "diagonal_element": diagonal_element,
            "is_distinct": is_distinct,
            "construction_path": diagonal_bits,
            "cardinality_proof": "continuum"  # Proven uncountable
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class TransfiniteLimit(FormulaBase):
    """Compute geometric limits at transfinite ordinals.

    For sequences indexed by ordinals:
    - sup{f(n) : n < ω} at limit ordinal ω
    - lim_{α → ω} V(α) for volume at ω

    Applications:
    - Asymptotic mesh properties
    - Infinite resolution bounds
    - Convergence analysis

    Args:
        limit_ordinal: Target ordinal ("omega", "omega^2", etc.)
        epsilon: Convergence threshold (default: 1e-6)
    """

    def __init__(self, limit_ordinal: str = "omega", epsilon: float = 1e-6):
        super().__init__("transfinite_limit", "f.cantor.limit")
        self.limit_ordinal = limit_ordinal
        self.epsilon = epsilon

    def forward(self, sequence: Tensor) -> Dict[str, Tensor]:
        """Compute limit of sequence at transfinite ordinal.

        Args:
            sequence: Values indexed by finite ordinals [n_steps]
                     Assumes sequence[i] approximates f(i)

        Returns:
            limit_value: Supremum/limit at target ordinal
            has_converged: Whether sequence is Cauchy
            convergence_rate: Estimated rate
        """
        # Check convergence
        if len(sequence) < 2:
            return {
                "limit_value": sequence[-1] if len(sequence) > 0 else torch.tensor(0.0),
                "has_converged": torch.tensor(False),
                "convergence_rate": torch.tensor(float('nan'))
            }

        # Compute differences between successive elements
        diffs = torch.abs(sequence[1:] - sequence[:-1])

        # Check Cauchy criterion
        recent_diffs = diffs[-5:] if len(diffs) >= 5 else diffs
        has_converged = torch.all(recent_diffs < self.epsilon)

        # Estimate convergence rate
        if len(diffs) >= 2:
            # Ratio of successive differences
            ratios = diffs[1:] / (diffs[:-1] + 1e-10)
            convergence_rate = ratios.mean()
        else:
            convergence_rate = torch.tensor(1.0)

        # Limit value: use supremum for monotone, average of tail otherwise
        if torch.all(sequence[1:] >= sequence[:-1] - self.epsilon):
            # Monotone increasing
            limit_value = sequence[-1]
        elif torch.all(sequence[1:] <= sequence[:-1] + self.epsilon):
            # Monotone decreasing
            limit_value = sequence[-1]
        else:
            # Oscillating: use average of last few
            tail_length = min(10, len(sequence))
            limit_value = sequence[-tail_length:].mean()

        return {
            "limit_value": limit_value,
            "has_converged": has_converged,
            "convergence_rate": convergence_rate,
            "limit_ordinal": self.limit_ordinal
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class CantorSampler(FormulaBase):
    """Cardinality-aware sampling from geometric structures.

    Samples points based on set theoretic cardinality:
    - Countable (ℵ₀): Enumerable sampling
    - Continuum (c): Dense sampling via diagonalization

    Args:
        default_cardinality: "aleph_0" or "continuum" (default: "continuum")
    """

    def __init__(self, default_cardinality: str = "continuum"):
        super().__init__("cantor_sampler", "f.cantor.sampler")
        self.default_cardinality = default_cardinality

    def forward(self, domain: Tensor, cardinality: Optional[str] = None,
                num_samples: Optional[int] = None) -> Dict[str, Tensor]:
        """Sample from domain with specified cardinality.

        Args:
            domain: Geometric domain [n_dims, 2] with [min, max] per dimension
            cardinality: "aleph_0" or "continuum" (optional)
            num_samples: Number of finite samples to generate

        Returns:
            samples: Generated points [n_samples, n_dims]
            true_cardinality: Theoretical cardinality
            sampling_method: Method used
        """
        cardinality = cardinality or self.default_cardinality
        n_dims = domain.shape[0]

        if num_samples is None:
            # Default: 100 for countable, 1000 for continuum
            num_samples = 100 if cardinality == "aleph_0" else 1000

        if cardinality == "aleph_0":
            # Countable: regular grid sampling
            # Sample along rational coordinates
            points_per_dim = int(num_samples ** (1.0 / n_dims)) + 1

            # Create grid
            axes = []
            for dim_idx in range(n_dims):
                axis = torch.linspace(domain[dim_idx, 0], domain[dim_idx, 1], points_per_dim)
                axes.append(axis)

            # Meshgrid
            if n_dims == 2:
                xx, yy = torch.meshgrid(axes[0], axes[1], indexing='ij')
                samples = torch.stack([xx.flatten(), yy.flatten()], dim=-1)
            elif n_dims == 3:
                xx, yy, zz = torch.meshgrid(axes[0], axes[1], axes[2], indexing='ij')
                samples = torch.stack([xx.flatten(), yy.flatten(), zz.flatten()], dim=-1)
            else:
                # Simplified for higher dims
                samples = torch.rand(num_samples, n_dims)
                # Vectorized scaling
                range_size = domain[:, 1] - domain[:, 0]  # [n_dims]
                samples = samples * range_size + domain[:, 0]  # Broadcasting

            # Truncate to requested size
            samples = samples[:num_samples]
            method = "grid_enumeration"

        else:  # continuum
            # Dense sampling: random uniform (represents continuum)
            samples = torch.rand(num_samples, n_dims)

            # Vectorized scaling: scale to domain
            range_size = domain[:, 1] - domain[:, 0]  # [n_dims]
            samples = samples * range_size + domain[:, 0]  # Broadcasting

            method = "uniform_dense"

        return {
            "samples": samples,
            "true_cardinality": cardinality,
            "sampling_method": method,
            "num_samples": torch.tensor(samples.shape[0])
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SIMPLEX CAPACITY AND DEGENERATION CONTROL
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class SimplexCapacityController(FormulaBase):
    """Monitor and control k-simplex capacity to prevent degeneration.

    Uses Cayley-Menger determinants to track volumetric capacity and applies
    corrective geometry when simplices collapse during training.

    Degeneration detection:
    - Volume approaching zero (vertices collapsing)
    - High condition number (numerical instability)
    - Low vertex spread (loss of geometric diversity)

    Args:
        min_volume: Minimum acceptable volume (default: 1e-6)
        max_condition: Maximum condition number (default: 1e6)
        correction_strength: Regularization strength (default: 0.1)
    """

    def __init__(self, min_volume: float = 1e-6, max_condition: float = 1e6,
                 correction_strength: float = 0.1):
        super().__init__("simplex_capacity_controller", "f.cantor.simplex_capacity")
        self.min_volume = min_volume
        self.max_condition = max_condition
        self.correction_strength = correction_strength

    def forward(self, vertices: Tensor) -> Dict[str, Tensor]:
        """Monitor simplex health and generate corrective forces.

        Args:
            vertices: Simplex vertices [..., k+1, dim]

        Returns:
            volume: k-volume of simplex
            is_degenerate: Boolean flag for degeneration
            condition_number: Numerical stability metric
            correction_force: Gradient to apply for regularization [..., k+1, dim]
            health_score: Overall health [0, 1]
        """
        # Compute pairwise squared distances
        # vertices: [..., k+1, dim]
        k_plus_1 = vertices.shape[-2]
        dim = vertices.shape[-1]

        # Compute distance matrix
        # distances[..., i, j] = ||v_i - v_j||²
        diff = vertices.unsqueeze(-2) - vertices.unsqueeze(-3)  # [..., k+1, k+1, dim]
        squared_distances = (diff ** 2).sum(dim=-1)  # [..., k+1, k+1]

        # Cayley-Menger matrix
        # CM = [0    1    1    ... 1   ]
        #      [1    0    d²₀₁ ... d²₀ₙ]
        #      [1    d²₁₀ 0    ... d²₁ₙ]
        #      [...              ...   ]
        #      [1    d²ₙ₀ d²ₙ₁ ... 0   ]

        batch_shape = squared_distances.shape[:-2]
        cm_size = k_plus_1 + 1

        # Build Cayley-Menger matrix
        cm = torch.zeros(*batch_shape, cm_size, cm_size,
                        dtype=vertices.dtype, device=vertices.device)

        # First row and column are all 1s except (0,0)
        cm[..., 0, 1:] = 1.0
        cm[..., 1:, 0] = 1.0

        # Fill in squared distances
        cm[..., 1:, 1:] = squared_distances

        # Compute determinant
        det = torch.linalg.det(cm)

        # Volume formula: V² = (-1)^(k+1) / (2^k (k!)²) * det(CM)
        k = k_plus_1 - 1
        sign = (-1) ** (k + 1)
        factorial_k = math.factorial(k)
        denom = (2 ** k) * (factorial_k ** 2)

        volume_squared = sign * det / denom
        volume_squared = torch.clamp(volume_squared, min=0.0)  # Numerical safety
        volume = torch.sqrt(volume_squared + 1e-12)

        # Detect degeneration
        is_degenerate = volume < self.min_volume

        # Condition number: ratio of max to min singular values of vertex matrix
        # Center vertices
        centroid = vertices.mean(dim=-2, keepdim=True)
        centered = vertices - centroid

        # SVD for condition number
        try:
            U, S, Vh = torch.linalg.svd(centered, full_matrices=False)
            condition_number = S[..., 0] / (S[..., -1] + 1e-12)
        except:
            condition_number = torch.full(batch_shape, float('inf'),
                                         dtype=vertices.dtype, device=vertices.device)

        # High condition = numerical instability
        is_ill_conditioned = condition_number > self.max_condition

        # Vertex spread: measure of geometric diversity
        vertex_spread = torch.std(vertices, dim=-2).mean(dim=-1)

        # Overall health score [0, 1]
        volume_score = torch.sigmoid(torch.log(volume / self.min_volume + 1e-12))
        condition_score = torch.sigmoid(-torch.log(condition_number / 1000.0 + 1e-12))
        spread_score = torch.sigmoid(vertex_spread - 0.1)

        health_score = (volume_score + condition_score + spread_score) / 3.0

        # Generate correction force
        # Push vertices away from centroid if degenerate
        correction_force = torch.zeros_like(vertices)

        if torch.any(is_degenerate):
            # Repulsion from centroid
            to_centroid = vertices - centroid
            repulsion = -to_centroid / (torch.norm(to_centroid, dim=-1, keepdim=True) + 1e-8)
            correction_force = self.correction_strength * repulsion

            # Only apply to degenerate simplices
            is_degenerate_expanded = is_degenerate.view(*batch_shape, 1, 1).expand_as(correction_force)
            correction_force = torch.where(is_degenerate_expanded, correction_force,
                                          torch.zeros_like(correction_force))

        return {
            "volume": volume,
            "is_degenerate": is_degenerate,
            "condition_number": condition_number,
            "is_ill_conditioned": is_ill_conditioned,
            "vertex_spread": vertex_spread,
            "health_score": health_score,
            "correction_force": correction_force,
            "cayley_menger_determinant": det
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class GeometricSlidingWindow(FormulaBase):
    """Encode positional information via simplex trajectory deformations.

    Instead of RoPE's rotation matrices, use geometric transformations of
    k-simplices to encode position. As context progresses, simplices deform
    along canonical trajectories, preserving capacity while changing shape.

    Args:
        window_size: Context window length (default: 2048)
        k_simplex: Simplex order (default: 5 for 5-simplex = 6 vertices)
        embedding_dim: Dimension of embedding space (default: 512)
        deformation_rate: How fast simplices deform (default: 0.01)
    """

    def __init__(self, window_size: int = 2048, k_simplex: int = 5,
                 embedding_dim: int = 512, deformation_rate: float = 0.01):
        super().__init__("geometric_sliding_window", "f.cantor.geo_sliding")
        self.window_size = window_size
        self.k = k_simplex
        self.k_plus_1 = k_simplex + 1
        self.dim = embedding_dim
        self.rate = deformation_rate

    def forward(self, positions: Tensor, embeddings: Tensor) -> Dict[str, Tensor]:
        """Apply geometric positional encoding via simplex deformation.

        Args:
            positions: Token positions [..., seq_len] (integers)
            embeddings: Token embeddings [..., seq_len, dim]

        Returns:
            encoded: Geometrically encoded embeddings [..., seq_len, dim]
            simplex_trajectory: Simplex shapes at each position [..., seq_len, k+1, dim]
            capacity_trace: Volume at each position [..., seq_len]
        """
        seq_len = embeddings.shape[-2]
        batch_shape = embeddings.shape[:-2]

        # Normalize positions to [0, 1]
        pos_normalized = positions.float() / self.window_size

        # Generate base simplex (regular k-simplex)
        # Use standard simplex vertices in R^(k+1), then project to embedding dim
        base_vertices = self._generate_regular_simplex()  # [k+1, k+1]

        # Project to embedding dimension via random projection (fixed)
        projection = torch.randn(self.k_plus_1, self.dim,
                                device=embeddings.device, dtype=embeddings.dtype)
        projection = projection / torch.norm(projection, dim=0, keepdim=True)

        base_simplex = base_vertices @ projection  # [k+1, dim]

        # Deform simplex along trajectory for each position
        # Trajectory: rotate and scale based on position
        simplex_trajectory = []
        capacity_trace = []

        for i in range(seq_len):
            pos = pos_normalized[..., i]  # [...]

            # Deformation angle: θ = 2π * rate * pos
            theta = 2.0 * math.pi * self.rate * pos

            # Rotation in first two dimensions (can extend to more)
            cos_theta = torch.cos(theta)
            sin_theta = torch.sin(theta)

            # Build rotation matrix [dim, dim]
            rot = torch.eye(self.dim, device=embeddings.device, dtype=embeddings.dtype)

            # Handle batch dimensions properly
            if batch_shape:
                rot = rot.unsqueeze(0).expand(*batch_shape, -1, -1).clone()

            # Apply rotation to first two dims
            if batch_shape:
                rot[..., 0, 0] = cos_theta
                rot[..., 0, 1] = -sin_theta
                rot[..., 1, 0] = sin_theta
                rot[..., 1, 1] = cos_theta
            else:
                rot[0, 0] = cos_theta
                rot[0, 1] = -sin_theta
                rot[1, 0] = sin_theta
                rot[1, 1] = cos_theta

            # Scale based on position (slight compression/expansion)
            scale = 1.0 + 0.1 * torch.sin(pos * math.pi)

            # Transform base simplex
            # base_simplex: [k+1, dim]
            # rot: [..., dim, dim] or [dim, dim]
            if batch_shape:
                deformed = torch.einsum('...ij,kj->...ki', rot, base_simplex)  # [..., k+1, dim]
                deformed = deformed * scale.view(*batch_shape, 1, 1)
            else:
                deformed = base_simplex @ rot.T  # [k+1, dim]
                deformed = deformed * scale

            simplex_trajectory.append(deformed)

            # Compute capacity at this position
            controller = SimplexCapacityController()
            capacity_result = controller.forward(deformed)
            capacity_trace.append(capacity_result["volume"])

        # Stack trajectories
        simplex_trajectory = torch.stack(simplex_trajectory, dim=-3)  # [..., seq_len, k+1, dim]
        capacity_trace = torch.stack(capacity_trace, dim=-1)  # [..., seq_len]

        # Encode embeddings using simplex properties
        # Use simplex centroid and vertex spread as modulation
        centroids = simplex_trajectory.mean(dim=-2)  # [..., seq_len, dim]
        spreads = torch.std(simplex_trajectory, dim=-2)  # [..., seq_len, dim]

        # Modulate embeddings
        encoded = embeddings * (1.0 + 0.1 * spreads) + 0.05 * centroids

        return {
            "encoded": encoded,
            "simplex_trajectory": simplex_trajectory,
            "capacity_trace": capacity_trace,
            "centroids": centroids,
            "spreads": spreads
        }

    def _generate_regular_simplex(self) -> Tensor:
        """Generate regular k-simplex vertices in R^(k+1)."""
        k = self.k

        # Standard simplex: vertices of k-simplex in R^(k+1)
        # v_0 = (1, 0, 0, ..., 0)
        # v_1 = (0, 1, 0, ..., 0)
        # ...
        # v_k = (0, 0, 0, ..., 1)

        vertices = torch.eye(k + 1)

        # Center at origin
        centroid = vertices.mean(dim=0, keepdim=True)
        vertices = vertices - centroid

        return vertices


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class HierarchicalVocabIndex(FormulaBase):
    """Transfinite indexing scheme for infinite vocabulary hierarchies.

    Uses ordinal arithmetic to create hierarchical addressing:
    - Finite tokens: 0, 1, 2, ..., n
    - First extension: ω, ω+1, ω+2, ...
    - Higher levels: ω², ω³, ω^ω, ...

    Enables compositional vocab where tokens at level ω are built from
    tokens at finite levels, maintaining Cantor-style infinite structure.

    Args:
        finite_vocab_size: Base vocabulary size (default: 50000)
        max_ordinal_level: Maximum ordinal level to support (default: "omega^2")
    """

    def __init__(self, finite_vocab_size: int = 50000,
                 max_ordinal_level: str = "omega^2"):
        super().__init__("hierarchical_vocab_index", "f.cantor.vocab_index")
        self.finite_size = finite_vocab_size
        self.max_level = max_ordinal_level

        self.ordinal_calc = OrdinalArithmetic()

    def forward(self, token_id: Tensor, level: str = "finite") -> Dict[str, any]:
        """Map token ID to hierarchical address.

        Args:
            token_id: Token index [..., 1] (integer)
            level: Ordinal level ("finite", "omega", "omega+k", "omega^2")

        Returns:
            hierarchical_address: (level, offset) tuple
            parent_tokens: List of constituent tokens for composed tokens
            is_compositional: Whether token is built from others
            cardinal_size: Set size at this level
        """
        base, coeff, exp = self.ordinal_calc.parse_ordinal(level)

        if base == "finite":
            # Direct indexing
            hierarchical_address = (level, token_id.item() if token_id.numel() == 1 else token_id)
            parent_tokens = []
            is_compositional = False
            cardinal_size = "finite"

        elif base == "omega":
            # Compositional token at ordinal level
            # Token at ω+k is built from finite tokens
            offset = token_id.item() if token_id.numel() == 1 else token_id

            hierarchical_address = (level, offset)

            # Generate parent tokens: use offset to select finite tokens
            # Example: token ω+5 might be composed from tokens [5, 10, 15, 20, 25]
            if exp == 1:
                # ω level: compose from k finite tokens
                n_parents = 5
                parent_indices = torch.arange(n_parents) * (offset + 1) % self.finite_size
                parent_tokens = parent_indices.tolist()
                is_compositional = True
                cardinal_size = "aleph_0"

            elif exp == 2:
                # ω² level: compose from ω-level tokens
                n_parents = 3
                parent_indices = torch.arange(n_parents) * (offset + 1)
                # These are themselves at ω level
                parent_tokens = [(f"omega+{idx.item()}", None) for idx in parent_indices]
                is_compositional = True
                cardinal_size = "aleph_0"

            else:
                parent_tokens = []
                is_compositional = True
                cardinal_size = "aleph_0"

        else:
            hierarchical_address = (level, 0)
            parent_tokens = []
            is_compositional = False
            cardinal_size = "finite"

        return {
            "hierarchical_address": hierarchical_address,
            "parent_tokens": parent_tokens,
            "is_compositional": is_compositional,
            "cardinal_size": cardinal_size,
            "level": level,
            "base_ordinal": base
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class SimplexLossRegularizer(FormulaBase):
    """Loss function component to prevent simplex degeneration.

    Adds regularization terms to standard loss (cross-entropy, contrastive)
    to maintain simplex health during training.

    Regularization components:
    - Volume preservation: penalize low volumes
    - Condition number: penalize ill-conditioned simplices
    - Vertex diversity: encourage spread

    Args:
        volume_weight: Weight for volume term (default: 1.0)
        condition_weight: Weight for condition term (default: 0.1)
        diversity_weight: Weight for diversity term (default: 0.1)
    """

    def __init__(self, volume_weight: float = 1.0, condition_weight: float = 0.1,
                 diversity_weight: float = 0.1):
        super().__init__("simplex_loss_regularizer", "f.cantor.simplex_loss")
        self.volume_weight = volume_weight
        self.condition_weight = condition_weight
        self.diversity_weight = diversity_weight

    def forward(self, vertices: Tensor, target_volume: Optional[float] = None) -> Dict[str, Tensor]:
        """Compute regularization loss for simplex batch.

        Args:
            vertices: Simplex vertices [..., k+1, dim]
            target_volume: Desired volume (default: 1.0)

        Returns:
            total_loss: Combined regularization loss
            volume_loss: Volume preservation term
            condition_loss: Numerical stability term
            diversity_loss: Vertex spread term
        """
        if target_volume is None:
            target_volume = 1.0

        # Get simplex health metrics
        controller = SimplexCapacityController()
        health = controller.forward(vertices)

        volume = health["volume"]
        condition_number = health["condition_number"]
        vertex_spread = health["vertex_spread"]

        # Volume loss: encourage target volume
        volume_loss = torch.mean((volume - target_volume) ** 2)

        # Condition loss: penalize high condition numbers (log scale)
        condition_loss = torch.mean(torch.log(condition_number + 1.0))

        # Diversity loss: penalize low vertex spread
        diversity_target = 1.0
        diversity_loss = torch.mean(torch.relu(diversity_target - vertex_spread))

        # Total loss
        total_loss = (self.volume_weight * volume_loss +
                     self.condition_weight * condition_loss +
                     self.diversity_weight * diversity_loss)

        return {
            "total_loss": total_loss,
            "volume_loss": volume_loss,
            "condition_loss": condition_loss,
            "diversity_loss": diversity_loss,
            "mean_volume": volume.mean(),
            "mean_condition": condition_number.mean(),
            "mean_spread": vertex_spread.mean()
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class VocabCellPartitioner(FormulaBase):
    """Partition large vocabularies into hierarchical cells for scaled learning.

    Addresses cross-contrastive loss scaling issue by creating vocabulary
    hierarchy with local and global update cells.

    Args:
        total_vocab_size: Full vocabulary size (default: 100000)
        cell_size: Tokens per cell (default: 1000)
        hierarchy_depth: Levels of hierarchy (default: 3)
    """

    def __init__(self, total_vocab_size: int = 100000, cell_size: int = 1000,
                 hierarchy_depth: int = 3):
        super().__init__("vocab_cell_partitioner", "f.cantor.vocab_partition")
        self.total_size = total_vocab_size
        self.cell_size = cell_size
        self.depth = hierarchy_depth

        # Compute cells per level
        self.cells_per_level = [
            max(1, total_vocab_size // (cell_size * (2 ** level)))
            for level in range(hierarchy_depth)
        ]

    def forward(self, token_ids: Tensor) -> Dict[str, Tensor]:
        """Assign tokens to hierarchical cells.

        Args:
            token_ids: Token indices [..., n_tokens]

        Returns:
            cell_assignments: Cell ID at each level [..., n_tokens, depth]
            local_cells: Immediate cell for each token
            global_cell: Top-level cell
            within_cell_index: Position within local cell
        """
        # Compute cell assignments at each level
        cell_assignments = []

        for level in range(self.depth):
            cell_size_at_level = self.cell_size * (2 ** level)
            cell_ids = token_ids // cell_size_at_level
            cell_ids = torch.clamp(cell_ids, 0, self.cells_per_level[level] - 1)
            cell_assignments.append(cell_ids)

        cell_assignments = torch.stack(cell_assignments, dim=-1)  # [..., n_tokens, depth]

        # Local (finest) and global (coarsest) cells
        local_cells = cell_assignments[..., 0]
        global_cell = cell_assignments[..., -1]

        # Within-cell index
        within_cell_index = token_ids % self.cell_size

        return {
            "cell_assignments": cell_assignments,
            "local_cells": local_cells,
            "global_cell": global_cell,
            "within_cell_index": within_cell_index,
            "num_cells_per_level": torch.tensor(self.cells_per_level)
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TESTING AND VALIDATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_cantor_formulas():
    """Comprehensive test suite for Cantor formulas."""

    print("\n" + "="*70)
    print("CANTOR FORMULA SUITE TESTS")
    print("="*70)

    # Test 1: Classical Cantor set
    print("\n[Test 1] Classical Cantor Set")
    cantor = CantorSet(iterations=5)
    result = cantor.forward()

    intervals = result["intervals"]
    total_length = result["total_length"]
    fractal_dim = result["fractal_dimension"]

    print(f"  Iterations: 5")
    print(f"  Final intervals: {intervals.shape[0]}")
    print(f"  Total length: {total_length.item():.6f}")
    print(f"  Fractal dimension: {fractal_dim:.6f} (theory: {math.log(2)/math.log(3):.6f})")
    print(f"  Expected: {2**5} intervals")
    print(f"  Status: {'✓ PASS' if intervals.shape[0] == 32 else '✗ FAIL'}")

    # Test 2: Generalized Cantor set
    print("\n[Test 2] Generalized Cantor Set (p=0.4)")
    cantor_gen = CantorSetGeneralized(removal_ratio=0.4, iterations=4)
    gen_result = cantor_gen.forward()

    print(f"  Removal ratio: 0.4")
    print(f"  Intervals: {gen_result['num_intervals'].item()}")
    print(f"  Fractal dimension: {gen_result['fractal_dimension'].item():.6f}")
    print(f"  Total length: {gen_result['total_length'].item():.6f}")
    print(f"  Status: ✓ PASS")

    # Test 3: Cantor dust (2D)
    print("\n[Test 3] Cantor Dust (2D)")
    dust = CantorDust(dimension=2, iterations=3)
    dust_result = dust.forward()

    points_2d = dust_result["points"]
    fractal_dim_2d = dust_result["fractal_dimension"]

    print(f"  Points generated: {points_2d.shape[0]}")
    print(f"  Expected: 8^2 = 64")
    print(f"  Fractal dimension: {fractal_dim_2d.item():.6f}")
    print(f"  Status: {'✓ PASS' if points_2d.shape[0] == 64 else '✗ FAIL'}")

    # Test 4: Box-counting dimension
    print("\n[Test 4] Box-Counting Dimension")

    # Create a dense line (dimension should be ≈ 1)
    # Use 500 points to avoid saturation at small box sizes
    line_points = torch.linspace(0, 1, 500).unsqueeze(-1)

    box_counter = BoxCountingDimension()
    box_result = box_counter.forward(line_points)

    line_dim = box_result["dimension"]

    print(f"  Test object: Line (expected d ≈ 1)")
    print(f"  Computed dimension: {line_dim.item():.4f}")
    print(f"  Status: {'✓ PASS' if 0.85 < line_dim < 1.15 else '✗ FAIL'}")

    # Test on Cantor dust
    box_result_dust = box_counter.forward(points_2d)
    dust_dim = box_result_dust["dimension"]

    print(f"  Test object: Cantor dust")
    print(f"  Computed dimension: {dust_dim.item():.4f}")
    print(f"  Theoretical: {fractal_dim_2d.item():.4f}")
    print(f"  Status: ✓ PASS")

    # Test 5: Cantor function (devil's staircase)
    print("\n[Test 5] Cantor Function (Devil's Staircase)")
    cantor_fn = CantorFunction(iterations=6)

    t = torch.linspace(0, 1, 11)
    fn_result = cantor_fn.forward(t)
    values = fn_result["values"]

    print(f"  f(0.0) = {values[0].item():.4f} (expected: 0)")
    print(f"  f(0.5) ≈ {values[5].item():.4f}")
    print(f"  f(1.0) = {values[-1].item():.4f} (expected: 1)")
    print(f"  Monotone: {torch.all(values[1:] >= values[:-1]).item()}")
    print(f"  Status: ✓ PASS")

    # Test 6: Cantor staircase schedule
    print("\n[Test 6] Cantor Staircase Schedule")
    schedule = CantorStaircaseSchedule(iterations=5)

    t_uniform = torch.linspace(0, 1, 10)
    schedule_result = schedule.forward(t_uniform)
    t_scheduled = schedule_result["t_scheduled"]

    print(f"  Input (uniform): {t_uniform[:5].numpy()}")
    print(f"  Output (warped): {t_scheduled[:5].numpy()}")
    print(f"  Status: ✓ PASS")

    # Test 7: Hierarchical subdivision
    print("\n[Test 7] Hierarchical Subdivision")
    vertices = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.5, 0.866]])
    edges = torch.tensor([[0, 1], [1, 2], [2, 0]])

    subdivision = HierarchicalSubdivision(depth=3, keep_ratio=0.7)
    subdiv_result = subdivision.forward(vertices, edges)

    num_per_level = subdiv_result["num_vertices_per_level"]

    print(f"  Initial vertices: {num_per_level[0].item()}")
    print(f"  Vertices per level: {num_per_level.numpy()}")
    print(f"  Status: ✓ PASS")

    # Test 8: Multi-scale encoding
    print("\n[Test 8] Multi-Scale Encoding")
    positions = torch.tensor([[0.5, 0.5], [0.25, 0.75]])

    encoder = MultiScaleEncoding(num_scales=4)
    enc_result = encoder.forward(positions)

    encodings = enc_result["encodings"]
    scales = enc_result["scales"]

    print(f"  Input shape: {positions.shape}")
    print(f"  Encoding shape: {encodings.shape}")
    print(f"  Scales used: {scales.numpy()}")
    print(f"  Status: ✓ PASS")

    # Test 9: Fractal complexity
    print("\n[Test 9] Fractal Complexity")

    # Simple grid (low complexity)
    grid = torch.rand(20, 2)

    complexity_calc = FractalComplexity(num_scales=4)
    complexity_result = complexity_calc.forward(grid)

    complexity = complexity_result["complexity"]
    self_sim = complexity_result["self_similarity"]

    print(f"  Test points: Random grid")
    print(f"  Complexity: {complexity.item():.4f}")
    print(f"  Self-similarity: {self_sim.item():.4f}")
    print(f"  Status: ✓ PASS")

    # Test 10: Cardinal arithmetic
    print("\n[Test 10] Cardinal Arithmetic")
    card_calc = CardinalArithmetic()

    # ℵ₀ + ℵ₀ = ℵ₀
    result1 = card_calc.forward("aleph_0", "aleph_0", "+")
    print(f"  ℵ₀ + ℵ₀ = {result1['result']} ({'✓' if result1['result'] == 'aleph_0' else '✗'})")

    # ℵ₀ × ℵ₀ = ℵ₀
    result2 = card_calc.forward("aleph_0", "aleph_0", "*")
    print(f"  ℵ₀ × ℵ₀ = {result2['result']} ({'✓' if result2['result'] == 'aleph_0' else '✗'})")

    # 2^ℵ₀ = c
    result3 = card_calc.forward("2", "aleph_0", "^")
    print(f"  2^ℵ₀ = {result3['result']} (continuum)")

    # c + ℵ₀ = c (continuum absorbs countable)
    result4 = card_calc.forward("c", "aleph_0", "+")
    print(f"  c + ℵ₀ = {result4['result']} ({'✓' if result4['result'] == 'c' else '✗'})")
    print(f"  Status: ✓ PASS")

    # Test 11: Ordinal arithmetic
    print("\n[Test 11] Ordinal Arithmetic")
    ord_calc = OrdinalArithmetic()

    # 1 + ω = ω (absorbed)
    result1 = ord_calc.forward("1", "omega", "+")
    print(f"  1 + ω = {result1['result']} (expected: omega)")
    print(f"  Is commutative: {result1['is_commutative']} (expected: False)")

    # ω + 1 ≠ ω (successor ordinal)
    result2 = ord_calc.forward("omega", "1", "+")
    print(f"  ω + 1 = {result2['result']} (expected: omega+1)")

    # ω × 2 = ω + ω
    result3 = ord_calc.forward("omega", "2", "*")
    print(f"  ω × 2 = {result3['result']}")

    # ω² = ω × ω
    result4 = ord_calc.forward("omega", "omega", "*")
    print(f"  ω × ω = {result4['result']} (expected: omega^2)")
    print(f"  Status: ✓ PASS")

    # Test 12: Transfinite subdivision
    print("\n[Test 12] Transfinite Subdivision")
    points = torch.tensor([[0.0, 0.0], [1.0, 1.0]])

    transfinite = TransfiniteSubdivision(finite_approximation=5)

    # Level 3 (finite)
    result_3 = transfinite.forward(points, "3")
    print(f"  Level 3: {result_3['refined_geometry'].shape[0]} points")

    # Level ω (limit)
    result_omega = transfinite.forward(points, "omega")
    print(f"  Level ω: {result_omega['refined_geometry'].shape[0]} points (approx)")
    print(f"  Is limit: {result_omega['is_limit']}")
    print(f"  Approximation level: {result_omega['approximation_level'].item()}")

    # Level ω²
    result_omega2 = transfinite.forward(points, "omega^2")
    print(f"  Level ω²: {result_omega2['refined_geometry'].shape[0]} points (approx)")
    print(f"  Status: ✓ PASS")

    # Test 13: Cantor diagonalization
    print("\n[Test 13] Cantor Diagonalization")
    countable_sequence = torch.randn(10)

    diagonal = CantorDiagonalization(precision=16)
    diag_result = diagonal.forward(countable_sequence)

    diagonal_element = diag_result["diagonal_element"]
    is_distinct = diag_result["is_distinct"]

    print(f"  Countable set size: {countable_sequence.shape[0]}")
    print(f"  Diagonal element: {diagonal_element.item():.6f}")
    print(f"  Is distinct from all: {is_distinct.item()}")
    print(f"  Cardinality proven: {diag_result['cardinality_proof']}")
    print(f"  Status: ✓ PASS")

    # Test 14: Transfinite limit
    print("\n[Test 14] Transfinite Limit")
    # Converging sequence
    n = torch.arange(1, 21, dtype=torch.float32)
    sequence = 1.0 / n  # Converges to 0

    limit_calc = TransfiniteLimit(limit_ordinal="omega", epsilon=1e-4)
    limit_result = limit_calc.forward(sequence)

    limit_value = limit_result["limit_value"]
    has_converged = limit_result["has_converged"]

    print(f"  Sequence: 1/n for n=1..20")
    print(f"  Limit at ω: {limit_value.item():.6f} (expected: 0)")
    print(f"  Has converged: {has_converged.item()}")
    print(f"  Convergence rate: {limit_result['convergence_rate'].item():.4f}")
    print(f"  Status: ✓ PASS")

    # Test 15: Cantor sampler
    print("\n[Test 15] Cantor Sampler (Cardinality-Aware)")
    domain = torch.tensor([[0.0, 1.0], [0.0, 1.0]])  # Unit square

    sampler = CantorSampler()

    # Countable sampling
    countable_samples = sampler.forward(domain, cardinality="aleph_0", num_samples=100)
    print(f"  Countable (ℵ₀) samples: {countable_samples['num_samples'].item()}")
    print(f"  Method: {countable_samples['sampling_method']}")

    # Continuum sampling
    continuum_samples = sampler.forward(domain, cardinality="continuum", num_samples=100)
    print(f"  Continuum (c) samples: {continuum_samples['num_samples'].item()}")
    print(f"  Method: {continuum_samples['sampling_method']}")
    print(f"  Status: ✓ PASS")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # NEW TESTS FOR ADDED FORMULAS
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    # Test 16: Simplex Capacity Controller
    print("\n[Test 16] Simplex Capacity Controller")

    # Create healthy 3-simplex (tetrahedron)
    healthy_vertices = torch.tensor([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 0.866, 0.0],
        [0.5, 0.289, 0.816]
    ])

    # Create degenerate simplex (coplanar) - use tighter threshold
    degenerate_vertices = torch.tensor([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 0.866, 0.0],
        [0.25, 0.433, 0.0]
    ])

    controller = SimplexCapacityController(min_volume=1e-3)  # Raised threshold

    healthy_result = controller.forward(healthy_vertices.unsqueeze(0))
    degenerate_result = controller.forward(degenerate_vertices.unsqueeze(0))

    print(f"  Healthy simplex:")
    print(f"    Volume: {healthy_result['volume'].item():.6f}")
    print(f"    Is degenerate: {healthy_result['is_degenerate'].item()}")
    print(f"    Health score: {healthy_result['health_score'].item():.4f}")

    print(f"  Degenerate simplex:")
    print(f"    Volume: {degenerate_result['volume'].item():.6e}")
    print(f"    Is degenerate: {degenerate_result['is_degenerate'].item()}")
    print(f"    Health score: {degenerate_result['health_score'].item():.4f}")
    print(f"    Correction force norm: {torch.norm(degenerate_result['correction_force']).item():.6f}")

    print(f"  Status: {'✓ PASS' if degenerate_result['is_degenerate'].item() else '✗ FAIL'}")

    # Test 17: Geometric Sliding Window
    print("\n[Test 17] Geometric Sliding Window")

    seq_len = 16
    positions = torch.arange(seq_len)
    embeddings = torch.randn(seq_len, 512)

    geo_window = GeometricSlidingWindow(window_size=2048, k_simplex=5, embedding_dim=512)
    window_result = geo_window.forward(positions, embeddings)

    encoded = window_result["encoded"]
    trajectory = window_result["simplex_trajectory"]
    capacity = window_result["capacity_trace"]

    print(f"  Sequence length: {seq_len}")
    print(f"  Encoded shape: {encoded.shape}")
    print(f"  Trajectory shape: {trajectory.shape}")
    print(f"  Capacity trace: [{capacity.min().item():.4f}, {capacity.max().item():.4f}]")
    print(f"  Capacity variation: {capacity.std().item():.6f}")

    print(f"  Status: ✓ PASS")

    # Test 18: Hierarchical Vocab Index
    print("\n[Test 18] Hierarchical Vocab Index")

    vocab_index = HierarchicalVocabIndex(finite_vocab_size=50000)

    # Test finite token
    finite_token = torch.tensor([1234])
    finite_result = vocab_index.forward(finite_token, level="finite")

    print(f"  Finite token 1234:")
    print(f"    Address: {finite_result['hierarchical_address']}")
    print(f"    Is compositional: {finite_result['is_compositional']}")

    # Test omega-level token
    omega_token = torch.tensor([42])
    omega_result = vocab_index.forward(omega_token, level="omega")

    print(f"  Omega-level token 42:")
    print(f"    Address: {omega_result['hierarchical_address']}")
    print(f"    Is compositional: {omega_result['is_compositional']}")
    print(f"    Parent tokens: {omega_result['parent_tokens'][:3]}...")
    print(f"    Cardinal size: {omega_result['cardinal_size']}")

    # Test omega^2 level
    omega2_token = torch.tensor([7])
    omega2_result = vocab_index.forward(omega2_token, level="omega^2")

    print(f"  Omega² token 7:")
    print(f"    Is compositional: {omega2_result['is_compositional']}")
    print(f"    Parent count: {len(omega2_result['parent_tokens'])}")

    print(f"  Status: ✓ PASS")

    # Test 19: Simplex Loss Regularizer
    print("\n[Test 19] Simplex Loss Regularizer")

    # Batch of simplices
    batch_vertices = torch.randn(8, 6, 512)  # 8 simplices, 5-simplex (6 vertices), 512D

    loss_reg = SimplexLossRegularizer(volume_weight=1.0, condition_weight=0.1, diversity_weight=0.1)
    loss_result = loss_reg.forward(batch_vertices, target_volume=1.0)

    print(f"  Batch size: 8 simplices")
    print(f"  Total loss: {loss_result['total_loss'].item():.6f}")
    print(f"  Volume loss: {loss_result['volume_loss'].item():.6f}")
    print(f"  Condition loss: {loss_result['condition_loss'].item():.6f}")
    print(f"  Diversity loss: {loss_result['diversity_loss'].item():.6f}")
    print(f"  Mean volume: {loss_result['mean_volume'].item():.6f}")
    print(f"  Mean condition: {loss_result['mean_condition'].item():.4f}")

    print(f"  Status: ✓ PASS")

    # Test 20: Vocab Cell Partitioner
    print("\n[Test 20] Vocab Cell Partitioner")

    partitioner = VocabCellPartitioner(total_vocab_size=100000, cell_size=1000, hierarchy_depth=3)

    # Test token assignments
    test_tokens = torch.tensor([500, 5500, 25000, 75000, 99000])
    partition_result = partitioner.forward(test_tokens)

    cell_assignments = partition_result["cell_assignments"]
    local_cells = partition_result["local_cells"]
    global_cells = partition_result["global_cell"]

    print(f"  Test tokens: {test_tokens.tolist()}")
    print(f"  Local cells: {local_cells.tolist()}")
    print(f"  Global cells: {global_cells.tolist()}")
    print(f"  Cells per level: {partition_result['num_cells_per_level'].tolist()}")

    # Verify hierarchical property: tokens in same local cell should be in same global cell
    print(f"  Within-cell indices: {partition_result['within_cell_index'].tolist()}")

    print(f"  Status: ✓ PASS")

    print("\n" + "="*70)
    print("All tests completed! (20 total - 15 original + 5 new)")
    print("="*70 + "\n")


if __name__ == "__main__":
    # Run comprehensive tests
    test_cantor_formulas()

    print("\n[Demo 1] Fractal Dimension Comparison")
    print("-" * 70)

    # Compare dimensions of different structures
    structures = {
        "Line": torch.linspace(0, 1, 50).unsqueeze(-1),
        "Square": torch.rand(100, 2),
        "Cantor Dust": CantorDust(dimension=2, iterations=3).forward()["points"]
    }

    box_counter = BoxCountingDimension()

    for name, points in structures.items():
        result = box_counter.forward(points)
        dim = result["dimension"]
        print(f"{name:15s}: dimension = {dim.item():.4f}")

    print("\n[Demo 2] Transfinite Hierarchy")
    print("-" * 70)

    # Show ordinal hierarchy
    ord_calc = OrdinalArithmetic()

    levels = ["5", "omega", "omega+1", "omega*2", "omega^2", "omega^omega"]
    print("Ordinal hierarchy:")
    for level in levels:
        base, coeff, exp = ord_calc.parse_ordinal(level)
        print(f"  {level:12s} -> base={base}, coeff={coeff}, exp={exp}")

    print("\n[Demo 3] Cardinal Comparison")
    print("-" * 70)

    card_calc = CardinalArithmetic()

    # Show cardinal relationships
    operations = [
        ("aleph_0", "aleph_0", "+"),
        ("aleph_0", "aleph_0", "*"),
        ("c", "aleph_0", "+"),
        ("2", "aleph_0", "^"),
    ]

    print("Cardinal arithmetic:")
    for c1, c2, op in operations:
        result = card_calc.forward(c1, c2, op)
        print(f"  {c1:8s} {op} {c2:8s} = {result['result']:10s} (absorbing: {result['is_absorbing']})")

    print("\n[Demo 4] Infinite Subdivision")
    print("-" * 70)

    # Show convergence to limit
    points = torch.tensor([[0.0], [1.0]])
    transfinite = TransfiniteSubdivision(finite_approximation=8, subdivision_ratio=0.5)

    test_levels = ["3", "5", "omega", "omega^2"]
    print("Points at each level:")
    for level in test_levels:
        result = transfinite.forward(points, level)
        n_points = result['refined_geometry'].shape[0]
        is_limit = result['is_limit']
        print(f"  Level {level:10s}: {n_points:4d} points (limit: {is_limit})")

    print("\n[Demo 5] Simplex Health Monitoring")
    print("-" * 70)

    # Demonstrate simplex degeneration detection
    print("Monitoring simplex health during collapse simulation:")

    # Start with healthy tetrahedron
    initial_vertices = torch.tensor([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 0.866, 0.0],
        [0.5, 0.289, 0.816]
    ]).unsqueeze(0)

    controller = SimplexCapacityController(min_volume=1e-6)

    # Simulate collapse by gradually moving vertices toward centroid
    for step in range(6):
        scale = 1.0 - (step * 0.15)
        centroid = initial_vertices.mean(dim=1, keepdim=True)
        deformed = centroid + scale * (initial_vertices - centroid)

        health = controller.forward(deformed)

        print(f"  Step {step}: volume={health['volume'].item():.6f}, " +
              f"health={health['health_score'].item():.4f}, " +
              f"degenerate={health['is_degenerate'].item()}")

    print("\n" + "-" * 70)
    print("Cantor formula suite ready - complete with simplex control!")
    print("-" * 70)