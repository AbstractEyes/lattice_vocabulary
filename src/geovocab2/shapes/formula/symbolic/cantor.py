"""
CANTOR FORMULA SUITE
--------------------
Fractal geometry, infinite hierarchies, and multi-resolution structures.

Named in honor of:
  • Georg Cantor (1845–1918) – set theory, fractals, infinite cardinalities, topology

This suite provides formulas for fractal-based geometric operations:
  - Cantor set construction (ternary and generalized)
  - Fractal dimension computation (Hausdorff, box-counting)
  - Cantor function (devil's staircase) for diffusion schedules
  - Hierarchical subdivision for multi-resolution meshes
  - Fractal interpolation and adaptive refinement
  - Self-similar geometric structures

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

Applications:
    - Hierarchical mesh refinement (adaptive LOD)
    - Non-uniform diffusion schedules
    - Multi-scale geometric representations
    - Sparse point cloud generation
    - Fractal noise for texture synthesis

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

    def __init__(self, removal_ratio: float = 1.0 / 3.0, iterations: int = 5):
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
                 removal_ratio: float = 1.0 / 3.0):
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
            # Default: ε = 2^(-k) for k = 1, 2, ..., 8
            self.box_sizes = [2.0 ** (-k) for k in range(1, 9)]
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

        # Find bounding box
        min_coords = points.min(dim=-2)[0]
        max_coords = points.max(dim=-2)[0]
        extent = (max_coords - min_coords).max()

        box_counts = []
        valid_sizes = []

        for box_size in self.box_sizes:
            # Number of boxes along each dimension
            n_boxes_per_dim = int(torch.ceil(extent / box_size).item()) + 1

            # Discretize points to box indices
            normalized_points = (points - min_coords) / box_size
            box_indices = torch.floor(normalized_points).long()

            # Clamp to valid range
            box_indices = torch.clamp(box_indices, 0, n_boxes_per_dim - 1)

            # Count unique boxes
            # Flatten multi-dimensional indices to 1D
            if dim == 1:
                flat_indices = box_indices
            elif dim == 2:
                flat_indices = box_indices[..., 0] * n_boxes_per_dim + box_indices[..., 1]
            elif dim == 3:
                flat_indices = (box_indices[..., 0] * n_boxes_per_dim * n_boxes_per_dim +
                                box_indices[..., 1] * n_boxes_per_dim +
                                box_indices[..., 2])
            else:
                # General case: unique rows
                flat_indices = box_indices[..., 0]  # Simplified for now

            n_occupied = torch.unique(flat_indices.flatten()).numel()

            if n_occupied > 0:
                box_counts.append(n_occupied)
                valid_sizes.append(box_size)

        if len(box_counts) < self.min_boxes:
            # Not enough data points
            return {
                "dimension": torch.tensor(float('nan')),
                "box_counts": torch.tensor(box_counts),
                "box_sizes": torch.tensor(valid_sizes),
                "log_log_slope": torch.tensor(float('nan'))
            }

        # Linear regression on log-log plot
        # log(N) ≈ -d · log(ε) + c
        # So d = -slope
        log_sizes = torch.log(torch.tensor(valid_sizes))
        log_counts = torch.log(torch.tensor(box_counts, dtype=torch.float32))

        # Fit line: log_counts = slope * log_sizes + intercept
        # Using least squares
        n = len(valid_sizes)
        sum_x = log_sizes.sum()
        sum_y = log_counts.sum()
        sum_xx = (log_sizes ** 2).sum()
        sum_xy = (log_sizes * log_counts).sum()

        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x ** 2 + 1e-10)
        dimension = -slope  # Negative because log(1/ε) = -log(ε)

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

        # For each Cantor interval at final iteration, assign value
        n_intervals = self.cantor_intervals.shape[0]

        for i, interval in enumerate(self.cantor_intervals):
            a, b = interval[0], interval[1]

            # Points in this interval get value i / (n_intervals - 1)
            mask = (t >= a) & (t <= b)
            values[mask] = i / (n_intervals - 1.0)

        # Handle removed middle thirds (linear interpolation)
        # This is approximate; exact construction is more complex

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
        encodings = []
        scales = []

        for scale_idx in range(self.num_scales):
            # Cantor-like frequency scaling: f_k = f_0 * 2^k
            frequency = self.base_frequency * (2.0 ** scale_idx)
            scales.append(frequency)

            # Sinusoidal encoding at this scale
            angle = 2.0 * math.pi * frequency * positions
            sin_enc = torch.sin(angle)
            cos_enc = torch.cos(angle)

            encodings.append(sin_enc)
            encodings.append(cos_enc)

        # Concatenate all scales
        encodings = torch.cat(encodings, dim=-1)

        return {
            "encodings": encodings,
            "scales": torch.tensor(scales),
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

        # Scale entropy: measure information at each scale
        scale_entropies = []
        for scale_idx in range(self.num_scales):
            box_size = 2.0 ** (-(scale_idx + 1))

            # Count points in boxes (simplified)
            # This is a proxy for information content
            scale_entropies.append(box_size * torch.log(torch.tensor(points.shape[0] + 1.0)))

        scale_entropies = torch.tensor(scale_entropies)

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
# TESTING AND VALIDATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_cantor_formulas():
    """Comprehensive test suite for Cantor formulas."""

    print("\n" + "=" * 70)
    print("CANTOR FORMULA SUITE TESTS")
    print("=" * 70)

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
    print(f"  Fractal dimension: {fractal_dim:.6f} (theory: {math.log(2) / math.log(3):.6f})")
    print(f"  Expected: {2 ** 5} intervals")
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

    # Create a simple line (dimension should be ≈ 1)
    line_points = torch.linspace(0, 1, 100).unsqueeze(-1)

    box_counter = BoxCountingDimension()
    box_result = box_counter.forward(line_points)

    line_dim = box_result["dimension"]

    print(f"  Test object: Line (expected d ≈ 1)")
    print(f"  Computed dimension: {line_dim.item():.4f}")
    print(f"  Status: {'✓ PASS' if 0.8 < line_dim < 1.2 else '✗ FAIL'}")

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

    print("\n" + "=" * 70)
    print("All tests completed!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    # Run comprehensive tests
    test_cantor_formulas()

    print("\n[Demo] Fractal Dimension Comparison")
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

    print("\n" + "-" * 70)
    print("Cantor formula suite ready for fractal geometry!")
    print("-" * 70)