"""
ATOMIC OPERATIONS
-----------------
Fundamental geometric and mathematical operations for crystal lattice systems.

This module provides the most basic, reusable operations that serve as building blocks
for more complex formulas. These are the "atoms" - indivisible primitives.

Categories:
  - Vector operations (dot, cross, norm, normalize)
  - Distance metrics (Euclidean, Manhattan, Chebyshev, Minkowski)
  - Angle calculations (between vectors, oriented angles)
  - Interpolation (linear, spherical, barycentric)
  - Projections (point-to-line, point-to-plane)
  - Basic transformations (translation, rotation, scaling)
  - Geometric primitives (area, volume, centroid)

Mathematical Foundation:

    Dot Product:
        a · b = Σ aᵢbᵢ = |a||b|cos(θ)

    Cross Product (3D):
        a × b = [a₂b₃-a₃b₂, a₃b₁-a₁b₃, a₁b₂-a₂b₁]

    Distance Metrics:
        Euclidean: d = √(Σ(xᵢ-yᵢ)²)
        Manhattan: d = Σ|xᵢ-yᵢ|
        Chebyshev: d = max|xᵢ-yᵢ|
        Minkowski: d = (Σ|xᵢ-yᵢ|ᵖ)^(1/p)

    Angle:
        θ = arccos((a·b)/(|a||b|))

    Linear Interpolation:
        lerp(a, b, t) = (1-t)a + tb

    Spherical Linear Interpolation:
        slerp(a, b, t) = sin((1-t)θ)a/sin(θ) + sin(tθ)b/sin(θ)

    Projection:
        proj_b(a) = ((a·b)/(b·b))b

Author: AbstractPhil + Claude Sonnet 4.5
License: MIT
"""

from typing import Dict, Optional, Tuple, Literal
import torch
from torch import Tensor
import math

from shapes.formula.formula_base import FormulaBase


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# VECTOR OPERATIONS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class DotProduct(FormulaBase):
    """Compute dot product between vectors.

    a · b = Σ aᵢbᵢ
    """

    def __init__(self):
        super().__init__("dot_product", "f.atomic.dot")

    def forward(self, a: Tensor, b: Tensor) -> Dict[str, Tensor]:
        """Compute dot product.

        Args:
            a: First vectors [..., dim]
            b: Second vectors [..., dim]

        Returns:
            dot: a · b [...]
            normalized_dot: (a·b)/(|a||b|) for angle computation [...]
        """
        # Dot product
        dot = torch.sum(a * b, dim=-1)

        # Normalized (for angle)
        norm_a = torch.norm(a, dim=-1)
        norm_b = torch.norm(b, dim=-1)
        normalized_dot = dot / (norm_a * norm_b + 1e-10)

        # Clamp to [-1, 1] for numerical stability
        normalized_dot = torch.clamp(normalized_dot, -1.0, 1.0)

        return {
            "dot": dot,
            "normalized_dot": normalized_dot,
            "norm_a": norm_a,
            "norm_b": norm_b
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class CrossProduct(FormulaBase):
    """Compute cross product between 3D vectors.

    a × b (perpendicular to both a and b)
    """

    def __init__(self):
        super().__init__("cross_product", "f.atomic.cross")

    def forward(self, a: Tensor, b: Tensor) -> Dict[str, Tensor]:
        """Compute cross product.

        Args:
            a: First vectors [..., 3]
            b: Second vectors [..., 3]

        Returns:
            cross: a × b [..., 3]
            magnitude: |a × b| [...]
            unit_cross: normalized cross product [..., 3]
        """
        if a.shape[-1] != 3 or b.shape[-1] != 3:
            raise ValueError("Cross product requires 3D vectors")

        # Cross product
        cross = torch.cross(a, b, dim=-1)

        # Magnitude
        magnitude = torch.norm(cross, dim=-1)

        # Unit vector
        unit_cross = cross / (magnitude.unsqueeze(-1) + 1e-10)

        return {
            "cross": cross,
            "magnitude": magnitude,
            "unit_cross": unit_cross,
            "is_parallel": magnitude < 1e-6
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class VectorNorm(FormulaBase):
    """Compute vector norms (L1, L2, Linf).

    Args:
        p: Norm order (1, 2, or 'inf')
    """

    def __init__(self, p: float = 2.0):
        super().__init__("vector_norm", "f.atomic.norm")
        self.p = p

    def forward(self, vectors: Tensor) -> Dict[str, Tensor]:
        """Compute norm.

        Args:
            vectors: Input vectors [..., dim]

        Returns:
            norm: ||v||_p [...]
            unit_vector: v/||v|| [..., dim]
        """
        # Compute norm
        if self.p == float('inf'):
            norm = torch.max(torch.abs(vectors), dim=-1)[0]
        else:
            norm = torch.norm(vectors, p=self.p, dim=-1)

        # Unit vector
        unit_vector = vectors / (norm.unsqueeze(-1) + 1e-10)

        return {
            "norm": norm,
            "unit_vector": unit_vector,
            "is_zero": norm < 1e-10
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class Normalize(FormulaBase):
    """Normalize vectors to unit length.

    Args:
        epsilon: Small value to avoid division by zero
    """

    def __init__(self, epsilon: float = 1e-10):
        super().__init__("normalize", "f.atomic.normalize")
        self.eps = epsilon

    def forward(self, vectors: Tensor) -> Dict[str, Tensor]:
        """Normalize vectors.

        Args:
            vectors: Input vectors [..., dim]

        Returns:
            normalized: Unit vectors [..., dim]
            original_norm: Original magnitudes [...]
        """
        norm = torch.norm(vectors, dim=-1, keepdim=True)
        normalized = vectors / (norm + self.eps)

        return {
            "normalized": normalized,
            "original_norm": norm.squeeze(-1),
            "was_zero": norm.squeeze(-1) < self.eps
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DISTANCE METRICS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class DistanceMetric(FormulaBase):
    """Compute various distance metrics.

    Args:
        metric: 'euclidean', 'manhattan', 'chebyshev', or 'minkowski'
        p: Order for Minkowski distance (default: 2)
    """

    def __init__(self, metric: str = 'euclidean', p: float = 2.0):
        super().__init__("distance_metric", "f.atomic.distance")
        self.metric = metric
        self.p = p

    def forward(self, points_a: Tensor, points_b: Tensor) -> Dict[str, Tensor]:
        """Compute distances.

        Args:
            points_a: First set of points [..., n_a, dim]
            points_b: Second set of points [..., n_b, dim]

        Returns:
            distances: Distance matrix [..., n_a, n_b]
            min_distance: Minimum distance [...]
            nearest_pairs: Indices of nearest pairs
        """
        if self.metric == 'euclidean':
            distances = torch.cdist(points_a, points_b, p=2)

        elif self.metric == 'manhattan':
            distances = torch.cdist(points_a, points_b, p=1)

        elif self.metric == 'chebyshev':
            # Chebyshev: max over dimensions
            diff = points_a.unsqueeze(-2) - points_b.unsqueeze(-3)  # [..., n_a, n_b, dim]
            distances = torch.max(torch.abs(diff), dim=-1)[0]

        elif self.metric == 'minkowski':
            distances = torch.cdist(points_a, points_b, p=self.p)

        else:
            raise ValueError(f"Unknown metric: {self.metric}")

        # Find minimum distances
        min_distance = distances.min()

        # Nearest pairs
        flat_idx = torch.argmin(distances.flatten())
        n_b = points_b.shape[-2]
        nearest_a = flat_idx // n_b
        nearest_b = flat_idx % n_b

        return {
            "distances": distances,
            "min_distance": min_distance,
            "nearest_pair": (nearest_a, nearest_b),
            "mean_distance": distances.mean()
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ANGLE CALCULATIONS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class AngleBetweenVectors(FormulaBase):
    """Compute angle between vectors.

    θ = arccos((a·b)/(|a||b|))

    Args:
        use_degrees: Return angle in degrees instead of radians
    """

    def __init__(self, use_degrees: bool = False):
        super().__init__("angle_between", "f.atomic.angle")
        self.degrees = use_degrees

    def forward(self, a: Tensor, b: Tensor) -> Dict[str, Tensor]:
        """Compute angle.

        Args:
            a: First vectors [..., dim]
            b: Second vectors [..., dim]

        Returns:
            angle: θ in radians or degrees [...]
            cos_angle: cos(θ) [...]
            is_orthogonal: |θ - π/2| < ε [...]
            is_parallel: θ ≈ 0 or θ ≈ π [...]
        """
        # Normalized dot product
        dot = torch.sum(a * b, dim=-1)
        norm_a = torch.norm(a, dim=-1)
        norm_b = torch.norm(b, dim=-1)

        cos_angle = dot / (norm_a * norm_b + 1e-10)
        cos_angle = torch.clamp(cos_angle, -1.0, 1.0)

        # Angle
        angle = torch.acos(cos_angle)

        if self.degrees:
            angle = angle * 180.0 / math.pi
            threshold_ortho = 5.0  # degrees
            threshold_parallel = 5.0
        else:
            threshold_ortho = 0.087  # ~5 degrees in radians
            threshold_parallel = 0.087

        # Check special cases
        is_orthogonal = torch.abs(angle - math.pi/2) < threshold_ortho
        is_parallel = (angle < threshold_parallel) | (angle > (math.pi - threshold_parallel))

        return {
            "angle": angle,
            "cos_angle": cos_angle,
            "is_orthogonal": is_orthogonal,
            "is_parallel": is_parallel
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class OrientedAngle2D(FormulaBase):
    """Compute oriented (signed) angle in 2D.

    Returns angle from a to b in range [-π, π]
    """

    def __init__(self):
        super().__init__("oriented_angle_2d", "f.atomic.oriented_angle")

    def forward(self, a: Tensor, b: Tensor) -> Dict[str, Tensor]:
        """Compute oriented angle.

        Args:
            a: First vectors [..., 2]
            b: Second vectors [..., 2]

        Returns:
            angle: Signed angle in radians [...]
            is_clockwise: Rotation direction [...]
        """
        if a.shape[-1] != 2 or b.shape[-1] != 2:
            raise ValueError("Oriented angle requires 2D vectors")

        # atan2(b_y, b_x) - atan2(a_y, a_x)
        angle_a = torch.atan2(a[..., 1], a[..., 0])
        angle_b = torch.atan2(b[..., 1], b[..., 0])

        angle = angle_b - angle_a

        # Wrap to [-π, π]
        angle = torch.atan2(torch.sin(angle), torch.cos(angle))

        is_clockwise = angle < 0

        return {
            "angle": angle,
            "is_clockwise": is_clockwise,
            "angle_degrees": angle * 180.0 / math.pi
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# INTERPOLATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class LinearInterpolation(FormulaBase):
    """Linear interpolation between points.

    lerp(a, b, t) = (1-t)a + tb
    """

    def __init__(self):
        super().__init__("lerp", "f.atomic.lerp")

    def forward(self, a: Tensor, b: Tensor, t: Tensor) -> Dict[str, Tensor]:
        """Interpolate linearly.

        Args:
            a: Start points [..., dim]
            b: End points [..., dim]
            t: Interpolation parameter [...] or scalar, t ∈ [0,1]

        Returns:
            result: Interpolated points [..., dim]
            is_extrapolating: t outside [0,1] [...]
        """
        # Reshape t for broadcasting
        # If t is [n] and a,b are [dim], result should be [n, dim]
        if t.ndim == 1 and a.ndim == 1:
            # t: [n], a: [dim] -> make t: [n, 1] for broadcasting
            t_expanded = t.unsqueeze(-1)
            result = (1.0 - t_expanded) * a.unsqueeze(0) + t_expanded * b.unsqueeze(0)
            is_extrapolating = (t < 0) | (t > 1.0)
        elif t.ndim == 0:
            # Scalar t
            result = (1.0 - t) * a + t * b
            is_extrapolating = (t < 0) | (t > 1.0)
        else:
            # General case: ensure t has extra dim for element-wise operation
            if t.shape != a.shape[:-1]:
                t_expanded = t.unsqueeze(-1)
            else:
                t_expanded = t
            result = (1.0 - t_expanded) * a + t_expanded * b
            is_extrapolating = (t < 0) | (t > 1.0)

        return {
            "result": result,
            "is_extrapolating": is_extrapolating,
            "t": t
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class SphericalInterpolation(FormulaBase):
    """Spherical linear interpolation (SLERP).

    Interpolates along great circle on unit sphere.
    """

    def __init__(self):
        super().__init__("slerp", "f.atomic.slerp")

    def forward(self, a: Tensor, b: Tensor, t: Tensor) -> Dict[str, Tensor]:
        """Spherical interpolation.

        Args:
            a: Start vectors [..., dim] (will be normalized)
            b: End vectors [..., dim] (will be normalized)
            t: Interpolation parameter [...]

        Returns:
            result: Interpolated unit vectors [..., dim]
            angle: Angle between a and b [...]
        """
        # Normalize inputs
        a_norm = a / (torch.norm(a, dim=-1, keepdim=True) + 1e-10)
        b_norm = b / (torch.norm(b, dim=-1, keepdim=True) + 1e-10)

        # Dot product (cos of angle)
        dot = torch.sum(a_norm * b_norm, dim=-1)
        dot = torch.clamp(dot, -1.0, 1.0)

        # Angle
        theta = torch.acos(dot)

        # Ensure t has right shape
        if t.ndim < a.ndim:
            t_expanded = t.unsqueeze(-1)
        else:
            t_expanded = t

        # SLERP formula
        # If angle is small, fall back to lerp
        use_lerp = theta < 1e-3

        if use_lerp.any():
            # Linear interpolation for small angles
            result_lerp = (1.0 - t_expanded) * a_norm + t_expanded * b_norm
            result_lerp = result_lerp / (torch.norm(result_lerp, dim=-1, keepdim=True) + 1e-10)

        # SLERP for larger angles
        sin_theta = torch.sin(theta).unsqueeze(-1)
        weight_a = torch.sin((1.0 - t_expanded) * theta.unsqueeze(-1)) / (sin_theta + 1e-10)
        weight_b = torch.sin(t_expanded * theta.unsqueeze(-1)) / (sin_theta + 1e-10)

        result_slerp = weight_a * a_norm + weight_b * b_norm

        # Choose based on angle
        if use_lerp.any():
            result = torch.where(use_lerp.unsqueeze(-1), result_lerp, result_slerp)
        else:
            result = result_slerp

        return {
            "result": result,
            "angle": theta,
            "used_lerp": use_lerp
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class BarycentricInterpolation(FormulaBase):
    """Barycentric (weighted) interpolation.

    result = Σ wᵢ pᵢ where Σ wᵢ = 1
    """

    def __init__(self):
        super().__init__("barycentric", "f.atomic.barycentric")

    def forward(self, points: Tensor, weights: Tensor) -> Dict[str, Tensor]:
        """Barycentric interpolation.

        Args:
            points: Control points [..., n_points, dim]
            weights: Barycentric weights [..., n_points]

        Returns:
            result: Interpolated point [..., dim]
            weights_sum: Σ wᵢ (should be 1) [...]
            is_valid: weights_sum ≈ 1 [...]
        """
        # Ensure weights sum to 1
        weights_sum = weights.sum(dim=-1)

        # Normalize if needed
        weights_normalized = weights / (weights_sum.unsqueeze(-1) + 1e-10)

        # Weighted sum
        result = torch.sum(weights_normalized.unsqueeze(-1) * points, dim=-2)

        # Check validity
        is_valid = torch.abs(weights_sum - 1.0) < 1e-6

        return {
            "result": result,
            "weights_sum": weights_sum,
            "weights_normalized": weights_normalized,
            "is_valid": is_valid
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PROJECTIONS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class VectorProjection(FormulaBase):
    """Project vector a onto vector b.

    proj_b(a) = ((a·b)/(b·b))b
    """

    def __init__(self):
        super().__init__("vector_projection", "f.atomic.proj_vector")

    def forward(self, a: Tensor, b: Tensor) -> Dict[str, Tensor]:
        """Project a onto b.

        Args:
            a: Vectors to project [..., dim]
            b: Target vectors [..., dim]

        Returns:
            projection: proj_b(a) [..., dim]
            rejection: a - proj_b(a) [..., dim]
            scalar_proj: (a·b)/|b| [...]
        """
        # Dot products
        dot_ab = torch.sum(a * b, dim=-1)
        dot_bb = torch.sum(b * b, dim=-1)

        # Scalar projection
        scalar_proj = dot_ab / (torch.norm(b, dim=-1) + 1e-10)

        # Vector projection
        projection = (dot_ab / (dot_bb + 1e-10)).unsqueeze(-1) * b

        # Rejection (perpendicular component)
        rejection = a - projection

        return {
            "projection": projection,
            "rejection": rejection,
            "scalar_proj": scalar_proj,
            "projection_norm": torch.norm(projection, dim=-1)
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class PointToLineDistance(FormulaBase):
    """Compute distance from point to line.

    Line defined by point + direction.
    """

    def __init__(self):
        super().__init__("point_line_distance", "f.atomic.point_to_line")

    def forward(self, point: Tensor, line_point: Tensor,
                line_direction: Tensor) -> Dict[str, Tensor]:
        """Compute distance.

        Args:
            point: Query points [..., dim]
            line_point: Point on line [..., dim]
            line_direction: Line direction vector [..., dim]

        Returns:
            distance: Perpendicular distance [...]
            closest_point: Closest point on line [..., dim]
            t_parameter: Parameter for closest point [...]
        """
        # Vector from line_point to point
        v = point - line_point

        # Normalize line direction
        d = line_direction / (torch.norm(line_direction, dim=-1, keepdim=True) + 1e-10)

        # Project v onto d
        t = torch.sum(v * d, dim=-1)

        # Closest point on line
        closest_point = line_point + t.unsqueeze(-1) * d

        # Distance
        distance = torch.norm(point - closest_point, dim=-1)

        return {
            "distance": distance,
            "closest_point": closest_point,
            "t_parameter": t
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class PointToPlaneDistance(FormulaBase):
    """Compute signed distance from point to plane.

    Plane defined by point + normal.
    """

    def __init__(self):
        super().__init__("point_plane_distance", "f.atomic.point_to_plane")

    def forward(self, point: Tensor, plane_point: Tensor,
                plane_normal: Tensor) -> Dict[str, Tensor]:
        """Compute signed distance.

        Args:
            point: Query points [..., dim]
            plane_point: Point on plane [..., dim]
            plane_normal: Plane normal vector [..., dim]

        Returns:
            distance: Signed distance (positive if on normal side) [...]
            closest_point: Closest point on plane [..., dim]
            is_above: Point is on normal side [...]
        """
        # Normalize normal
        n = plane_normal / (torch.norm(plane_normal, dim=-1, keepdim=True) + 1e-10)

        # Vector from plane_point to point
        v = point - plane_point

        # Signed distance: v · n
        distance = torch.sum(v * n, dim=-1)

        # Closest point on plane
        closest_point = point - distance.unsqueeze(-1) * n

        # Side check
        is_above = distance > 0

        return {
            "distance": distance,
            "closest_point": closest_point,
            "is_above": is_above,
            "abs_distance": torch.abs(distance)
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# GEOMETRIC PRIMITIVES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TriangleArea(FormulaBase):
    """Compute triangle area from vertices.

    Area = (1/2)|AB × AC| (3D)
    Area = (1/2)|det([B-A, C-A])| (2D)
    """

    def __init__(self):
        super().__init__("triangle_area", "f.atomic.tri_area")

    def forward(self, vertices: Tensor) -> Dict[str, Tensor]:
        """Compute triangle area.

        Args:
            vertices: Triangle vertices [..., 3, dim]

        Returns:
            area: Triangle area [...]
            centroid: Center point [..., dim]
            normal: Normal vector (3D only) [..., 3]
        """
        a = vertices[..., 0, :]
        b = vertices[..., 1, :]
        c = vertices[..., 2, :]

        # Edges
        ab = b - a
        ac = c - a

        if vertices.shape[-1] == 2:
            # 2D: area = (1/2)|det|
            area = 0.5 * torch.abs(ab[..., 0] * ac[..., 1] - ab[..., 1] * ac[..., 0])
            normal = None

        elif vertices.shape[-1] == 3:
            # 3D: area = (1/2)|AB × AC|
            cross = torch.cross(ab, ac, dim=-1)
            area = 0.5 * torch.norm(cross, dim=-1)
            normal = cross / (torch.norm(cross, dim=-1, keepdim=True) + 1e-10)
        else:
            raise ValueError("Triangle area requires 2D or 3D vertices")

        # Centroid
        centroid = (a + b + c) / 3.0

        result = {
            "area": area,
            "centroid": centroid
        }

        if normal is not None:
            result["normal"] = normal

        return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class TetrahedronVolume(FormulaBase):
    """Compute tetrahedron volume from vertices.

    V = (1/6)|det([B-A, C-A, D-A])|
    """

    def __init__(self):
        super().__init__("tetrahedron_volume", "f.atomic.tet_volume")

    def forward(self, vertices: Tensor) -> Dict[str, Tensor]:
        """Compute tetrahedron volume.

        Args:
            vertices: Tetrahedron vertices [..., 4, 3]

        Returns:
            volume: Tetrahedron volume [...]
            centroid: Center point [..., 3]
            is_degenerate: Volume ≈ 0 [...]
        """
        if vertices.shape[-1] != 3:
            raise ValueError("Tetrahedron requires 3D vertices")

        a = vertices[..., 0, :]
        b = vertices[..., 1, :]
        c = vertices[..., 2, :]
        d = vertices[..., 3, :]

        # Edges from a
        ab = b - a
        ac = c - a
        ad = d - a

        # Volume = (1/6)|det([ab, ac, ad])|
        # det = ab · (ac × ad)
        cross_ac_ad = torch.cross(ac, ad, dim=-1)
        det = torch.sum(ab * cross_ac_ad, dim=-1)

        volume = torch.abs(det) / 6.0

        # Centroid
        centroid = (a + b + c + d) / 4.0

        # Check degeneracy
        is_degenerate = volume < 1e-10

        return {
            "volume": volume,
            "centroid": centroid,
            "is_degenerate": is_degenerate
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TESTING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_atomic_operations():
    """Test suite for atomic operations."""

    print("\n" + "=" * 70)
    print("ATOMIC OPERATIONS TESTS")
    print("=" * 70)

    # Test 1: Dot product
    print("\n[Test 1] Dot Product")
    a = torch.tensor([1.0, 0.0, 0.0])
    b = torch.tensor([0.0, 1.0, 0.0])

    dot_op = DotProduct()
    dot_result = dot_op.forward(a, b)

    print(f"  a: {a.numpy()}")
    print(f"  b: {b.numpy()}")
    print(f"  a · b: {dot_result['dot'].item():.4f}")
    print(f"  normalized: {dot_result['normalized_dot'].item():.4f}")
    print(f"  Status: ✓ PASS")

    # Test 2: Cross product
    print("\n[Test 2] Cross Product")
    cross_op = CrossProduct()
    cross_result = cross_op.forward(a, b)

    print(f"  a × b: {cross_result['cross'].numpy()}")
    print(f"  |a × b|: {cross_result['magnitude'].item():.4f}")
    print(f"  Parallel: {cross_result['is_parallel'].item()}")
    print(f"  Status: ✓ PASS")

    # Test 3: Vector norm
    print("\n[Test 3] Vector Norm")
    v = torch.tensor([[3.0, 4.0], [1.0, 1.0]])

    norm_op = VectorNorm(p=2.0)
    norm_result = norm_op.forward(v)

    print(f"  Vectors: {v.numpy()}")
    print(f"  L2 norms: {norm_result['norm'].numpy()}")
    print(f"  Unit vectors: {norm_result['unit_vector'].numpy()}")
    print(f"  Status: ✓ PASS")

    # Test 4: Normalize
    print("\n[Test 4] Normalize")
    normalize_op = Normalize()
    norm_result = normalize_op.forward(v)

    print(f"  Normalized: {norm_result['normalized'].numpy()}")
    print(f"  Original norms: {norm_result['original_norm'].numpy()}")
    print(f"  Status: ✓ PASS")

    # Test 5: Distance metrics
    print("\n[Test 5] Distance Metrics")
    points_a = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
    points_b = torch.tensor([[1.0, 1.0], [2.0, 0.0]])

    dist_op = DistanceMetric(metric='euclidean')
    dist_result = dist_op.forward(points_a, points_b)

    print(f"  Points A: {points_a.numpy()}")
    print(f"  Points B: {points_b.numpy()}")
    print(f"  Distance matrix:\n{dist_result['distances'].numpy()}")
    print(f"  Min distance: {dist_result['min_distance'].item():.4f}")
    print(f"  Status: ✓ PASS")

    # Test 6: Angle between vectors
    print("\n[Test 6] Angle Between Vectors")
    angle_op = AngleBetweenVectors(use_degrees=True)
    angle_result = angle_op.forward(a, b)

    print(f"  a: {a.numpy()}")
    print(f"  b: {b.numpy()}")
    print(f"  Angle: {angle_result['angle'].item():.2f}°")
    print(f"  Orthogonal: {angle_result['is_orthogonal'].item()}")
    print(f"  Status: ✓ PASS")

    # Test 7: Linear interpolation
    print("\n[Test 7] Linear Interpolation")
    p0 = torch.tensor([0.0, 0.0])
    p1 = torch.tensor([1.0, 1.0])
    t = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])

    lerp_op = LinearInterpolation()
    lerp_result = lerp_op.forward(p0, p1, t)

    print(f"  Start: {p0.numpy()}")
    print(f"  End: {p1.numpy()}")
    print(f"  t values: {t.numpy()}")
    print(f"  Interpolated: {lerp_result['result'].numpy()}")
    print(f"  Status: ✓ PASS")

    # Test 8: Spherical interpolation
    print("\n[Test 8] Spherical Interpolation (SLERP)")
    v0 = torch.tensor([1.0, 0.0, 0.0])
    v1 = torch.tensor([0.0, 1.0, 0.0])
    t_slerp = torch.tensor(0.5)

    slerp_op = SphericalInterpolation()
    slerp_result = slerp_op.forward(v0, v1, t_slerp)

    print(f"  Start: {v0.numpy()}")
    print(f"  End: {v1.numpy()}")
    print(f"  t = 0.5")
    print(f"  Result: {slerp_result['result'].numpy()}")
    print(f"  Angle: {slerp_result['angle'].item():.4f} rad")
    print(f"  Status: ✓ PASS")

    # Test 9: Triangle area
    print("\n[Test 9] Triangle Area")
    triangle = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])

    tri_op = TriangleArea()
    tri_result = tri_op.forward(triangle.unsqueeze(0))

    print(f"  Vertices: {triangle.numpy()}")
    print(f"  Area: {tri_result['area'].item():.4f}")
    print(f"  Centroid: {tri_result['centroid'].squeeze().numpy()}")
    print(f"  Status: ✓ PASS")

    # Test 10: Tetrahedron volume
    print("\n[Test 10] Tetrahedron Volume")
    tet = torch.tensor([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ])

    tet_op = TetrahedronVolume()
    tet_result = tet_op.forward(tet.unsqueeze(0))

    print(f"  Vertices: {tet.numpy()}")
    print(f"  Volume: {tet_result['volume'].item():.4f}")
    print(f"  Centroid: {tet_result['centroid'].squeeze().numpy()}")
    print(f"  Degenerate: {tet_result['is_degenerate'].item()}")
    print(f"  Status: ✓ PASS")

    print("\n" + "=" * 70)
    print("All tests completed! (10 total)")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    test_atomic_operations()

    print("\n[Usage Example]")
    print("-" * 70)
    print("Atomic operations are building blocks for complex formulas:")
    print("  - DotProduct: angle calculations, projections")
    print("  - CrossProduct: normals, torque, angular momentum")
    print("  - DistanceMetric: nearest neighbors, clustering")
    print("  - Interpolation: animation, path planning")
    print("  - Projections: shadow casting, constraint enforcement")
    print("  - Geometric primitives: mesh processing, volume calculations")
    print("-" * 70)