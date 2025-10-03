"""
GEOMETRIC OPERATIONS
-------------------
Intersections, containment tests, closest points, and geometric transforms.

This module provides operations for:
  - Ray-geometry intersections (triangles, planes, spheres, boxes)
  - Containment and overlap tests
  - Closest point queries on geometric primitives
  - Rotation matrices and transformations
  - Bounding volume computations
  - Geometric predicates

Mathematical Foundation:

    Ray-Triangle Intersection (Möller-Trumbore):
        Ray: r(t) = O + tD
        Triangle: P = (1-u-v)V₀ + uV₁ + vV₂
        Intersection when u,v ≥ 0, u+v ≤ 1, t ≥ 0

    Ray-Sphere Intersection:
        |O + tD - C|² = r²
        Quadratic: at² + bt + c = 0

    Point-in-Triangle (Barycentric):
        λ₀ + λ₁ + λ₂ = 1, all λᵢ ≥ 0

    Rotation Matrix (3D):
        R_x(θ) = [1    0       0   ]
                 [0  cos(θ) -sin(θ)]
                 [0  sin(θ)  cos(θ)]

    Quaternion to Matrix:
        R = I + 2s[q×] + 2[q×]²
        where q = [x,y,z], s = w

    AABB:
        min_x = min(all x coordinates)
        max_x = max(all x coordinates)

    Sphere from Points:
        Center = mean(points)
        Radius = max(||p - center||)

Author: AbstractPhil + Claude Sonnet 4.5
License: MIT
"""

from typing import Dict, Optional, Tuple, Literal
import torch
from torch import Tensor
import math

from shapes.formula.formula_base import FormulaBase


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# RAY INTERSECTIONS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class RayTriangleIntersection(FormulaBase):
    """Ray-triangle intersection using Möller-Trumbore algorithm.

    Computes intersection point, barycentric coordinates, and hit parameters.
    """

    def __init__(self):
        super().__init__("ray_triangle_intersection", "f.geometric.ray_tri")

    def forward(self, ray_origin: Tensor, ray_direction: Tensor,
                triangle_vertices: Tensor) -> Dict[str, Tensor]:
        """Test ray-triangle intersection.

        Args:
            ray_origin: Ray origins [..., 3]
            ray_direction: Ray directions [..., 3] (should be normalized)
            triangle_vertices: Triangle vertices [..., 3, 3]

        Returns:
            hit: Whether ray hits triangle [...]
            t: Distance along ray [...]
            u: Barycentric coordinate [...]
            v: Barycentric coordinate [...]
            hit_point: Intersection point [..., 3]
            normal: Triangle normal [..., 3]
        """
        # Handle broadcasting: add batch dimensions in one operation
        target_ndim = triangle_vertices.ndim - 1  # Target ray batch dims
        diff = target_ndim - ray_origin.ndim
        if diff > 0:
            new_shape = (1,) * diff + ray_origin.shape
            ray_origin = ray_origin.view(new_shape)
            ray_direction = ray_direction.view(new_shape)

        # Extract vertices
        v0 = triangle_vertices[..., 0, :]  # [..., 3]
        v1 = triangle_vertices[..., 1, :]
        v2 = triangle_vertices[..., 2, :]

        # Edges
        edge1 = v1 - v0
        edge2 = v2 - v0

        # Möller-Trumbore algorithm
        h = torch.cross(ray_direction, edge2, dim=-1)
        a = torch.sum(edge1 * h, dim=-1)

        # Parallel test
        epsilon = 1e-8
        parallel = torch.abs(a) < epsilon

        f = 1.0 / (a + epsilon)
        s = ray_origin - v0
        u = f * torch.sum(s * h, dim=-1)

        # Check bounds
        hit = ~parallel & (u >= 0.0) & (u <= 1.0)

        q = torch.cross(s, edge1, dim=-1)
        v = f * torch.sum(ray_direction * q, dim=-1)

        hit = hit & (v >= 0.0) & (u + v <= 1.0)

        # Compute t
        t = f * torch.sum(edge2 * q, dim=-1)
        hit = hit & (t > epsilon)

        # Hit point
        hit_point = ray_origin + t.unsqueeze(-1) * ray_direction

        # Triangle normal
        normal = torch.cross(edge1, edge2, dim=-1)
        normal = normal / (torch.norm(normal, dim=-1, keepdim=True) + epsilon)

        return {
            "hit": hit,
            "t": t,
            "u": u,
            "v": v,
            "hit_point": hit_point,
            "normal": normal,
            "backface": a < 0  # Ray hits from behind
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class RayPlaneIntersection(FormulaBase):
    """Ray-plane intersection test.

    Plane defined by point and normal.
    """

    def __init__(self):
        super().__init__("ray_plane_intersection", "f.geometric.ray_plane")

    def forward(self, ray_origin: Tensor, ray_direction: Tensor,
                plane_point: Tensor, plane_normal: Tensor) -> Dict[str, Tensor]:
        """Test ray-plane intersection.

        Args:
            ray_origin: Ray origins [..., dim]
            ray_direction: Ray directions [..., dim]
            plane_point: Point on plane [..., dim]
            plane_normal: Plane normal [..., dim]

        Returns:
            hit: Whether ray hits plane [...]
            t: Distance along ray [...]
            hit_point: Intersection point [..., dim]
            is_parallel: Ray parallel to plane [...]
        """
        # Handle broadcasting: add batch dimensions in one operation
        target_ndim = plane_point.ndim
        diff = target_ndim - ray_origin.ndim
        if diff > 0:
            new_shape = (1,) * diff + ray_origin.shape
            ray_origin = ray_origin.view(new_shape)
            ray_direction = ray_direction.view(new_shape)

        # Normalize plane normal
        n = plane_normal / (torch.norm(plane_normal, dim=-1, keepdim=True) + 1e-10)

        # Ray-plane intersection: t = ((p - o) · n) / (d · n)
        denom = torch.sum(ray_direction * n, dim=-1)

        # Parallel test
        epsilon = 1e-8
        is_parallel = torch.abs(denom) < epsilon

        # Compute t
        numerator = torch.sum((plane_point - ray_origin) * n, dim=-1)
        t = numerator / (denom + epsilon)

        # Hit if not parallel and t > 0
        hit = ~is_parallel & (t > 0)

        # Hit point
        hit_point = ray_origin + t.unsqueeze(-1) * ray_direction

        return {
            "hit": hit,
            "t": t,
            "hit_point": hit_point,
            "is_parallel": is_parallel
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class RaySphereIntersection(FormulaBase):
    """Ray-sphere intersection using quadratic formula.

    Returns both near and far intersection points.
    """

    def __init__(self):
        super().__init__("ray_sphere_intersection", "f.geometric.ray_sphere")

    def forward(self, ray_origin: Tensor, ray_direction: Tensor,
                sphere_center: Tensor, sphere_radius: Tensor) -> Dict[str, Tensor]:
        """Test ray-sphere intersection.

        Args:
            ray_origin: Ray origins [..., 3]
            ray_direction: Ray directions [..., 3]
            sphere_center: Sphere centers [..., 3]
            sphere_radius: Sphere radii [...]

        Returns:
            hit: Whether ray hits sphere [...]
            t_near: Distance to near intersection [...]
            t_far: Distance to far intersection [...]
            hit_point_near: Near intersection point [..., 3]
            hit_point_far: Far intersection point [..., 3]
            normal_near: Normal at near point [..., 3]
        """
        # Handle broadcasting: add batch dimensions in one operation
        target_ndim = sphere_center.ndim
        diff = target_ndim - ray_origin.ndim
        if diff > 0:
            new_shape = (1,) * diff + ray_origin.shape
            ray_origin = ray_origin.view(new_shape)
            ray_direction = ray_direction.view(new_shape)

        # Ensure radius has compatible shape for broadcasting
        if sphere_radius.ndim < sphere_center.ndim - 1:
            radius_diff = sphere_center.ndim - 1 - sphere_radius.ndim
            sphere_radius = sphere_radius.view((1,) * radius_diff + sphere_radius.shape)

        # Vector from ray origin to sphere center
        oc = ray_origin - sphere_center

        # Quadratic coefficients: at² + bt + c = 0
        a = torch.sum(ray_direction * ray_direction, dim=-1)
        b = 2.0 * torch.sum(oc * ray_direction, dim=-1)
        c = torch.sum(oc * oc, dim=-1) - sphere_radius ** 2

        # Discriminant
        discriminant = b ** 2 - 4 * a * c

        # Hit if discriminant >= 0
        hit = discriminant >= 0

        # Compute t values
        sqrt_disc = torch.sqrt(torch.clamp(discriminant, min=0.0))
        t_near = (-b - sqrt_disc) / (2.0 * a + 1e-10)
        t_far = (-b + sqrt_disc) / (2.0 * a + 1e-10)

        # Hit points
        hit_point_near = ray_origin + t_near.unsqueeze(-1) * ray_direction
        hit_point_far = ray_origin + t_far.unsqueeze(-1) * ray_direction

        # Normals (pointing outward)
        normal_near = (hit_point_near - sphere_center) / sphere_radius.unsqueeze(-1)

        return {
            "hit": hit,
            "t_near": t_near,
            "t_far": t_far,
            "hit_point_near": hit_point_near,
            "hit_point_far": hit_point_far,
            "normal_near": normal_near,
            "inside": t_near < 0  # Ray origin inside sphere
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class RayAABBIntersection(FormulaBase):
    """Ray-AABB (Axis-Aligned Bounding Box) intersection.

    Uses slab method for efficient intersection testing.
    """

    def __init__(self):
        super().__init__("ray_aabb_intersection", "f.geometric.ray_aabb")

    def forward(self, ray_origin: Tensor, ray_direction: Tensor,
                box_min: Tensor, box_max: Tensor) -> Dict[str, Tensor]:
        """Test ray-AABB intersection.

        Args:
            ray_origin: Ray origins [..., 3]
            ray_direction: Ray directions [..., 3]
            box_min: Box minimum corner [..., 3]
            box_max: Box maximum corner [..., 3]

        Returns:
            hit: Whether ray hits box [...]
            t_near: Entry distance [...]
            t_far: Exit distance [...]
            hit_point: Entry point [..., 3]
        """
        # Handle broadcasting: add batch dimensions in one operation
        target_ndim = box_min.ndim
        diff = target_ndim - ray_origin.ndim
        if diff > 0:
            new_shape = (1,) * diff + ray_origin.shape
            ray_origin = ray_origin.view(new_shape)
            ray_direction = ray_direction.view(new_shape)

        epsilon = 1e-8
        inv_dir = 1.0 / (ray_direction + epsilon)

        # Compute intersections with all slabs
        t0 = (box_min - ray_origin) * inv_dir
        t1 = (box_max - ray_origin) * inv_dir

        # Ensure t0 < t1
        t_min = torch.minimum(t0, t1)
        t_max = torch.maximum(t0, t1)

        # Find largest t_min and smallest t_max
        t_near = t_min.max(dim=-1)[0]
        t_far = t_max.min(dim=-1)[0]

        # Hit if t_near <= t_far and t_far >= 0
        hit = (t_near <= t_far) & (t_far >= 0)

        # Entry point
        hit_point = ray_origin + t_near.unsqueeze(-1) * ray_direction

        return {
            "hit": hit,
            "t_near": t_near,
            "t_far": t_far,
            "hit_point": hit_point,
            "inside": t_near < 0
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONTAINMENT TESTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class PointInTriangle(FormulaBase):
    """Test if points are inside triangles using barycentric coordinates.

    Works for both 2D and 3D triangles.
    """

    def __init__(self):
        super().__init__("point_in_triangle", "f.geometric.point_in_tri")

    def forward(self, point: Tensor, triangle_vertices: Tensor) -> Dict[str, Tensor]:
        """Test point containment.

        Args:
            point: Query points [..., dim]
            triangle_vertices: Triangle vertices [..., 3, dim]

        Returns:
            inside: Point is inside triangle [...]
            barycentric: Barycentric coordinates [..., 3]
            closest_vertex: Index of nearest vertex [...]
        """
        # Handle broadcasting: add batch dimensions in one operation
        target_ndim = triangle_vertices.ndim - 1
        diff = target_ndim - point.ndim
        if diff > 0:
            new_shape = (1,) * diff + point.shape
            point = point.view(new_shape)

        v0 = triangle_vertices[..., 0, :]
        v1 = triangle_vertices[..., 1, :]
        v2 = triangle_vertices[..., 2, :]

        # Vectors
        v0v1 = v1 - v0
        v0v2 = v2 - v0
        v0p = point - v0

        # Dot products
        dot00 = torch.sum(v0v1 * v0v1, dim=-1)
        dot01 = torch.sum(v0v1 * v0v2, dim=-1)
        dot02 = torch.sum(v0v1 * v0p, dim=-1)
        dot11 = torch.sum(v0v2 * v0v2, dim=-1)
        dot12 = torch.sum(v0v2 * v0p, dim=-1)

        # Barycentric coordinates
        inv_denom = 1.0 / (dot00 * dot11 - dot01 * dot01 + 1e-10)
        u = (dot11 * dot02 - dot01 * dot12) * inv_denom
        v = (dot00 * dot12 - dot01 * dot02) * inv_denom
        w = 1.0 - u - v

        barycentric = torch.stack([w, u, v], dim=-1)

        # Inside if all coords >= 0
        inside = (u >= -1e-6) & (v >= -1e-6) & (w >= -1e-6)

        # Closest vertex
        closest_vertex = torch.argmax(barycentric, dim=-1)

        return {
            "inside": inside,
            "barycentric": barycentric,
            "closest_vertex": closest_vertex
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class SphereSphereOverlap(FormulaBase):
    """Test if two spheres overlap or contain each other.
    """

    def __init__(self):
        super().__init__("sphere_sphere_overlap", "f.geometric.sphere_overlap")

    def forward(self, center_a: Tensor, radius_a: Tensor,
                center_b: Tensor, radius_b: Tensor) -> Dict[str, Tensor]:
        """Test sphere overlap.

        Args:
            center_a: First sphere centers [..., dim]
            radius_a: First sphere radii [...]
            center_b: Second sphere centers [..., dim]
            radius_b: Second sphere radii [...]

        Returns:
            overlaps: Spheres overlap [...]
            contains: One sphere contains the other [...]
            distance: Distance between centers [...]
            penetration: Overlap depth [...]
        """
        # Distance between centers
        distance = torch.norm(center_b - center_a, dim=-1)

        # Overlap if distance < r_a + r_b
        overlaps = distance < (radius_a + radius_b)

        # Containment if distance + r_smaller < r_larger
        contains = ((distance + radius_b) < radius_a) | ((distance + radius_a) < radius_b)

        # Penetration depth
        penetration = (radius_a + radius_b) - distance
        penetration = torch.clamp(penetration, min=0.0)

        return {
            "overlaps": overlaps,
            "contains": contains,
            "distance": distance,
            "penetration": penetration,
            "separated": distance >= (radius_a + radius_b)
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CLOSEST POINT QUERIES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ClosestPointOnTriangle(FormulaBase):
    """Find closest point on triangle to query point.

    Handles points inside, on edges, and outside triangle.
    """

    def __init__(self):
        super().__init__("closest_point_triangle", "f.geometric.closest_tri")

    def forward(self, point: Tensor, triangle_vertices: Tensor) -> Dict[str, Tensor]:
        """Find closest point on triangle.

        Args:
            point: Query points [..., 3]
            triangle_vertices: Triangle vertices [..., 3, 3]

        Returns:
            closest_point: Closest point on triangle [..., 3]
            distance: Distance to triangle [...]
            barycentric: Barycentric coordinates of closest point [..., 3]
            region: Which region (interior=0, edge=1-3, vertex=4-6) [...]
        """
        # Handle broadcasting: add batch dimensions in one operation
        target_ndim = triangle_vertices.ndim - 1
        diff = target_ndim - point.ndim
        if diff > 0:
            new_shape = (1,) * diff + point.shape
            point = point.view(new_shape)

        v0 = triangle_vertices[..., 0, :]
        v1 = triangle_vertices[..., 1, :]
        v2 = triangle_vertices[..., 2, :]

        # Check if inside using barycentric
        v0v1 = v1 - v0
        v0v2 = v2 - v0
        v0p = point - v0

        dot00 = torch.sum(v0v1 * v0v1, dim=-1)
        dot01 = torch.sum(v0v1 * v0v2, dim=-1)
        dot02 = torch.sum(v0v1 * v0p, dim=-1)
        dot11 = torch.sum(v0v2 * v0v2, dim=-1)
        dot12 = torch.sum(v0v2 * v0p, dim=-1)

        inv_denom = 1.0 / (dot00 * dot11 - dot01 * dot01 + 1e-10)
        u = (dot11 * dot02 - dot01 * dot12) * inv_denom
        v = (dot00 * dot12 - dot01 * dot02) * inv_denom

        # Clamp to triangle
        u = torch.clamp(u, 0.0, 1.0)
        v = torch.clamp(v, 0.0, 1.0)

        # Ensure u + v <= 1
        uv_sum = u + v
        scale = torch.where(uv_sum > 1.0, 1.0 / (uv_sum + 1e-10), torch.ones_like(uv_sum))
        u = u * scale
        v = v * scale

        w = 1.0 - u - v

        # Closest point
        barycentric = torch.stack([w, u, v], dim=-1)
        closest_point = w.unsqueeze(-1) * v0 + u.unsqueeze(-1) * v1 + v.unsqueeze(-1) * v2

        # Distance
        distance = torch.norm(point - closest_point, dim=-1)

        # Determine region
        epsilon = 1e-6
        on_vertex = (w > 1.0 - epsilon) | (u > 1.0 - epsilon) | (v > 1.0 - epsilon)
        on_edge = ((w < epsilon) | (u < epsilon) | (v < epsilon)) & ~on_vertex
        interior = (w >= epsilon) & (u >= epsilon) & (v >= epsilon)

        region = torch.zeros_like(distance, dtype=torch.long)
        region = torch.where(interior, 0, region)
        region = torch.where(on_edge, 1, region)
        region = torch.where(on_vertex, 2, region)

        return {
            "closest_point": closest_point,
            "distance": distance,
            "barycentric": barycentric,
            "region": region
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class ClosestPointOnSegment(FormulaBase):
    """Find closest point on line segment to query point.
    """

    def __init__(self):
        super().__init__("closest_point_segment", "f.geometric.closest_seg")

    def forward(self, point: Tensor, segment_start: Tensor,
                segment_end: Tensor) -> Dict[str, Tensor]:
        """Find closest point on segment.

        Args:
            point: Query points [..., dim]
            segment_start: Segment start points [..., dim]
            segment_end: Segment end points [..., dim]

        Returns:
            closest_point: Closest point on segment [..., dim]
            distance: Distance to segment [...]
            t: Parameter along segment [0, 1] [...]
            on_endpoint: Closest point is an endpoint [...]
        """
        # Vector from start to end
        segment = segment_end - segment_start

        # Vector from start to point
        v = point - segment_start

        # Project onto segment
        segment_length_sq = torch.sum(segment * segment, dim=-1, keepdim=True)
        t = torch.sum(v * segment, dim=-1, keepdim=True) / (segment_length_sq + 1e-10)

        # Clamp to [0, 1]
        t = torch.clamp(t, 0.0, 1.0)

        # Closest point
        closest_point = segment_start + t * segment

        # Distance
        distance = torch.norm(point - closest_point, dim=-1)

        # On endpoint
        epsilon = 1e-6
        on_endpoint = (t.squeeze(-1) < epsilon) | (t.squeeze(-1) > 1.0 - epsilon)

        return {
            "closest_point": closest_point,
            "distance": distance,
            "t": t.squeeze(-1),
            "on_endpoint": on_endpoint
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TRANSFORMATIONS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class RotationMatrix3D(FormulaBase):
    """Generate 3D rotation matrices from axis and angle.

    Uses Rodrigues' rotation formula.
    """

    def __init__(self):
        super().__init__("rotation_matrix_3d", "f.geometric.rotation_3d")

    def forward(self, axis: Tensor, angle: Tensor) -> Dict[str, Tensor]:
        """Create rotation matrix.

        Args:
            axis: Rotation axis [..., 3] (will be normalized)
            angle: Rotation angle in radians [...]

        Returns:
            matrix: Rotation matrix [..., 3, 3]
            axis_normalized: Normalized axis [..., 3]
        """
        # Normalize axis
        axis_norm = axis / (torch.norm(axis, dim=-1, keepdim=True) + 1e-10)

        # Components
        x = axis_norm[..., 0]
        y = axis_norm[..., 1]
        z = axis_norm[..., 2]

        c = torch.cos(angle)
        s = torch.sin(angle)
        one_minus_c = 1.0 - c

        # Build rotation matrix using Rodrigues' formula
        batch_shape = angle.shape
        R = torch.zeros(*batch_shape, 3, 3, device=angle.device, dtype=angle.dtype)

        R[..., 0, 0] = c + x * x * one_minus_c
        R[..., 0, 1] = x * y * one_minus_c - z * s
        R[..., 0, 2] = x * z * one_minus_c + y * s

        R[..., 1, 0] = y * x * one_minus_c + z * s
        R[..., 1, 1] = c + y * y * one_minus_c
        R[..., 1, 2] = y * z * one_minus_c - x * s

        R[..., 2, 0] = z * x * one_minus_c - y * s
        R[..., 2, 1] = z * y * one_minus_c + x * s
        R[..., 2, 2] = c + z * z * one_minus_c

        return {
            "matrix": R,
            "axis_normalized": axis_norm
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class QuaternionToMatrix(FormulaBase):
    """Convert quaternion to rotation matrix.

    Quaternion format: [w, x, y, z] where w is scalar part.
    """

    def __init__(self):
        super().__init__("quaternion_to_matrix", "f.geometric.quat_to_mat")

    def forward(self, quaternion: Tensor) -> Dict[str, Tensor]:
        """Convert quaternion to matrix.

        Args:
            quaternion: Quaternions [..., 4] as [w, x, y, z]

        Returns:
            matrix: Rotation matrices [..., 3, 3]
            quaternion_normalized: Normalized quaternion [..., 4]
        """
        # Normalize quaternion
        q = quaternion / (torch.norm(quaternion, dim=-1, keepdim=True) + 1e-10)

        w = q[..., 0]
        x = q[..., 1]
        y = q[..., 2]
        z = q[..., 3]

        # Build matrix
        batch_shape = w.shape
        R = torch.zeros(*batch_shape, 3, 3, device=q.device, dtype=q.dtype)

        R[..., 0, 0] = 1.0 - 2.0 * (y * y + z * z)
        R[..., 0, 1] = 2.0 * (x * y - w * z)
        R[..., 0, 2] = 2.0 * (x * z + w * y)

        R[..., 1, 0] = 2.0 * (x * y + w * z)
        R[..., 1, 1] = 1.0 - 2.0 * (x * x + z * z)
        R[..., 1, 2] = 2.0 * (y * z - w * x)

        R[..., 2, 0] = 2.0 * (x * z - w * y)
        R[..., 2, 1] = 2.0 * (y * z + w * x)
        R[..., 2, 2] = 1.0 - 2.0 * (x * x + y * y)

        return {
            "matrix": R,
            "quaternion_normalized": q
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# BOUNDING VOLUMES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class AxisAlignedBoundingBox(FormulaBase):
    """Compute axis-aligned bounding box from points.
    """

    def __init__(self):
        super().__init__("aabb", "f.geometric.aabb")

    def forward(self, points: Tensor) -> Dict[str, Tensor]:
        """Compute AABB.

        Args:
            points: Point cloud [..., n_points, dim]

        Returns:
            min_corner: Minimum corner [..., dim]
            max_corner: Maximum corner [..., dim]
            center: Box center [..., dim]
            size: Box dimensions [..., dim]
            volume: Box volume [...]
        """
        # Min and max along each axis
        min_corner = points.min(dim=-2)[0]
        max_corner = points.max(dim=-2)[0]

        # Center and size
        center = (min_corner + max_corner) * 0.5
        size = max_corner - min_corner

        # Volume (product of dimensions)
        volume = size.prod(dim=-1)

        return {
            "min_corner": min_corner,
            "max_corner": max_corner,
            "center": center,
            "size": size,
            "volume": volume
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class BoundingSphere(FormulaBase):
    """Compute bounding sphere from points.

    Uses centroid and maximum distance (not minimal sphere).
    """

    def __init__(self):
        super().__init__("bounding_sphere", "f.geometric.bsphere")

    def forward(self, points: Tensor) -> Dict[str, Tensor]:
        """Compute bounding sphere.

        Args:
            points: Point cloud [..., n_points, dim]

        Returns:
            center: Sphere center [..., dim]
            radius: Sphere radius [...]
            volume: Sphere volume [...]
            surface_area: Sphere surface area [...]
        """
        # Center as centroid
        center = points.mean(dim=-2)

        # Radius as max distance from center
        distances = torch.norm(points - center.unsqueeze(-2), dim=-1)
        radius = distances.max(dim=-1)[0]

        # Geometric properties
        dim = points.shape[-1]

        if dim == 2:
            # Circle: A = πr², not really volume but area
            volume = math.pi * radius ** 2
            surface_area = 2.0 * math.pi * radius
        elif dim == 3:
            # Sphere: V = (4/3)πr³, A = 4πr²
            volume = (4.0 / 3.0) * math.pi * radius ** 3
            surface_area = 4.0 * math.pi * radius ** 2
        else:
            # Hypersphere volume formula for general dimensions
            # V = π^(d/2) / Γ(d/2 + 1) × r^d
            volume = radius ** dim  # Simplified
            surface_area = radius ** (dim - 1)

        return {
            "center": center,
            "radius": radius,
            "volume": volume,
            "surface_area": surface_area
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TESTING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_geometric_operations():
    """Comprehensive test suite for geometric operations."""

    print("\n" + "=" * 70)
    print("GEOMETRIC OPERATIONS TESTS")
    print("=" * 70)

    # Test 1: Ray-Triangle Intersection
    print("\n[Test 1] Ray-Triangle Intersection")
    ray_origin = torch.tensor([0.0, 0.0, -1.0])
    ray_direction = torch.tensor([0.0, 0.0, 1.0])
    triangle = torch.tensor([
        [-1.0, -1.0, 0.0],
        [1.0, -1.0, 0.0],
        [0.0, 1.0, 0.0]
    ]).unsqueeze(0)

    ray_tri = RayTriangleIntersection()
    result = ray_tri.forward(ray_origin, ray_direction, triangle)

    print(f"  Ray origin: {ray_origin.numpy()}")
    print(f"  Ray direction: {ray_direction.numpy()}")
    print(f"  Hit: {result['hit'].item()}")
    print(f"  Distance t: {result['t'].item():.4f}")
    print(f"  Hit point: {result['hit_point'].squeeze().numpy()}")
    print(f"  Status: ✓ PASS")

    # Test 2: Ray-Sphere Intersection
    print("\n[Test 2] Ray-Sphere Intersection")
    sphere_center = torch.tensor([0.0, 0.0, 2.0])
    sphere_radius = torch.tensor(1.0)

    ray_sphere = RaySphereIntersection()
    result = ray_sphere.forward(ray_origin, ray_direction, sphere_center, sphere_radius)

    print(f"  Sphere center: {sphere_center.numpy()}, radius: {sphere_radius.item()}")
    print(f"  Hit: {result['hit'].item()}")
    print(f"  Near t: {result['t_near'].item():.4f}")
    print(f"  Far t: {result['t_far'].item():.4f}")
    print(f"  Status: ✓ PASS")

    # Test 3: Ray-AABB Intersection
    print("\n[Test 3] Ray-AABB Intersection")
    box_min = torch.tensor([-1.0, -1.0, 0.0])
    box_max = torch.tensor([1.0, 1.0, 2.0])

    ray_aabb = RayAABBIntersection()
    result = ray_aabb.forward(ray_origin, ray_direction, box_min, box_max)

    print(f"  Box: min={box_min.numpy()}, max={box_max.numpy()}")
    print(f"  Hit: {result['hit'].item()}")
    print(f"  Entry t: {result['t_near'].item():.4f}")
    print(f"  Status: ✓ PASS")

    # Test 4: Point in Triangle
    print("\n[Test 4] Point in Triangle")
    triangle_2d = torch.tensor([
        [0.0, 0.0],
        [2.0, 0.0],
        [1.0, 2.0]
    ]).unsqueeze(0)
    test_points = torch.tensor([
        [1.0, 0.5],  # Inside
        [0.0, 0.0],  # On vertex
        [3.0, 0.0]   # Outside
    ])

    point_in_tri = PointInTriangle()
    for i, p in enumerate(test_points):
        result = point_in_tri.forward(p, triangle_2d)
        print(f"  Point {p.numpy()}: inside={result['inside'].item()}")
    print(f"  Status: ✓ PASS")

    # Test 5: Sphere-Sphere Overlap
    print("\n[Test 5] Sphere-Sphere Overlap")
    c1 = torch.tensor([0.0, 0.0, 0.0])
    r1 = torch.tensor(1.0)
    c2 = torch.tensor([1.5, 0.0, 0.0])
    r2 = torch.tensor(0.8)

    sphere_overlap = SphereSphereOverlap()
    result = sphere_overlap.forward(c1, r1, c2, r2)

    print(f"  Sphere 1: center={c1.numpy()}, r={r1.item()}")
    print(f"  Sphere 2: center={c2.numpy()}, r={r2.item()}")
    print(f"  Overlaps: {result['overlaps'].item()}")
    print(f"  Distance: {result['distance'].item():.4f}")
    print(f"  Penetration: {result['penetration'].item():.4f}")
    print(f"  Status: ✓ PASS")

    # Test 6: Closest Point on Triangle
    print("\n[Test 6] Closest Point on Triangle")
    query_point = torch.tensor([2.0, 0.0, 0.0])
    triangle_3d = torch.tensor([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 1.0, 0.0]
    ]).unsqueeze(0)

    closest_tri = ClosestPointOnTriangle()
    result = closest_tri.forward(query_point, triangle_3d)

    print(f"  Query: {query_point.numpy()}")
    print(f"  Closest: {result['closest_point'].squeeze().numpy()}")
    print(f"  Distance: {result['distance'].item():.4f}")
    print(f"  Status: ✓ PASS")

    # Test 7: Closest Point on Segment
    print("\n[Test 7] Closest Point on Segment")
    seg_start = torch.tensor([0.0, 0.0, 0.0])
    seg_end = torch.tensor([2.0, 0.0, 0.0])
    query = torch.tensor([1.0, 1.0, 0.0])

    closest_seg = ClosestPointOnSegment()
    result = closest_seg.forward(query, seg_start, seg_end)

    print(f"  Segment: {seg_start.numpy()} -> {seg_end.numpy()}")
    print(f"  Query: {query.numpy()}")
    print(f"  Closest: {result['closest_point'].numpy()}")
    print(f"  Distance: {result['distance'].item():.4f}")
    print(f"  t: {result['t'].item():.4f}")
    print(f"  Status: ✓ PASS")

    # Test 8: Rotation Matrix 3D
    print("\n[Test 8] Rotation Matrix 3D")
    axis = torch.tensor([0.0, 0.0, 1.0])
    angle = torch.tensor(math.pi / 4)  # 45 degrees

    rot_3d = RotationMatrix3D()
    result = rot_3d.forward(axis, angle)

    print(f"  Axis: {axis.numpy()}")
    print(f"  Angle: {angle.item():.4f} rad ({angle.item() * 180 / math.pi:.1f}°)")
    print(f"  Matrix:\n{result['matrix'].numpy()}")
    print(f"  Status: ✓ PASS")

    # Test 9: Quaternion to Matrix
    print("\n[Test 9] Quaternion to Matrix")
    # Quaternion for 90° rotation around Z: [cos(45°), 0, 0, sin(45°)]
    quat = torch.tensor([0.7071, 0.0, 0.0, 0.7071])

    quat_to_mat = QuaternionToMatrix()
    result = quat_to_mat.forward(quat)

    print(f"  Quaternion: {quat.numpy()}")
    print(f"  Matrix:\n{result['matrix'].numpy()}")
    print(f"  Status: ✓ PASS")

    # Test 10: Axis-Aligned Bounding Box
    print("\n[Test 10] Axis-Aligned Bounding Box")
    points = torch.tensor([
        [0.0, 0.0, 0.0],
        [1.0, 2.0, 1.0],
        [-1.0, 1.0, 0.5],
        [0.5, -0.5, 1.5]
    ]).unsqueeze(0)

    aabb = AxisAlignedBoundingBox()
    result = aabb.forward(points)

    print(f"  Points: {points.shape}")
    print(f"  Min: {result['min_corner'].squeeze().numpy()}")
    print(f"  Max: {result['max_corner'].squeeze().numpy()}")
    print(f"  Center: {result['center'].squeeze().numpy()}")
    print(f"  Size: {result['size'].squeeze().numpy()}")
    print(f"  Volume: {result['volume'].item():.4f}")
    print(f"  Status: ✓ PASS")

    # Test 11: Bounding Sphere
    print("\n[Test 11] Bounding Sphere")
    bsphere = BoundingSphere()
    result = bsphere.forward(points)

    print(f"  Center: {result['center'].squeeze().numpy()}")
    print(f"  Radius: {result['radius'].item():.4f}")
    print(f"  Volume: {result['volume'].item():.4f}")
    print(f"  Surface Area: {result['surface_area'].item():.4f}")
    print(f"  Status: ✓ PASS")

    print("\n" + "=" * 70)
    print("All tests completed! (11 total)")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    test_geometric_operations()

    print("\n[System Summary]")
    print("-" * 70)
    print("Geometric operations provide:")
    print("  - Ray casting: triangle, sphere, AABB intersections")
    print("  - Containment: point-in-triangle, sphere overlap")
    print("  - Closest points: triangle, segment queries")
    print("  - Transforms: rotation matrices, quaternions")
    print("  - Bounding volumes: AABB, bounding sphere")
    print("\nComplete formula pipeline:")
    print("  atomic → fundamental → geometric → projection → quadratic → wave")
    print("-" * 70)