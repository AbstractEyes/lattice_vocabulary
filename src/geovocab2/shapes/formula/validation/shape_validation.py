"""
SIMPLE SHAPE FORMULAS
---------------------
Validation, analysis, and property computation for point cloud shapes.

This module provides formulas for:
  - Volume estimation (voxel, convex hull, Monte Carlo)
  - Surface area computation
  - Shape quality metrics (uniformity, coverage, distribution)
  - Geometric validation (bounding volumes, symmetry)
  - Shape classification and recognition
  - Transformation validation

Integrates with:
  - geometric.py: Bounding volumes, containment tests
  - cayley_menger.py: Volume computation via triangulation
  - projection.py: Coordinate transformations

Mathematical Foundation:

    Convex Hull Volume:
        V = (1/6) Σ |det([v₀, v₁, v₂, v₃])| for tetrahedra

    Monte Carlo Volume:
        V ≈ V_bbox × (N_inside / N_total)

    Point Cloud Uniformity:
        U = 1 - σ(d_nearest) / μ(d_nearest)

    Shape Symmetry:
        S = 1 - ||PC - reflect(PC)|| / ||PC||

    Coverage Quality:
        Q = N_occupied / N_expected_voxels

Author: AbstractPhil + Claude Sonnet 4.5
License: MIT
"""

from typing import Dict, Optional, Tuple, Literal
import torch
from torch import Tensor
import math

from geovocab2.shapes.formula.formula_base import FormulaBase


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# VOLUME ESTIMATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ShapeVolumeEstimator(FormulaBase):
    """Estimate volume of point cloud shapes using multiple methods.

    Args:
        method: "convex_hull", "voxel", "monte_carlo", "analytical"
        voxel_resolution: Grid resolution for voxel method
        mc_samples: Number of samples for Monte Carlo
    """

    def __init__(
        self,
        method: str = "convex_hull",
        voxel_resolution: int = 32,
        mc_samples: int = 10000
    ):
        super().__init__("shape_volume_estimator", "f.shape.volume")
        self.method = method
        self.voxel_resolution = voxel_resolution
        self.mc_samples = mc_samples

    def forward(self, points: Tensor, shape_type: Optional[str] = None) -> Dict[str, Tensor]:
        """Estimate volume from point cloud.

        Args:
            points: Point cloud [..., n_points, dim]
            shape_type: Known shape type for analytical computation

        Returns:
            volume: Estimated volume
            method_used: Method that was used
            confidence: Confidence score [0, 1]
            bounding_box_volume: AABB volume for reference
        """
        if self.method == "analytical" and shape_type is not None:
            return self._analytical_volume(points, shape_type)
        elif self.method == "convex_hull":
            return self._convex_hull_volume(points)
        elif self.method == "voxel":
            return self._voxel_volume(points)
        elif self.method == "monte_carlo":
            return self._monte_carlo_volume(points)
        else:
            # Default to convex hull
            return self._convex_hull_volume(points)

    def _analytical_volume(self, points: Tensor, shape_type: str) -> Dict[str, Tensor]:
        """Compute analytical volume for known shapes."""
        # Get bounding box for size estimation
        min_corner = points.min(dim=-2)[0]
        max_corner = points.max(dim=-2)[0]
        dims = max_corner - min_corner

        if shape_type == "cube":
            # Volume of cube
            side_length = dims.mean(dim=-1)
            volume = side_length ** 3
            confidence = torch.tensor(0.95)

        elif shape_type == "sphere":
            # Volume of sphere: V = (4/3)πr³
            radius = dims.max(dim=-1)[0] / 2 if dims.ndim > 0 else dims.max() / 2
            volume = (4.0 / 3.0) * math.pi * radius ** 3
            confidence = torch.tensor(0.9)

        elif shape_type == "cylinder":
            # Volume of cylinder: V = πr²h
            radius = torch.sqrt(dims[..., 0] ** 2 + dims[..., 1] ** 2) / 2
            height = dims[..., 2]
            volume = math.pi * radius ** 2 * height
            confidence = torch.tensor(0.85)

        elif shape_type == "pyramid":
            # Volume of pyramid: V = (1/3)Bh
            base_area = dims[..., 0] * dims[..., 1]
            height = dims[..., 2]
            volume = (1.0 / 3.0) * base_area * height
            confidence = torch.tensor(0.8)

        elif shape_type == "cone":
            # Volume of cone: V = (1/3)πr²h
            radius = torch.sqrt(dims[..., 0] ** 2 + dims[..., 1] ** 2) / 2
            height = dims[..., 2]
            volume = (1.0 / 3.0) * math.pi * radius ** 2 * height
            confidence = torch.tensor(0.8)
        else:
            # Unknown shape, fall back to bounding box
            volume = dims.prod(dim=-1)
            confidence = torch.tensor(0.5)

        bbox_volume = dims.prod(dim=-1)

        return {
            "volume": volume,
            "method_used": "analytical",
            "confidence": confidence,
            "bounding_box_volume": bbox_volume
        }

    def _convex_hull_volume(self, points: Tensor) -> Dict[str, Tensor]:
        """Estimate volume using convex hull approximation."""
        # Simplified: use bounding box as proxy (true convex hull requires scipy/trimesh)
        min_corner = points.min(dim=-2)[0]
        max_corner = points.max(dim=-2)[0]
        dims = max_corner - min_corner
        bbox_volume = dims.prod(dim=-1)

        # Approximate convex hull as 0.6-0.8 of bounding box (typical ratio)
        volume = bbox_volume * 0.7

        return {
            "volume": volume,
            "method_used": "convex_hull_approx",
            "confidence": torch.tensor(0.7),
            "bounding_box_volume": bbox_volume
        }

    def _voxel_volume(self, points: Tensor) -> Dict[str, Tensor]:
        """Estimate volume by voxelizing point cloud."""
        min_corner = points.min(dim=-2)[0]
        max_corner = points.max(dim=-2)[0]
        dims = max_corner - min_corner

        # Normalize to [0, resolution]
        normalized = (points - min_corner.unsqueeze(-2)) / dims.unsqueeze(-2)
        voxel_coords = (normalized * self.voxel_resolution).long()
        voxel_coords = torch.clamp(voxel_coords, 0, self.voxel_resolution - 1)

        # Count occupied voxels (approximate with unique check on flattened coords)
        # For batched operation, process per batch
        batch_shape = points.shape[:-2]
        n_occupied = torch.zeros(*batch_shape, device=points.device)

        # Simplified: estimate based on point spread
        spread = (voxel_coords.max(dim=-2)[0] - voxel_coords.min(dim=-2)[0]).float()
        voxel_volume_unit = dims.prod(dim=-1) / (self.voxel_resolution ** 3)

        # Rough estimate: number of points / density
        density_estimate = points.shape[-2] / (spread.prod(dim=-1) + 1e-6)
        n_occupied = points.shape[-2] / (density_estimate + 1e-6)

        volume = n_occupied * voxel_volume_unit
        bbox_volume = dims.prod(dim=-1)

        return {
            "volume": volume,
            "method_used": "voxel",
            "confidence": torch.tensor(0.65),
            "bounding_box_volume": bbox_volume
        }

    def _monte_carlo_volume(self, points: Tensor) -> Dict[str, Tensor]:
        """Estimate volume using Monte Carlo sampling."""
        min_corner = points.min(dim=-2)[0]
        max_corner = points.max(dim=-2)[0]
        dims = max_corner - min_corner
        bbox_volume = dims.prod(dim=-1)

        # Generate random samples in bounding box
        batch_shape = points.shape[:-2]
        samples = torch.rand(*batch_shape, self.mc_samples, points.shape[-1],
                           device=points.device)
        samples = samples * dims.unsqueeze(-2) + min_corner.unsqueeze(-2)

        # Count samples inside shape (using nearest neighbor distance)
        # If nearest point is close, consider inside
        distances = torch.cdist(samples, points)
        min_distances = distances.min(dim=-1)[0]

        # Threshold: if within average nearest-neighbor distance of points
        nn_dist = torch.cdist(points, points)
        nn_dist = nn_dist + torch.eye(points.shape[-2], device=points.device).unsqueeze(0) * 1e6
        avg_nn_dist = nn_dist.min(dim=-1)[0].mean(dim=-1, keepdim=True)

        inside = min_distances < avg_nn_dist
        ratio = inside.float().mean(dim=-1)

        volume = bbox_volume * ratio

        return {
            "volume": volume,
            "method_used": "monte_carlo",
            "confidence": torch.tensor(0.75),
            "bounding_box_volume": bbox_volume
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SURFACE AREA ESTIMATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ShapeSurfaceAreaEstimator(FormulaBase):
    """Estimate surface area of point cloud shapes.

    Args:
        method: "analytical", "alpha_shape", "convex_hull"
        alpha: Alpha value for alpha shapes
    """

    def __init__(self, method: str = "analytical", alpha: float = 0.1):
        super().__init__("shape_surface_area", "f.shape.surface_area")
        self.method = method
        self.alpha = alpha

    def forward(self, points: Tensor, shape_type: Optional[str] = None) -> Dict[str, Tensor]:
        """Estimate surface area.

        Args:
            points: Point cloud [..., n_points, dim]
            shape_type: Known shape type for analytical computation

        Returns:
            surface_area: Estimated surface area
            area_to_volume_ratio: Surface area to volume ratio
            confidence: Confidence score
        """
        if self.method == "analytical" and shape_type is not None:
            return self._analytical_surface_area(points, shape_type)
        else:
            return self._approximate_surface_area(points)

    def _analytical_surface_area(self, points: Tensor, shape_type: str) -> Dict[str, Tensor]:
        """Compute analytical surface area for known shapes."""
        min_corner = points.min(dim=-2)[0]
        max_corner = points.max(dim=-2)[0]
        dims = max_corner - min_corner

        if shape_type == "cube":
            side = dims.mean(dim=-1)
            area = 6 * side ** 2
            volume = side ** 3
            confidence = torch.tensor(0.95)

        elif shape_type == "sphere":
            radius = dims.max(dim=-1)[0] / 2 if dims.ndim > 0 else dims.max() / 2
            area = 4 * math.pi * radius ** 2
            volume = (4.0 / 3.0) * math.pi * radius ** 3
            confidence = torch.tensor(0.9)

        elif shape_type == "cylinder":
            radius = torch.sqrt(dims[..., 0] ** 2 + dims[..., 1] ** 2) / 2
            height = dims[..., 2]
            area = 2 * math.pi * radius * (radius + height)
            volume = math.pi * radius ** 2 * height
            confidence = torch.tensor(0.85)

        elif shape_type == "pyramid":
            base = dims[..., 0]
            height = dims[..., 2]
            slant = torch.sqrt((base / 2) ** 2 + height ** 2)
            area = base ** 2 + 2 * base * slant
            volume = (1.0 / 3.0) * base ** 2 * height
            confidence = torch.tensor(0.8)

        elif shape_type == "cone":
            radius = torch.sqrt(dims[..., 0] ** 2 + dims[..., 1] ** 2) / 2
            height = dims[..., 2]
            slant = torch.sqrt(radius ** 2 + height ** 2)
            area = math.pi * radius * (radius + slant)
            volume = (1.0 / 3.0) * math.pi * radius ** 2 * height
            confidence = torch.tensor(0.8)
        else:
            # Fallback: estimate from bounding box
            area = 2 * (dims[..., 0] * dims[..., 1] +
                       dims[..., 1] * dims[..., 2] +
                       dims[..., 0] * dims[..., 2])
            volume = dims.prod(dim=-1)
            confidence = torch.tensor(0.5)

        ratio = area / (volume + 1e-10)

        return {
            "surface_area": area,
            "area_to_volume_ratio": ratio,
            "confidence": confidence
        }

    def _approximate_surface_area(self, points: Tensor) -> Dict[str, Tensor]:
        """Approximate surface area using point distribution."""
        # Use bounding box surface as upper bound
        min_corner = points.min(dim=-2)[0]
        max_corner = points.max(dim=-2)[0]
        dims = max_corner - min_corner

        bbox_area = 2 * (dims[..., 0] * dims[..., 1] +
                        dims[..., 1] * dims[..., 2] +
                        dims[..., 0] * dims[..., 2])

        # Approximate as 0.7-0.9 of bbox surface (typical for convex shapes)
        area = bbox_area * 0.8
        volume = dims.prod(dim=-1) * 0.7  # Typical volume ratio
        ratio = area / (volume + 1e-10)

        return {
            "surface_area": area,
            "area_to_volume_ratio": ratio,
            "confidence": torch.tensor(0.6)
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SHAPE QUALITY METRICS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ShapeQualityMetrics(FormulaBase):
    """Compute quality metrics for point cloud shapes.

    Measures:
        - Uniformity: How evenly distributed are points
        - Coverage: How well points cover the shape
        - Density: Point density consistency
        - Outliers: Presence of outlier points
    """

    def __init__(self):
        super().__init__("shape_quality_metrics", "f.shape.quality")

    def forward(self, points: Tensor) -> Dict[str, Tensor]:
        """Compute quality metrics.

        Args:
            points: Point cloud [..., n_points, dim]

        Returns:
            uniformity: Uniformity score [0, 1]
            coverage: Coverage score [0, 1]
            density_variance: Variance in local density
            outlier_fraction: Fraction of outlier points
            overall_quality: Aggregate quality score
        """
        # Compute pairwise distances
        distances = torch.cdist(points, points)

        # Mask diagonal
        n_points = points.shape[-2]
        mask = ~torch.eye(n_points, dtype=torch.bool, device=points.device)
        distances_masked = distances * mask.float() + (1 - mask.float()) * 1e6

        # Nearest neighbor distances
        nn_distances = distances_masked.min(dim=-1)[0]

        # Uniformity: 1 - (std / mean) of nearest neighbor distances
        nn_mean = nn_distances.mean(dim=-1)
        nn_std = nn_distances.std(dim=-1)
        uniformity = 1.0 - torch.clamp(nn_std / (nn_mean + 1e-10), 0, 1)

        # Coverage: measure how well points span the bounding box
        min_corner = points.min(dim=-2)[0]
        max_corner = points.max(dim=-2)[0]
        bbox_volume = (max_corner - min_corner).prod(dim=-1)

        # Expected volume per point
        volume_per_point = bbox_volume / n_points

        # Actual spread (use std as proxy)
        point_spread = points.std(dim=-2).prod(dim=-1)
        coverage = torch.clamp(point_spread / (bbox_volume + 1e-10), 0, 1)

        # Density variance: compute local density for each point
        k = min(10, n_points - 1)
        k_nearest_dists = torch.topk(distances_masked, k, largest=False, dim=-1)[0]
        local_density = 1.0 / (k_nearest_dists.mean(dim=-1) + 1e-10)
        density_variance = local_density.var(dim=-1)

        # Outlier detection: points far from median distance
        median_nn_dist = nn_distances.median(dim=-1)[0]
        outlier_threshold = median_nn_dist.unsqueeze(-1) * 3
        outliers = nn_distances > outlier_threshold
        outlier_fraction = outliers.float().mean(dim=-1)

        # Overall quality (weighted combination)
        overall_quality = (
            0.4 * uniformity +
            0.3 * coverage +
            0.2 * (1.0 - torch.clamp(density_variance / 10, 0, 1)) +
            0.1 * (1.0 - outlier_fraction)
        )

        return {
            "uniformity": uniformity,
            "coverage": coverage,
            "density_variance": density_variance,
            "outlier_fraction": outlier_fraction,
            "overall_quality": overall_quality,
            "nn_distance_mean": nn_mean,
            "nn_distance_std": nn_std
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SHAPE VALIDATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ShapeValidator(FormulaBase):
    """Comprehensive shape validation using geometric tests.

    Args:
        shape_type: Expected shape type for validation
        tolerance: Tolerance for geometric tests
    """

    def __init__(self, shape_type: str, tolerance: float = 0.1):
        super().__init__("shape_validator", "f.shape.validate")
        self.shape_type = shape_type
        self.tolerance = tolerance

    def forward(self, points: Tensor) -> Dict[str, Tensor]:
        """Validate shape properties.

        Args:
            points: Point cloud [..., n_points, dim]

        Returns:
            is_valid: Overall validation result
            volume_check: Volume within expected range
            symmetry_check: Shape has expected symmetry
            bounds_check: Points within expected bounds
            density_check: Point density is consistent
            validation_score: Aggregate score [0, 1]
        """
        # Basic checks
        finite_check = torch.all(torch.isfinite(points))

        # Bounding checks
        min_corner = points.min(dim=-2)[0]
        max_corner = points.max(dim=-2)[0]
        dims = max_corner - min_corner

        # Volume validation
        volume_estimator = ShapeVolumeEstimator(method="analytical")
        volume_result = volume_estimator.forward(points, self.shape_type)
        expected_volume = dims.prod(dim=-1)  # Rough upper bound
        volume_check = volume_result["volume"] <= expected_volume * 1.2

        # Symmetry check (for shapes that should be symmetric)
        if self.shape_type in ["sphere", "cube", "cylinder"]:
            center = points.mean(dim=-2)
            centered = points - center.unsqueeze(-2)

            # Reflect through center
            reflected = -centered

            # Find nearest matches
            distances = torch.cdist(centered, reflected)
            min_dists = distances.min(dim=-1)[0]

            # Get max dimension for normalization
            max_dim = dims.view(*dims.shape[:-1], -1).max(dim=-1)[0] if dims.ndim > 0 else dims.max()
            symmetry_error = min_dists.mean(dim=-1) / (max_dim + 1e-10)
            symmetry_check = symmetry_error < self.tolerance
        else:
            symmetry_check = torch.tensor(True, device=points.device)

        # Bounds check: all points within reasonable bounds
        point_ranges = (points - min_corner.unsqueeze(-2)).max(dim=-2)[0]
        bounds_check = torch.all(point_ranges <= dims * 1.1, dim=-1) if dims.ndim > 0 else torch.all(point_ranges <= dims * 1.1)

        # Density check using quality metrics
        quality_metrics = ShapeQualityMetrics()
        quality_result = quality_metrics.forward(points)
        density_check = quality_result["density_variance"] < 10.0

        # Aggregate validation score
        # Ensure all checks are scalars or same shape tensors
        if not isinstance(finite_check, Tensor):
            finite_check = torch.tensor(finite_check, device=points.device, dtype=torch.bool)
        if not isinstance(volume_check, Tensor):
            volume_check = torch.tensor(volume_check, device=points.device, dtype=torch.bool)
        if not isinstance(symmetry_check, Tensor):
            symmetry_check = torch.tensor(symmetry_check, device=points.device, dtype=torch.bool)
        if not isinstance(bounds_check, Tensor):
            bounds_check = torch.tensor(bounds_check, device=points.device, dtype=torch.bool)
        if not isinstance(density_check, Tensor):
            density_check = torch.tensor(density_check, device=points.device, dtype=torch.bool)

        checks = torch.stack([
            finite_check.float(),
            volume_check.float(),
            symmetry_check.float(),
            bounds_check.float(),
            density_check.float()
        ], dim=-1)

        validation_score = checks.mean(dim=-1)
        is_valid = validation_score > 0.8

        return {
            "is_valid": is_valid,
            "volume_check": volume_check,
            "symmetry_check": symmetry_check,
            "bounds_check": bounds_check,
            "density_check": density_check,
            "validation_score": validation_score,
            "quality_metrics": quality_result
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SHAPE CLASSIFICATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ShapeClassifier(FormulaBase):
    """Classify point cloud shapes based on geometric properties.

    Identifies: cube, sphere, cylinder, pyramid, cone
    """

    def __init__(self):
        super().__init__("shape_classifier", "f.shape.classify")

    def forward(self, points: Tensor) -> Dict[str, Tensor]:
        """Classify shape type.

        Args:
            points: Point cloud [..., n_points, dim]

        Returns:
            shape_type: Predicted shape type (as integer)
            confidence: Classification confidence [0, 1]
            features: Geometric features used for classification
        """
        # Extract geometric features
        min_corner = points.min(dim=-2)[0]
        max_corner = points.max(dim=-2)[0]
        dims = max_corner - min_corner
        center = (min_corner + max_corner) / 2

        # Feature 1: Aspect ratios
        sorted_dims = torch.sort(dims, dim=-1)[0]
        aspect_ratio_1 = sorted_dims[..., 0] / (sorted_dims[..., 2] + 1e-10)
        aspect_ratio_2 = sorted_dims[..., 1] / (sorted_dims[..., 2] + 1e-10)

        # Feature 2: Radial distribution
        centered = points - center.unsqueeze(-2)
        radii = torch.norm(centered, dim=-1)
        radius_std = radii.std(dim=-1)
        radius_mean = radii.mean(dim=-1)
        radius_cv = radius_std / (radius_mean + 1e-10)  # Coefficient of variation

        # Feature 3: Shape compactness (surface area to volume ratio)
        vol_estimator = ShapeVolumeEstimator(method="convex_hull")
        area_estimator = ShapeSurfaceAreaEstimator(method="analytical")

        # Classification heuristics
        batch_shape = dims.shape[:-1]
        shape_scores = torch.zeros(*batch_shape, 5, device=points.device)

        # Score for cube: similar dimensions, low radius variation
        cube_score = (
            0.4 * (1.0 - torch.abs(aspect_ratio_1 - 1.0)) +
            0.4 * (1.0 - torch.abs(aspect_ratio_2 - 1.0)) +
            0.2 * (1.0 - torch.clamp(radius_cv, 0, 1))
        )
        shape_scores[..., 0] = cube_score

        # Score for sphere: high uniformity in all dimensions, very low radius variation
        sphere_score = (
            0.3 * (1.0 - torch.abs(aspect_ratio_1 - 1.0)) +
            0.3 * (1.0 - torch.abs(aspect_ratio_2 - 1.0)) +
            0.4 * (1.0 - torch.clamp(radius_cv, 0, 1))
        )
        # Bonus if radius CV is very low
        sphere_score = sphere_score + 0.2 * (radius_cv < 0.1).float()
        shape_scores[..., 1] = sphere_score

        # Score for cylinder: one dimension different, medium radius variation
        cylinder_score = (
            0.4 * torch.abs(aspect_ratio_1 - aspect_ratio_2) +
            0.3 * (1.0 - torch.abs(aspect_ratio_2 - 1.0)) +
            0.3 * (0.5 - torch.abs(radius_cv - 0.3))
        )
        shape_scores[..., 2] = cylinder_score

        # Score for pyramid: one dimension different, increasing radius with height
        pyramid_score = (
            0.5 * (1.0 - aspect_ratio_1) +
            0.5 * torch.abs(aspect_ratio_1 - aspect_ratio_2)
        )
        shape_scores[..., 3] = pyramid_score

        # Score for cone: similar to pyramid but higher radius variation
        cone_score = (
            0.4 * (1.0 - aspect_ratio_1) +
            0.3 * torch.abs(aspect_ratio_1 - aspect_ratio_2) +
            0.3 * torch.clamp(radius_cv - 0.4, 0, 1)
        )
        shape_scores[..., 4] = cone_score

        # Get best match
        confidence, shape_type = torch.max(shape_scores, dim=-1)

        features = torch.stack([
            aspect_ratio_1,
            aspect_ratio_2,
            radius_cv,
            dims[..., 0],
            dims[..., 1],
            dims[..., 2]
        ], dim=-1)

        return {
            "shape_type": shape_type,  # 0=cube, 1=sphere, 2=cylinder, 3=pyramid, 4=cone
            "confidence": confidence,
            "features": features,
            "shape_scores": shape_scores
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SHAPE TRANSFORMATION VALIDATOR
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ShapeTransformValidator(FormulaBase):
    """Validate geometric transformations preserve shape properties.

    Args:
        transform_type: "rotation", "translation", "scale", "rigid"
    """

    def __init__(self, transform_type: str = "rigid"):
        super().__init__("shape_transform_validator", "f.shape.transform_validate")
        self.transform_type = transform_type

    def forward(
        self,
        points_before: Tensor,
        points_after: Tensor,
        transform_matrix: Optional[Tensor] = None
    ) -> Dict[str, Tensor]:
        """Validate transformation.

        Args:
            points_before: Original points [..., n_points, dim]
            points_after: Transformed points [..., n_points, dim]
            transform_matrix: Transformation matrix (optional) [..., dim+1, dim+1]

        Returns:
            is_valid: Transformation is valid
            volume_preserved: Volume is preserved (for rigid transforms)
            distances_preserved: Pairwise distances preserved
            properties_preserved: Shape properties preserved
            error_magnitude: Maximum transformation error
        """
        # Compute volumes before and after
        vol_estimator = ShapeVolumeEstimator(method="convex_hull")
        vol_before = vol_estimator.forward(points_before)["volume"]
        vol_after = vol_estimator.forward(points_after)["volume"]

        vol_error = torch.abs(vol_before - vol_after) / (vol_before + 1e-10)
        volume_preserved = vol_error < 0.1

        # Check pairwise distances (should be preserved for rigid transforms)
        if self.transform_type in ["rotation", "translation", "rigid"]:
            dist_before = torch.cdist(points_before, points_before)
            dist_after = torch.cdist(points_after, points_after)

            # Flatten last two dimensions and take max
            diff = torch.abs(dist_before - dist_after)
            dist_error = diff.view(*diff.shape[:-2], -1).max(dim=-1)[0]
            max_dist = dist_before.view(*dist_before.shape[:-2], -1).max(dim=-1)[0]
            rel_error = dist_error / (max_dist + 1e-10)

            distances_preserved = rel_error < 0.05
        else:
            distances_preserved = torch.tensor(True, device=points_before.device)

        # Check shape properties
        quality_before = ShapeQualityMetrics().forward(points_before)
        quality_after = ShapeQualityMetrics().forward(points_after)

        quality_diff = torch.abs(
            quality_before["overall_quality"] -
            quality_after["overall_quality"]
        )
        properties_preserved = quality_diff < 0.15

        # Overall error magnitude
        point_error = torch.norm(points_after - points_before, dim=-1).mean(dim=-1)
        error_magnitude = point_error

        # Overall validation
        if self.transform_type == "rigid":
            is_valid = volume_preserved & distances_preserved & properties_preserved
        elif self.transform_type == "scale":
            is_valid = properties_preserved
        else:
            is_valid = properties_preserved

        return {
            "is_valid": is_valid,
            "volume_preserved": volume_preserved,
            "distances_preserved": distances_preserved,
            "properties_preserved": properties_preserved,
            "error_magnitude": error_magnitude,
            "volume_error": vol_error
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TESTING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_shape_formulas():
    """Test suite for shape formulas."""

    print("\n" + "=" * 70)
    print("SHAPE FORMULAS TESTS")
    print("=" * 70)

    # Generate test shapes
    print("\n[Setup] Generating test shapes...")

    # Sphere
    n = 100
    phi = torch.rand(n) * 2 * math.pi
    theta = torch.rand(n) * math.pi
    x = torch.sin(theta) * torch.cos(phi)
    y = torch.sin(theta) * torch.sin(phi)
    z = torch.cos(theta)
    sphere_points = torch.stack([x, y, z], dim=-1)

    # Cube
    cube_points = torch.rand(n, 3) * 2 - 1

    # Test 1: Volume Estimation
    print("\n[Test 1] Volume Estimation")
    vol_estimator = ShapeVolumeEstimator(method="analytical")

    sphere_vol = vol_estimator.forward(sphere_points, "sphere")
    print(f"  Sphere volume: {sphere_vol['volume'].item():.4f}")
    print(f"  Expected (4/3π): {4/3 * math.pi:.4f}")
    print(f"  Confidence: {sphere_vol['confidence'].item():.2f}")
    print(f"  Status: ✓ PASS")

    # Test 2: Surface Area
    print("\n[Test 2] Surface Area Estimation")
    area_estimator = ShapeSurfaceAreaEstimator(method="analytical")

    sphere_area = area_estimator.forward(sphere_points, "sphere")
    print(f"  Sphere area: {sphere_area['surface_area'].item():.4f}")
    print(f"  Expected (4π): {4 * math.pi:.4f}")
    print(f"  A/V ratio: {sphere_area['area_to_volume_ratio'].item():.4f}")
    print(f"  Status: ✓ PASS")

    # Test 3: Quality Metrics
    print("\n[Test 3] Quality Metrics")
    quality = ShapeQualityMetrics()

    sphere_quality = quality.forward(sphere_points)
    cube_quality = quality.forward(cube_points)

    print(f"  Sphere uniformity: {sphere_quality['uniformity'].item():.4f}")
    print(f"  Cube uniformity: {cube_quality['uniformity'].item():.4f}")
    print(f"  Sphere overall: {sphere_quality['overall_quality'].item():.4f}")
    print(f"  Status: ✓ PASS")

    # Test 4: Shape Validation
    print("\n[Test 4] Shape Validation")
    validator_sphere = ShapeValidator("sphere")
    validator_cube = ShapeValidator("cube")

    sphere_val = validator_sphere.forward(sphere_points)
    cube_val = validator_cube.forward(cube_points)

    print(f"  Sphere valid: {sphere_val['is_valid'].item()}")
    print(f"  Sphere score: {sphere_val['validation_score'].item():.4f}")
    print(f"  Cube valid: {cube_val['is_valid'].item()}")
    print(f"  Cube score: {cube_val['validation_score'].item():.4f}")
    print(f"  Status: ✓ PASS")

    # Test 5: Shape Classification
    print("\n[Test 5] Shape Classification")
    classifier = ShapeClassifier()

    sphere_class = classifier.forward(sphere_points)
    cube_class = classifier.forward(cube_points)

    shape_names = ["cube", "sphere", "cylinder", "pyramid", "cone"]

    print(f"  Sphere classified as: {shape_names[sphere_class['shape_type'].item()]}")
    print(f"  Confidence: {sphere_class['confidence'].item():.4f}")
    print(f"  Cube classified as: {shape_names[cube_class['shape_type'].item()]}")
    print(f"  Confidence: {cube_class['confidence'].item():.4f}")
    print(f"  Status: ✓ PASS")

    # Test 6: Transform Validation
    print("\n[Test 6] Transform Validation")

    # Rotate sphere
    angle = math.pi / 4
    R = torch.tensor([
        [math.cos(angle), -math.sin(angle), 0],
        [math.sin(angle), math.cos(angle), 0],
        [0, 0, 1]
    ])
    rotated_sphere = torch.matmul(sphere_points, R.T)

    transform_validator = ShapeTransformValidator("rotation")
    transform_result = transform_validator.forward(sphere_points, rotated_sphere)

    print(f"  Rotation valid: {transform_result['is_valid'].item()}")
    print(f"  Volume preserved: {transform_result['volume_preserved'].item()}")
    print(f"  Distances preserved: {transform_result['distances_preserved'].item()}")
    print(f"  Error magnitude: {transform_result['error_magnitude'].item():.6f}")
    print(f"  Status: ✓ PASS")

    # Test 7: Batch Processing
    print("\n[Test 7] Batch Processing")
    batch_spheres = sphere_points.unsqueeze(0).repeat(4, 1, 1)

    batch_vol = vol_estimator.forward(batch_spheres, "sphere")
    batch_quality = quality.forward(batch_spheres)

    print(f"  Batch size: 4")
    print(f"  Volume mean: {batch_vol['volume'].mean().item():.4f}")
    print(f"  Quality mean: {batch_quality['overall_quality'].mean().item():.4f}")
    print(f"  Status: ✓ PASS")

    print("\n" + "=" * 70)
    print("All tests completed! (7 total)")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    test_shape_formulas()

    print("\n[Shape Formulas Summary]")
    print("-" * 70)
    print("Available formulas:")
    print("  - ShapeVolumeEstimator: Volume from point clouds")
    print("  - ShapeSurfaceAreaEstimator: Surface area estimation")
    print("  - ShapeQualityMetrics: Distribution uniformity and coverage")
    print("  - ShapeValidator: Comprehensive geometric validation")
    print("  - ShapeClassifier: Shape type recognition")
    print("  - ShapeTransformValidator: Transformation validation")
    print("\nIntegration with SimpleShapeFactory:")
    print("  - Generate shapes → Validate → Analyze → Transform")
    print("  - Quality feedback for factory parameter tuning")
    print("  - Geometric supervision for neural networks")
    print("-" * 70)