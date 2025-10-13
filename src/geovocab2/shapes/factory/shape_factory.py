"""
SimpleShapeFactory
------------------
Factory for generating common geometric shapes (cubes, spheres, cylinders, pyramids).

Generates point cloud representations of basic shapes with configurable resolution
and embedding dimensions. Follows FactoryBase pattern with backend flexibility.

Supported Shapes:
    - cube: Axis-aligned hypercube
    - sphere: Points uniformly distributed on sphere surface
    - cylinder: Circular cylinder with caps
    - pyramid: Square-base pyramid
    - cone: Circular cone

Integrated Validation:
    - Geometric property validation
    - Quality metrics (uniformity, coverage)
    - Volume and surface area verification
    - Shape classification confirmation

License: MIT
"""

import numpy as np
from typing import Optional, Tuple, Union, Literal, Dict, Any

try:
    from .factory_base import FactoryBase, HAS_TORCH
except ImportError:
    from factory_base import FactoryBase, HAS_TORCH

if HAS_TORCH:
    import torch

ShapeType = Literal["cube", "sphere", "cylinder", "pyramid", "cone"]

# Import validation formulas
try:
    from ..formula.engineering.shape_validation import (
        ShapeVolumeEstimator,
        ShapeSurfaceAreaEstimator,
        ShapeQualityMetrics,
        ShapeValidator,
        ShapeClassifier
    )
    HAS_VALIDATION = True
except ImportError:
    HAS_VALIDATION = False


class SimpleShapeFactory(FactoryBase):
    """
    Generate simple geometric shapes as point clouds.

    Args:
        shape_type: Type of shape to generate
        embed_dim: Embedding space dimension (2 or 3 for most shapes)
        resolution: Number of points to generate (approximate)
        scale: Scaling factor for shape dimensions
        validate_output: Whether to validate generated shapes
        quality_threshold: Minimum quality score [0, 1] for validation
        max_retries: Max regeneration attempts if validation fails
    """

    def __init__(
        self,
        shape_type: ShapeType,
        embed_dim: int = 3,
        resolution: int = 100,
        scale: float = 1.0,
        validate_output: bool = False,
        quality_threshold: float = 0.7,
        max_retries: int = 3,
    ):
        valid_shapes = ["cube", "sphere", "cylinder", "pyramid", "cone"]
        if shape_type not in valid_shapes:
            raise ValueError(f"shape_type must be one of {valid_shapes}")

        if embed_dim < 2:
            raise ValueError(f"embed_dim must be >= 2, got {embed_dim}")

        if resolution < 4:
            raise ValueError(f"resolution must be >= 4, got {resolution}")

        super().__init__(
            name=f"shape_{shape_type}_d{embed_dim}",
            uid=f"factory.shape.{shape_type}.d{embed_dim}"
        )

        self.shape_type = shape_type
        self.embed_dim = embed_dim
        self.resolution = resolution
        self.scale = scale
        self.validate_output = validate_output
        self.quality_threshold = quality_threshold
        self.max_retries = max_retries

        # Initialize validators if available
        if HAS_VALIDATION and validate_output:
            self.volume_estimator = ShapeVolumeEstimator(method="analytical")
            self.area_estimator = ShapeSurfaceAreaEstimator(method="analytical")
            self.quality_metrics = ShapeQualityMetrics()
            self.validator = ShapeValidator(shape_type, tolerance=0.15)
            self.classifier = ShapeClassifier()
        else:
            self.volume_estimator = None
            self.area_estimator = None
            self.quality_metrics = None
            self.validator = None
            self.classifier = None

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # NumPy Backend
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def build_numpy(
        self,
        *,
        dtype=np.float32,
        seed: Optional[int] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Build shape using NumPy.

        Returns:
            Array of shape (N, embed_dim) representing shape vertices
        """
        rng = np.random.default_rng(seed)

        if self.shape_type == "cube":
            points = self._generate_cube_numpy(rng, dtype)
        elif self.shape_type == "sphere":
            points = self._generate_sphere_numpy(rng, dtype)
        elif self.shape_type == "cylinder":
            points = self._generate_cylinder_numpy(rng, dtype)
        elif self.shape_type == "pyramid":
            points = self._generate_pyramid_numpy(rng, dtype)
        elif self.shape_type == "cone":
            points = self._generate_cone_numpy(rng, dtype)
        else:
            raise ValueError(f"Unknown shape: {self.shape_type}")

        return points * self.scale

    def _generate_cube_numpy(self, rng, dtype) -> np.ndarray:
        """Generate cube vertices (2^d corners + face samples)."""
        # Generate all corner vertices
        n_corners = 2 ** min(self.embed_dim, 10)  # Cap for high dims
        corners = np.array(
            [[(-1)**((i >> j) & 1) for j in range(self.embed_dim)]
             for i in range(n_corners)],
            dtype=dtype
        )

        # Sample additional points on faces
        n_face_samples = max(4, self.resolution - n_corners)
        face_points = rng.uniform(-1, 1, size=(n_face_samples, self.embed_dim))

        # Project to cube surface (randomly pick dimension to clamp)
        for i in range(n_face_samples):
            dim = rng.integers(0, self.embed_dim)
            face_points[i, dim] = rng.choice([-1.0, 1.0])

        points = np.vstack([corners, face_points.astype(dtype)])
        return points

    def _generate_sphere_numpy(self, rng, dtype) -> np.ndarray:
        """Generate points uniformly on sphere surface."""
        # Use Gaussian method for uniform sphere sampling
        points = rng.standard_normal((self.resolution, self.embed_dim))
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        points = points / (norms + 1e-10)
        return points.astype(dtype)

    def _generate_cylinder_numpy(self, rng, dtype) -> np.ndarray:
        """Generate cylinder along z-axis (requires embed_dim >= 3)."""
        if self.embed_dim < 3:
            raise ValueError("Cylinder requires embed_dim >= 3")

        # Split points: curved surface + two caps
        n_surface = int(0.7 * self.resolution)
        n_cap = (self.resolution - n_surface) // 2

        # Curved surface
        theta = rng.uniform(0, 2*np.pi, n_surface)
        z = rng.uniform(-1, 1, n_surface)
        x = np.cos(theta)
        y = np.sin(theta)
        surface = np.column_stack([x, y, z])

        # Top cap (z=1)
        r_top = np.sqrt(rng.uniform(0, 1, n_cap))
        theta_top = rng.uniform(0, 2*np.pi, n_cap)
        top_cap = np.column_stack([
            r_top * np.cos(theta_top),
            r_top * np.sin(theta_top),
            np.ones(n_cap)
        ])

        # Bottom cap (z=-1)
        r_bot = np.sqrt(rng.uniform(0, 1, n_cap))
        theta_bot = rng.uniform(0, 2*np.pi, n_cap)
        bot_cap = np.column_stack([
            r_bot * np.cos(theta_bot),
            r_bot * np.sin(theta_bot),
            -np.ones(n_cap)
        ])

        points_3d = np.vstack([surface, top_cap, bot_cap])

        # Embed in higher dimensions if needed
        if self.embed_dim > 3:
            points = np.zeros((len(points_3d), self.embed_dim), dtype=dtype)
            points[:, :3] = points_3d
        else:
            points = points_3d

        return points.astype(dtype)

    def _generate_pyramid_numpy(self, rng, dtype) -> np.ndarray:
        """Generate square pyramid (requires embed_dim >= 3)."""
        if self.embed_dim < 3:
            raise ValueError("Pyramid requires embed_dim >= 3")

        # Apex at (0, 0, 1), base at z=-1
        apex = np.array([[0.0, 0.0, 1.0]])

        # Base corners
        base_corners = np.array([
            [-1, -1, -1],
            [ 1, -1, -1],
            [ 1,  1, -1],
            [-1,  1, -1]
        ])

        # Sample points on triangular faces
        n_face = (self.resolution - 5) // 4
        face_points = []

        for i in range(4):
            corner1 = base_corners[i]
            corner2 = base_corners[(i+1) % 4]

            # Random barycentric coordinates for triangular face
            for _ in range(n_face):
                u, v = rng.random(2)
                if u + v > 1:
                    u, v = 1-u, 1-v
                w = 1 - u - v

                point = w * apex[0] + u * corner1 + v * corner2
                face_points.append(point)

        # Sample base
        n_base = self.resolution - 5 - len(face_points)
        base_points = rng.uniform(-1, 1, size=(n_base, 2))
        base_z = np.full((n_base, 1), -1.0)
        base_samples = np.hstack([base_points, base_z])

        points_3d = np.vstack([apex, base_corners, np.array(face_points), base_samples])

        # Embed in higher dimensions if needed
        if self.embed_dim > 3:
            points = np.zeros((len(points_3d), self.embed_dim), dtype=dtype)
            points[:, :3] = points_3d
        else:
            points = points_3d

        return points.astype(dtype)

    def _generate_cone_numpy(self, rng, dtype) -> np.ndarray:
        """Generate cone along z-axis (requires embed_dim >= 3)."""
        if self.embed_dim < 3:
            raise ValueError("Cone requires embed_dim >= 3")

        # Split points: curved surface + base
        n_surface = int(0.7 * self.resolution)
        n_base = self.resolution - n_surface

        # Apex at (0, 0, 1)
        apex = np.array([[0.0, 0.0, 1.0]])

        # Curved surface: radius decreases linearly with height
        z = rng.uniform(-1, 1, n_surface)
        r = (1 - z) / 2  # radius = 0 at z=1, radius = 1 at z=-1
        theta = rng.uniform(0, 2*np.pi, n_surface)

        surface = np.column_stack([
            r * np.cos(theta),
            r * np.sin(theta),
            z
        ])

        # Base (z=-1)
        r_base = np.sqrt(rng.uniform(0, 1, n_base))
        theta_base = rng.uniform(0, 2*np.pi, n_base)
        base = np.column_stack([
            r_base * np.cos(theta_base),
            r_base * np.sin(theta_base),
            -np.ones(n_base)
        ])

        points_3d = np.vstack([apex, surface, base])

        # Embed in higher dimensions if needed
        if self.embed_dim > 3:
            points = np.zeros((len(points_3d), self.embed_dim), dtype=dtype)
            points[:, :3] = points_3d
        else:
            points = points_3d

        return points.astype(dtype)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PyTorch Backend
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def build_torch(
        self,
        *,
        device: str = "cpu",
        dtype: Optional["torch.dtype"] = None,
        seed: Optional[int] = None,
        **kwargs
    ) -> "torch.Tensor":
        """Build shape using PyTorch (direct on-device generation)."""
        if not HAS_TORCH:
            raise RuntimeError("PyTorch required for build_torch")

        target_dtype = dtype or self._infer_torch_dtype(device)
        dev = torch.device(device)

        # Generate on CPU first (RNG is CPU-only)
        if seed is not None:
            gen = torch.Generator(device="cpu")
            gen.manual_seed(seed)
        else:
            gen = None

        if self.shape_type == "cube":
            points = self._generate_cube_torch(gen, target_dtype)
        elif self.shape_type == "sphere":
            points = self._generate_sphere_torch(gen, target_dtype)
        elif self.shape_type == "cylinder":
            points = self._generate_cylinder_torch(gen, target_dtype)
        elif self.shape_type == "pyramid":
            points = self._generate_pyramid_torch(gen, target_dtype)
        elif self.shape_type == "cone":
            points = self._generate_cone_torch(gen, target_dtype)
        else:
            raise ValueError(f"Unknown shape: {self.shape_type}")

        return (points * self.scale).to(dev)

    def _generate_cube_torch(self, gen, dtype) -> "torch.Tensor":
        """Generate cube vertices."""
        n_corners = 2 ** min(self.embed_dim, 10)

        # Generate corners
        indices = torch.arange(n_corners, dtype=torch.long)
        corners = torch.zeros((n_corners, self.embed_dim), dtype=dtype)

        for j in range(self.embed_dim):
            corners[:, j] = (-1.0) ** ((indices >> j) & 1).float()

        # Face samples
        n_face = max(4, self.resolution - n_corners)
        face_points = torch.rand((n_face, self.embed_dim), generator=gen, dtype=dtype) * 2 - 1

        # Project to faces
        for i in range(n_face):
            dim = torch.randint(0, self.embed_dim, (1,), generator=gen).item()
            face_points[i, dim] = torch.randint(0, 2, (1,), generator=gen).item() * 2.0 - 1.0

        return torch.cat([corners, face_points], dim=0)

    def _generate_sphere_torch(self, gen, dtype) -> "torch.Tensor":
        """Generate points on sphere surface."""
        points = torch.randn((self.resolution, self.embed_dim), generator=gen, dtype=dtype)
        norms = torch.linalg.norm(points, dim=1, keepdim=True)
        return points / (norms + 1e-10)

    def _generate_cylinder_torch(self, gen, dtype) -> "torch.Tensor":
        """Generate cylinder."""
        if self.embed_dim < 3:
            raise ValueError("Cylinder requires embed_dim >= 3")

        n_surface = int(0.7 * self.resolution)
        n_cap = (self.resolution - n_surface) // 2

        # Surface
        theta = torch.rand(n_surface, generator=gen, dtype=dtype) * 2 * np.pi
        z = torch.rand(n_surface, generator=gen, dtype=dtype) * 2 - 1
        surface = torch.stack([torch.cos(theta), torch.sin(theta), z], dim=1)

        # Top cap
        r_top = torch.sqrt(torch.rand(n_cap, generator=gen, dtype=dtype))
        theta_top = torch.rand(n_cap, generator=gen, dtype=dtype) * 2 * np.pi
        top_cap = torch.stack([
            r_top * torch.cos(theta_top),
            r_top * torch.sin(theta_top),
            torch.ones(n_cap, dtype=dtype)
        ], dim=1)

        # Bottom cap
        r_bot = torch.sqrt(torch.rand(n_cap, generator=gen, dtype=dtype))
        theta_bot = torch.rand(n_cap, generator=gen, dtype=dtype) * 2 * np.pi
        bot_cap = torch.stack([
            r_bot * torch.cos(theta_bot),
            r_bot * torch.sin(theta_bot),
            -torch.ones(n_cap, dtype=dtype)
        ], dim=1)

        points_3d = torch.cat([surface, top_cap, bot_cap], dim=0)

        if self.embed_dim > 3:
            points = torch.zeros((len(points_3d), self.embed_dim), dtype=dtype)
            points[:, :3] = points_3d
        else:
            points = points_3d

        return points

    def _generate_pyramid_torch(self, gen, dtype) -> "torch.Tensor":
        """Generate square pyramid."""
        if self.embed_dim < 3:
            raise ValueError("Pyramid requires embed_dim >= 3")

        apex = torch.tensor([[0.0, 0.0, 1.0]], dtype=dtype)
        base_corners = torch.tensor([
            [-1, -1, -1],
            [ 1, -1, -1],
            [ 1,  1, -1],
            [-1,  1, -1]
        ], dtype=dtype)

        # Sample faces
        n_face = (self.resolution - 5) // 4
        face_points_list = []

        for i in range(4):
            corner1 = base_corners[i]
            corner2 = base_corners[(i+1) % 4]

            uv = torch.rand((n_face, 2), generator=gen, dtype=dtype)
            mask = (uv[:, 0] + uv[:, 1]) > 1
            uv[mask] = 1 - uv[mask]

            w = 1 - uv.sum(dim=1, keepdim=True)
            points = w * apex + uv[:, 0:1] * corner1 + uv[:, 1:2] * corner2
            face_points_list.append(points)

        face_points = torch.cat(face_points_list, dim=0)

        # Base samples
        n_base = self.resolution - 5 - len(face_points)
        base_xy = torch.rand((n_base, 2), generator=gen, dtype=dtype) * 2 - 1
        base_z = torch.full((n_base, 1), -1.0, dtype=dtype)
        base_samples = torch.cat([base_xy, base_z], dim=1)

        points_3d = torch.cat([apex, base_corners, face_points, base_samples], dim=0)

        if self.embed_dim > 3:
            points = torch.zeros((len(points_3d), self.embed_dim), dtype=dtype)
            points[:, :3] = points_3d
        else:
            points = points_3d

        return points

    def _generate_cone_torch(self, gen, dtype) -> "torch.Tensor":
        """Generate cone."""
        if self.embed_dim < 3:
            raise ValueError("Cone requires embed_dim >= 3")

        n_surface = int(0.7 * self.resolution)
        n_base = self.resolution - n_surface

        apex = torch.tensor([[0.0, 0.0, 1.0]], dtype=dtype)

        # Surface
        z = torch.rand(n_surface, generator=gen, dtype=dtype) * 2 - 1
        r = (1 - z) / 2
        theta = torch.rand(n_surface, generator=gen, dtype=dtype) * 2 * np.pi
        surface = torch.stack([r * torch.cos(theta), r * torch.sin(theta), z], dim=1)

        # Base
        r_base = torch.sqrt(torch.rand(n_base, generator=gen, dtype=dtype))
        theta_base = torch.rand(n_base, generator=gen, dtype=dtype) * 2 * np.pi
        base = torch.stack([
            r_base * torch.cos(theta_base),
            r_base * torch.sin(theta_base),
            -torch.ones(n_base, dtype=dtype)
        ], dim=1)

        points_3d = torch.cat([apex, surface, base], dim=0)

        if self.embed_dim > 3:
            points = torch.zeros((len(points_3d), self.embed_dim), dtype=dtype)
            points[:, :3] = points_3d
        else:
            points = points_3d

        return points

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Validation
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def compute_validation_metrics(
        self,
        output: Union[np.ndarray, "torch.Tensor"]
    ) -> Dict[str, Any]:
        """Compute comprehensive validation metrics.

        Args:
            output: Generated shape points

        Returns:
            Dictionary with validation metrics:
                - volume: Estimated volume
                - surface_area: Estimated surface area
                - quality: Quality metrics dict
                - validation: Validation results dict
                - classification: Classification results dict
        """
        if not HAS_VALIDATION:
            return {
                "validation_available": False,
                "message": "Validation formulas not available"
            }

        # Convert to torch if needed
        if isinstance(output, np.ndarray):
            points = torch.from_numpy(output)
        else:
            points = output

        # Compute all metrics
        results = {}

        # Volume estimation
        if self.volume_estimator is not None:
            vol_result = self.volume_estimator.forward(points, self.shape_type)
            results["volume"] = {
                "value": vol_result["volume"].item() if hasattr(vol_result["volume"], "item") else float(vol_result["volume"]),
                "confidence": vol_result["confidence"].item(),
                "method": vol_result["method_used"]
            }

        # Surface area
        if self.area_estimator is not None:
            area_result = self.area_estimator.forward(points, self.shape_type)
            results["surface_area"] = {
                "value": area_result["surface_area"].item() if hasattr(area_result["surface_area"], "item") else float(area_result["surface_area"]),
                "area_to_volume_ratio": area_result["area_to_volume_ratio"].item() if hasattr(area_result["area_to_volume_ratio"], "item") else float(area_result["area_to_volume_ratio"]),
                "confidence": area_result["confidence"].item()
            }

        # Quality metrics
        if self.quality_metrics is not None:
            quality_result = self.quality_metrics.forward(points)
            results["quality"] = {
                "overall": quality_result["overall_quality"].item() if hasattr(quality_result["overall_quality"], "item") else float(quality_result["overall_quality"]),
                "uniformity": quality_result["uniformity"].item() if hasattr(quality_result["uniformity"], "item") else float(quality_result["uniformity"]),
                "coverage": quality_result["coverage"].item() if hasattr(quality_result["coverage"], "item") else float(quality_result["coverage"]),
                "outlier_fraction": quality_result["outlier_fraction"].item() if hasattr(quality_result["outlier_fraction"], "item") else float(quality_result["outlier_fraction"])
            }

        # Validation
        if self.validator is not None:
            val_result = self.validator.forward(points)
            results["validation"] = {
                "is_valid": val_result["is_valid"].item() if hasattr(val_result["is_valid"], "item") else bool(val_result["is_valid"]),
                "score": val_result["validation_score"].item() if hasattr(val_result["validation_score"], "item") else float(val_result["validation_score"]),
                "volume_check": val_result["volume_check"].item() if hasattr(val_result["volume_check"], "item") else bool(val_result["volume_check"]),
                "symmetry_check": val_result["symmetry_check"].item() if hasattr(val_result["symmetry_check"], "item") else bool(val_result["symmetry_check"]),
                "bounds_check": val_result["bounds_check"].item() if hasattr(val_result["bounds_check"], "item") else bool(val_result["bounds_check"])
            }

        # Classification
        if self.classifier is not None:
            class_result = self.classifier.forward(points)
            shape_names = ["cube", "sphere", "cylinder", "pyramid", "cone"]
            predicted_idx = class_result["shape_type"].item() if hasattr(class_result["shape_type"], "item") else int(class_result["shape_type"])
            results["classification"] = {
                "predicted_type": shape_names[predicted_idx],
                "expected_type": self.shape_type,
                "correct": shape_names[predicted_idx] == self.shape_type,
                "confidence": class_result["confidence"].item() if hasattr(class_result["confidence"], "item") else float(class_result["confidence"])
            }

        return results

    def validate(
        self,
        output: Union[np.ndarray, "torch.Tensor"]
    ) -> Tuple[bool, str]:
        """
        Validate shape output using integrated formulas.

        Args:
            output: Constructed object from build()

        Returns:
            (is_valid, error_message)
        """
        # Check dimensions
        if output.ndim != 2:
            return False, f"Expected 2D array, got shape {output.shape}"

        if output.shape[1] != self.embed_dim:
            return False, f"Expected embed_dim={self.embed_dim}, got {output.shape[1]}"

        if output.shape[0] < 4:
            return False, f"Too few points: {output.shape[0]}"

        # Check for NaN/Inf
        if isinstance(output, np.ndarray):
            if not np.all(np.isfinite(output)):
                return False, "Contains NaN or Inf"

            # Check spread
            std = np.std(output, axis=0)
            if np.any(std < 1e-8):
                return False, "Points too concentrated (degenerate)"
        else:
            if not torch.all(torch.isfinite(output)):
                return False, "Contains NaN or Inf"

            std = torch.std(output, dim=0)
            if torch.any(std < 1e-8):
                return False, "Points too concentrated (degenerate)"

        # Use advanced validation if enabled
        if self.validate_output and HAS_VALIDATION:
            metrics = self.compute_validation_metrics(output)

            # Check quality threshold
            if "quality" in metrics:
                quality_score = metrics["quality"]["overall"]
                if quality_score < self.quality_threshold:
                    return False, f"Quality score {quality_score:.3f} below threshold {self.quality_threshold}"

            # Check validation result
            if "validation" in metrics:
                if not metrics["validation"]["is_valid"]:
                    return False, f"Shape validation failed (score: {metrics['validation']['score']:.3f})"

            # Check classification correctness
            if "classification" in metrics:
                if not metrics["classification"]["correct"]:
                    return False, f"Classified as {metrics['classification']['predicted_type']}, expected {self.shape_type}"

        return True, ""

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Metadata
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def info(self):
        base_info = super().info()
        base_info.update({
            "description": f"{self.shape_type} shape factory (embed_dim={self.embed_dim})",
            "shape_type": self.shape_type,
            "embedding_dimension": self.embed_dim,
            "resolution": self.resolution,
            "scale": self.scale,
            "output_shape": f"(~{self.resolution}, {self.embed_dim})"
        })
        return base_info


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Example Usage
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    print("=" * 70)
    print("SIMPLE SHAPE FACTORY DEMONSTRATION")
    print("=" * 70)

    # Test all shapes
    shapes = ["cube", "sphere", "cylinder", "pyramid", "cone"]

    for shape in shapes:
        print(f"\n[{shape.upper()}] 3D with 50 points")
        factory = SimpleShapeFactory(shape, embed_dim=3, resolution=50)

        points = factory.build(backend="numpy", seed=42, validate=True)
        print(f"  Shape: {points.shape}")
        print(f"  Range: [{points.min():.3f}, {points.max():.3f}]")
        print(f"  Centroid: {points.mean(axis=0)}")

    # High-res sphere
    print("\n[HIGH-RES SPHERE] 1000 points")
    sphere_factory = SimpleShapeFactory("sphere", embed_dim=3, resolution=1000, scale=2.0)
    sphere = sphere_factory.build(backend="numpy", validate=True)
    print(f"  Shape: {sphere.shape}")
    print(f"  Radius check: {np.linalg.norm(sphere, axis=1)[:5]}")

    if HAS_TORCH:
        print("\n[PYTORCH] Cylinder on CUDA" if torch.cuda.is_available() else "\n[PYTORCH] Cylinder on CPU")
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

        cyl_factory = SimpleShapeFactory("cylinder", embed_dim=3, resolution=200)
        cylinder = cyl_factory.build(backend="torch", device=device, validate=True)

        print(f"  Device: {cylinder.device}")
        print(f"  Dtype: {cylinder.dtype}")
        print(f"  Shape: {cylinder.shape}")

    print("\n" + "=" * 70)
    print("SimpleShapeFactory ready for geometric processing pipelines")
    print("=" * 70)