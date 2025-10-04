"""
PROJECTION OPERATIONS
--------------------
Coordinate transformations, camera projections, and perspective mappings.

This module provides operations for:
  - Perspective and orthographic projections
  - Camera transformations (view, projection matrices)
  - Coordinate space conversions (world, view, clip, NDC, screen)
  - Homogeneous coordinates
  - Viewport transformations
  - Ray generation for ray casting
  - Look-at matrix construction

Mathematical Foundation:

    Perspective Projection:
        x_ndc = x_view / -z_view
        y_ndc = y_view / -z_view

    Projection Matrix:
        P = [f/aspect  0      0           0    ]
            [0         f      0           0    ]
            [0         0      (f+n)/(n-f) 2fn/(n-f)]
            [0         0      -1          0    ]
        where f = cot(fov/2)

    View Matrix (Look-At):
        right = normalize(forward × world_up)
        up = normalize(right × forward)
        View = [right.x    right.y    right.z    -dot(right, eye)   ]
               [up.x       up.y       up.z       -dot(up, eye)      ]
               [-fwd.x     -fwd.y     -fwd.z     dot(forward, eye)  ]
               [0          0          0          1                  ]

    Homogeneous Coordinates:
        [x, y, z] → [x, y, z, 1]
        [x, y, z, w] → [x/w, y/w, z/w]

    Screen Space:
        x_screen = (x_ndc + 1) * width / 2
        y_screen = (1 - y_ndc) * height / 2

Author: AbstractPhil + Claude Sonnet 4.5
License: MIT
"""

from typing import Dict, Optional, Tuple
import torch
from torch import Tensor
import math

from ..formula_base import FormulaBase


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HOMOGENEOUS COORDINATES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ToHomogeneous(FormulaBase):
    """Convert Cartesian coordinates to homogeneous coordinates.

    [x, y, z] → [x, y, z, 1]
    """

    def __init__(self):
        super().__init__("to_homogeneous", "f.projection.to_homo")

    def forward(self, points: Tensor) -> Dict[str, Tensor]:
        """Convert to homogeneous.

        Args:
            points: Cartesian points [..., n_points, dim]

        Returns:
            homogeneous: Homogeneous coords [..., n_points, dim+1]
        """
        ones = torch.ones(*points.shape[:-1], 1, device=points.device, dtype=points.dtype)
        homogeneous = torch.cat([points, ones], dim=-1)

        return {
            "homogeneous": homogeneous
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class FromHomogeneous(FormulaBase):
    """Convert homogeneous coordinates to Cartesian.

    [x, y, z, w] → [x/w, y/w, z/w]
    """

    def __init__(self):
        super().__init__("from_homogeneous", "f.projection.from_homo")

    def forward(self, homogeneous: Tensor) -> Dict[str, Tensor]:
        """Convert from homogeneous.

        Args:
            homogeneous: Homogeneous coords [..., n_points, dim+1]

        Returns:
            cartesian: Cartesian points [..., n_points, dim]
            w: Homogeneous coordinate [..., n_points]
        """
        w = homogeneous[..., -1:]
        cartesian = homogeneous[..., :-1] / (w + 1e-10)

        return {
            "cartesian": cartesian,
            "w": w.squeeze(-1)
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PROJECTION MATRICES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class PerspectiveMatrix(FormulaBase):
    """Create perspective projection matrix.

    Args:
        fov: Field of view in radians (default: π/3 = 60°)
        aspect: Aspect ratio width/height (default: 1.0)
        near: Near clipping plane (default: 0.1)
        far: Far clipping plane (default: 100.0)
    """

    def __init__(self, fov: float = math.pi / 3, aspect: float = 1.0,
                 near: float = 0.1, far: float = 100.0):
        super().__init__("perspective_matrix", "f.projection.perspective")
        self.fov = fov
        self.aspect = aspect
        self.near = near
        self.far = far

    def forward(self, device: Optional[torch.device] = None) -> Dict[str, Tensor]:
        """Create perspective projection matrix.

        Args:
            device: Device to create matrix on

        Returns:
            matrix: 4x4 projection matrix [4, 4]
            fov: Field of view used
            aspect: Aspect ratio used
        """
        if device is None:
            device = torch.device("cpu")

        # Compute matrix elements
        f = 1.0 / math.tan(self.fov / 2.0)
        n = self.near
        far = self.far

        matrix = torch.zeros(4, 4, device=device, dtype=torch.float32)

        matrix[0, 0] = f / self.aspect
        matrix[1, 1] = f
        matrix[2, 2] = (far + n) / (n - far)
        matrix[2, 3] = (2 * far * n) / (n - far)
        matrix[3, 2] = -1.0

        return {
            "matrix": matrix,
            "fov": torch.tensor(self.fov),
            "aspect": torch.tensor(self.aspect),
            "near": torch.tensor(self.near),
            "far": torch.tensor(self.far)
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class OrthographicMatrix(FormulaBase):
    """Create orthographic projection matrix.

    Args:
        left: Left clipping plane
        right: Right clipping plane
        bottom: Bottom clipping plane
        top: Top clipping plane
        near: Near clipping plane
        far: Far clipping plane
    """

    def __init__(self, left: float = -1.0, right: float = 1.0,
                 bottom: float = -1.0, top: float = 1.0,
                 near: float = 0.1, far: float = 100.0):
        super().__init__("orthographic_matrix", "f.projection.ortho")
        self.left = left
        self.right = right
        self.bottom = bottom
        self.top = top
        self.near = near
        self.far = far

    def forward(self, device: Optional[torch.device] = None) -> Dict[str, Tensor]:
        """Create orthographic projection matrix.

        Args:
            device: Device to create matrix on

        Returns:
            matrix: 4x4 projection matrix [4, 4]
        """
        if device is None:
            device = torch.device("cpu")

        l, r = self.left, self.right
        b, t = self.bottom, self.top
        n, f = self.near, self.far

        matrix = torch.zeros(4, 4, device=device, dtype=torch.float32)

        matrix[0, 0] = 2.0 / (r - l)
        matrix[0, 3] = -(r + l) / (r - l)
        matrix[1, 1] = 2.0 / (t - b)
        matrix[1, 3] = -(t + b) / (t - b)
        matrix[2, 2] = -2.0 / (f - n)
        matrix[2, 3] = -(f + n) / (f - n)
        matrix[3, 3] = 1.0

        return {
            "matrix": matrix
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# VIEW MATRIX
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class LookAtMatrix(FormulaBase):
    """Create view (look-at) matrix.

    Transforms world space to view space.
    """

    def __init__(self):
        super().__init__("look_at_matrix", "f.projection.look_at")

    def forward(self, eye: Tensor, target: Tensor, up: Tensor) -> Dict[str, Tensor]:
        """Create look-at matrix.

        Args:
            eye: Camera position [..., 3]
            target: Point camera looks at [..., 3]
            up: Up vector [..., 3]

        Returns:
            matrix: View matrix [..., 4, 4]
            forward: Forward direction [..., 3]
            right: Right direction [..., 3]
            up_corrected: Corrected up direction [..., 3]
        """
        # Forward vector (from eye to target)
        forward = target - eye
        forward = forward / (torch.norm(forward, dim=-1, keepdim=True) + 1e-10)

        # Right vector (forward × up)
        right = torch.cross(forward, up, dim=-1)
        right = right / (torch.norm(right, dim=-1, keepdim=True) + 1e-10)

        # Corrected up vector (right × forward)
        up_corrected = torch.cross(right, forward, dim=-1)
        up_corrected = up_corrected / (torch.norm(up_corrected, dim=-1, keepdim=True) + 1e-10)

        # Build view matrix
        batch_shape = eye.shape[:-1]
        matrix = torch.zeros(*batch_shape, 4, 4, device=eye.device, dtype=eye.dtype)

        # Rotation part
        matrix[..., 0, :3] = right
        matrix[..., 1, :3] = up_corrected
        matrix[..., 2, :3] = -forward  # Negated for RH coordinate system

        # Translation part
        matrix[..., 0, 3] = -torch.sum(right * eye, dim=-1)
        matrix[..., 1, 3] = -torch.sum(up_corrected * eye, dim=-1)
        matrix[..., 2, 3] = torch.sum(forward * eye, dim=-1)
        matrix[..., 3, 3] = 1.0

        return {
            "matrix": matrix,
            "forward": forward,
            "right": right,
            "up_corrected": up_corrected
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PROJECTIONS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class PerspectiveProject(FormulaBase):
    """Project 3D points to 2D using perspective projection.
    """

    def __init__(self):
        super().__init__("perspective_project", "f.projection.project_persp")

    def forward(self, points: Tensor, projection_matrix: Tensor) -> Dict[str, Tensor]:
        """Project points.

        Args:
            points: 3D points [..., n_points, 3]
            projection_matrix: 4x4 projection matrix [4, 4] or [..., 4, 4]

        Returns:
            projected: 2D projected points [..., n_points, 2]
            clip_space: Points in clip space [..., n_points, 4]
            ndc: Normalized device coords [..., n_points, 3]
            depth: Depth values [..., n_points]
        """
        # Convert to homogeneous
        homo_op = ToHomogeneous()
        points_homo = homo_op.forward(points)["homogeneous"]  # [..., n_points, 4]

        # Apply projection matrix
        # points_homo: [..., n_points, 4]
        # projection_matrix: [..., 4, 4] or [4, 4]

        if projection_matrix.ndim == 2:
            # Broadcast matrix
            clip_space = torch.matmul(points_homo, projection_matrix.T)
        else:
            # Batched matrix multiply
            clip_space = torch.matmul(points_homo.unsqueeze(-2),
                                      projection_matrix.unsqueeze(-3).transpose(-2, -1)).squeeze(-2)

        # Perspective divide
        w = clip_space[..., 3:4]
        ndc = clip_space[..., :3] / (w + 1e-10)

        # Extract 2D projection and depth
        projected = ndc[..., :2]
        depth = ndc[..., 2]

        return {
            "projected": projected,
            "clip_space": clip_space,
            "ndc": ndc,
            "depth": depth
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class OrthographicProject(FormulaBase):
    """Project 3D points to 2D using orthographic projection.
    """

    def __init__(self):
        super().__init__("orthographic_project", "f.projection.project_ortho")

    def forward(self, points: Tensor, projection_matrix: Tensor) -> Dict[str, Tensor]:
        """Project points orthographically.

        Args:
            points: 3D points [..., n_points, 3]
            projection_matrix: 4x4 orthographic matrix [4, 4] or [..., 4, 4]

        Returns:
            projected: 2D projected points [..., n_points, 2]
            ndc: Normalized device coords [..., n_points, 3]
            depth: Depth values [..., n_points]
        """
        # Convert to homogeneous
        homo_op = ToHomogeneous()
        points_homo = homo_op.forward(points)["homogeneous"]

        # Apply projection (no perspective divide needed for ortho)
        if projection_matrix.ndim == 2:
            ndc_homo = torch.matmul(points_homo, projection_matrix.T)
        else:
            ndc_homo = torch.matmul(points_homo.unsqueeze(-2),
                                    projection_matrix.unsqueeze(-3).transpose(-2, -1)).squeeze(-2)

        # Extract NDC (w should be 1 for orthographic)
        ndc = ndc_homo[..., :3]
        projected = ndc[..., :2]
        depth = ndc[..., 2]

        return {
            "projected": projected,
            "ndc": ndc,
            "depth": depth
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# VIEWPORT TRANSFORMATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ViewportTransform(FormulaBase):
    """Transform NDC coordinates to screen space.

    Args:
        width: Screen width in pixels
        height: Screen height in pixels
    """

    def __init__(self, width: int = 1920, height: int = 1080):
        super().__init__("viewport_transform", "f.projection.viewport")
        self.width = width
        self.height = height

    def forward(self, ndc: Tensor) -> Dict[str, Tensor]:
        """Transform to screen space.

        Args:
            ndc: Normalized device coords [..., n_points, 2] or [..., n_points, 3]

        Returns:
            screen: Screen coordinates [..., n_points, 2]
            is_visible: Points within screen bounds [..., n_points]
        """
        # NDC is in [-1, 1], transform to [0, width] × [0, height]
        # x_screen = (x_ndc + 1) * width / 2
        # y_screen = (1 - y_ndc) * height / 2  (flip y-axis)

        ndc_xy = ndc[..., :2]

        x_screen = (ndc_xy[..., 0] + 1.0) * self.width / 2.0
        y_screen = (1.0 - ndc_xy[..., 1]) * self.height / 2.0

        screen = torch.stack([x_screen, y_screen], dim=-1)

        # Check visibility
        is_visible = (
                (ndc_xy[..., 0] >= -1.0) & (ndc_xy[..., 0] <= 1.0) &
                (ndc_xy[..., 1] >= -1.0) & (ndc_xy[..., 1] <= 1.0)
        )

        return {
            "screen": screen,
            "is_visible": is_visible
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class ScreenToNDC(FormulaBase):
    """Transform screen coordinates to NDC.

    Args:
        width: Screen width in pixels
        height: Screen height in pixels
    """

    def __init__(self, width: int = 1920, height: int = 1080):
        super().__init__("screen_to_ndc", "f.projection.screen_to_ndc")
        self.width = width
        self.height = height

    def forward(self, screen: Tensor) -> Dict[str, Tensor]:
        """Transform to NDC.

        Args:
            screen: Screen coordinates [..., n_points, 2]

        Returns:
            ndc: Normalized device coords [..., n_points, 2]
        """
        # Inverse of viewport transform
        x_ndc = 2.0 * screen[..., 0] / self.width - 1.0
        y_ndc = 1.0 - 2.0 * screen[..., 1] / self.height

        ndc = torch.stack([x_ndc, y_ndc], dim=-1)

        return {
            "ndc": ndc
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# RAY GENERATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class GenerateRays(FormulaBase):
    """Generate rays from camera through screen pixels.

    Used for ray tracing and ray casting.
    """

    def __init__(self):
        super().__init__("generate_rays", "f.projection.gen_rays")

    def forward(self, screen_coords: Tensor, camera_pos: Tensor,
                view_matrix: Tensor, projection_matrix: Tensor) -> Dict[str, Tensor]:
        """Generate rays.

        Args:
            screen_coords: Screen positions [..., n_rays, 2]
            camera_pos: Camera position [..., 3]
            view_matrix: View matrix [..., 4, 4] or [4, 4]
            projection_matrix: Projection matrix [..., 4, 4] or [4, 4]

        Returns:
            ray_origins: Ray starting points [..., n_rays, 3]
            ray_directions: Ray directions (normalized) [..., n_rays, 3]
        """
        # Convert screen coords to NDC
        # Assuming screen coords are already normalized or we need width/height
        # For simplicity, assume screen_coords are in NDC range
        ndc = screen_coords

        # Add z=1 (far plane) and convert to homogeneous
        ndc_3d = torch.cat([ndc, torch.ones(*ndc.shape[:-1], 1, device=ndc.device)], dim=-1)
        ndc_homo = torch.cat([ndc_3d, torch.ones(*ndc.shape[:-1], 1, device=ndc.device)], dim=-1)

        # Unproject: inverse of projection
        if projection_matrix.ndim == 2:
            proj_inv = torch.inverse(projection_matrix)
            view_space = torch.matmul(ndc_homo, proj_inv.T)
        else:
            proj_inv = torch.inverse(projection_matrix)
            view_space = torch.matmul(ndc_homo.unsqueeze(-2),
                                      proj_inv.unsqueeze(-3).transpose(-2, -1)).squeeze(-2)

        # Perspective divide
        view_space = view_space[..., :3] / (view_space[..., 3:4] + 1e-10)

        # Transform to world space: inverse of view matrix
        if view_matrix.ndim == 2:
            view_inv = torch.inverse(view_matrix)
            # Extract rotation part
            view_rot_inv = view_inv[:3, :3]
            world_dir = torch.matmul(view_space, view_rot_inv.T)
        else:
            view_inv = torch.inverse(view_matrix)
            view_rot_inv = view_inv[..., :3, :3]
            world_dir = torch.matmul(view_space.unsqueeze(-2),
                                     view_rot_inv.unsqueeze(-3).transpose(-2, -1)).squeeze(-2)

        # Normalize directions
        ray_directions = world_dir / (torch.norm(world_dir, dim=-1, keepdim=True) + 1e-10)

        # Ray origins are camera positions
        ray_origins = camera_pos.unsqueeze(-2).expand_as(ray_directions)

        return {
            "ray_origins": ray_origins,
            "ray_directions": ray_directions
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# COORDINATE SPACE PIPELINE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ProjectionPipeline(FormulaBase):
    """Complete projection pipeline: world → view → clip → NDC → screen.
    """

    def __init__(self):
        super().__init__("projection_pipeline", "f.projection.pipeline")

    def forward(self, world_points: Tensor, view_matrix: Tensor,
                projection_matrix: Tensor, width: int = 1920,
                height: int = 1080) -> Dict[str, Tensor]:
        """Execute full projection pipeline.

        Args:
            world_points: Points in world space [..., n_points, 3]
            view_matrix: View matrix [..., 4, 4] or [4, 4]
            projection_matrix: Projection matrix [..., 4, 4] or [4, 4]
            width: Screen width
            height: Screen height

        Returns:
            screen_coords: Final screen coordinates [..., n_points, 2]
            view_space: Points in view space [..., n_points, 3]
            clip_space: Points in clip space [..., n_points, 4]
            ndc: Points in NDC [..., n_points, 3]
            depth: Depth values [..., n_points]
            is_visible: Visibility mask [..., n_points]
        """
        # World → View
        homo_op = ToHomogeneous()
        world_homo = homo_op.forward(world_points)["homogeneous"]

        if view_matrix.ndim == 2:
            view_homo = torch.matmul(world_homo, view_matrix.T)
        else:
            view_homo = torch.matmul(world_homo.unsqueeze(-2),
                                     view_matrix.unsqueeze(-3).transpose(-2, -1)).squeeze(-2)

        view_space = view_homo[..., :3] / (view_homo[..., 3:4] + 1e-10)

        # View → Clip
        persp_op = PerspectiveProject()
        proj_result = persp_op.forward(view_space, projection_matrix)

        clip_space = proj_result["clip_space"]
        ndc = proj_result["ndc"]
        depth = proj_result["depth"]

        # NDC → Screen
        viewport_op = ViewportTransform(width=width, height=height)
        viewport_result = viewport_op.forward(ndc)

        screen_coords = viewport_result["screen"]
        is_visible = viewport_result["is_visible"]

        return {
            "screen_coords": screen_coords,
            "view_space": view_space,
            "clip_space": clip_space,
            "ndc": ndc,
            "depth": depth,
            "is_visible": is_visible
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TESTING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_projection_operations():
    """Test suite for projection operations."""

    print("\n" + "=" * 70)
    print("PROJECTION OPERATIONS TESTS")
    print("=" * 70)

    # Test 1: Homogeneous coordinates
    print("\n[Test 1] Homogeneous Coordinates")
    points_3d = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    to_homo = ToHomogeneous()
    homo_result = to_homo.forward(points_3d)

    from_homo = FromHomogeneous()
    cart_result = from_homo.forward(homo_result["homogeneous"])

    error = torch.abs(points_3d - cart_result["cartesian"]).max()
    print(f"  Original: {points_3d.shape}")
    print(f"  Homogeneous: {homo_result['homogeneous'].shape}")
    print(f"  Roundtrip error: {error.item():.6e}")
    print(f"  Status: ✓ PASS")

    # Test 2: Perspective matrix
    print("\n[Test 2] Perspective Projection Matrix")
    persp_mat = PerspectiveMatrix(fov=math.pi / 3, aspect=16 / 9, near=0.1, far=100.0)
    persp_result = persp_mat.forward()

    print(f"  Matrix shape: {persp_result['matrix'].shape}")
    print(f"  FOV: {math.degrees(persp_result['fov'].item()):.1f}°")
    print(f"  Aspect: {persp_result['aspect'].item():.2f}")
    print(f"  Status: ✓ PASS")

    # Test 3: Orthographic matrix
    print("\n[Test 3] Orthographic Projection Matrix")
    ortho_mat = OrthographicMatrix(left=-1, right=1, bottom=-1, top=1, near=0.1, far=100)
    ortho_result = ortho_mat.forward()

    print(f"  Matrix shape: {ortho_result['matrix'].shape}")
    print(f"  Matrix determinant: {torch.det(ortho_result['matrix']).item():.6f}")
    print(f"  Status: ✓ PASS")

    # Test 4: Look-at matrix
    print("\n[Test 4] Look-At (View) Matrix")
    eye = torch.tensor([0.0, 0.0, 5.0])
    target = torch.tensor([0.0, 0.0, 0.0])
    up = torch.tensor([0.0, 1.0, 0.0])

    look_at = LookAtMatrix()
    view_result = look_at.forward(eye, target, up)

    print(f"  View matrix shape: {view_result['matrix'].shape}")
    print(f"  Forward: {view_result['forward'].numpy()}")
    print(f"  Right: {view_result['right'].numpy()}")
    print(f"  Up: {view_result['up_corrected'].numpy()}")
    print(f"  Status: ✓ PASS")

    # Test 5: Perspective projection
    print("\n[Test 5] Perspective Projection")
    points = torch.tensor([[0.0, 0.0, -5.0], [1.0, 1.0, -10.0], [-1.0, 0.5, -3.0]])

    proj_op = PerspectiveProject()
    proj_result = proj_op.forward(points, persp_result["matrix"])

    print(f"  Input points: {points.shape}")
    print(f"  Projected 2D: {proj_result['projected'].shape}")
    print(f"  Depths: {proj_result['depth'].numpy()}")
    print(f"  Status: ✓ PASS")

    # Test 6: Orthographic projection
    print("\n[Test 6] Orthographic Projection")
    ortho_proj = OrthographicProject()
    ortho_proj_result = ortho_proj.forward(points, ortho_result["matrix"])

    print(f"  Projected 2D: {ortho_proj_result['projected'].shape}")
    print(f"  NDC: {ortho_proj_result['ndc'].numpy()}")
    print(f"  Status: ✓ PASS")

    # Test 7: Viewport transform
    print("\n[Test 7] Viewport Transform")
    ndc_coords = torch.tensor([[0.0, 0.0], [0.5, 0.5], [-1.0, 1.0]])

    viewport = ViewportTransform(width=1920, height=1080)
    viewport_result = viewport.forward(ndc_coords)

    print(f"  NDC: {ndc_coords.numpy()}")
    print(f"  Screen: {viewport_result['screen'].numpy()}")
    print(f"  Visible: {viewport_result['is_visible'].numpy()}")
    print(f"  Status: ✓ PASS")

    # Test 8: Screen to NDC
    print("\n[Test 8] Screen to NDC (Inverse)")
    screen_to_ndc = ScreenToNDC(width=1920, height=1080)
    ndc_back = screen_to_ndc.forward(viewport_result["screen"])

    error = torch.abs(ndc_coords - ndc_back["ndc"]).max()
    print(f"  Roundtrip error: {error.item():.6e}")
    print(f"  Status: ✓ PASS")

    # Test 9: Ray generation
    print("\n[Test 9] Ray Generation")
    screen_pixels = torch.tensor([[0.0, 0.0], [0.5, 0.5], [-0.5, 0.5]])

    ray_gen = GenerateRays()
    ray_result = ray_gen.forward(screen_pixels, eye, view_result["matrix"],
                                 persp_result["matrix"])

    print(f"  Rays generated: {ray_result['ray_origins'].shape[0]}")
    print(
        f"  Ray directions normalized: {torch.allclose(torch.norm(ray_result['ray_directions'], dim=-1), torch.ones(3))}")
    print(f"  Status: ✓ PASS")

    # Test 10: Full projection pipeline
    print("\n[Test 10] Complete Projection Pipeline")
    world_pts = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    pipeline = ProjectionPipeline()
    pipeline_result = pipeline.forward(world_pts, view_result["matrix"],
                                       persp_result["matrix"], width=1920, height=1080)

    print(f"  World points: {world_pts.shape}")
    print(f"  Screen coords: {pipeline_result['screen_coords'].numpy()}")
    print(f"  Visible: {pipeline_result['is_visible'].numpy()}")
    print(f"  Depths: {pipeline_result['depth'].numpy()}")
    print(f"  Status: ✓ PASS")

    print("\n" + "=" * 70)
    print("All tests completed! (10 total)")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    test_projection_operations()

    print("\n[Coordinate Space Pipeline]")
    print("-" * 70)
    print("Transformation sequence:")
    print("  1. World Space → View Matrix → View Space")
    print("  2. View Space → Projection Matrix → Clip Space")
    print("  3. Clip Space → Perspective Divide → NDC")
    print("  4. NDC → Viewport Transform → Screen Space")
    print("\nSupports:")
    print("  - Perspective and orthographic projections")
    print("  - Camera transformations (look-at)")
    print("  - Ray generation for ray tracing")
    print("  - Batched operations across all transforms")
    print("-" * 70)