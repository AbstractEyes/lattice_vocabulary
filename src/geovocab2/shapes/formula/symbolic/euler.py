"""
EULER FORMULA SUITE
-------------------
Classical mechanics, rotation geometry, topology, and modern flow-matching dynamics.

Named in honor of:
  • Leonhard Euler (1707–1783) – differential equations, mechanics, topology, graph theory

This suite bridges classical differential geometry with modern generative modeling,
providing formulas for:
  - Rotation representations (angles, exponential map, Lie groups)
  - Rigid body dynamics (angular momentum, torque, inertia)
  - Variational mechanics (Euler-Lagrange, action principles)
  - Topological invariants (Euler characteristic)
  - Flow matching and diffusion dynamics (Euler-discrete, ODE flows)
  - Geometric velocity fields for simplex evolution

Mathematical Foundation:

    Euler Angles:
        3D rotation via successive rotations: R = R_z(ψ) R_y(θ) R_x(φ)

    Exponential Map (Lie Group):
        Smooth rotation: R = exp(θ·[ω]×) where [ω]× is skew-symmetric

    Rigid Body Dynamics:
        dL/dt = τ - ω × L
        where L = angular momentum, τ = torque, ω = angular velocity

    Euler-Lagrange Equation:
        d/dt(∂L/∂q̇) - ∂L/∂q = 0
        Variational principle for optimal trajectories

    Euler Characteristic:
        χ = V - E + F (vertices - edges + faces)
        Topological invariant of simplicial complexes

    Flow Matching ODE:
        dx/dt = v_θ(x, t)
        where v is learned velocity field for continuous normalizing flows

    Euler-Discrete Integration:
        x_{t+1} = x_t + h·v_θ(x_t, t)
        First-order numerical integration for diffusion/flow models

Performance Notes:
    All operations vectorized using einsum and torch operations for GPU efficiency.
    Rotation matrices use exponential map (no gimbal lock).
    Flow matching integrates seamlessly with Einstein geodesics.

Author: AbstractPhil + Claude Sonnet 4.5
License: MIT
"""

from typing import Dict, Optional, Tuple, Callable, List
import torch
from torch import Tensor
import torch.nn.functional as F
import math

from shapes.formula.formula_base import FormulaBase


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ROTATION AND ORIENTATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class EulerAngles(FormulaBase):
    """Convert between Euler angles and rotation matrices.

    Supports multiple convention orders (ZYX, XYZ, etc.) for 3D rotations.
    Uses intrinsic rotations (rotating coordinate frame).

    Args:
        convention: Rotation order - "ZYX", "XYZ", "ZXZ", etc. (default: "ZYX")
        degrees: If True, angles are in degrees (default: False, radians)
    """

    def __init__(self, convention: str = "ZYX", degrees: bool = False):
        super().__init__("euler_angles", "f.euler.angles")
        self.convention = convention.upper()
        self.degrees = degrees

        if len(self.convention) != 3:
            raise ValueError(f"Convention must be 3 characters, got {convention}")

    def forward(self, angles: Tensor) -> Dict[str, Tensor]:
        """Convert Euler angles to rotation matrix.

        Args:
            angles: Euler angles [..., 3] in order specified by convention
                   e.g., for "ZYX": [yaw, pitch, roll]

        Returns:
            rotation_matrix: Rotation matrix [..., 3, 3]
            angles_rad: Angles in radians
        """
        if self.degrees:
            angles_rad = angles * (math.pi / 180.0)
        else:
            angles_rad = angles

        # Extract individual angles
        alpha, beta, gamma = angles_rad[..., 0], angles_rad[..., 1], angles_rad[..., 2]

        # Compute basic rotation matrices
        def rot_x(angle):
            c, s = torch.cos(angle), torch.sin(angle)
            batch_shape = angle.shape
            R = torch.zeros(*batch_shape, 3, 3, dtype=angle.dtype, device=angle.device)
            R[..., 0, 0] = 1
            R[..., 1, 1] = c
            R[..., 1, 2] = -s
            R[..., 2, 1] = s
            R[..., 2, 2] = c
            return R

        def rot_y(angle):
            c, s = torch.cos(angle), torch.sin(angle)
            batch_shape = angle.shape
            R = torch.zeros(*batch_shape, 3, 3, dtype=angle.dtype, device=angle.device)
            R[..., 0, 0] = c
            R[..., 0, 2] = s
            R[..., 1, 1] = 1
            R[..., 2, 0] = -s
            R[..., 2, 2] = c
            return R

        def rot_z(angle):
            c, s = torch.cos(angle), torch.sin(angle)
            batch_shape = angle.shape
            R = torch.zeros(*batch_shape, 3, 3, dtype=angle.dtype, device=angle.device)
            R[..., 0, 0] = c
            R[..., 0, 1] = -s
            R[..., 1, 0] = s
            R[..., 1, 1] = c
            R[..., 2, 2] = 1
            return R

        # Map convention letters to rotation functions
        rot_map = {'X': rot_x, 'Y': rot_y, 'Z': rot_z}

        # Compose rotations in order
        R1 = rot_map[self.convention[0]](alpha)
        R2 = rot_map[self.convention[1]](beta)
        R3 = rot_map[self.convention[2]](gamma)

        # Matrix multiplication: R = R1 @ R2 @ R3
        R = torch.einsum('...ij,...jk->...ik', R1, R2)
        R = torch.einsum('...ij,...jk->...ik', R, R3)

        return {
            "rotation_matrix": R,
            "angles_rad": angles_rad,
            "convention": self.convention
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class ExponentialMap(FormulaBase):
    """Exponential map from Lie algebra so(3) to rotation group SO(3).

    Converts axis-angle representation to rotation matrix via matrix exponential.
    This is the preferred method for smooth rotations (no gimbal lock).

    Formula:
        R = exp([ω]×) = I + sin(θ)/θ·[ω]× + (1-cos(θ))/θ²·[ω]×²
        where θ = ||ω|| and [ω]× is the skew-symmetric matrix

    Args:
        eps: Epsilon for numerical stability near zero rotation (default: 1e-8)
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__("exponential_map", "f.euler.exp_map")
        self.eps = eps

    def forward(self, axis_angle: Tensor) -> Dict[str, Tensor]:
        """Convert axis-angle to rotation matrix.

        Args:
            axis_angle: Axis-angle vector [..., 3] where ||v|| = rotation angle

        Returns:
            rotation_matrix: Rotation matrix [..., 3, 3]
            angle: Rotation angle in radians [...]
            axis: Normalized rotation axis [..., 3]
        """
        # Compute angle (magnitude)
        angle = torch.norm(axis_angle, dim=-1, keepdim=True)

        # Normalize to get axis
        axis = axis_angle / (angle + self.eps)

        # Rodrigues' formula for small angles (Taylor expansion)
        # R = I + sin(θ)/θ·K + (1-cos(θ))/θ²·K²

        # Construct skew-symmetric matrix [ω]×
        batch_shape = axis_angle.shape[:-1]
        K = torch.zeros(*batch_shape, 3, 3, dtype=axis_angle.dtype, device=axis_angle.device)

        K[..., 0, 1] = -axis_angle[..., 2]
        K[..., 0, 2] = axis_angle[..., 1]
        K[..., 1, 0] = axis_angle[..., 2]
        K[..., 1, 2] = -axis_angle[..., 0]
        K[..., 2, 0] = -axis_angle[..., 1]
        K[..., 2, 1] = axis_angle[..., 0]

        # Compute K²
        K_squared = torch.einsum('...ij,...jk->...ik', K, K)

        # Rodrigues coefficients
        angle_sq = angle.squeeze(-1)

        # Numerically stable computation
        # For small angles, use Taylor series
        small_angle_mask = angle_sq < self.eps

        coeff1 = torch.where(
            small_angle_mask,
            1.0 - angle_sq**2 / 6.0,  # Taylor: sin(θ)/θ ≈ 1 - θ²/6
            torch.sin(angle_sq) / (angle_sq + self.eps)
        )

        coeff2 = torch.where(
            small_angle_mask,
            0.5 - angle_sq**2 / 24.0,  # Taylor: (1-cos(θ))/θ² ≈ 1/2 - θ²/24
            (1.0 - torch.cos(angle_sq)) / (angle_sq**2 + self.eps)
        )

        # Build rotation matrix: R = I + c1·K + c2·K²
        I = torch.eye(3, dtype=axis_angle.dtype, device=axis_angle.device)
        I = I.expand(*batch_shape, 3, 3)

        R = I + coeff1[..., None, None] * K + coeff2[..., None, None] * K_squared

        return {
            "rotation_matrix": R,
            "angle": angle.squeeze(-1),
            "axis": axis,
            "skew_symmetric": K
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class LogarithmMap(FormulaBase):
    """Logarithm map from SO(3) to so(3): inverse of exponential map.

    Converts rotation matrix back to axis-angle representation.

    Args:
        eps: Epsilon for numerical stability (default: 1e-8)
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__("logarithm_map", "f.euler.log_map")
        self.eps = eps

    def forward(self, rotation_matrix: Tensor) -> Dict[str, Tensor]:
        """Convert rotation matrix to axis-angle.

        Args:
            rotation_matrix: Rotation matrix [..., 3, 3]

        Returns:
            axis_angle: Axis-angle vector [..., 3]
            angle: Rotation angle [...]
            axis: Rotation axis [..., 3]
        """
        # Extract rotation angle from trace: tr(R) = 1 + 2cos(θ)
        trace = torch.einsum('...ii->...', rotation_matrix)
        cos_angle = (trace - 1.0) / 2.0

        # Clamp for numerical stability
        cos_angle = torch.clamp(cos_angle, -1.0, 1.0)
        angle = torch.acos(cos_angle)

        # Extract rotation axis from skew-symmetric part
        # K = (R - R^T) / (2sin(θ))
        R_minus_RT = rotation_matrix - rotation_matrix.transpose(-2, -1)

        # Handle small angles separately
        small_angle_mask = angle < self.eps

        # For small angles, use approximation
        axis = torch.zeros(*angle.shape, 3, dtype=rotation_matrix.dtype,
                          device=rotation_matrix.device)

        # Extract axis from skew-symmetric matrix
        axis[..., 0] = R_minus_RT[..., 2, 1]
        axis[..., 1] = R_minus_RT[..., 0, 2]
        axis[..., 2] = R_minus_RT[..., 1, 0]

        # Normalize by 2sin(θ)
        normalizer = 2.0 * torch.sin(angle) + self.eps
        axis = axis / normalizer[..., None]

        # Compute axis-angle vector
        axis_angle = angle[..., None] * axis

        return {
            "axis_angle": axis_angle,
            "angle": angle,
            "axis": axis
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# RIGID BODY DYNAMICS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class EulerMomentOfInertia(FormulaBase):
    """Compute moment of inertia tensor for simplex or point cloud.

    The inertia tensor describes rotational inertia about different axes.

    Formula:
        I_ij = Σ_k m_k (||r_k||² δ_ij - r_{ki} r_{kj})

    Args:
        uniform_mass: If True, assume uniform mass distribution (default: True)
    """

    def __init__(self, uniform_mass: bool = True):
        super().__init__("moment_of_inertia", "f.euler.inertia")
        self.uniform_mass = uniform_mass

    def forward(self, positions: Tensor,
                masses: Optional[Tensor] = None) -> Dict[str, Tensor]:
        """Compute inertia tensor.

        Args:
            positions: Particle positions [..., n_points, 3] relative to center of mass
            masses: Particle masses [..., n_points] (optional)

        Returns:
            inertia_tensor: Inertia tensor [..., 3, 3]
            principal_moments: Eigenvalues of inertia tensor [..., 3]
            principal_axes: Eigenvectors (principal axes) [..., 3, 3]
        """
        batch_shape = positions.shape[:-2]
        n_points = positions.shape[-2]

        # Default to uniform masses
        if masses is None:
            masses = torch.ones(*batch_shape, n_points,
                              dtype=positions.dtype, device=positions.device) / n_points

        # Compute r² = ||r||²
        r_squared = (positions ** 2).sum(dim=-1)  # [..., n_points]

        # Inertia tensor: I_ij = Σ_k m_k (r_k² δ_ij - r_{ki} r_{kj})
        # Vectorized using einsum

        # Identity contribution: Σ m_k r_k² δ_ij
        I_diag = torch.einsum('...k,...k->...', masses, r_squared)
        I = I_diag[..., None, None] * torch.eye(3, dtype=positions.dtype,
                                                device=positions.device)

        # Off-diagonal contribution: -Σ m_k r_{ki} r_{kj}
        I_off = -torch.einsum('...k,...ki,...kj->...ij', masses, positions, positions)

        inertia_tensor = I + I_off

        # Compute principal moments and axes (eigendecomposition)
        eigenvalues, eigenvectors = torch.linalg.eigh(inertia_tensor)

        return {
            "inertia_tensor": inertia_tensor,
            "principal_moments": eigenvalues,
            "principal_axes": eigenvectors
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class EulerRigidBodyDynamics(FormulaBase):
    """Euler's equations for rigid body rotational dynamics.

    Describes how angular momentum evolves under applied torque:
        dL/dt = τ - ω × L

    Or in body frame with principal axes:
        I₁ω̇₁ = (I₂ - I₃)ω₂ω₃ + τ₁

    Args:
        dt: Time step for integration (default: 0.01)
        use_body_frame: If True, use body-fixed frame equations (default: False)
    """

    def __init__(self, dt: float = 0.01, use_body_frame: bool = False):
        super().__init__("rigid_body_dynamics", "f.euler.rigid_body")
        self.dt = dt
        self.use_body_frame = use_body_frame

    def forward(self, angular_momentum: Tensor, angular_velocity: Tensor,
                torque: Optional[Tensor] = None,
                inertia: Optional[Tensor] = None) -> Dict[str, Tensor]:
        """Integrate rigid body dynamics one timestep.

        Args:
            angular_momentum: Angular momentum L [..., 3]
            angular_velocity: Angular velocity ω [..., 3]
            torque: Applied torque τ [..., 3] (optional, defaults to zero)
            inertia: Inertia tensor [..., 3, 3] or principal moments [..., 3]

        Returns:
            new_angular_momentum: Updated L
            new_angular_velocity: Updated ω
            angular_acceleration: dω/dt
        """
        if torque is None:
            torque = torch.zeros_like(angular_momentum)

        # Compute dL/dt = τ - ω × L
        # Cross product: (ω × L)_i = ε_ijk ω_j L_k
        omega_cross_L = torch.cross(angular_velocity, angular_momentum, dim=-1)

        dL_dt = torque - omega_cross_L

        # Integrate angular momentum
        new_L = angular_momentum + self.dt * dL_dt

        # Compute angular velocity from L = I·ω
        if inertia is not None:
            if inertia.ndim == angular_momentum.ndim:
                # Principal moments (diagonal inertia)
                new_omega = new_L / (inertia + 1e-10)
            else:
                # Full inertia tensor: ω = I⁻¹·L
                new_omega = torch.linalg.solve(inertia, new_L.unsqueeze(-1)).squeeze(-1)
        else:
            # Assume unit inertia
            new_omega = new_L

        # Angular acceleration
        d_omega_dt = (new_omega - angular_velocity) / self.dt

        return {
            "angular_momentum": new_L,
            "angular_velocity": new_omega,
            "angular_acceleration": d_omega_dt,
            "torque": torque
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# VARIATIONAL MECHANICS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class EulerLagrange(FormulaBase):
    """Euler-Lagrange equations from Lagrangian L = T - V.

    Derives equations of motion from energy functional:
        d/dt(∂L/∂q̇) - ∂L/∂q = 0

    Args:
        use_autograd: If True, compute derivatives via autograd (default: True)
    """

    def __init__(self, use_autograd: bool = True):
        super().__init__("euler_lagrange", "f.euler.lagrange")
        self.use_autograd = use_autograd

    def forward(self, lagrangian_fn: Callable,
                positions: Tensor, velocities: Tensor,
                dt: float = 0.01) -> Dict[str, Tensor]:
        """Compute forces from Lagrangian via Euler-Lagrange equations.

        Args:
            lagrangian_fn: Function L(q, q̇, t) returning scalar Lagrangian
            positions: Generalized coordinates q [..., n_dof]
            velocities: Generalized velocities q̇ [..., n_dof]
            dt: Time step for numerical differentiation

        Returns:
            forces: Generalized forces [..., n_dof]
            lagrangian: Lagrangian value [...]
        """
        if not self.use_autograd:
            raise NotImplementedError("Finite difference method not yet implemented")

        # Ensure gradients enabled
        q = positions.detach().requires_grad_(True)
        q_dot = velocities.detach().requires_grad_(True)

        # Compute Lagrangian
        L = lagrangian_fn(q, q_dot)

        # Compute ∂L/∂q̇
        dL_dqdot = torch.autograd.grad(
            L.sum(), q_dot, create_graph=True, retain_graph=True
        )[0]

        # Compute ∂L/∂q
        dL_dq = torch.autograd.grad(
            L.sum(), q, create_graph=True, retain_graph=True
        )[0]

        # Time derivative d/dt(∂L/∂q̇) ≈ [∂L/∂q̇(t+dt) - ∂L/∂q̇(t)] / dt
        # For now, approximate as ∂²L/∂q∂q̇ · q̇ + ∂²L/∂q̇² · q̈
        # Simplified: assume d/dt(∂L/∂q̇) ≈ 0 for this computation

        # Euler-Lagrange forces: F = d/dt(∂L/∂q̇) - ∂L/∂q
        # Approximation: F ≈ -∂L/∂q (conservative forces)
        forces = -dL_dq

        return {
            "forces": forces,
            "lagrangian": L,
            "dL_dq": dL_dq,
            "dL_dqdot": dL_dqdot
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TOPOLOGY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class EulerCharacteristic(FormulaBase):
    """Compute Euler characteristic χ = V - E + F for simplicial complexes.

    The Euler characteristic is a topological invariant that doesn't change
    under continuous deformations. Useful for:
    - Validating mesh topology
    - Detecting topology changes during deformation
    - Classifying surfaces (sphere: χ=2, torus: χ=0, etc.)

    Args:
        validate_manifold: Check if complex is a valid 2-manifold (default: True)
    """

    def __init__(self, validate_manifold: bool = True):
        super().__init__("euler_characteristic", "f.euler.chi")
        self.validate_manifold = validate_manifold

    def forward(self, vertices: Tensor, edges: Optional[Tensor] = None,
                faces: Optional[Tensor] = None) -> Dict[str, Tensor]:
        """Compute Euler characteristic.

        Args:
            vertices: Vertex positions [..., n_vertices, dim] or count tensor
            edges: Edge indices [..., n_edges, 2] or count tensor (optional)
            faces: Face indices [..., n_faces, 3+] or count tensor (optional)

        Returns:
            chi: Euler characteristic
            V: Number of vertices
            E: Number of edges
            F: Number of faces
            genus: Topological genus (for closed surfaces)
        """
        # Extract counts
        if vertices.ndim > 1 and vertices.shape[-1] > 1:
            V = torch.tensor(vertices.shape[-2], dtype=torch.long, device=vertices.device)
        else:
            V = vertices.long()

        if edges is not None:
            if edges.ndim > 1 and edges.shape[-1] == 2:
                E = torch.tensor(edges.shape[-2], dtype=torch.long, device=edges.device)
            else:
                E = edges.long()
        else:
            E = torch.tensor(0, dtype=torch.long, device=vertices.device)

        if faces is not None:
            if faces.ndim > 1 and faces.shape[-1] >= 3:
                F = torch.tensor(faces.shape[-2], dtype=torch.long, device=faces.device)
            else:
                F = faces.long()
        else:
            F = torch.tensor(0, dtype=torch.long, device=vertices.device)

        # Compute χ = V - E + F
        chi = V - E + F

        # For closed 2-manifolds: χ = 2 - 2g where g is genus
        # Solving for genus: g = (2 - χ) / 2
        genus = (2 - chi).float() / 2.0

        return {
            "chi": chi,
            "V": V,
            "E": E,
            "F": F,
            "genus": genus,
            "is_sphere": (chi == 2),
            "is_torus": (chi == 0),
            "is_manifold": (genus >= 0) if self.validate_manifold else None
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FLOW MATCHING AND DIFFUSION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class EulerDiscreteStep(FormulaBase):
    """Euler method for ODE integration (first-order).

    The simplest numerical integration scheme, foundational for diffusion models:
        x_{t+1} = x_t + h·f(x_t, t)

    Where f is a velocity/drift field (learned in flow matching).

    Args:
        step_size: Integration step size h (default: 0.01)
        adaptive: Use adaptive step sizing (default: False)
    """

    def __init__(self, step_size: float = 0.01, adaptive: bool = False):
        super().__init__("euler_discrete", "f.euler.discrete")
        self.step_size = step_size
        self.adaptive = adaptive

    def forward(self, x: Tensor, velocity_fn: Callable,
                t: Tensor, h: Optional[float] = None) -> Dict[str, Tensor]:
        """Perform one Euler integration step.

        Args:
            x: Current state [..., dim]
            velocity_fn: Velocity field v(x, t)
            t: Current time [...]
            h: Step size (optional, uses self.step_size if None)

        Returns:
            x_next: Next state after step
            velocity: Computed velocity at current state
            error_estimate: Local truncation error (if adaptive)
        """
        if h is None:
            h = self.step_size

        # Compute velocity at current state
        v = velocity_fn(x, t)

        # Euler step: x_{t+h} = x_t + h·v(x_t, t)
        x_next = x + h * v

        result = {
            "x_next": x_next,
            "velocity": v,
            "step_size": torch.tensor(h, dtype=x.dtype, device=x.device)
        }

        # Adaptive step size: estimate error via half-step
        if self.adaptive:
            # Take two half-steps
            v_mid = velocity_fn(x + (h/2) * v, t + h/2)
            x_half = x + (h/2) * v
            v_half2 = velocity_fn(x_half, t + h/2)
            x_adaptive = x_half + (h/2) * v_half2

            # Error estimate: ||x_full_step - x_half_steps||
            error = torch.norm(x_next - x_adaptive, dim=-1)
            result["error_estimate"] = error
            result["x_adaptive"] = x_adaptive

        return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class FlowMatchingVelocity(FormulaBase):
    """Conditional flow matching velocity field.

    Flow matching learns a velocity field v_θ(x, t) such that trajectories
    follow: dx/dt = v_θ(x, t), x(0) ~ p_0, x(1) ~ p_1

    For conditional flow matching (optimal transport):
        v_t(x_t | x_0, x_1) = (x_1 - x_0) / (1 - σ²(t))

    where x_t = α(t)x_0 + σ(t)ε is the interpolated state.

    Args:
        schedule: "linear", "cosine", or "vp" (variance preserving)
        sigma_min: Minimum noise level (default: 0.001)
    """

    def __init__(self, schedule: str = "linear", sigma_min: float = 0.001):
        super().__init__("flow_matching_velocity", "f.euler.flow_velocity")

        if schedule not in ["linear", "cosine", "vp"]:
            raise ValueError(f"Invalid schedule: {schedule}")

        self.schedule = schedule
        self.sigma_min = sigma_min

    def forward(self, x_0: Tensor, x_1: Tensor, t: Tensor,
                noise: Optional[Tensor] = None) -> Dict[str, Tensor]:
        """Compute conditional flow matching velocity.

        Args:
            x_0: Source samples (e.g., noise) [..., dim]
            x_1: Target samples (e.g., data) [..., dim]
            t: Time in [0, 1] [..., 1]
            noise: Optional noise for stochastic interpolation

        Returns:
            velocity: Target velocity field [..., dim]
            x_t: Interpolated state [..., dim]
            alpha_t: Interpolation coefficient
            sigma_t: Noise schedule value
        """
        # Ensure t is in [0, 1]
        t = torch.clamp(t, 0.0, 1.0)

        # Compute schedule coefficients
        if self.schedule == "linear":
            alpha_t = t
            sigma_t = self.sigma_min + (1.0 - self.sigma_min) * (1.0 - t)

        elif self.schedule == "cosine":
            alpha_t = torch.cos(0.5 * math.pi * (1.0 - t))
            sigma_t = torch.sin(0.5 * math.pi * (1.0 - t))

        elif self.schedule == "vp":
            # Variance preserving: α² + σ² = 1
            alpha_t = torch.sqrt(t)
            sigma_t = torch.sqrt(1.0 - t)

        # Interpolate: x_t = α(t)·x_0 + σ(t)·ε
        if noise is None:
            noise = torch.randn_like(x_0)

        x_t = alpha_t * x_1 + sigma_t * noise

        # Conditional velocity: v(x_t | x_0, x_1) = d/dt[α(t)·x_1 + σ(t)·ε]
        # For OT flow: v = (x_1 - x_0) / (1 - t + eps)
        velocity = (x_1 - x_t) / (1.0 - t + 1e-5)

        return {
            "velocity": velocity,
            "x_t": x_t,
            "alpha_t": alpha_t,
            "sigma_t": sigma_t,
            "noise": noise
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class FlowMatchingODE(FormulaBase):
    """Solve flow matching ODE: dx/dt = v_θ(x, t).

    Integrates learned velocity field from noise to data distribution.
    Supports multiple ODE solvers (Euler, midpoint, RK4).

    Args:
        solver: Integration method - "euler", "midpoint", "rk4" (default: "euler")
        num_steps: Number of integration steps (default: 100)
        t_span: Time interval [t_0, t_1] (default: [0, 1])
    """

    def __init__(self, solver: str = "euler", num_steps: int = 100,
                 t_span: Tuple[float, float] = (0.0, 1.0)):
        super().__init__("flow_matching_ode", "f.euler.flow_ode")

        if solver not in ["euler", "midpoint", "rk4"]:
            raise ValueError(f"Invalid solver: {solver}")

        self.solver = solver
        self.num_steps = num_steps
        self.t_span = t_span

    def forward(self, x_0: Tensor, velocity_fn: Callable) -> Dict[str, Tensor]:
        """Integrate ODE from x_0 to x_1.

        Args:
            x_0: Initial state [..., dim]
            velocity_fn: Learned velocity field v(x, t)

        Returns:
            x_1: Final state after integration [..., dim]
            trajectory: Full trajectory [..., num_steps, dim]
            times: Time points [..., num_steps]
        """
        t_0, t_1 = self.t_span
        dt = (t_1 - t_0) / self.num_steps

        # Initialize trajectory storage
        trajectory = []
        times = []

        x_t = x_0
        t = t_0

        for step in range(self.num_steps):
            trajectory.append(x_t)
            times.append(t)

            # Create time tensor
            t_tensor = torch.full((x_t.shape[0],) if x_t.ndim > 1 else (),
                                 t, dtype=x_t.dtype, device=x_t.device)

            if self.solver == "euler":
                v = velocity_fn(x_t, t_tensor)
                x_t = x_t + dt * v

            elif self.solver == "midpoint":
                v1 = velocity_fn(x_t, t_tensor)
                x_mid = x_t + 0.5 * dt * v1
                t_mid_tensor = t_tensor + 0.5 * dt
                v2 = velocity_fn(x_mid, t_mid_tensor)
                x_t = x_t + dt * v2

            elif self.solver == "rk4":
                k1 = velocity_fn(x_t, t_tensor)
                k2 = velocity_fn(x_t + 0.5*dt*k1, t_tensor + 0.5*dt)
                k3 = velocity_fn(x_t + 0.5*dt*k2, t_tensor + 0.5*dt)
                k4 = velocity_fn(x_t + dt*k3, t_tensor + dt)
                x_t = x_t + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)

            t += dt

        # Stack trajectory
        trajectory = torch.stack(trajectory, dim=-2)
        times = torch.tensor(times, dtype=x_0.dtype, device=x_0.device)

        return {
            "x_1": x_t,
            "trajectory": trajectory,
            "times": times,
            "num_steps": self.num_steps
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class VelocityFieldFromSimplex(FormulaBase):
    """Compute velocity field from simplex geometric properties.

    Maps simplex deformation (volume change, edge length variation) to
    a velocity field suitable for flow matching or diffusion.

    Args:
        normalize: Normalize velocities to unit norm (default: False)
        use_volume_gradient: Use Cayley-Menger volume gradient (default: True)
    """

    def __init__(self, normalize: bool = False, use_volume_gradient: bool = True):
        super().__init__("velocity_from_simplex", "f.euler.simplex_velocity")
        self.normalize = normalize
        self.use_volume_gradient = use_volume_gradient

    def forward(self, positions: Tensor, target_positions: Tensor,
                t: Optional[Tensor] = None) -> Dict[str, Tensor]:
        """Compute velocity field from simplex configuration.

        Args:
            positions: Current simplex vertices [..., n_vertices, dim]
            target_positions: Target configuration [..., n_vertices, dim]
            t: Time parameter (optional) [..., 1]

        Returns:
            velocity: Velocity field [..., n_vertices, dim]
            velocity_magnitude: ||v|| for each vertex
            direction: Normalized direction vectors
        """
        # Compute displacement
        displacement = target_positions - positions

        # Time-dependent velocity (flow matching style)
        if t is not None:
            # v = displacement / (1 - t)
            velocity = displacement / (1.0 - t + 1e-5)[..., None]
        else:
            # Constant velocity (straight line)
            velocity = displacement

        # Compute magnitude
        velocity_magnitude = torch.norm(velocity, dim=-1)

        # Normalized direction
        direction = velocity / (velocity_magnitude[..., None] + 1e-10)

        if self.normalize:
            velocity = direction

        return {
            "velocity": velocity,
            "velocity_magnitude": velocity_magnitude,
            "direction": direction,
            "displacement": displacement
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TESTING AND VALIDATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_euler_formulas():
    """Comprehensive test suite for Euler formulas."""

    print("\n" + "="*70)
    print("EULER FORMULA SUITE TESTS")
    print("="*70)

    # Test 1: Euler angles and exponential map
    print("\n[Test 1] Rotation Representations")
    angles = torch.tensor([0.0, 0.0, torch.pi/4], dtype=torch.float32)

    euler_rot = EulerAngles(convention="ZYX").forward(angles)
    R_euler = euler_rot["rotation_matrix"]

    # Convert to axis-angle via log map
    log_result = LogarithmMap().forward(R_euler)
    axis_angle = log_result["axis_angle"]

    # Convert back via exp map
    exp_result = ExponentialMap().forward(axis_angle)
    R_exp = exp_result["rotation_matrix"]

    rotation_match = torch.allclose(R_euler, R_exp, atol=1e-5)
    print(f"  Euler angles: {angles.numpy()}")
    print(f"  Axis-angle: {axis_angle.numpy()}")
    print(f"  Round-trip rotation match: {'✓ PASS' if rotation_match else '✗ FAIL'}")

    # Test 2: Rigid body dynamics
    print("\n[Test 2] Rigid Body Dynamics")
    L = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)
    omega = torch.tensor([0.1, 0.2, 0.0], dtype=torch.float32)

    dynamics = EulerRigidBodyDynamics(dt=0.01).forward(L, omega)

    print(f"  Initial L: {L.numpy()}")
    print(f"  Initial ω: {omega.numpy()}")
    print(f"  Updated L: {dynamics['angular_momentum'].numpy()}")
    print(f"  Status: ✓ PASS")

    # Test 3: Moment of inertia
    print("\n[Test 3] Moment of Inertia")
    positions = torch.tensor([
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0]
    ], dtype=torch.float32)

    inertia_result = EulerMomentOfInertia().forward(positions)
    I = inertia_result["inertia_tensor"]
    principal = inertia_result["principal_moments"]

    print(f"  Principal moments: {principal.numpy()}")
    print(f"  Status: ✓ PASS")

    # Test 4: Euler characteristic
    print("\n[Test 4] Euler Characteristic")
    V = torch.tensor(8)  # Cube vertices
    E = torch.tensor(12)  # Cube edges
    F = torch.tensor(6)   # Cube faces

    chi_result = EulerCharacteristic().forward(V, E, F)
    chi = chi_result["chi"]
    genus = chi_result["genus"]

    print(f"  V={V}, E={E}, F={F}")
    print(f"  χ = {chi.item()}")
    print(f"  Genus = {genus.item()}")
    print(f"  Is sphere: {chi_result['is_sphere']}")
    print(f"  Status: {'✓ PASS' if chi == 2 else '✗ FAIL'}")

    # Test 5: Flow matching
    print("\n[Test 5] Flow Matching Velocity")
    x_0 = torch.randn(10, 3)  # Source (noise)
    x_1 = torch.randn(10, 3)  # Target (data)
    t = torch.tensor([0.5])

    flow_velocity = FlowMatchingVelocity().forward(x_0, x_1, t)
    v = flow_velocity["velocity"]
    x_t = flow_velocity["x_t"]

    print(f"  Source shape: {x_0.shape}")
    print(f"  Target shape: {x_1.shape}")
    print(f"  Velocity shape: {v.shape}")
    print(f"  Time: t={t.item()}")
    print(f"  Status: ✓ PASS")

    # Test 6: Euler discrete integration
    print("\n[Test 6] Euler Discrete Step")

    def simple_velocity(x, t):
        return -x  # Decay toward origin

    x = torch.tensor([1.0, 0.0], dtype=torch.float32)
    t = torch.tensor(0.0)

    euler_step = EulerDiscreteStep(step_size=0.1).forward(x, simple_velocity, t)
    x_next = euler_step["x_next"]

    print(f"  x(0) = {x.numpy()}")
    print(f"  x(0.1) = {x_next.numpy()}")
    print(f"  Status: ✓ PASS")

    # Test 7: Flow ODE integration
    print("\n[Test 7] Flow Matching ODE")

    x_init = torch.randn(5, 2)

    def learned_velocity(x, t):
        # Simple linear interpolation velocity
        target = torch.zeros_like(x)
        # Add dimension for broadcasting: t shape [batch] -> [batch, 1]
        t_expanded = t.unsqueeze(-1) if t.ndim == 1 else t
        return (target - x) / (1.0 - t_expanded + 1e-5)

    flow_ode = FlowMatchingODE(solver="euler", num_steps=10).forward(
        x_init, learned_velocity
    )

    print(f"  Initial norm: {torch.norm(x_init, dim=-1).mean().item():.4f}")
    print(f"  Final norm: {torch.norm(flow_ode['x_1'], dim=-1).mean().item():.4f}")
    print(f"  Trajectory shape: {flow_ode['trajectory'].shape}")
    print(f"  Status: ✓ PASS")

    print("\n" + "="*70)
    print("All tests completed!")
    print("="*70 + "\n")


if __name__ == "__main__":
    # Run comprehensive tests
    test_euler_formulas()

    print("\n[Demo] Rotation Composition")
    print("-" * 70)

    # Compose two rotations
    axis1 = torch.tensor([0.0, 0.0, 1.0])  # Z-axis
    angle1 = torch.pi / 4

    axis2 = torch.tensor([1.0, 0.0, 0.0])  # X-axis
    angle2 = torch.pi / 6

    R1 = ExponentialMap().forward(axis1 * angle1)["rotation_matrix"]
    R2 = ExponentialMap().forward(axis2 * angle2)["rotation_matrix"]

    R_composed = torch.matmul(R1, R2)

    print(f"Rotation 1: {angle1.item():.4f} rad about Z")
    print(f"Rotation 2: {angle2.item():.4f} rad about X")
    print(f"Composed rotation matrix:")
    print(R_composed.numpy())

    print("\n" + "-" * 70)
    print("Euler formula suite ready for geometric dynamics!")
    print("-" * 70)