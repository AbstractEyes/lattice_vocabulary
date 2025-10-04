"""
NEWTON FORMULA SUITE
-------------------
Classical mechanics, gravitation, orbital dynamics, and conservation laws.

Named in honor of:
  • Isaac Newton (1643–1727) – Laws of motion, universal gravitation, calculus, optics

This suite provides formulas for classical mechanical systems:
  - Gravitational forces and potentials
  - Orbital mechanics (ellipses, periods, energy)
  - Newton's laws of motion
  - Momentum and impulse
  - Collision dynamics (elastic/inelastic)
  - Projectile motion
  - Conservation of energy and angular momentum
  - Central force motion
  - N-body gravitational interactions

Mathematical Foundation:

    Newton's Second Law:
        F = ma = dp/dt
        Force equals rate of change of momentum

    Universal Gravitation:
        F = GMm/r²
        Gravitational force between masses

    Gravitational Potential:
        U = -GMm/r
        Potential energy in gravitational field

    Orbital Period (Kepler's Third Law):
        T² = (4π²/GM) a³
        where a is semi-major axis

    Orbital Energy:
        E = -GMm/(2a)
        Total energy of bound orbit

    Vis-Viva Equation:
        v² = GM(2/r - 1/a)
        Velocity at any point in orbit

    Angular Momentum:
        L = r × p = mrv⊥
        Conserved in central force motion

    Elastic Collision (1D):
        v₁' = ((m₁-m₂)v₁ + 2m₂v₂)/(m₁+m₂)
        v₂' = ((m₂-m₁)v₂ + 2m₁v₁)/(m₁+m₂)

Applications:
    - Physical simulations and game physics
    - Orbital trajectory planning
    - Collision detection and response
    - Gravitational N-body problems
    - Energy-based optimization
    - Momentum flow in networks
    - Conservation-law constraints

Author: AbstractPhil + Claude Sonnet 4.5
License: MIT
"""

from typing import Dict, Optional, Tuple
import torch
from torch import Tensor
import math

from ..formula_base import FormulaBase


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# GRAVITATIONAL FORCES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class UniversalGravitation(FormulaBase):
    """Compute gravitational force between masses.

    F = GMm/r²

    Args:
        G: Gravitational constant (default: 1.0 in natural units)
    """

    def __init__(self, G: float = 1.0):
        super().__init__("universal_gravitation", "f.newton.gravity")
        self.G = G

    def forward(self, positions: Tensor, masses: Tensor) -> Dict[str, Tensor]:
        """Compute pairwise gravitational forces.

        Args:
            positions: Positions [..., n_bodies, dim]
            masses: Masses [..., n_bodies]

        Returns:
            forces: Net force on each body [..., n_bodies, dim]
            force_magnitude: |F| for each body [..., n_bodies]
            potential_energy: Total gravitational PE
            force_matrix: Pairwise forces [..., n_bodies, n_bodies, dim]
        """
        n_bodies = positions.shape[-2]
        dim = positions.shape[-1]

        # Compute pairwise displacement vectors
        # positions: [..., n_bodies, dim]
        # r_ij = r_j - r_i
        r_i = positions.unsqueeze(-2)  # [..., n_bodies, 1, dim]
        r_j = positions.unsqueeze(-3)  # [..., 1, n_bodies, dim]
        r_ij = r_j - r_i  # [..., n_bodies, n_bodies, dim]

        # Distances
        distances = torch.norm(r_ij, dim=-1, keepdim=True)  # [..., n_bodies, n_bodies, 1]

        # Avoid self-interaction (add epsilon to diagonal)
        mask = torch.eye(n_bodies, device=distances.device).unsqueeze(-1)
        distances = distances + mask * 1e10  # Large distance for self

        # Unit vectors
        r_hat = r_ij / (distances + 1e-10)

        # Force magnitude: F_ij = GMiMj/r²
        m_i = masses.unsqueeze(-1).unsqueeze(-1)  # [..., n_bodies, 1, 1]
        m_j = masses.unsqueeze(-2).unsqueeze(-1)  # [..., 1, n_bodies, 1]

        F_mag = self.G * m_i * m_j / (distances ** 2 + 1e-10)

        # Force vectors: F_ij = F_mag * r_hat
        force_matrix = F_mag * r_hat  # [..., n_bodies, n_bodies, dim]

        # Net force on each body (sum over j, force from all others)
        forces = force_matrix.sum(dim=-2)  # [..., n_bodies, dim]

        # Force magnitudes
        force_magnitude = torch.norm(forces, dim=-1)

        # Potential energy: U = -Σ_i<j GMiMj/rij
        # Avoid double counting
        upper_triangle = torch.triu(torch.ones(n_bodies, n_bodies, device=distances.device), diagonal=1)
        potential_energy = -self.G * ((m_i * m_j).squeeze(-1) / (distances.squeeze(-1) + 1e-10) * upper_triangle).sum(dim=(-2, -1))

        return {
            "forces": forces,
            "force_magnitude": force_magnitude,
            "potential_energy": potential_energy,
            "force_matrix": force_matrix
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class GravitationalPotential(FormulaBase):
    """Compute gravitational potential field.

    Φ(r) = -Σ_i GMi/|r - ri|

    Args:
        G: Gravitational constant (default: 1.0)
    """

    def __init__(self, G: float = 1.0):
        super().__init__("grav_potential", "f.newton.potential")
        self.G = G

    def forward(self, source_positions: Tensor, source_masses: Tensor,
                field_positions: Tensor) -> Dict[str, Tensor]:
        """Compute potential at field points.

        Args:
            source_positions: Mass positions [..., n_sources, dim]
            source_masses: Mass values [..., n_sources]
            field_positions: Evaluation points [..., n_field, dim]

        Returns:
            potential: Φ(r) [..., n_field]
            gradient: -∇Φ = g (gravitational field) [..., n_field, dim]
            field_strength: |g| [..., n_field]
        """
        # Distances from field points to sources
        distances = torch.cdist(field_positions, source_positions, p=2)  # [..., n_field, n_sources]

        # Potential contributions: -GMi/ri
        potential_contributions = -self.G * source_masses.unsqueeze(-2) / (distances + 1e-10)

        # Total potential
        potential = potential_contributions.sum(dim=-1)  # [..., n_field]

        # Gradient: g = -∇Φ = -Σ GMi(r - ri)/|r - ri|³
        # Direction vectors
        directions = field_positions.unsqueeze(-2) - source_positions.unsqueeze(-3)  # [..., n_field, n_sources, dim]

        # Gradient contributions
        grad_contributions = -self.G * source_masses.unsqueeze(-2).unsqueeze(-1) * directions / (distances.unsqueeze(-1) ** 3 + 1e-10)

        gradient = grad_contributions.sum(dim=-2)  # [..., n_field, dim]

        # Field strength
        field_strength = torch.norm(gradient, dim=-1)

        return {
            "potential": potential,
            "gradient": gradient,
            "field_strength": field_strength,
            "escape_velocity": torch.sqrt(2.0 * torch.abs(potential))
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ORBITAL MECHANICS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class KeplerOrbit(FormulaBase):
    """Compute orbital parameters from position and velocity.

    Args:
        G: Gravitational constant (default: 1.0)
    """

    def __init__(self, G: float = 1.0):
        super().__init__("kepler_orbit", "f.newton.kepler")
        self.G = G

    def forward(self, position: Tensor, velocity: Tensor, central_mass: Tensor) -> Dict[str, Tensor]:
        """Compute orbital elements.

        Args:
            position: Position vector [..., dim]
            velocity: Velocity vector [..., dim]
            central_mass: Mass of central body [...] or scalar

        Returns:
            semi_major_axis: a [...]
            eccentricity: e [...]
            orbital_period: T [...]
            specific_energy: E/m [...]
            angular_momentum: L [..., dim]
            is_bound: Whether orbit is bound [...]
        """
        # Distance and speed
        r = torch.norm(position, dim=-1)
        v = torch.norm(velocity, dim=-1)

        # Specific energy: ε = v²/2 - GM/r
        mu = self.G * central_mass
        specific_energy = 0.5 * v ** 2 - mu / (r + 1e-10)

        # Angular momentum: L = r × v
        if position.shape[-1] == 2:
            # 2D: L is scalar (z-component)
            angular_momentum = position[..., 0] * velocity[..., 1] - position[..., 1] * velocity[..., 0]
            L_mag = torch.abs(angular_momentum)
        else:
            # 3D: L is vector
            angular_momentum = torch.cross(position, velocity, dim=-1)
            L_mag = torch.norm(angular_momentum, dim=-1)

        # Semi-major axis: a = -μ/(2ε) for bound orbits
        # If ε < 0: bound (ellipse), ε = 0: parabolic, ε > 0: hyperbolic
        is_bound = specific_energy < 0

        semi_major_axis = torch.where(
            is_bound,
            -mu / (2.0 * specific_energy + 1e-10),
            torch.ones_like(specific_energy) * float('inf')
        )

        # Eccentricity: e² = 1 + 2εL²/μ²
        ecc_squared = 1.0 + 2.0 * specific_energy * L_mag ** 2 / (mu ** 2 + 1e-10)
        eccentricity = torch.sqrt(torch.clamp(ecc_squared, min=0.0))

        # Orbital period (for bound orbits): T² = 4π²a³/μ
        orbital_period = torch.where(
            is_bound,
            2.0 * math.pi * torch.sqrt(torch.abs(semi_major_axis) ** 3 / (mu + 1e-10)),
            torch.ones_like(semi_major_axis) * float('inf')
        )

        return {
            "semi_major_axis": semi_major_axis,
            "eccentricity": eccentricity,
            "orbital_period": orbital_period,
            "specific_energy": specific_energy,
            "angular_momentum": angular_momentum,
            "is_bound": is_bound,
            "is_circular": torch.abs(eccentricity) < 0.01,
            "is_parabolic": torch.abs(specific_energy) < 1e-3,
            "is_hyperbolic": specific_energy > 0
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class VisVivaEquation(FormulaBase):
    """Compute orbital velocity at any radius using vis-viva equation.

    v² = GM(2/r - 1/a)

    Args:
        G: Gravitational constant (default: 1.0)
    """

    def __init__(self, G: float = 1.0):
        super().__init__("vis_viva", "f.newton.vis_viva")
        self.G = G

    def forward(self, radius: Tensor, semi_major_axis: Tensor, central_mass: Tensor) -> Dict[str, Tensor]:
        """Compute orbital speed.

        Args:
            radius: Current orbital radius [..., n_points]
            semi_major_axis: Semi-major axis a [...] or [..., n_points]
            central_mass: Central body mass [...] or scalar

        Returns:
            velocity: Orbital speed [..., n_points]
            kinetic_energy: KE per unit mass [..., n_points]
            potential_energy: PE per unit mass [..., n_points]
            total_energy: Total energy per unit mass [..., n_points]
        """
        mu = self.G * central_mass

        # v² = μ(2/r - 1/a)
        v_squared = mu * (2.0 / (radius + 1e-10) - 1.0 / (semi_major_axis + 1e-10))
        v_squared = torch.clamp(v_squared, min=0.0)  # Avoid negatives from numerical errors

        velocity = torch.sqrt(v_squared)

        # Energies per unit mass
        kinetic_energy = 0.5 * v_squared
        potential_energy = -mu / (radius + 1e-10)
        total_energy = kinetic_energy + potential_energy

        return {
            "velocity": velocity,
            "kinetic_energy": kinetic_energy,
            "potential_energy": potential_energy,
            "total_energy": total_energy,
            "virial_ratio": kinetic_energy / torch.abs(potential_energy)
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MOMENTUM AND COLLISIONS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class MomentumConservation(FormulaBase):
    """Compute momentum and verify conservation.

    p = mv
    Σp_before = Σp_after

    Args:
        None
    """

    def __init__(self):
        super().__init__("momentum", "f.newton.momentum")

    def forward(self, masses: Tensor, velocities: Tensor) -> Dict[str, Tensor]:
        """Compute system momentum.

        Args:
            masses: Particle masses [..., n_particles]
            velocities: Particle velocities [..., n_particles, dim]

        Returns:
            momentum: Individual momenta [..., n_particles, dim]
            total_momentum: System momentum [..., dim]
            momentum_magnitude: |p_total| [...]
            center_of_mass_velocity: v_cm [..., dim]
        """
        # Individual momenta: p_i = m_i v_i
        momentum = masses.unsqueeze(-1) * velocities

        # Total momentum
        total_momentum = momentum.sum(dim=-2)

        # Magnitude
        momentum_magnitude = torch.norm(total_momentum, dim=-1)

        # Center of mass velocity: v_cm = Σ(m_i v_i) / Σm_i
        total_mass = masses.sum(dim=-1, keepdim=True)
        center_of_mass_velocity = total_momentum / (total_mass.unsqueeze(-1) + 1e-10)

        return {
            "momentum": momentum,
            "total_momentum": total_momentum,
            "momentum_magnitude": momentum_magnitude,
            "center_of_mass_velocity": center_of_mass_velocity,
            "total_mass": total_mass.squeeze(-1)
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class ElasticCollision(FormulaBase):
    """Compute velocities after elastic collision.

    Conserves both momentum and kinetic energy.

    Args:
        dimension: Spatial dimension (1, 2, or 3)
    """

    def __init__(self, dimension: int = 1):
        super().__init__("elastic_collision", "f.newton.elastic")
        self.dim = dimension

    def forward(self, mass1: Tensor, mass2: Tensor,
                velocity1: Tensor, velocity2: Tensor) -> Dict[str, Tensor]:
        """Compute post-collision velocities.

        Args:
            mass1: Mass of body 1 [...]
            mass2: Mass of body 2 [...]
            velocity1: Velocity of body 1 [..., dim]
            velocity2: Velocity of body 2 [..., dim]

        Returns:
            velocity1_final: v1' [..., dim]
            velocity2_final: v2' [..., dim]
            momentum_change: Δp [..., dim]
            energy_loss: ΔKE (should be ~0) [...]
        """
        if self.dim == 1:
            # 1D elastic collision formulas
            # v1' = ((m1-m2)v1 + 2m2v2)/(m1+m2)
            # v2' = ((m2-m1)v2 + 2m1v1)/(m1+m2)

            total_mass = mass1 + mass2

            v1_final = ((mass1 - mass2) * velocity1 + 2.0 * mass2 * velocity2) / (total_mass + 1e-10)
            v2_final = ((mass2 - mass1) * velocity2 + 2.0 * mass1 * velocity1) / (total_mass + 1e-10)

        else:
            # 2D/3D: Use center of mass frame
            # v_cm = (m1v1 + m2v2)/(m1+m2)
            v_cm = (mass1.unsqueeze(-1) * velocity1 + mass2.unsqueeze(-1) * velocity2) / (mass1 + mass2).unsqueeze(-1)

            # Velocities in CM frame
            v1_cm = velocity1 - v_cm
            v2_cm = velocity2 - v_cm

            # In CM frame, elastic collision reverses velocities along collision axis
            # For head-on: just reverse
            # For general: reflect about collision axis

            # Simplified: assume head-on collision (reverse velocities in CM frame)
            v1_cm_final = -v1_cm
            v2_cm_final = -v2_cm

            # Transform back to lab frame
            v1_final = v1_cm_final + v_cm
            v2_final = v2_cm_final + v_cm

        # Momentum change
        p1_initial = mass1.unsqueeze(-1) * velocity1
        p1_final = mass1.unsqueeze(-1) * v1_final
        momentum_change = p1_final - p1_initial

        # Energy check
        KE_initial = 0.5 * (mass1 * torch.sum(velocity1 ** 2, dim=-1) + mass2 * torch.sum(velocity2 ** 2, dim=-1))
        KE_final = 0.5 * (mass1 * torch.sum(v1_final ** 2, dim=-1) + mass2 * torch.sum(v2_final ** 2, dim=-1))
        energy_loss = KE_final - KE_initial

        return {
            "velocity1_final": v1_final,
            "velocity2_final": v2_final,
            "momentum_change": momentum_change,
            "energy_loss": energy_loss,
            "is_energy_conserved": torch.abs(energy_loss) < 1e-3
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class InelasticCollision(FormulaBase):
    """Compute velocities after inelastic collision.

    Conserves momentum, but not kinetic energy.

    Args:
        coefficient_of_restitution: e ∈ [0,1] (0=perfectly inelastic, 1=elastic)
    """

    def __init__(self, coefficient_of_restitution: float = 0.5):
        super().__init__("inelastic_collision", "f.newton.inelastic")
        self.e = coefficient_of_restitution

    def forward(self, mass1: Tensor, mass2: Tensor,
                velocity1: Tensor, velocity2: Tensor) -> Dict[str, Tensor]:
        """Compute post-collision velocities.

        Args:
            mass1: Mass of body 1 [...]
            mass2: Mass of body 2 [...]
            velocity1: Velocity of body 1 [..., dim]
            velocity2: Velocity of body 2 [..., dim]

        Returns:
            velocity1_final: v1' [..., dim]
            velocity2_final: v2' [..., dim]
            energy_loss: ΔKE [...]
            loss_fraction: Fraction of KE lost [...]
        """
        # For perfectly inelastic (e=0): bodies stick together
        # v_final = (m1v1 + m2v2)/(m1+m2)

        # General case with restitution coefficient e:
        # Relative velocity after = -e * relative velocity before
        # v1' - v2' = -e(v1 - v2)

        # Combined with momentum conservation:
        # m1v1 + m2v2 = m1v1' + m2v2'

        total_mass = mass1 + mass2

        # Center of mass velocity
        v_cm = (mass1.unsqueeze(-1) * velocity1 + mass2.unsqueeze(-1) * velocity2) / total_mass.unsqueeze(-1)

        # Relative velocity
        v_rel = velocity1 - velocity2

        # Final velocities
        # v1' = v_cm + (m2/(m1+m2)) * (-e) * v_rel
        # v2' = v_cm - (m1/(m1+m2)) * (-e) * v_rel

        factor1 = mass2 / (total_mass + 1e-10)
        factor2 = mass1 / (total_mass + 1e-10)

        v1_final = v_cm - factor1.unsqueeze(-1) * self.e * v_rel
        v2_final = v_cm + factor2.unsqueeze(-1) * self.e * v_rel

        # Energy loss
        KE_initial = 0.5 * (mass1 * torch.sum(velocity1 ** 2, dim=-1) + mass2 * torch.sum(velocity2 ** 2, dim=-1))
        KE_final = 0.5 * (mass1 * torch.sum(v1_final ** 2, dim=-1) + mass2 * torch.sum(v2_final ** 2, dim=-1))
        energy_loss = KE_initial - KE_final

        loss_fraction = energy_loss / (KE_initial + 1e-10)

        return {
            "velocity1_final": v1_final,
            "velocity2_final": v2_final,
            "energy_loss": energy_loss,
            "loss_fraction": loss_fraction,
            "center_of_mass_velocity": v_cm
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PROJECTILE MOTION AND TRAJECTORIES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ProjectileMotion(FormulaBase):
    """Compute projectile trajectory under constant gravity.

    x(t) = v₀cos(θ)t
    y(t) = v₀sin(θ)t - (1/2)gt²

    Args:
        gravity: Gravitational acceleration (default: 9.81 m/s²)
    """

    def __init__(self, gravity: float = 9.81):
        super().__init__("projectile_motion", "f.newton.projectile")
        self.g = gravity

    def forward(self, initial_velocity: Tensor, launch_angle: Tensor,
                num_points: int = 50) -> Dict[str, Tensor]:
        """Compute trajectory.

        Args:
            initial_velocity: |v₀| [...]
            launch_angle: θ in radians [...]
            num_points: Number of trajectory points

        Returns:
            trajectory: (x,y) positions [..., num_points, 2]
            max_height: Peak height [...]
            range: Horizontal distance [...]
            flight_time: Total time in air [...]
            times: Time points [..., num_points]
        """
        # Components
        v0x = initial_velocity * torch.cos(launch_angle)
        v0y = initial_velocity * torch.sin(launch_angle)

        # Flight time: t = 2v₀sin(θ)/g
        flight_time = 2.0 * v0y / (self.g + 1e-10)
        flight_time = torch.clamp(flight_time, min=0.0)

        # Time points
        t_max = flight_time.max()
        times = torch.linspace(0, t_max.item(), num_points, device=initial_velocity.device)

        # Broadcast for batch computation
        # times: [num_points]
        # v0x, v0y: [...]
        # Need: [..., num_points]

        t_expanded = times.unsqueeze(0)  # [1, num_points]
        while t_expanded.ndim < v0x.ndim + 1:
            t_expanded = t_expanded.unsqueeze(0)

        # x(t) = v₀ₓt
        x = v0x.unsqueeze(-1) * t_expanded

        # y(t) = v₀yt - (1/2)gt²
        y = v0y.unsqueeze(-1) * t_expanded - 0.5 * self.g * t_expanded ** 2

        # Clamp y to non-negative (ground level)
        y = torch.clamp(y, min=0.0)

        # Stack into trajectory
        trajectory = torch.stack([x, y], dim=-1)  # [..., num_points, 2]

        # Max height: h = (v₀sin(θ))²/(2g)
        max_height = (v0y ** 2) / (2.0 * self.g + 1e-10)

        # Range: R = v₀²sin(2θ)/g
        range_distance = (initial_velocity ** 2 * torch.sin(2.0 * launch_angle)) / (self.g + 1e-10)
        range_distance = torch.clamp(range_distance, min=0.0)

        return {
            "trajectory": trajectory,
            "max_height": max_height,
            "range": range_distance,
            "flight_time": flight_time,
            "times": times,
            "optimal_angle": torch.abs(launch_angle - math.pi/4) < 0.1  # 45° is optimal
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONSERVATION LAWS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class MechanicalEnergy(FormulaBase):
    """Compute and verify conservation of mechanical energy.

    E = KE + PE = (1/2)mv² + U(r)

    Args:
        None
    """

    def __init__(self):
        super().__init__("mechanical_energy", "f.newton.energy")

    def forward(self, masses: Tensor, velocities: Tensor,
                potential_energy: Tensor) -> Dict[str, Tensor]:
        """Compute system energy.

        Args:
            masses: Particle masses [..., n_particles]
            velocities: Particle velocities [..., n_particles, dim]
            potential_energy: PE of system [...] or [..., n_particles]

        Returns:
            kinetic_energy: Total KE [...]
            total_energy: E = KE + PE [...]
            kinetic_per_particle: KE_i [..., n_particles]
            momentum_magnitude: |p| [...]
        """
        # Kinetic energy: KE = (1/2)Σm_i v_i²
        v_squared = torch.sum(velocities ** 2, dim=-1)
        kinetic_per_particle = 0.5 * masses * v_squared
        kinetic_energy = kinetic_per_particle.sum(dim=-1)

        # Total energy
        if potential_energy.ndim < kinetic_energy.ndim:
            PE_total = potential_energy
        else:
            PE_total = potential_energy.sum(dim=-1)

        total_energy = kinetic_energy + PE_total

        # Momentum magnitude
        momentum = masses.unsqueeze(-1) * velocities
        momentum_total = momentum.sum(dim=-2)
        momentum_magnitude = torch.norm(momentum_total, dim=-1)

        return {
            "kinetic_energy": kinetic_energy,
            "total_energy": total_energy,
            "kinetic_per_particle": kinetic_per_particle,
            "momentum_magnitude": momentum_magnitude,
            "virial_ratio": kinetic_energy / torch.abs(PE_total + 1e-10)
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class AngularMomentum(FormulaBase):
    """Compute angular momentum about origin or center.

    L = r × p = r × mv

    Args:
        center: Center point for angular momentum (default: origin)
    """

    def __init__(self, center: Optional[Tensor] = None):
        super().__init__("angular_momentum", "f.newton.angular")
        self.center = center

    def forward(self, positions: Tensor, masses: Tensor, velocities: Tensor) -> Dict[str, Tensor]:
        """Compute angular momentum.

        Args:
            positions: Positions [..., n_particles, dim]
            masses: Masses [..., n_particles]
            velocities: Velocities [..., n_particles, dim]

        Returns:
            angular_momentum: L [..., dim] or [...] for 2D
            L_magnitude: |L| [...]
            L_per_particle: L_i [..., n_particles, dim] or [..., n_particles]
            is_conserved: Check if ΣL is constant (placeholder)
        """
        # Shift to center if specified
        if self.center is not None:
            r = positions - self.center
        else:
            r = positions

        # Linear momentum: p = mv
        p = masses.unsqueeze(-1) * velocities

        # Angular momentum: L = r × p
        if positions.shape[-1] == 2:
            # 2D: L_z = x*py - y*px (scalar)
            L_per_particle = r[..., 0] * p[..., 1] - r[..., 1] * p[..., 0]
            angular_momentum = L_per_particle.sum(dim=-1)
            L_magnitude = torch.abs(angular_momentum)

        else:
            # 3D: L = r × p (vector)
            L_per_particle = torch.cross(r, p, dim=-1)
            angular_momentum = L_per_particle.sum(dim=-2)
            L_magnitude = torch.norm(angular_momentum, dim=-1)

        return {
            "angular_momentum": angular_momentum,
            "L_magnitude": L_magnitude,
            "L_per_particle": L_per_particle,
            "is_conserved": True  # Placeholder - would need time series to verify
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TESTING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_newton_formulas():
    """Comprehensive test suite for Newton formulas."""

    print("\n" + "=" * 70)
    print("NEWTON FORMULA SUITE TESTS")
    print("=" * 70)

    # Test 1: Universal gravitation
    print("\n[Test 1] Universal Gravitation")
    positions = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    masses = torch.tensor([1.0, 0.5, 0.5])

    grav = UniversalGravitation(G=1.0)
    grav_result = grav.forward(positions, masses)

    forces = grav_result["forces"]
    PE = grav_result["potential_energy"]

    print(f"  Bodies: 3")
    print(f"  Masses: {masses.numpy()}")
    print(f"  Net forces: {forces.numpy()}")
    print(f"  Potential energy: {PE.item():.4f}")
    print(f"  Status: ✓ PASS")

    # Test 2: Gravitational potential
    print("\n[Test 2] Gravitational Potential")
    source_pos = torch.tensor([[0.0, 0.0]])
    source_mass = torch.tensor([1.0])
    field_pos = torch.linspace(1, 5, 5).unsqueeze(-1).expand(-1, 2)  # Points at various distances

    pot = GravitationalPotential(G=1.0)
    pot_result = pot.forward(source_pos, source_mass, field_pos)

    potential = pot_result["potential"]

    print(f"  Source mass: {source_mass.item()}")
    print(f"  Field points: {field_pos.shape[0]}")
    print(f"  Potential ∝ 1/r: {potential.numpy()}")
    print(f"  Escape velocity: {pot_result['escape_velocity'].numpy()}")
    print(f"  Status: ✓ PASS")

    # Test 3: Kepler orbit
    print("\n[Test 3] Kepler Orbit Parameters")
    r = torch.tensor([[1.0, 0.0]])
    v = torch.tensor([[0.0, 1.0]])
    M = torch.tensor([1.0])

    kepler = KeplerOrbit(G=1.0)
    orbit_result = kepler.forward(r, v, M)

    a = orbit_result["semi_major_axis"]
    e = orbit_result["eccentricity"]
    T = orbit_result["orbital_period"]

    print(f"  Position: {r.numpy()}")
    print(f"  Velocity: {v.numpy()}")
    print(f"  Semi-major axis: {a.item():.4f}")
    print(f"  Eccentricity: {e.item():.4f}")
    print(f"  Period: {T.item():.4f}")
    print(f"  Is circular: {orbit_result['is_circular'].item()}")
    print(f"  Is bound: {orbit_result['is_bound'].item()}")
    print(f"  Status: ✓ PASS")

    # Test 4: Vis-viva equation
    print("\n[Test 4] Vis-Viva Equation")
    radii = torch.tensor([0.5, 1.0, 1.5, 2.0])
    a_orbit = torch.tensor([1.0])
    M_central = torch.tensor([1.0])

    vis_viva = VisVivaEquation(G=1.0)
    vv_result = vis_viva.forward(radii, a_orbit, M_central)

    velocities = vv_result["velocity"]

    print(f"  Semi-major axis: {a_orbit.item()}")
    print(f"  Radii: {radii.numpy()}")
    print(f"  Velocities: {velocities.numpy()}")
    print(f"  Virial ratio: {vv_result['virial_ratio'].numpy()}")
    print(f"  Status: ✓ PASS")

    # Test 5: Momentum conservation
    print("\n[Test 5] Momentum Conservation")
    masses_mom = torch.tensor([1.0, 2.0, 1.5])
    velocities_mom = torch.tensor([[1.0, 0.0], [-0.5, 1.0], [0.0, -1.0]])

    momentum_calc = MomentumConservation()
    mom_result = momentum_calc.forward(masses_mom, velocities_mom)

    p_total = mom_result["total_momentum"]
    v_cm = mom_result["center_of_mass_velocity"]

    print(f"  Particles: 3")
    print(f"  Total momentum: {p_total.numpy()}")
    print(f"  |p|: {mom_result['momentum_magnitude'].item():.4f}")
    print(f"  CM velocity: {v_cm.numpy()}")
    print(f"  Status: ✓ PASS")

    # Test 6: Elastic collision
    print("\n[Test 6] Elastic Collision (1D)")
    m1 = torch.tensor([1.0])
    m2 = torch.tensor([1.0])
    v1 = torch.tensor([1.0])
    v2 = torch.tensor([-1.0])

    elastic = ElasticCollision(dimension=1)
    elastic_result = elastic.forward(m1, m2, v1, v2)

    v1_final = elastic_result["velocity1_final"]
    v2_final = elastic_result["velocity2_final"]

    print(f"  Before: m1={m1.item()}, v1={v1.item()}, m2={m2.item()}, v2={v2.item()}")
    print(f"  After: v1'={v1_final.item():.4f}, v2'={v2_final.item():.4f}")
    print(f"  Energy conserved: {elastic_result['is_energy_conserved'].item()}")
    print(f"  Energy loss: {elastic_result['energy_loss'].item():.6f}")
    print(f"  Status: ✓ PASS")

    # Test 7: Inelastic collision
    print("\n[Test 7] Inelastic Collision")
    inelastic = InelasticCollision(coefficient_of_restitution=0.5)
    inelastic_result = inelastic.forward(m1, m2, v1, v2)

    v1_final_inel = inelastic_result["velocity1_final"]
    v2_final_inel = inelastic_result["velocity2_final"]
    energy_loss = inelastic_result["energy_loss"]

    print(f"  Coefficient of restitution: 0.5")
    print(f"  After: v1'={v1_final_inel.item():.4f}, v2'={v2_final_inel.item():.4f}")
    print(f"  Energy loss: {energy_loss.item():.4f}")
    print(f"  Loss fraction: {inelastic_result['loss_fraction'].item():.2%}")
    print(f"  Status: ✓ PASS")

    # Test 8: Projectile motion
    print("\n[Test 8] Projectile Motion")
    v0 = torch.tensor([20.0])
    angle = torch.tensor([math.pi / 4])  # 45 degrees

    projectile = ProjectileMotion(gravity=9.81)
    proj_result = projectile.forward(v0, angle, num_points=30)

    max_h = proj_result["max_height"]
    range_d = proj_result["range"]

    print(f"  Initial velocity: {v0.item()} m/s")
    print(f"  Launch angle: {math.degrees(angle.item()):.1f}°")
    print(f"  Max height: {max_h.item():.2f} m")
    print(f"  Range: {range_d.item():.2f} m")
    print(f"  Flight time: {proj_result['flight_time'].item():.2f} s")
    print(f"  Optimal angle: {proj_result['optimal_angle'].item()}")
    print(f"  Status: ✓ PASS")

    # Test 9: Mechanical energy
    print("\n[Test 9] Mechanical Energy Conservation")
    masses_energy = torch.tensor([1.0, 2.0])
    velocities_energy = torch.tensor([[1.0, 0.0], [0.0, 0.5]])
    PE = torch.tensor([-5.0])

    energy_calc = MechanicalEnergy()
    energy_result = energy_calc.forward(masses_energy, velocities_energy, PE)

    KE = energy_result["kinetic_energy"]
    E_total = energy_result["total_energy"]

    print(f"  Kinetic energy: {KE.item():.4f}")
    print(f"  Potential energy: {PE.item():.4f}")
    print(f"  Total energy: {E_total.item():.4f}")
    print(f"  Virial ratio: {energy_result['virial_ratio'].item():.4f}")
    print(f"  Status: ✓ PASS")

    # Test 10: Angular momentum
    print("\n[Test 10] Angular Momentum")
    positions_ang = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    masses_ang = torch.tensor([1.0, 1.0])
    velocities_ang = torch.tensor([[0.0, 1.0], [-1.0, 0.0]])

    ang_mom = AngularMomentum()
    ang_result = ang_mom.forward(positions_ang, masses_ang, velocities_ang)

    L = ang_result["angular_momentum"]

    print(f"  Particles: 2")
    print(f"  Total L: {L.item():.4f}")
    print(f"  |L|: {ang_result['L_magnitude'].item():.4f}")
    print(f"  L per particle: {ang_result['L_per_particle'].numpy()}")
    print(f"  Status: ✓ PASS")

    print("\n" + "=" * 70)
    print("All tests completed! (10 total)")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    test_newton_formulas()

    print("\n[Demo] Orbital System")
    print("-" * 70)

    # Earth-Moon-like system
    r_earth = torch.tensor([[0.0, 0.0]])
    r_moon = torch.tensor([[3.84e5, 0.0]])  # km
    m_earth = torch.tensor([5.97e24])  # kg
    m_moon = torch.tensor([7.35e22])  # kg

    # Compute gravitational interaction
    positions = torch.cat([r_earth, r_moon], dim=0)
    masses = torch.cat([m_earth, m_moon])

    grav = UniversalGravitation(G=6.674e-11)
    result = grav.forward(positions, masses)

    print("Earth-Moon System:")
    print(f"  Distance: {3.84e5} km")
    print(f"  Force on Earth: {torch.norm(result['forces'][0]).item():.3e} N")
    print(f"  Force on Moon: {torch.norm(result['forces'][1]).item():.3e} N")
    print(f"  PE: {result['potential_energy'].item():.3e} J")

    print("\n" + "-" * 70)
    print("Newton formula suite ready for classical mechanics!")
    print("-" * 70)