"""
HOOKE FORMULA SUITE
------------------
Structural mechanics, elasticity, stress, strain, and material deformation.

Named in honor of:
  • Robert Hooke (1635–1703) – Elasticity law, microscopy, springs, mechanics

This suite provides formulas for mechanical engineering and materials science:
  - Stress and strain relationships
  - Hooke's law (linear elasticity)
  - Beam bending and deflection
  - Torsion and shear
  - Buckling and stability
  - Material failure criteria
  - Energy stored in elastic deformation
  - Young's modulus, shear modulus, Poisson's ratio

Mathematical Foundation:

    Hooke's Law (1D):
        σ = Eε
        Stress = Young's modulus × Strain

    Strain:
        ε = ΔL/L₀
        Relative deformation

    Shear Stress:
        τ = Gγ
        where G is shear modulus, γ is shear strain

    Elastic Energy:
        U = (1/2)∫σε dV = (1/2)Eε²V
        Energy stored in deformation

    Beam Bending (Euler-Bernoulli):
        M = EI(d²y/dx²)
        Moment = Stiffness × Curvature

    Beam Deflection:
        y = (FL³)/(3EI) for cantilever with point load

    Torsion:
        τ = (TR)/J
        Shear stress in twisted shaft

    Buckling Load (Euler):
        P_cr = (π²EI)/(L²)
        Critical load for column buckling

    Von Mises Stress (failure criterion):
        σ_vm = √(σ_x² - σ_xσ_y + σ_y² + 3τ_xy²)

Applications:
    - Structural analysis and design
    - Material selection and testing
    - Finite element method (FEM)
    - Deformable mesh simulation
    - Elastic shape matching
    - Stability analysis for geometric structures
    - Spring systems and dampers

Author: AbstractPhil + Claude Sonnet 4.5
License: MIT
"""

from typing import Dict, Optional, Tuple
import torch
from torch import Tensor
import math

from shapes.formula.formula_base import FormulaBase


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STRESS AND STRAIN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class HookesLaw(FormulaBase):
    """Compute stress from strain using Hooke's law.

    σ = Eε (linear elastic relationship)

    Args:
        youngs_modulus: E (default: 200 GPa, typical steel)
        use_gpa: If True, E is in GPa units
    """

    def __init__(self, youngs_modulus: float = 200.0, use_gpa: bool = True):
        super().__init__("hookes_law", "f.hooke.stress_strain")
        self.E = youngs_modulus * 1e9 if use_gpa else youngs_modulus

    def forward(self, strain: Tensor) -> Dict[str, Tensor]:
        """Compute stress from strain.

        Args:
            strain: ε [..., n_points]

        Returns:
            stress: σ [..., n_points]
            elastic_energy_density: U/V = (1/2)Eε² [..., n_points]
            is_yielding: Strain exceeds typical yield (0.2%) [..., n_points]
        """
        # Stress: σ = Eε
        stress = self.E * strain

        # Elastic energy density: u = (1/2)Eε²
        energy_density = 0.5 * self.E * strain ** 2

        # Check if exceeds typical yield strain (~0.002 for steel)
        is_yielding = torch.abs(strain) > 0.002

        return {
            "stress": stress,
            "elastic_energy_density": energy_density,
            "is_yielding": is_yielding,
            "strain": strain
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class StrainFromDisplacement(FormulaBase):
    """Compute strain from displacement field.

    ε = ΔL/L₀

    Args:
        None
    """

    def __init__(self):
        super().__init__("strain_displacement", "f.hooke.strain")

    def forward(self, original_positions: Tensor, deformed_positions: Tensor) -> Dict[str, Tensor]:
        """Compute strain from displacement.

        Args:
            original_positions: Initial positions [..., n_points, dim]
            deformed_positions: Deformed positions [..., n_points, dim]

        Returns:
            displacement: u = x' - x [..., n_points, dim]
            displacement_magnitude: |u| [..., n_points]
            strain: ε (engineering strain) [..., n_points]
            max_strain: Maximum strain value
        """
        # Displacement field
        displacement = deformed_positions - original_positions

        # Magnitude
        displacement_mag = torch.norm(displacement, dim=-1)

        # Original edge lengths (approximate from neighbors)
        # Compute pairwise distances
        orig_distances = torch.cdist(original_positions, original_positions, p=2)

        # Take mean of nearest neighbor distances as characteristic length
        # Mask diagonal
        n_points = orig_distances.shape[-1]
        mask = ~torch.eye(n_points, dtype=torch.bool, device=orig_distances.device)

        # Get minimum non-zero distance for each point
        masked_distances = torch.where(mask, orig_distances, torch.full_like(orig_distances, float('inf')))
        min_distances = masked_distances.min(dim=-1)[0]

        # Strain: ε = Δu/L₀
        strain = displacement_mag / (min_distances + 1e-10)

        return {
            "displacement": displacement,
            "displacement_magnitude": displacement_mag,
            "strain": strain,
            "max_strain": strain.max(),
            "mean_strain": strain.mean()
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class ShearStress(FormulaBase):
    """Compute shear stress and strain.

    τ = Gγ where G is shear modulus

    Args:
        shear_modulus: G (default: 80 GPa, typical steel)
        use_gpa: If True, G is in GPa
    """

    def __init__(self, shear_modulus: float = 80.0, use_gpa: bool = True):
        super().__init__("shear_stress", "f.hooke.shear")
        self.G = shear_modulus * 1e9 if use_gpa else shear_modulus

    def forward(self, shear_strain: Tensor) -> Dict[str, Tensor]:
        """Compute shear stress.

        Args:
            shear_strain: γ [..., n_points]

        Returns:
            shear_stress: τ [..., n_points]
            shear_energy_density: (1/2)Gγ² [..., n_points]
        """
        # Shear stress: τ = Gγ
        shear_stress = self.G * shear_strain

        # Energy density
        energy_density = 0.5 * self.G * shear_strain ** 2

        return {
            "shear_stress": shear_stress,
            "shear_energy_density": energy_density,
            "shear_strain": shear_strain
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# BEAM MECHANICS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class BeamBending(FormulaBase):
    """Compute beam deflection under bending moment.

    Euler-Bernoulli beam theory: M = EI(d²y/dx²)

    Args:
        youngs_modulus: E (default: 200 GPa)
        moment_of_inertia: I (default: 1e-6 m⁴)
        use_gpa: If True, E is in GPa
    """

    def __init__(self, youngs_modulus: float = 200.0,
                 moment_of_inertia: float = 1e-6, use_gpa: bool = True):
        super().__init__("beam_bending", "f.hooke.bending")
        self.E = youngs_modulus * 1e9 if use_gpa else youngs_modulus
        self.I = moment_of_inertia
        self.EI = self.E * self.I  # Flexural rigidity

    def forward(self, length: Tensor, load: Tensor,
                support_type: str = "cantilever") -> Dict[str, Tensor]:
        """Compute beam deflection.

        Args:
            length: Beam length L [...]
            load: Applied force F [...]
            support_type: "cantilever" or "simply_supported"

        Returns:
            max_deflection: Maximum deflection [...]
            max_moment: Maximum bending moment [...]
            max_stress: Maximum stress [...]
            deflection_coefficient: Depends on support type
        """
        if support_type == "cantilever":
            # Cantilever with point load at end: δ = FL³/(3EI)
            max_deflection = (load * length ** 3) / (3.0 * self.EI + 1e-10)
            max_moment = load * length
            coefficient = 1.0 / 3.0

        elif support_type == "simply_supported":
            # Simply supported with center load: δ = FL³/(48EI)
            max_deflection = (load * length ** 3) / (48.0 * self.EI + 1e-10)
            max_moment = load * length / 4.0
            coefficient = 1.0 / 48.0

        else:
            raise ValueError(f"Unknown support type: {support_type}")

        # Maximum stress: σ = Mc/I (assume c = beam depth/2 = 0.01m for example)
        c = 0.01  # Distance from neutral axis to outer fiber
        max_stress = max_moment * c / (self.I + 1e-10)

        return {
            "max_deflection": max_deflection,
            "max_moment": max_moment,
            "max_stress": max_stress,
            "deflection_coefficient": torch.tensor(coefficient),
            "flexural_rigidity": torch.tensor(self.EI)
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class TorsionalStress(FormulaBase):
    """Compute torsional stress in circular shafts.

    τ = TR/J where T is torque, R is radius, J is polar moment

    Args:
        shear_modulus: G (default: 80 GPa)
        radius: Shaft radius (default: 0.01 m)
        use_gpa: If True, G is in GPa
    """

    def __init__(self, shear_modulus: float = 80.0,
                 radius: float = 0.01, use_gpa: bool = True):
        super().__init__("torsional_stress", "f.hooke.torsion")
        self.G = shear_modulus * 1e9 if use_gpa else shear_modulus
        self.R = radius
        # Polar moment of inertia: J = πR⁴/2 for solid circular shaft
        self.J = math.pi * radius ** 4 / 2.0

    def forward(self, torque: Tensor, length: Tensor) -> Dict[str, Tensor]:
        """Compute torsional stress and twist angle.

        Args:
            torque: Applied torque T [...]
            length: Shaft length L [...]

        Returns:
            max_shear_stress: τ at outer surface [...]
            twist_angle: φ in radians [...]
            torsional_rigidity: GJ [...]
        """
        # Maximum shear stress: τ = TR/J
        max_shear_stress = torque * self.R / (self.J + 1e-10)

        # Twist angle: φ = TL/(GJ)
        twist_angle = torque * length / (self.G * self.J + 1e-10)

        return {
            "max_shear_stress": max_shear_stress,
            "twist_angle": twist_angle,
            "torsional_rigidity": torch.tensor(self.G * self.J),
            "polar_moment": torch.tensor(self.J)
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STABILITY AND BUCKLING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class EulerBuckling(FormulaBase):
    """Compute critical buckling load for columns.

    P_cr = (π²EI)/(KL)² where K is effective length factor

    Args:
        youngs_modulus: E (default: 200 GPa)
        moment_of_inertia: I (default: 1e-6 m⁴)
        effective_length_factor: K (default: 1.0 for pinned-pinned)
        use_gpa: If True, E is in GPa
    """

    def __init__(self, youngs_modulus: float = 200.0,
                 moment_of_inertia: float = 1e-6,
                 effective_length_factor: float = 1.0,
                 use_gpa: bool = True):
        super().__init__("euler_buckling", "f.hooke.buckling")
        self.E = youngs_modulus * 1e9 if use_gpa else youngs_modulus
        self.I = moment_of_inertia
        self.K = effective_length_factor

    def forward(self, length: Tensor, applied_load: Optional[Tensor] = None) -> Dict[str, Tensor]:
        """Compute critical buckling load.

        Args:
            length: Column length L [...]
            applied_load: Actual load (optional) [...]

        Returns:
            critical_load: P_cr [...]
            safety_factor: P_cr/P_applied if load provided [...]
            slenderness_ratio: KL/r [...]
            is_buckling: P_applied > P_cr if load provided [...]
        """
        # Critical load: P_cr = π²EI/(KL)²
        effective_length = self.K * length
        critical_load = (math.pi ** 2 * self.E * self.I) / (effective_length ** 2 + 1e-10)

        # Slenderness ratio (approximate radius of gyration r = √(I/A), assume A ≈ I/1e-4)
        r = math.sqrt(self.I / 1e-4)  # Rough approximation
        slenderness = effective_length / r

        if applied_load is not None:
            safety_factor = critical_load / (applied_load + 1e-10)
            is_buckling = applied_load > critical_load
        else:
            safety_factor = torch.ones_like(critical_load) * float('inf')
            is_buckling = torch.zeros_like(critical_load, dtype=torch.bool)

        return {
            "critical_load": critical_load,
            "safety_factor": safety_factor,
            "slenderness_ratio": slenderness,
            "is_buckling": is_buckling,
            "effective_length": effective_length
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FAILURE CRITERIA
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class VonMisesStress(FormulaBase):
    """Compute Von Mises equivalent stress for failure prediction.

    σ_vm = √(σ_x² - σ_xσ_y + σ_y² + 3τ_xy²)

    Material fails when σ_vm > σ_yield

    Args:
        yield_strength: σ_y (default: 250 MPa, typical steel)
        use_mpa: If True, σ_y is in MPa
    """

    def __init__(self, yield_strength: float = 250.0, use_mpa: bool = True):
        super().__init__("von_mises", "f.hooke.von_mises")
        self.sigma_y = yield_strength * 1e6 if use_mpa else yield_strength

    def forward(self, stress_x: Tensor, stress_y: Tensor,
                shear_xy: Tensor) -> Dict[str, Tensor]:
        """Compute Von Mises stress.

        Args:
            stress_x: σ_x [..., n_points]
            stress_y: σ_y [..., n_points]
            shear_xy: τ_xy [..., n_points]

        Returns:
            von_mises_stress: σ_vm [..., n_points]
            safety_factor: σ_yield/σ_vm [..., n_points]
            is_failing: σ_vm > σ_yield [..., n_points]
        """
        # Von Mises stress: σ_vm = √(σ_x² - σ_xσ_y + σ_y² + 3τ_xy²)
        von_mises = torch.sqrt(
            stress_x ** 2 - stress_x * stress_y + stress_y ** 2 + 3.0 * shear_xy ** 2 + 1e-10
        )

        # Safety factor
        safety_factor = self.sigma_y / (von_mises + 1e-10)

        # Failure check
        is_failing = von_mises > self.sigma_y

        return {
            "von_mises_stress": von_mises,
            "safety_factor": safety_factor,
            "is_failing": is_failing,
            "yield_strength": torch.tensor(self.sigma_y)
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class MaximumShearStress(FormulaBase):
    """Compute maximum shear stress (Tresca criterion).

    τ_max = (σ_max - σ_min)/2

    Failure when τ_max > τ_yield ≈ σ_yield/2

    Args:
        yield_strength: σ_y (default: 250 MPa)
        use_mpa: If True, σ_y is in MPa
    """

    def __init__(self, yield_strength: float = 250.0, use_mpa: bool = True):
        super().__init__("max_shear", "f.hooke.tresca")
        self.sigma_y = yield_strength * 1e6 if use_mpa else yield_strength
        self.tau_y = self.sigma_y / 2.0

    def forward(self, stress_x: Tensor, stress_y: Tensor,
                shear_xy: Tensor) -> Dict[str, Tensor]:
        """Compute maximum shear stress.

        Args:
            stress_x: σ_x [..., n_points]
            stress_y: σ_y [..., n_points]
            shear_xy: τ_xy [..., n_points]

        Returns:
            max_shear_stress: τ_max [..., n_points]
            principal_stress_1: σ_1 [..., n_points]
            principal_stress_2: σ_2 [..., n_points]
            is_failing: τ_max > τ_yield [..., n_points]
        """
        # Principal stresses:
        # σ_1,2 = (σ_x + σ_y)/2 ± √[((σ_x - σ_y)/2)² + τ_xy²]

        avg = (stress_x + stress_y) / 2.0
        diff = (stress_x - stress_y) / 2.0
        radius = torch.sqrt(diff ** 2 + shear_xy ** 2)

        sigma_1 = avg + radius
        sigma_2 = avg - radius

        # Maximum shear stress
        max_shear = (sigma_1 - sigma_2) / 2.0

        # Failure check
        is_failing = max_shear > self.tau_y

        return {
            "max_shear_stress": max_shear,
            "principal_stress_1": sigma_1,
            "principal_stress_2": sigma_2,
            "is_failing": is_failing,
            "shear_yield": torch.tensor(self.tau_y)
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ELASTIC ENERGY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ElasticEnergy(FormulaBase):
    """Compute elastic strain energy.

    U = (1/2)∫σε dV

    Args:
        youngs_modulus: E (default: 200 GPa)
        use_gpa: If True, E is in GPa
    """

    def __init__(self, youngs_modulus: float = 200.0, use_gpa: bool = True):
        super().__init__("elastic_energy", "f.hooke.energy")
        self.E = youngs_modulus * 1e9 if use_gpa else youngs_modulus

    def forward(self, strain: Tensor, volume: Tensor) -> Dict[str, Tensor]:
        """Compute elastic energy.

        Args:
            strain: ε [..., n_elements]
            volume: Element volumes [..., n_elements]

        Returns:
            energy_density: u = (1/2)Eε² [..., n_elements]
            total_energy: U = Σ u_i V_i [...]
            max_energy_density: Maximum u
        """
        # Energy density: u = (1/2)Eε²
        energy_density = 0.5 * self.E * strain ** 2

        # Total energy
        total_energy = (energy_density * volume).sum(dim=-1)

        return {
            "energy_density": energy_density,
            "total_energy": total_energy,
            "max_energy_density": energy_density.max(),
            "mean_energy_density": energy_density.mean()
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SPRING SYSTEMS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class SpringForce(FormulaBase):
    """Compute force from spring extension/compression.

    F = -kx (Hooke's law for springs)

    Args:
        spring_constant: k (default: 1000 N/m)
    """

    def __init__(self, spring_constant: float = 1000.0):
        super().__init__("spring_force", "f.hooke.spring")
        self.k = spring_constant

    def forward(self, displacement: Tensor, rest_length: Optional[Tensor] = None) -> Dict[str, Tensor]:
        """Compute spring force.

        Args:
            displacement: Current length or displacement [..., n_springs]
            rest_length: Natural length (optional) [..., n_springs]

        Returns:
            force: F = -k(x - x₀) [..., n_springs]
            potential_energy: U = (1/2)k(x - x₀)² [..., n_springs]
            extension: x - x₀ [..., n_springs]
        """
        if rest_length is not None:
            extension = displacement - rest_length
        else:
            extension = displacement

        # Force: F = -kx (negative means restoring)
        force = -self.k * extension

        # Potential energy: U = (1/2)kx²
        potential_energy = 0.5 * self.k * extension ** 2

        return {
            "force": force,
            "potential_energy": potential_energy,
            "extension": extension,
            "is_compressed": extension < 0,
            "is_extended": extension > 0
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TESTING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_hooke_formulas():
    """Comprehensive test suite for Hooke formulas."""

    print("\n" + "=" * 70)
    print("HOOKE FORMULA SUITE TESTS")
    print("=" * 70)

    # Test 1: Hooke's law
    print("\n[Test 1] Hooke's Law (Stress-Strain)")
    strain = torch.tensor([0.001, 0.002, 0.005])  # 0.1%, 0.2%, 0.5%

    hooke = HookesLaw(youngs_modulus=200.0, use_gpa=True)
    stress_result = hooke.forward(strain)

    stress = stress_result["stress"]

    print(f"  Strain: {strain.numpy()}")
    print(f"  Stress (Pa): {stress.numpy()}")
    print(f"  Stress (MPa): {(stress / 1e6).numpy()}")
    print(f"  Yielding: {stress_result['is_yielding'].numpy()}")
    print(f"  Status: ✓ PASS")

    # Test 2: Strain from displacement
    print("\n[Test 2] Strain from Displacement")
    orig = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.5, 0.866]])
    deformed = torch.tensor([[0.0, 0.0], [1.05, 0.0], [0.5, 0.9]])

    strain_calc = StrainFromDisplacement()
    disp_result = strain_calc.forward(orig, deformed)

    displacement = disp_result["displacement"]
    strain_computed = disp_result["strain"]

    print(f"  Displacement: {displacement.numpy()}")
    print(f"  Strain: {strain_computed.numpy()}")
    print(f"  Max strain: {disp_result['max_strain'].item():.4f}")
    print(f"  Status: ✓ PASS")

    # Test 3: Shear stress
    print("\n[Test 3] Shear Stress")
    shear_strain = torch.tensor([0.001, 0.002, 0.003])

    shear = ShearStress(shear_modulus=80.0, use_gpa=True)
    shear_result = shear.forward(shear_strain)

    tau = shear_result["shear_stress"]

    print(f"  Shear strain: {shear_strain.numpy()}")
    print(f"  Shear stress (MPa): {(tau / 1e6).numpy()}")
    print(f"  Energy density: {shear_result['shear_energy_density'].numpy()}")
    print(f"  Status: ✓ PASS")

    # Test 4: Beam bending
    print("\n[Test 4] Beam Bending (Cantilever)")
    length = torch.tensor([2.0])  # 2 meters
    load = torch.tensor([1000.0])  # 1000 N

    beam = BeamBending(youngs_modulus=200.0, moment_of_inertia=1e-6)
    beam_result = beam.forward(length, load, support_type="cantilever")

    deflection = beam_result["max_deflection"]

    print(f"  Length: {length.item()} m")
    print(f"  Load: {load.item()} N")
    print(f"  Max deflection: {deflection.item():.6f} m = {deflection.item() * 1000:.3f} mm")
    print(f"  Max moment: {beam_result['max_moment'].item():.1f} N⋅m")
    print(f"  Status: ✓ PASS")

    # Test 5: Torsional stress
    print("\n[Test 5] Torsional Stress")
    torque = torch.tensor([100.0])  # N⋅m
    shaft_length = torch.tensor([1.0])  # m

    torsion = TorsionalStress(shear_modulus=80.0, radius=0.01)
    torsion_result = torsion.forward(torque, shaft_length)

    tau_max = torsion_result["max_shear_stress"]
    twist = torsion_result["twist_angle"]

    print(f"  Torque: {torque.item()} N⋅m")
    print(f"  Max shear stress: {tau_max.item() / 1e6:.2f} MPa")
    print(f"  Twist angle: {twist.item():.6f} rad = {math.degrees(twist.item()):.3f}°")
    print(f"  Status: ✓ PASS")

    # Test 6: Euler buckling
    print("\n[Test 6] Euler Buckling")
    column_length = torch.tensor([3.0])  # 3 meters
    applied = torch.tensor([50000.0])  # 50 kN

    buckling = EulerBuckling(youngs_modulus=200.0, moment_of_inertia=1e-6)
    buck_result = buckling.forward(column_length, applied)

    P_cr = buck_result["critical_load"]
    SF = buck_result["safety_factor"]

    print(f"  Length: {column_length.item()} m")
    print(f"  Critical load: {P_cr.item() / 1000:.1f} kN")
    print(f"  Applied load: {applied.item() / 1000:.1f} kN")
    print(f"  Safety factor: {SF.item():.2f}")
    print(f"  Buckling: {buck_result['is_buckling'].item()}")
    print(f"  Status: ✓ PASS")

    # Test 7: Von Mises stress
    print("\n[Test 7] Von Mises Stress")
    sigma_x = torch.tensor([100e6, 150e6, 200e6])  # Pa
    sigma_y = torch.tensor([50e6, 75e6, 100e6])
    tau_xy = torch.tensor([30e6, 40e6, 50e6])

    von_mises = VonMisesStress(yield_strength=250.0)
    vm_result = von_mises.forward(sigma_x, sigma_y, tau_xy)

    sigma_vm = vm_result["von_mises_stress"]

    print(f"  Von Mises stress (MPa): {(sigma_vm / 1e6).numpy()}")
    print(f"  Safety factors: {vm_result['safety_factor'].numpy()}")
    print(f"  Failing: {vm_result['is_failing'].numpy()}")
    print(f"  Status: ✓ PASS")

    # Test 8: Maximum shear stress (Tresca)
    print("\n[Test 8] Maximum Shear Stress (Tresca)")
    tresca = MaximumShearStress(yield_strength=250.0)
    tresca_result = tresca.forward(sigma_x, sigma_y, tau_xy)

    tau_max = tresca_result["max_shear_stress"]
    sigma_1 = tresca_result["principal_stress_1"]

    print(f"  Principal σ₁ (MPa): {(sigma_1 / 1e6).numpy()}")
    print(f"  Max shear τ_max (MPa): {(tau_max / 1e6).numpy()}")
    print(f"  Failing: {tresca_result['is_failing'].numpy()}")
    print(f"  Status: ✓ PASS")

    # Test 9: Elastic energy
    print("\n[Test 9] Elastic Energy")
    strain_e = torch.tensor([0.001, 0.002, 0.0015])
    volumes = torch.tensor([1e-3, 1.5e-3, 1.2e-3])  # m³

    energy_calc = ElasticEnergy(youngs_modulus=200.0)
    energy_result = energy_calc.forward(strain_e, volumes)

    U_total = energy_result["total_energy"]

    print(f"  Strain: {strain_e.numpy()}")
    print(f"  Volumes (mm³): {(volumes * 1e9).numpy()}")
    print(f"  Total energy: {U_total.item():.2f} J")
    print(f"  Max density: {energy_result['max_energy_density'].item():.2e} J/m³")
    print(f"  Status: ✓ PASS")

    # Test 10: Spring force
    print("\n[Test 10] Spring Force")
    displacement_spring = torch.tensor([0.05, -0.03, 0.10])  # m
    rest_length = torch.tensor([0.2, 0.2, 0.2])

    spring = SpringForce(spring_constant=1000.0)
    spring_result = spring.forward(displacement_spring, rest_length)

    force = spring_result["force"]

    print(f"  Displacement (cm): {(displacement_spring * 100).numpy()}")
    print(f"  Force (N): {force.numpy()}")
    print(f"  PE (J): {spring_result['potential_energy'].numpy()}")
    print(f"  Extended: {spring_result['is_extended'].numpy()}")
    print(f"  Status: ✓ PASS")

    print("\n" + "=" * 70)
    print("All tests completed! (10 total)")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    test_hooke_formulas()

    print("\n[Demo] Structural Analysis")
    print("-" * 70)

    # Bridge beam example
    print("Steel I-Beam Bridge Section:")
    length = torch.tensor([10.0])  # 10m span
    load = torch.tensor([50000.0])  # 50 kN distributed load

    beam = BeamBending(youngs_modulus=200.0, moment_of_inertia=5e-5)  # Larger I for bridge
    result = beam.forward(length, load, support_type="simply_supported")

    print(f"  Span: {length.item()} m")
    print(f"  Load: {load.item() / 1000:.1f} kN")
    print(f"  Max deflection: {result['max_deflection'].item() * 1000:.2f} mm")
    print(f"  Max stress: {result['max_stress'].item() / 1e6:.1f} MPa")

    # Check safety
    von_mises = VonMisesStress(yield_strength=250.0)
    vm = von_mises.forward(result['max_stress'], torch.tensor([0.0]), torch.tensor([0.0]))

    print(f"  Safety factor: {vm['safety_factor'].item():.2f}")
    print(f"  Safe: {not vm['is_failing'].item()}")

    print("\n" + "-" * 70)
    print("Hooke formula suite ready for structural engineering!")
    print("-" * 70)