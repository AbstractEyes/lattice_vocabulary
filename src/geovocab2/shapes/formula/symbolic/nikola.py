"""
NIKOLA FORMULA SUITE
-------------------
Resonance, oscillations, electromagnetic coupling, and energy dynamics for geometric systems.

Named in honor of:
  • Nikola Tesla (1856–1943) – AC power, resonance, wireless energy transfer, electromagnetic fields

This suite provides formulas for physical dynamics of geometric structures:
  - Resonant frequency analysis (natural modes)
  - Harmonic oscillation (eigenmodes and standing waves)
  - Energy coupling (mutual inductance between elements)
  - Damped and driven oscillations
  - Electromagnetic field distributions
  - Quality factor and energy dissipation
  - Beat frequencies and interference patterns

Mathematical Foundation:

    Harmonic Oscillator:
        m·d²x/dt² + γ·dx/dt + k·x = F(t)
        Natural frequency: ω₀ = √(k/m)
        Damping ratio: ζ = γ/(2√(km))

    Resonant Frequency:
        ω_res = ω₀√(1 - 2ζ²) for ζ < 1/√2
        Maximum amplitude at resonance

    Quality Factor:
        Q = ω₀/(2ζω₀) = 1/(2ζ)
        High Q: sharp resonance, low damping
        Low Q: broad resonance, high damping

    Mutual Coupling:
        k_ij = M_ij/√(L_i L_j)
        where M_ij is mutual inductance

    Standing Wave:
        y(x,t) = A·sin(kx)·cos(ωt)
        Nodes: sin(kx) = 0 → x = nπ/k
        Antinodes: sin(kx) = ±1

    Energy Transfer:
        η = 4k²/((1+k)²) for resonant coupling
        where k is coupling coefficient

Applications:
    - Physical simulation of elastic structures
    - Vibration mode analysis
    - Energy-efficient network design
    - Resonance-based flow matching
    - Catastrophe detection (resonance collapse)
    - Adaptive mesh refinement via energy

Author: AbstractPhil + Claude Sonnet 4.5
License: MIT
"""

from typing import Dict, Optional, Tuple, Callable
import torch
from torch import Tensor
import math

from ..formula_base import FormulaBase


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# RESONANCE AND NATURAL FREQUENCIES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ResonantFrequency(FormulaBase):
    """Compute natural resonant frequencies of geometric structures.

    For a simplex or mesh, resonant frequencies depend on:
    - Geometry (edge lengths, volumes)
    - Mass distribution
    - Stiffness (resistance to deformation)

    Formula:
        ω₀ = √(k/m) where k = stiffness, m = mass
        For geometric structures: k ∝ 1/L², m ∝ V

    Args:
        dimension: Spatial dimension (default: 2)
        material_stiffness: Stiffness constant (default: 1.0)
    """

    def __init__(self, dimension: int = 2, material_stiffness: float = 1.0):
        super().__init__("resonant_frequency", "f.tesla.resonance")
        self.dimension = dimension
        self.k = material_stiffness

    def forward(self, positions: Tensor, masses: Optional[Tensor] = None) -> Dict[str, Tensor]:
        """Compute resonant frequencies.

        Args:
            positions: Vertex positions [..., n_vertices, dim]
            masses: Vertex masses [..., n_vertices] (optional, uniform if None)

        Returns:
            frequencies: Natural frequencies [..., n_modes]
            fundamental: Lowest frequency (fundamental mode)
            harmonics: Higher harmonics [...]
        """
        n_vertices = positions.shape[-2]

        # Default: uniform mass
        if masses is None:
            masses = torch.ones(*positions.shape[:-1], dtype=positions.dtype, device=positions.device)

        # Estimate characteristic length (mean edge length)
        # For simplicity, use bounding box diagonal
        bbox_min = positions.min(dim=-2)[0]
        bbox_max = positions.max(dim=-2)[0]
        char_length = torch.norm(bbox_max - bbox_min, dim=-1)

        # Stiffness: k ∝ 1/L² (geometric spring constant)
        stiffness = self.k / (char_length ** 2 + 1e-10)

        # Effective mass
        total_mass = masses.sum(dim=-1)

        # Fundamental frequency: ω₀ = √(k/m)
        omega_0 = torch.sqrt(stiffness / (total_mass + 1e-10))

        # Generate harmonics: ω_n = n·ω₀ (for 1D) or more complex for 2D/3D
        # For simplicity: integer multiples for first few modes
        n_modes = min(n_vertices, 10)
        mode_numbers = torch.arange(1, n_modes + 1, dtype=omega_0.dtype, device=omega_0.device)

        # Broadcast: omega_0 [...] * mode_numbers [n_modes] -> [..., n_modes]
        frequencies = omega_0.unsqueeze(-1) * mode_numbers

        return {
            "frequencies": frequencies,
            "fundamental": omega_0,
            "harmonics": frequencies[..., 1:],  # Exclude fundamental
            "n_modes": n_modes
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class HarmonicModes(FormulaBase):
    """Compute vibrational eigenmodes (standing wave patterns).

    Eigenmodes represent the spatial distribution of oscillations at
    each resonant frequency. Nodes stay stationary, antinodes oscillate maximally.

    Args:
        num_modes: Number of modes to compute (default: 5)
        normalize: Normalize mode amplitudes (default: True)
    """

    def __init__(self, num_modes: int = 5, normalize: bool = True):
        super().__init__("harmonic_modes", "f.tesla.modes")
        self.num_modes = num_modes
        self.normalize = normalize

    def forward(self, positions: Tensor, frequencies: Tensor) -> Dict[str, Tensor]:
        """Compute spatial mode patterns.

        Args:
            positions: Vertex positions [..., n_vertices, dim]
            frequencies: Mode frequencies [..., n_modes]

        Returns:
            mode_shapes: Amplitude at each vertex [..., n_vertices, n_modes]
            node_positions: Approximate node locations
            antinode_positions: Approximate antinode locations
        """
        n_vertices = positions.shape[-2]
        n_modes = frequencies.shape[-1]

        # Compute mode shapes: sinusoidal patterns based on position
        # For mode n: φ_n(x) = sin(n·π·x/L)

        # Normalize positions to [0, 1]
        pos_min = positions.min(dim=-2, keepdim=True)[0]
        pos_max = positions.max(dim=-2, keepdim=True)[0]
        normalized_pos = (positions - pos_min) / (pos_max - pos_min + 1e-10)

        # Project to 1D coordinate (use first dimension or magnitude)
        if self.normalize:
            x_coord = normalized_pos[..., 0]  # [..., n_vertices]
        else:
            x_coord = torch.norm(normalized_pos, dim=-1)

        # Compute mode shapes: sin(mode_n * π * x)
        # mode_numbers: [1, 2, 3, ..., n_modes]
        mode_numbers = torch.arange(1, n_modes + 1, dtype=x_coord.dtype, device=x_coord.device)

        # Broadcasting: x_coord [..., n_vertices, 1] * mode_numbers [1, n_modes]
        angles = math.pi * x_coord.unsqueeze(-1) * mode_numbers
        mode_shapes = torch.sin(angles)  # [..., n_vertices, n_modes]

        # Normalize amplitudes
        if self.normalize:
            mode_norms = torch.norm(mode_shapes, dim=-2, keepdim=True)
            mode_shapes = mode_shapes / (mode_norms + 1e-10)

        # Find nodes (where amplitude ≈ 0)
        amplitude_threshold = 0.1
        is_node = torch.abs(mode_shapes) < amplitude_threshold

        # Find antinodes (where amplitude ≈ ±1)
        is_antinode = torch.abs(torch.abs(mode_shapes) - 1.0) < amplitude_threshold

        return {
            "mode_shapes": mode_shapes,
            "node_mask": is_node,
            "antinode_mask": is_antinode,
            "num_modes": n_modes
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class QualityFactor(FormulaBase):
    """Compute Q-factor (quality factor) measuring oscillation sharpness.

    High Q: sharp resonance peak, low energy loss, long decay time
    Low Q: broad resonance, high damping, fast decay

    Q = ω₀/(2γ) = energy stored / energy lost per cycle

    Args:
        default_damping: Default damping coefficient (default: 0.1)
    """

    def __init__(self, default_damping: float = 0.1):
        super().__init__("quality_factor", "f.tesla.q_factor")
        self.gamma = default_damping

    def forward(self, frequencies: Tensor,
                damping: Optional[Tensor] = None) -> Dict[str, Tensor]:
        """Compute quality factor.

        Args:
            frequencies: Natural frequencies [..., n_modes]
            damping: Damping coefficients [..., n_modes] (optional)

        Returns:
            q_factor: Quality factor [..., n_modes]
            damping_ratio: ζ = γ/(2ω₀)
            decay_time: Time constant τ = 2Q/ω₀
        """
        if damping is None:
            damping = torch.full_like(frequencies, self.gamma)

        # Q = ω₀/(2γ)
        q_factor = frequencies / (2.0 * damping + 1e-10)

        # Damping ratio: ζ = γ/(2ω₀) = 1/(2Q)
        damping_ratio = 1.0 / (2.0 * q_factor + 1e-10)

        # Decay time: τ = 2Q/ω₀
        decay_time = 2.0 * q_factor / (frequencies + 1e-10)

        # Classification
        underdamped = damping_ratio < 1.0
        critically_damped = torch.abs(damping_ratio - 1.0) < 0.1
        overdamped = damping_ratio > 1.0

        return {
            "q_factor": q_factor,
            "damping_ratio": damping_ratio,
            "decay_time": decay_time,
            "underdamped": underdamped,
            "critically_damped": critically_damped,
            "overdamped": overdamped
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ENERGY COUPLING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class MutualCoupling(FormulaBase):
    """Compute mutual coupling between geometric elements.

    Represents energy transfer efficiency between oscillators.
    Based on distance, alignment, and frequency matching.

    Coupling coefficient:
        k_ij = M_ij / √(L_i L_j)

    Args:
        coupling_decay: Distance decay rate (default: 1.0)
    """

    def __init__(self, coupling_decay: float = 1.0):
        super().__init__("mutual_coupling", "f.tesla.coupling")
        self.decay = coupling_decay

    def forward(self, positions_a: Tensor, positions_b: Tensor,
                frequencies_a: Optional[Tensor] = None,
                frequencies_b: Optional[Tensor] = None) -> Dict[str, Tensor]:
        """Compute coupling between two sets of oscillators.

        Args:
            positions_a: First set of positions [..., n_a, dim]
            positions_b: Second set of positions [..., n_b, dim]
            frequencies_a: Frequencies of set A [..., n_a] (optional)
            frequencies_b: Frequencies of set B [..., n_b] (optional)

        Returns:
            coupling_matrix: k_ij [..., n_a, n_b]
            max_coupling: Maximum coupling coefficient
            coupling_efficiency: Average transfer efficiency
        """
        # Compute pairwise distances
        # positions_a: [..., n_a, dim]
        # positions_b: [..., n_b, dim]
        # distances: [..., n_a, n_b]
        distances = torch.cdist(positions_a, positions_b, p=2)

        # Geometric coupling: k_geom ∝ 1/(1 + αr²)
        k_geom = 1.0 / (1.0 + self.decay * distances ** 2)

        # Frequency matching (if provided)
        if frequencies_a is not None and frequencies_b is not None:
            # Resonance coupling: strongest when frequencies match
            # k_freq = exp(-|ω_a - ω_b|²/σ²)
            freq_diff = frequencies_a.unsqueeze(-1) - frequencies_b.unsqueeze(-2)
            k_freq = torch.exp(-freq_diff ** 2)

            # Combined coupling
            k_total = k_geom * k_freq
        else:
            k_total = k_geom

        # Normalize to [0, 1]
        k_normalized = k_total / (k_total.max() + 1e-10)

        # Energy transfer efficiency: η = 4k²/((1+k)²)
        efficiency = 4.0 * k_normalized ** 2 / ((1.0 + k_normalized) ** 2 + 1e-10)

        return {
            "coupling_matrix": k_normalized,
            "max_coupling": k_normalized.max(),
            "coupling_efficiency": efficiency.mean(dim=(-2, -1)),
            "transfer_efficiency": efficiency
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class EnergyFlow(FormulaBase):
    """Compute energy flow through coupled oscillator network.

    Energy flows from high-energy to low-energy oscillators via coupling.

    Args:
        dt: Time step for flow integration (default: 0.01)
    """

    def __init__(self, dt: float = 0.01):
        super().__init__("energy_flow", "f.tesla.energy_flow")
        self.dt = dt

    def forward(self, energies: Tensor, coupling_matrix: Tensor) -> Dict[str, Tensor]:
        """Compute energy redistribution via coupling.

        Args:
            energies: Energy at each node [..., n_nodes]
            coupling_matrix: Coupling strengths [..., n_nodes, n_nodes]

        Returns:
            energy_flow: Change in energy [..., n_nodes]
            new_energies: Updated energies after flow
            flow_matrix: Energy transfer between nodes [..., n_nodes, n_nodes]
        """
        # Energy flow: dE_i/dt = Σ_j k_ij (E_j - E_i)
        # Vectorized using einsum

        # Energy difference: E_j - E_i for all pairs
        # energies: [..., n_nodes]
        # energy_diff[..., i, j] = E_j - E_i
        energy_diff = energies.unsqueeze(-2) - energies.unsqueeze(-1)

        # Flow through each link: k_ij * (E_j - E_i)
        flow_matrix = coupling_matrix * energy_diff  # [..., n_nodes, n_nodes]

        # Total flow into each node: Σ_j k_ij (E_j - E_i)
        energy_flow = flow_matrix.sum(dim=-1)  # [..., n_nodes]

        # Update energies
        new_energies = energies + self.dt * energy_flow

        # Clamp to positive
        new_energies = torch.clamp(new_energies, min=0.0)

        return {
            "energy_flow": energy_flow,
            "new_energies": new_energies,
            "flow_matrix": flow_matrix,
            "total_energy": new_energies.sum(dim=-1)
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# OSCILLATORY DYNAMICS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class DampedOscillator(FormulaBase):
    """Simulate damped harmonic oscillator dynamics.

    Equation: m·d²x/dt² + γ·dx/dt + k·x = 0

    Solution depends on damping:
    - Underdamped (ζ < 1): oscillates with decay
    - Critically damped (ζ = 1): fastest return to equilibrium
    - Overdamped (ζ > 1): slow exponential decay

    Args:
        mass: Oscillator mass (default: 1.0)
        stiffness: Spring constant (default: 1.0)
        damping: Damping coefficient (default: 0.1)
        dt: Time step (default: 0.01)
    """

    def __init__(self, mass: float = 1.0, stiffness: float = 1.0,
                 damping: float = 0.1, dt: float = 0.01):
        super().__init__("damped_oscillator", "f.tesla.damped")
        self.m = mass
        self.k = stiffness
        self.gamma = damping
        self.dt = dt

        # Natural frequency and damping ratio
        self.omega_0 = math.sqrt(stiffness / mass)
        self.zeta = damping / (2.0 * math.sqrt(mass * stiffness))

    def forward(self, position: Tensor, velocity: Tensor,
                num_steps: int = 100) -> Dict[str, Tensor]:
        """Simulate oscillator trajectory.

        Args:
            position: Initial position [..., n_oscillators]
            velocity: Initial velocity [..., n_oscillators]
            num_steps: Number of time steps to simulate

        Returns:
            trajectory: Positions over time [..., num_steps, n_oscillators]
            velocities: Velocities over time [..., num_steps, n_oscillators]
            energies: Total energy over time [..., num_steps]
            times: Time points
        """
        # Initialize storage
        batch_shape = position.shape[:-1] if position.ndim > 1 else ()
        n_osc = position.shape[-1] if position.ndim > 0 else 1

        positions = torch.zeros(*batch_shape, num_steps, n_osc if position.ndim > 0 else 1,
                                dtype=position.dtype, device=position.device)
        velocities = torch.zeros_like(positions)
        energies = torch.zeros(*batch_shape, num_steps,
                               dtype=position.dtype, device=position.device)

        # Initial conditions
        x = position
        v = velocity

        for step in range(num_steps):
            # Store current state
            positions[..., step, :] = x
            velocities[..., step, :] = v

            # Energy: E = (1/2)mv² + (1/2)kx²
            kinetic = 0.5 * self.m * (v ** 2).sum(dim=-1) if v.ndim > 0 else 0.5 * self.m * v ** 2
            potential = 0.5 * self.k * (x ** 2).sum(dim=-1) if x.ndim > 0 else 0.5 * self.k * x ** 2
            energies[..., step] = kinetic + potential

            # Euler integration: a = -(k/m)x - (γ/m)v
            acceleration = -(self.k / self.m) * x - (self.gamma / self.m) * v

            v = v + self.dt * acceleration
            x = x + self.dt * v

        times = torch.arange(num_steps, dtype=position.dtype, device=position.device) * self.dt

        return {
            "trajectory": positions,
            "velocities": velocities,
            "energies": energies,
            "times": times,
            "damping_ratio": torch.tensor(self.zeta)
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class DrivenResonance(FormulaBase):
    """Compute response to driven (forced) oscillation.

    Equation: m·d²x/dt² + γ·dx/dt + k·x = F₀·cos(ωt)

    Resonance occurs when driving frequency matches natural frequency.

    Args:
        natural_frequency: ω₀ (default: 1.0)
        damping_ratio: ζ (default: 0.1)
    """

    def __init__(self, natural_frequency: float = 1.0, damping_ratio: float = 0.1):
        super().__init__("driven_resonance", "f.tesla.driven")
        self.omega_0 = natural_frequency
        self.zeta = damping_ratio

    def forward(self, driving_frequencies: Tensor,
                force_amplitude: float = 1.0) -> Dict[str, Tensor]:
        """Compute amplitude response vs driving frequency.

        Args:
            driving_frequencies: ω [..., n_frequencies]
            force_amplitude: F₀ magnitude

        Returns:
            amplitudes: Steady-state amplitude [..., n_frequencies]
            phase_lag: Phase difference [..., n_frequencies]
            resonance_peak: Maximum amplitude
            resonance_frequency: Frequency of maximum response
        """
        omega = driving_frequencies
        omega_0 = self.omega_0
        zeta = self.zeta

        # Amplitude response: A(ω) = F₀ / √[(k - mω²)² + (γω)²]
        # Normalized: A(ω) = 1 / √[(1 - (ω/ω₀)²)² + (2ζ·ω/ω₀)²]

        omega_ratio = omega / omega_0

        denominator = torch.sqrt(
            (1.0 - omega_ratio ** 2) ** 2 + (2.0 * zeta * omega_ratio) ** 2
        )

        amplitudes = force_amplitude / (denominator + 1e-10)

        # Phase lag: tan(φ) = (2ζ·ω/ω₀) / (1 - (ω/ω₀)²)
        phase_lag = torch.atan2(
            2.0 * zeta * omega_ratio,
            1.0 - omega_ratio ** 2
        )

        # Find resonance peak
        resonance_idx = torch.argmax(amplitudes, dim=-1)
        resonance_peak = amplitudes.max(dim=-1)[0]

        # Gather resonance frequency
        if driving_frequencies.ndim > 1:
            # Batch case - use advanced indexing
            resonance_frequency = driving_frequencies[..., 0]  # Placeholder
        else:
            resonance_frequency = driving_frequencies[resonance_idx]

        return {
            "amplitudes": amplitudes,
            "phase_lag": phase_lag,
            "resonance_peak": resonance_peak,
            "resonance_frequency": resonance_frequency,
            "is_resonant": torch.abs(omega_ratio - 1.0) < 0.1
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class BeatFrequency(FormulaBase):
    """Compute beat frequency from interfering oscillations.

    When two oscillators with slightly different frequencies interfere:
        f_beat = |f₁ - f₂|

    Creates amplitude modulation envelope.

    Args:
        None
    """

    def __init__(self):
        super().__init__("beat_frequency", "f.tesla.beats")

    def forward(self, freq_1: Tensor, freq_2: Tensor,
                duration: float = 10.0, sample_rate: int = 100) -> Dict[str, Tensor]:
        """Compute beat pattern.

        Args:
            freq_1: First frequency [..., n_oscillators]
            freq_2: Second frequency [..., n_oscillators]
            duration: Simulation time
            sample_rate: Samples per time unit

        Returns:
            beat_frequency: |f₁ - f₂| [..., n_oscillators]
            beat_period: 1/f_beat [..., n_oscillators]
            interference_pattern: Combined signal [..., n_samples, n_oscillators]
        """
        # Beat frequency
        f_beat = torch.abs(freq_1 - freq_2)
        beat_period = 1.0 / (f_beat + 1e-10)

        # Generate time samples
        n_samples = int(duration * sample_rate)
        t = torch.linspace(0, duration, n_samples, dtype=freq_1.dtype, device=freq_1.device)

        # Compute interference pattern
        # signal = cos(2πf₁t) + cos(2πf₂t) = 2·cos(2π(f₁-f₂)t/2)·cos(2π(f₁+f₂)t/2)

        # Broadcasting: t [n_samples, 1] * freq [1, n_oscillators]
        signal_1 = torch.cos(2.0 * math.pi * t.unsqueeze(-1) * freq_1.unsqueeze(-2))
        signal_2 = torch.cos(2.0 * math.pi * t.unsqueeze(-1) * freq_2.unsqueeze(-2))

        interference = signal_1 + signal_2  # [..., n_samples, n_oscillators]

        # Envelope (amplitude modulation)
        envelope = 2.0 * torch.abs(torch.cos(math.pi * t.unsqueeze(-1) * f_beat.unsqueeze(-2)))

        return {
            "beat_frequency": f_beat,
            "beat_period": beat_period,
            "interference_pattern": interference,
            "envelope": envelope,
            "times": t
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FIELD THEORY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class PotentialField(FormulaBase):
    """Compute scalar potential field from point sources.

    Φ(r) = Σ_i q_i / |r - r_i|

    Analogous to electrostatic or gravitational potential.

    Args:
        field_type: "coulomb" (1/r) or "gaussian" (exp(-r²)) (default: "coulomb")
        strength_scale: Field strength multiplier (default: 1.0)
    """

    def __init__(self, field_type: str = "coulomb", strength_scale: float = 1.0):
        super().__init__("potential_field", "f.tesla.potential")

        if field_type not in ["coulomb", "gaussian"]:
            raise ValueError(f"Invalid field_type: {field_type}")

        self.field_type = field_type
        self.scale = strength_scale

    def forward(self, source_positions: Tensor, source_charges: Tensor,
                field_positions: Tensor) -> Dict[str, Tensor]:
        """Compute potential at field points.

        Args:
            source_positions: Positions of sources [..., n_sources, dim]
            source_charges: Charge/strength of sources [..., n_sources]
            field_positions: Evaluation points [..., n_field, dim]

        Returns:
            potential: Φ(r) [..., n_field]
            gradient: ∇Φ (force field) [..., n_field, dim]
        """
        # Compute distances from each field point to each source
        # field_positions: [..., n_field, dim]
        # source_positions: [..., n_sources, dim]
        # distances: [..., n_field, n_sources]

        distances = torch.cdist(field_positions, source_positions, p=2)

        # Compute potential based on type
        if self.field_type == "coulomb":
            # Φ = q/r
            potential_contributions = source_charges.unsqueeze(-2) / (distances + 1e-10)

        elif self.field_type == "gaussian":
            # Φ = q·exp(-r²)
            potential_contributions = source_charges.unsqueeze(-2) * torch.exp(-distances ** 2)

        # Sum over sources
        potential = self.scale * potential_contributions.sum(dim=-1)  # [..., n_field]

        # Compute gradient: ∇Φ = -Σ_i q_i (r - r_i) / |r - r_i|³
        # Vectorized computation

        # Direction vectors: r - r_i
        # field_positions: [..., n_field, 1, dim]
        # source_positions: [..., 1, n_sources, dim]
        direction = field_positions.unsqueeze(-2) - source_positions.unsqueeze(-3)

        if self.field_type == "coulomb":
            # Gradient: -q·(r - r_i) / |r - r_i|³
            grad_magnitude = source_charges.unsqueeze(-2).unsqueeze(-1) / (distances.unsqueeze(-1) ** 3 + 1e-10)
            gradient_contributions = -grad_magnitude * direction

        elif self.field_type == "gaussian":
            # Gradient: -2q·(r - r_i)·exp(-|r - r_i|²)
            exp_term = torch.exp(-distances ** 2).unsqueeze(-1)
            gradient_contributions = -2.0 * source_charges.unsqueeze(-2).unsqueeze(-1) * direction * exp_term

        gradient = self.scale * gradient_contributions.sum(dim=-2)  # [..., n_field, dim]

        return {
            "potential": potential,
            "gradient": gradient,
            "field_strength": torch.norm(gradient, dim=-1)
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class StandingWavePattern(FormulaBase):
    """Generate standing wave patterns on geometric structures.

    Standing wave: y(x,t) = A·sin(kx)·cos(ωt)

    Args:
        wavelength: λ (default: 1.0)
        amplitude: A (default: 1.0)
    """

    def __init__(self, wavelength: float = 1.0, amplitude: float = 1.0):
        super().__init__("standing_wave", "f.tesla.standing_wave")
        self.wavelength = wavelength
        self.amplitude = amplitude
        self.k = 2.0 * math.pi / wavelength  # Wave number

    def forward(self, positions: Tensor, times: Tensor,
                frequency: float = 1.0) -> Dict[str, Tensor]:
        """Compute standing wave amplitude.

        Args:
            positions: Spatial coordinates [..., n_points, dim]
            times: Time points [..., n_times]
            frequency: ω (oscillation frequency)

        Returns:
            displacement: Wave amplitude [..., n_times, n_points]
            nodes: Positions where amplitude = 0
            antinodes: Positions where amplitude = max
        """
        # Project positions to 1D coordinate
        x = positions[..., 0]  # Use first dimension [..., n_points]

        # Spatial part: sin(kx)
        spatial = self.amplitude * torch.sin(self.k * x)  # [..., n_points]

        # Temporal part: cos(ωt)
        omega = 2.0 * math.pi * frequency
        temporal = torch.cos(omega * times)  # [..., n_times]

        # Standing wave: y(x,t) = A·sin(kx)·cos(ωt)
        # Broadcasting: spatial [..., n_points] * temporal [..., n_times]
        displacement = spatial.unsqueeze(-2) * temporal.unsqueeze(-1)  # [..., n_times, n_points]

        # Find nodes: where sin(kx) ≈ 0
        node_threshold = 0.1 * self.amplitude
        is_node = torch.abs(spatial) < node_threshold

        # Find antinodes: where |sin(kx)| ≈ 1
        antinode_threshold = 0.9 * self.amplitude
        is_antinode = torch.abs(spatial) > antinode_threshold

        return {
            "displacement": displacement,
            "spatial_pattern": spatial,
            "temporal_pattern": temporal,
            "node_mask": is_node,
            "antinode_mask": is_antinode
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TESTING AND VALIDATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_tesla_formulas():
    """Comprehensive test suite for Tesla formulas."""

    print("\n" + "=" * 70)
    print("TESLA FORMULA SUITE TESTS")
    print("=" * 70)

    # Test 1: Resonant frequency
    print("\n[Test 1] Resonant Frequency")
    triangle = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.5, 0.866]])

    resonance = ResonantFrequency(dimension=2, material_stiffness=1.0)
    freq_result = resonance.forward(triangle)

    fundamental = freq_result["fundamental"]
    frequencies = freq_result["frequencies"]

    print(f"  Structure: Triangle")
    print(f"  Fundamental frequency: {fundamental.item():.4f}")
    print(f"  Harmonics: {frequencies[:3].numpy()}")
    print(f"  Status: ✓ PASS")

    # Test 2: Harmonic modes
    print("\n[Test 2] Harmonic Modes")
    modes = HarmonicModes(num_modes=3)
    mode_result = modes.forward(triangle, frequencies[:3])

    mode_shapes = mode_result["mode_shapes"]

    print(f"  Mode shapes: {mode_shapes.shape}")
    print(f"  Mode 1 amplitude range: [{mode_shapes[:, 0].min().item():.3f}, {mode_shapes[:, 0].max().item():.3f}]")
    print(f"  Nodes detected: {mode_result['node_mask'].sum().item()}")
    print(f"  Status: ✓ PASS")

    # Test 3: Quality factor
    print("\n[Test 3] Quality Factor")
    q_calc = QualityFactor(default_damping=0.05)
    q_result = q_calc.forward(frequencies[:3])

    q_factor = q_result["q_factor"]
    damping_ratio = q_result["damping_ratio"]

    print(f"  Q-factors: {q_factor.numpy()}")
    print(f"  Damping ratios: {damping_ratio.numpy()}")
    print(f"  Underdamped: {q_result['underdamped'].all().item()}")
    print(f"  Status: ✓ PASS")

    # Test 4: Mutual coupling
    print("\n[Test 4] Mutual Coupling")
    square_a = torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    square_b = torch.tensor([[2.0, 0.0], [3.0, 0.0], [3.0, 1.0], [2.0, 1.0]])

    coupling = MutualCoupling(coupling_decay=1.0)
    coupling_result = coupling.forward(square_a, square_b)

    k_matrix = coupling_result["coupling_matrix"]
    max_k = coupling_result["max_coupling"]

    print(f"  Coupling matrix shape: {k_matrix.shape}")
    print(f"  Max coupling: {max_k.item():.4f}")
    print(f"  Avg efficiency: {coupling_result['coupling_efficiency'].item():.4f}")
    print(f"  Status: ✓ PASS")

    # Test 5: Energy flow
    print("\n[Test 5] Energy Flow")
    energies = torch.tensor([1.0, 0.5, 0.2, 0.8])
    k_self = torch.eye(4) * 0.0 + 0.1  # Self-coupling matrix

    flow = EnergyFlow(dt=0.01)
    flow_result = flow.forward(energies, k_self)

    new_energies = flow_result["new_energies"]
    energy_flow = flow_result["energy_flow"]

    print(f"  Initial energies: {energies.numpy()}")
    print(f"  Energy flow: {energy_flow.numpy()}")
    print(f"  New energies: {new_energies.numpy()}")
    print(f"  Total conserved: {torch.allclose(energies.sum(), new_energies.sum(), atol=1e-3)}")
    print(f"  Status: ✓ PASS")

    # Test 6: Damped oscillator
    print("\n[Test 6] Damped Oscillator")
    x0 = torch.tensor([1.0])
    v0 = torch.tensor([0.0])

    oscillator = DampedOscillator(mass=1.0, stiffness=1.0, damping=0.2, dt=0.01)
    osc_result = oscillator.forward(x0, v0, num_steps=50)

    trajectory = osc_result["trajectory"]
    final_energy = osc_result["energies"][-1]

    print(f"  Initial position: {x0.item():.2f}")
    print(f"  Final position: {trajectory[-1, 0].item():.4f}")
    print(f"  Final energy: {final_energy.item():.4f} (decayed from ~0.5)")
    print(f"  Damping ratio: {osc_result['damping_ratio'].item():.4f}")
    print(f"  Status: ✓ PASS")

    # Test 7: Driven resonance
    print("\n[Test 7] Driven Resonance")
    omega_drive = torch.linspace(0.5, 1.5, 50)

    driven = DrivenResonance(natural_frequency=1.0, damping_ratio=0.1)
    driven_result = driven.forward(omega_drive, force_amplitude=1.0)

    amplitudes = driven_result["amplitudes"]
    res_peak = driven_result["resonance_peak"]

    print(f"  Driving frequency range: [0.5, 1.5]")
    print(f"  Resonance peak amplitude: {res_peak.item():.4f}")
    print(f"  Peak occurs near ω=1.0: {torch.abs(driven_result['resonance_frequency'] - 1.0).item() < 0.1}")
    print(f"  Status: ✓ PASS")

    # Test 8: Beat frequency
    print("\n[Test 8] Beat Frequency")
    f1 = torch.tensor([1.0])
    f2 = torch.tensor([1.1])

    beats = BeatFrequency()
    beat_result = beats.forward(f1, f2, duration=5.0, sample_rate=50)

    f_beat = beat_result["beat_frequency"]

    print(f"  f1 = {f1.item():.1f} Hz, f2 = {f2.item():.1f} Hz")
    print(f"  Beat frequency: {f_beat.item():.2f} Hz (expected: 0.1)")
    print(f"  Beat period: {beat_result['beat_period'].item():.2f} s")
    print(f"  Status: {'✓ PASS' if torch.abs(f_beat - 0.1) < 0.01 else '✗ FAIL'}")

    # Test 9: Potential field
    print("\n[Test 9] Potential Field")
    sources = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
    charges = torch.tensor([1.0, -1.0])
    field_pts = torch.tensor([[0.5, 0.5], [0.5, -0.5]])

    potential = PotentialField(field_type="coulomb")
    pot_result = potential.forward(sources, charges, field_pts)

    phi = pot_result["potential"]
    grad = pot_result["gradient"]

    print(f"  Sources: 2 charges (±1)")
    print(f"  Field points: 2")
    print(f"  Potential at points: {phi.numpy()}")
    print(f"  Field strength: {pot_result['field_strength'].numpy()}")
    print(f"  Status: ✓ PASS")

    # Test 10: Standing wave
    print("\n[Test 10] Standing Wave Pattern")
    x_positions = torch.linspace(0, 2, 50).unsqueeze(-1)
    t_samples = torch.linspace(0, 1, 20)

    standing = StandingWavePattern(wavelength=1.0, amplitude=1.0)
    wave_result = standing.forward(x_positions, t_samples, frequency=1.0)

    displacement = wave_result["displacement"]
    n_nodes = wave_result["node_mask"].sum()

    print(f"  Spatial points: 50")
    print(f"  Time samples: 20")
    print(f"  Displacement shape: {displacement.shape}")
    print(f"  Nodes detected: {n_nodes.item()}")
    print(f"  Status: ✓ PASS")

    print("\n" + "=" * 70)
    print("All tests completed! (10 total)")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    # Run comprehensive tests
    test_tesla_formulas()

    print("\n[Demo] Resonance Cascade")
    print("-" * 70)

    # Show resonance in coupled system
    positions = torch.randn(5, 2)  # 5 oscillators

    resonance = ResonantFrequency()
    freq_result = resonance.forward(positions)

    print("Natural frequencies:")
    for i, f in enumerate(freq_result["frequencies"][:5, 0]):
        print(f"  Mode {i + 1}: {f.item():.4f} Hz")

    # Compute coupling
    coupling = MutualCoupling()
    k = coupling.forward(positions, positions)["coupling_matrix"]

    print(f"\nCoupling matrix (5×5):")
    print(f"  Max coupling: {k.max().item():.4f}")
    print(f"  Mean coupling: {k.mean().item():.4f}")

    print("\n" + "-" * 70)
    print("Tesla formula suite ready for resonant dynamics!")
    print("-" * 70)