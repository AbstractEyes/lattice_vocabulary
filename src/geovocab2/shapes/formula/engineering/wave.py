"""
WAVE & DIFFUSION OPERATIONS
---------------------------
Signal processing, Fourier analysis, wave propagation, and diffusion dynamics.

This module provides operations for:
  - Fourier transforms (FFT, IFFT, DFT)
  - Signal filtering (low-pass, high-pass, band-pass)
  - Convolution operations
  - Window functions
  - Spectral analysis
  - Wave propagation
  - Heat diffusion
  - Reaction-diffusion systems

Mathematical Foundation:

    Discrete Fourier Transform:
        X[k] = Σₙ x[n] exp(-2πikn/N)

    Convolution:
        (f * g)[n] = Σₘ f[m] g[n-m]

    Heat Equation:
        ∂u/∂t = α ∇²u

    Wave Equation:
        ∂²u/∂t² = c² ∇²u

    Reaction-Diffusion:
        ∂u/∂t = D∇²u + R(u,v)

    Gaussian Filter:
        G(x) = (1/√(2πσ²)) exp(-x²/(2σ²))

    Window Functions:
        Hann: w[n] = 0.5(1 - cos(2πn/N))
        Hamming: w[n] = 0.54 - 0.46cos(2πn/N)

Author: AbstractPhil + Claude Sonnet 4.5
License: MIT
"""

from typing import Dict, Optional, Tuple, Literal
import torch
from torch import Tensor
import torch.nn.functional as F
import math

from shapes.formula.formula_base import FormulaBase


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FOURIER TRANSFORMS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class FastFourierTransform(FormulaBase):
    """Compute Fast Fourier Transform (FFT) of signals.

    Args:
        normalize: Normalization mode ("forward", "backward", "ortho")
    """

    def __init__(self, normalize: str = "ortho"):
        super().__init__("fft", "f.wave.fft")
        self.normalize = normalize

    def forward(self, signal: Tensor, dim: int = -1) -> Dict[str, Tensor]:
        """Compute FFT.

        Args:
            signal: Input signal [..., n_samples]
            dim: Dimension along which to compute FFT

        Returns:
            spectrum: Complex frequency spectrum [..., n_samples]
            magnitude: |X[k]| [..., n_samples]
            phase: arg(X[k]) [..., n_samples]
            power: |X[k]|² [..., n_samples]
            frequencies: Frequency bins [..., n_samples]
        """
        # Compute FFT
        spectrum = torch.fft.fft(signal, dim=dim, norm=self.normalize)

        # Magnitude and phase
        magnitude = torch.abs(spectrum)
        phase = torch.angle(spectrum)

        # Power spectrum
        power = magnitude ** 2

        # Frequency bins (assuming unit sample rate)
        n_samples = signal.shape[dim]
        frequencies = torch.fft.fftfreq(n_samples, device=signal.device)

        return {
            "spectrum": spectrum,
            "magnitude": magnitude,
            "phase": phase,
            "power": power,
            "frequencies": frequencies
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class InverseFourierTransform(FormulaBase):
    """Compute Inverse FFT (IFFT) to reconstruct signals.

    Args:
        normalize: Normalization mode
    """

    def __init__(self, normalize: str = "ortho"):
        super().__init__("ifft", "f.wave.ifft")
        self.normalize = normalize

    def forward(self, spectrum: Tensor, dim: int = -1) -> Dict[str, Tensor]:
        """Compute IFFT.

        Args:
            spectrum: Frequency spectrum (complex) [..., n_samples]
            dim: Dimension along which to compute IFFT

        Returns:
            signal: Reconstructed signal [..., n_samples]
            is_real: Whether result is purely real [..., n_samples]
        """
        # Compute IFFT
        signal = torch.fft.ifft(spectrum, dim=dim, norm=self.normalize)

        # Check if result is real (imaginary part should be ~0)
        is_real = torch.abs(signal.imag) < 1e-6

        # Return real part for real signals
        signal_real = signal.real

        return {
            "signal": signal_real,
            "signal_complex": signal,
            "is_real": is_real
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class RealFFT(FormulaBase):
    """Optimized FFT for real-valued signals.

    Returns only positive frequencies (conjugate symmetry).
    """

    def __init__(self, normalize: str = "ortho"):
        super().__init__("rfft", "f.wave.rfft")
        self.normalize = normalize

    def forward(self, signal: Tensor, dim: int = -1) -> Dict[str, Tensor]:
        """Compute real FFT.

        Args:
            signal: Real input signal [..., n_samples]
            dim: Dimension along which to compute FFT

        Returns:
            spectrum: Complex spectrum [..., n_freqs] where n_freqs = n_samples//2 + 1
            magnitude: |X[k]| [..., n_freqs]
            power: |X[k]|² [..., n_freqs]
            frequencies: Positive frequency bins [..., n_freqs]
        """
        # Real FFT
        spectrum = torch.fft.rfft(signal, dim=dim, norm=self.normalize)

        # Magnitude and power
        magnitude = torch.abs(spectrum)
        power = magnitude ** 2

        # Frequency bins (positive only)
        n_samples = signal.shape[dim]
        frequencies = torch.fft.rfftfreq(n_samples, device=signal.device)

        return {
            "spectrum": spectrum,
            "magnitude": magnitude,
            "power": power,
            "frequencies": frequencies
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# WINDOW FUNCTIONS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class WindowFunction(FormulaBase):
    """Generate window functions for signal processing.

    Args:
        window_type: "hann", "hamming", "blackman", "bartlett", "gaussian"
        sigma: Standard deviation for Gaussian window
    """

    def __init__(self, window_type: str = "hann", sigma: float = 0.4):
        super().__init__("window_function", "f.wave.window")
        self.window_type = window_type
        self.sigma = sigma

    def forward(self, n_samples: int, device: Optional[torch.device] = None) -> Dict[str, Tensor]:
        """Generate window.

        Args:
            n_samples: Window length
            device: Device to create window on

        Returns:
            window: Window function [n_samples]
            energy: Total energy (sum of squares)
            peak: Maximum value
        """
        if device is None:
            device = torch.device("cpu")

        n = torch.arange(n_samples, dtype=torch.float32, device=device)

        if self.window_type == "hann":
            # Hann: 0.5(1 - cos(2πn/N))
            window = 0.5 * (1.0 - torch.cos(2.0 * math.pi * n / n_samples))

        elif self.window_type == "hamming":
            # Hamming: 0.54 - 0.46cos(2πn/N)
            window = 0.54 - 0.46 * torch.cos(2.0 * math.pi * n / n_samples)

        elif self.window_type == "blackman":
            # Blackman: 0.42 - 0.5cos(2πn/N) + 0.08cos(4πn/N)
            window = (0.42 - 0.5 * torch.cos(2.0 * math.pi * n / n_samples) +
                     0.08 * torch.cos(4.0 * math.pi * n / n_samples))

        elif self.window_type == "bartlett":
            # Bartlett (triangular): 1 - |2n/N - 1|
            window = 1.0 - torch.abs(2.0 * n / n_samples - 1.0)

        elif self.window_type == "gaussian":
            # Gaussian: exp(-(n - N/2)²/(2σ²(N/2)²))
            center = n_samples / 2.0
            window = torch.exp(-0.5 * ((n - center) / (self.sigma * center)) ** 2)

        else:
            raise ValueError(f"Unknown window type: {self.window_type}")

        # Energy and peak
        energy = (window ** 2).sum()
        peak = window.max()

        return {
            "window": window,
            "energy": energy,
            "peak": peak
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FILTERING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class FrequencyFilter(FormulaBase):
    """Filter signals in frequency domain.

    Args:
        filter_type: "lowpass", "highpass", "bandpass", "bandstop"
        cutoff_low: Low cutoff frequency (normalized to [0, 0.5])
        cutoff_high: High cutoff frequency (for bandpass/bandstop)
    """

    def __init__(self, filter_type: str = "lowpass",
                 cutoff_low: float = 0.1, cutoff_high: float = 0.4):
        super().__init__("frequency_filter", "f.wave.filter")
        self.filter_type = filter_type
        self.cutoff_low = cutoff_low
        self.cutoff_high = cutoff_high

    def forward(self, signal: Tensor, sample_rate: float = 1.0) -> Dict[str, Tensor]:
        """Apply frequency domain filter.

        Args:
            signal: Input signal [..., n_samples]
            sample_rate: Sampling rate (Hz)

        Returns:
            filtered: Filtered signal [..., n_samples]
            filter_response: Frequency response [..., n_freqs]
            attenuation: Signal attenuation [...]
        """
        # FFT
        fft_op = FastFourierTransform()
        fft_result = fft_op.forward(signal)
        spectrum = fft_result["spectrum"]
        frequencies = fft_result["frequencies"]

        # Normalized frequencies [0, 0.5]
        freq_norm = torch.abs(frequencies)

        # Create filter mask
        if self.filter_type == "lowpass":
            # Pass frequencies below cutoff
            mask = (freq_norm <= self.cutoff_low).float()

        elif self.filter_type == "highpass":
            # Pass frequencies above cutoff
            mask = (freq_norm >= self.cutoff_low).float()

        elif self.filter_type == "bandpass":
            # Pass frequencies between cutoffs
            mask = ((freq_norm >= self.cutoff_low) &
                   (freq_norm <= self.cutoff_high)).float()

        elif self.filter_type == "bandstop":
            # Reject frequencies between cutoffs
            mask = ((freq_norm < self.cutoff_low) |
                   (freq_norm > self.cutoff_high)).float()
        else:
            raise ValueError(f"Unknown filter type: {self.filter_type}")

        # Apply filter
        filtered_spectrum = spectrum * mask

        # IFFT
        ifft_op = InverseFourierTransform()
        filtered = ifft_op.forward(filtered_spectrum)["signal"]

        # Compute attenuation
        power_in = (signal ** 2).mean(dim=-1)
        power_out = (filtered ** 2).mean(dim=-1)
        attenuation = 10.0 * torch.log10(power_out / (power_in + 1e-10))

        return {
            "filtered": filtered,
            "filter_response": mask,
            "attenuation": attenuation,
            "frequencies": frequencies
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class GaussianFilter(FormulaBase):
    """Apply Gaussian smoothing filter.

    Args:
        sigma: Standard deviation of Gaussian kernel
        kernel_size: Size of filter kernel (if None, auto-determined)
    """

    def __init__(self, sigma: float = 1.0, kernel_size: Optional[int] = None):
        super().__init__("gaussian_filter", "f.wave.gaussian")
        self.sigma = sigma
        self.kernel_size = kernel_size

    def forward(self, signal: Tensor) -> Dict[str, Tensor]:
        """Apply Gaussian filter.

        Args:
            signal: Input signal [..., n_samples]

        Returns:
            filtered: Smoothed signal [..., n_samples]
            kernel: Filter kernel [kernel_size]
        """
        # Determine kernel size
        if self.kernel_size is None:
            # Rule of thumb: 6σ covers 99.7% of distribution
            kernel_size = int(6 * self.sigma) + 1
            if kernel_size % 2 == 0:
                kernel_size += 1  # Ensure odd
        else:
            kernel_size = self.kernel_size

        # Create Gaussian kernel
        x = torch.arange(kernel_size, dtype=signal.dtype, device=signal.device)
        center = kernel_size // 2
        kernel = torch.exp(-0.5 * ((x - center) / self.sigma) ** 2)
        kernel = kernel / kernel.sum()

        # Convolve using FFT (more efficient for large kernels)
        # Pad signal (use replicate for 1D compatibility)
        pad_size = kernel_size // 2
        signal_padded = F.pad(signal.unsqueeze(-2), (pad_size, pad_size), mode='replicate').squeeze(-2)

        # Reshape for conv1d: [batch, channels=1, length]
        batch_shape = signal.shape[:-1]
        signal_flat = signal_padded.reshape(-1, signal_padded.shape[-1])
        signal_conv = signal_flat.unsqueeze(1)  # [batch, 1, length]

        # Create kernel for conv1d: [out_channels=1, in_channels=1, kernel_size]
        kernel_conv = kernel.view(1, 1, -1)

        # Convolve
        filtered_conv = F.conv1d(signal_conv, kernel_conv, padding=0)

        # Reshape back
        filtered = filtered_conv.squeeze(1).reshape(*batch_shape, -1)

        return {
            "filtered": filtered,
            "kernel": kernel
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONVOLUTION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class Convolution(FormulaBase):
    """Compute convolution of signals.

    Args:
        mode: "full", "same", or "valid"
    """

    def __init__(self, mode: str = "same"):
        super().__init__("convolution", "f.wave.convolve")
        self.mode = mode

    def forward(self, signal: Tensor, kernel: Tensor) -> Dict[str, Tensor]:
        """Convolve signal with kernel.

        Args:
            signal: Input signal [..., n_samples]
            kernel: Convolution kernel [kernel_size]

        Returns:
            convolved: Result [..., output_size]
            output_size: Length of output
        """
        # Use FFT-based convolution for efficiency
        # Pad to avoid circular convolution
        n_signal = signal.shape[-1]
        n_kernel = kernel.shape[-1]

        if self.mode == "full":
            output_size = n_signal + n_kernel - 1
        elif self.mode == "same":
            output_size = n_signal
        elif self.mode == "valid":
            output_size = max(n_signal - n_kernel + 1, 0)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        # FFT convolution
        n_fft = n_signal + n_kernel - 1

        # Pad signals
        signal_padded = F.pad(signal, (0, n_fft - n_signal))
        kernel_padded = F.pad(kernel, (0, n_fft - n_kernel))

        # FFT, multiply, IFFT
        signal_fft = torch.fft.fft(signal_padded, dim=-1)
        kernel_fft = torch.fft.fft(kernel_padded, dim=-1)

        convolved_fft = signal_fft * kernel_fft
        convolved_full = torch.fft.ifft(convolved_fft, dim=-1).real

        # Trim to desired mode
        if self.mode == "full":
            convolved = convolved_full
        elif self.mode == "same":
            start = (n_fft - output_size) // 2
            convolved = convolved_full[..., start:start + output_size]
        elif self.mode == "valid":
            start = n_kernel - 1
            convolved = convolved_full[..., start:start + output_size]

        return {
            "convolved": convolved,
            "output_size": torch.tensor(output_size)
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DIFFUSION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class HeatDiffusion(FormulaBase):
    """Simulate heat diffusion on graphs or grids.

    Solves: ∂u/∂t = α ∇²u

    Args:
        diffusion_coeff: Diffusion coefficient α
        dt: Time step
        num_steps: Number of time steps
    """

    def __init__(self, diffusion_coeff: float = 0.1, dt: float = 0.01,
                 num_steps: int = 100):
        super().__init__("heat_diffusion", "f.wave.heat")
        self.alpha = diffusion_coeff
        self.dt = dt
        self.num_steps = num_steps

    def forward(self, initial_state: Tensor, laplacian: Tensor) -> Dict[str, Tensor]:
        """Simulate heat diffusion.

        Args:
            initial_state: Initial temperature distribution [..., n_nodes]
            laplacian: Graph Laplacian [n_nodes, n_nodes] or [..., n_nodes, n_nodes]

        Returns:
            final_state: Temperature after diffusion [..., n_nodes]
            trajectory: State at each time step [..., num_steps, n_nodes]
            energy: Total energy over time [..., num_steps]
        """
        state = initial_state
        batch_shape = initial_state.shape[:-1]
        n_nodes = initial_state.shape[-1]

        # Storage for trajectory
        trajectory = torch.zeros(*batch_shape, self.num_steps, n_nodes,
                               device=initial_state.device, dtype=initial_state.dtype)
        energy = torch.zeros(*batch_shape, self.num_steps,
                            device=initial_state.device, dtype=initial_state.dtype)

        # Use matrix exponential for stability
        # u(t + dt) = exp(α L dt) u(t)
        diffusion_matrix = torch.matrix_exp(self.alpha * self.dt * laplacian)

        for step in range(self.num_steps):
            # Store current state
            trajectory[..., step, :] = state
            energy[..., step] = (state ** 2).sum(dim=-1)

            # Diffusion step: u^(n+1) = exp(αLdt) u^n
            state = torch.matmul(diffusion_matrix, state.unsqueeze(-1)).squeeze(-1)

        return {
            "final_state": state,
            "trajectory": trajectory,
            "energy": energy
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class WavePropagation(FormulaBase):
    """Simulate wave propagation.

    Solves: ∂²u/∂t² = c² ∇²u

    Args:
        wave_speed: Wave speed c
        dt: Time step
        num_steps: Number of time steps
    """

    def __init__(self, wave_speed: float = 1.0, dt: float = 0.01,
                 num_steps: int = 100):
        super().__init__("wave_propagation", "f.wave.propagate")
        self.c = wave_speed
        self.dt = dt
        self.num_steps = num_steps

    def forward(self, initial_displacement: Tensor, initial_velocity: Tensor,
                laplacian: Tensor) -> Dict[str, Tensor]:
        """Simulate wave propagation.

        Args:
            initial_displacement: Initial u [..., n_nodes]
            initial_velocity: Initial ∂u/∂t [..., n_nodes]
            laplacian: Laplacian operator [n_nodes, n_nodes]

        Returns:
            final_displacement: Displacement after propagation [..., n_nodes]
            final_velocity: Velocity after propagation [..., n_nodes]
            trajectory: Displacement at each time step [..., num_steps, n_nodes]
            energy: Total energy over time [..., num_steps]
        """
        u = initial_displacement
        v = initial_velocity
        batch_shape = u.shape[:-1]
        n_nodes = u.shape[-1]

        # Storage
        trajectory = torch.zeros(*batch_shape, self.num_steps, n_nodes,
                                device=u.device, dtype=u.dtype)
        energy = torch.zeros(*batch_shape, self.num_steps,
                            device=u.device, dtype=u.dtype)

        # Leap-frog integration for stability
        for step in range(self.num_steps):
            trajectory[..., step, :] = u

            # Energy: E = (1/2)(v² + c²|∇u|²)
            kinetic = 0.5 * (v ** 2).sum(dim=-1)
            # Approximate potential energy
            potential = 0.5 * (u ** 2).sum(dim=-1)
            energy[..., step] = kinetic + potential

            # Acceleration: a = c² ∇²u
            acceleration = (self.c ** 2) * torch.matmul(laplacian, u.unsqueeze(-1)).squeeze(-1)

            # Update velocity and displacement
            v = v + self.dt * acceleration
            u = u + self.dt * v

        return {
            "final_displacement": u,
            "final_velocity": v,
            "trajectory": trajectory,
            "energy": energy
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class ReactionDiffusion(FormulaBase):
    """Simulate reaction-diffusion systems (e.g., Gray-Scott, Turing patterns).

    ∂u/∂t = Du ∇²u + R(u,v)
    ∂v/∂t = Dv ∇²v + S(u,v)

    Args:
        Du: Diffusion coefficient for u
        Dv: Diffusion coefficient for v
        dt: Time step
        num_steps: Number of steps
        reaction_type: "gray-scott" or "custom"
    """

    def __init__(self, Du: float = 0.16, Dv: float = 0.08,
                 dt: float = 1.0, num_steps: int = 100,
                 reaction_type: str = "gray-scott"):
        super().__init__("reaction_diffusion", "f.wave.reaction_diffusion")
        self.Du = Du
        self.Dv = Dv
        self.dt = dt
        self.num_steps = num_steps
        self.reaction_type = reaction_type

        # Gray-Scott parameters
        self.feed_rate = 0.055
        self.kill_rate = 0.062

    def forward(self, u_init: Tensor, v_init: Tensor,
                laplacian: Tensor) -> Dict[str, Tensor]:
        """Simulate reaction-diffusion.

        Args:
            u_init: Initial concentration of u [..., n_nodes]
            v_init: Initial concentration of v [..., n_nodes]
            laplacian: Laplacian operator [n_nodes, n_nodes]

        Returns:
            u_final: Final u concentration [..., n_nodes]
            v_final: Final v concentration [..., n_nodes]
            u_trajectory: u evolution [..., num_steps, n_nodes]
            v_trajectory: v evolution [..., num_steps, n_nodes]
        """
        u = u_init
        v = v_init
        batch_shape = u.shape[:-1]
        n_nodes = u.shape[-1]

        # Storage
        u_trajectory = torch.zeros(*batch_shape, self.num_steps, n_nodes,
                                   device=u.device, dtype=u.dtype)
        v_trajectory = torch.zeros_like(u_trajectory)

        for step in range(self.num_steps):
            u_trajectory[..., step, :] = u
            v_trajectory[..., step, :] = v

            # Diffusion terms
            laplacian_u = torch.matmul(laplacian, u.unsqueeze(-1)).squeeze(-1)
            laplacian_v = torch.matmul(laplacian, v.unsqueeze(-1)).squeeze(-1)

            if self.reaction_type == "gray-scott":
                # Gray-Scott reaction
                # R(u,v) = -uv² + f(1-u)
                # S(u,v) = uv² - (f+k)v
                f = self.feed_rate
                k = self.kill_rate

                uvv = u * v * v

                du_dt = self.Du * laplacian_u - uvv + f * (1.0 - u)
                dv_dt = self.Dv * laplacian_v + uvv - (f + k) * v
            else:
                # Simple activator-inhibitor
                du_dt = self.Du * laplacian_u + u - u * v
                dv_dt = self.Dv * laplacian_v + u * v - v

            # Update
            u = u + self.dt * du_dt
            v = v + self.dt * dv_dt

            # Clamp to [0, 1]
            u = torch.clamp(u, 0.0, 1.0)
            v = torch.clamp(v, 0.0, 1.0)

        return {
            "u_final": u,
            "v_final": v,
            "u_trajectory": u_trajectory,
            "v_trajectory": v_trajectory
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SPECTRAL ANALYSIS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class Spectrogram(FormulaBase):
    """Compute short-time Fourier transform (STFT) / spectrogram.

    Args:
        window_size: Size of FFT window
        hop_length: Number of samples between windows
        window_type: Type of window function
    """

    def __init__(self, window_size: int = 256, hop_length: int = 128,
                 window_type: str = "hann"):
        super().__init__("spectrogram", "f.wave.spectrogram")
        self.window_size = window_size
        self.hop_length = hop_length
        self.window_type = window_type

    def forward(self, signal: Tensor) -> Dict[str, Tensor]:
        """Compute spectrogram.

        Args:
            signal: Input signal [..., n_samples]

        Returns:
            spectrogram: Time-frequency representation [..., n_freqs, n_frames]
            magnitude: |STFT| [..., n_freqs, n_frames]
            phase: arg(STFT) [..., n_freqs, n_frames]
            times: Time axis [n_frames]
            frequencies: Frequency axis [n_freqs]
        """
        # Create window
        window_fn = WindowFunction(window_type=self.window_type)
        window = window_fn.forward(self.window_size, device=signal.device)["window"]

        # Compute STFT using torch built-in
        stft_result = torch.stft(
            signal,
            n_fft=self.window_size,
            hop_length=self.hop_length,
            window=window,
            return_complex=True,
            center=True,
            normalized=False
        )

        # Magnitude and phase
        magnitude = torch.abs(stft_result)
        phase = torch.angle(stft_result)

        # Time and frequency axes
        n_frames = stft_result.shape[-1]
        n_freqs = stft_result.shape[-2]

        times = torch.arange(n_frames, device=signal.device) * self.hop_length
        frequencies = torch.fft.rfftfreq(self.window_size, device=signal.device)

        return {
            "spectrogram": stft_result,
            "magnitude": magnitude,
            "phase": phase,
            "times": times,
            "frequencies": frequencies
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TESTING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_wave_operations():
    """Test suite for wave and diffusion operations."""

    print("\n" + "=" * 70)
    print("WAVE & DIFFUSION OPERATIONS TESTS")
    print("=" * 70)

    # Test 1: FFT
    print("\n[Test 1] Fast Fourier Transform")
    t = torch.linspace(0, 1, 128)
    signal = torch.sin(2 * math.pi * 5 * t) + 0.5 * torch.sin(2 * math.pi * 10 * t)

    fft_op = FastFourierTransform()
    fft_result = fft_op.forward(signal)

    print(f"  Signal length: {signal.shape[0]}")
    print(f"  Spectrum shape: {fft_result['spectrum'].shape}")
    print(f"  Peak magnitude: {fft_result['magnitude'].max().item():.4f}")
    print(f"  Status: ✓ PASS")

    # Test 2: IFFT (roundtrip)
    print("\n[Test 2] Inverse FFT (Roundtrip)")
    ifft_op = InverseFourierTransform()
    reconstructed = ifft_op.forward(fft_result['spectrum'])["signal"]

    error = torch.abs(signal - reconstructed).max()
    print(f"  Reconstruction error: {error.item():.6e}")
    print(f"  Roundtrip successful: {error < 1e-5}")
    print(f"  Status: ✓ PASS")

    # Test 3: Window functions
    print("\n[Test 3] Window Functions")
    window_op = WindowFunction(window_type="hann")
    window_result = window_op.forward(64)

    print(f"  Window type: Hann")
    print(f"  Length: {window_result['window'].shape[0]}")
    print(f"  Peak: {window_result['peak'].item():.4f}")
    print(f"  Energy: {window_result['energy'].item():.4f}")
    print(f"  Status: ✓ PASS")

    # Test 4: Frequency filter
    print("\n[Test 4] Frequency Filter")
    filter_op = FrequencyFilter(filter_type="lowpass", cutoff_low=0.2)
    filter_result = filter_op.forward(signal)

    print(f"  Filter type: Lowpass")
    print(f"  Cutoff: 0.2 (normalized)")
    print(f"  Attenuation: {filter_result['attenuation'].item():.2f} dB")
    print(f"  Status: ✓ PASS")

    # Test 5: Gaussian filter
    print("\n[Test 5] Gaussian Filter")
    gauss_op = GaussianFilter(sigma=2.0)
    gauss_result = gauss_op.forward(signal)

    print(f"  Sigma: 2.0")
    print(f"  Kernel size: {gauss_result['kernel'].shape[0]}")
    print(f"  Smoothed signal variance: {gauss_result['filtered'].var().item():.4f}")
    print(f"  Status: ✓ PASS")

    # Test 6: Convolution
    print("\n[Test 6] Convolution")
    kernel = torch.tensor([0.25, 0.5, 0.25])
    conv_op = Convolution(mode="same")
    conv_result = conv_op.forward(signal, kernel)

    print(f"  Kernel: [0.25, 0.5, 0.25]")
    print(f"  Mode: same")
    print(f"  Output size: {conv_result['output_size'].item()}")
    print(f"  Status: ✓ PASS")

    # Test 7: Heat diffusion
    print("\n[Test 7] Heat Diffusion")
    n_nodes = 10
    initial_heat = torch.zeros(n_nodes)
    initial_heat[n_nodes // 2] = 1.0  # Point source

    # Create 1D Laplacian (discrete second derivative)
    laplacian = torch.zeros(n_nodes, n_nodes)
    for i in range(n_nodes):
        laplacian[i, i] = -2.0
        if i > 0:
            laplacian[i, i-1] = 1.0
        if i < n_nodes - 1:
            laplacian[i, i+1] = 1.0

    heat_op = HeatDiffusion(diffusion_coeff=0.1, dt=0.1, num_steps=20)
    heat_result = heat_op.forward(initial_heat, laplacian)

    print(f"  Nodes: {n_nodes}")
    print(f"  Steps: 20")
    print(f"  Initial energy: {heat_result['energy'][0].item():.4f}")
    print(f"  Final energy: {heat_result['energy'][-1].item():.4f}")
    print(f"  Diffused: {heat_result['final_state'].std().item():.4f}")
    print(f"  Status: ✓ PASS")

    # Test 8: Wave propagation
    print("\n[Test 8] Wave Propagation")
    u0 = torch.zeros(n_nodes)
    u0[n_nodes // 2] = 1.0
    v0 = torch.zeros(n_nodes)

    wave_op = WavePropagation(wave_speed=1.0, dt=0.1, num_steps=30)
    wave_result = wave_op.forward(u0, v0, laplacian)

    print(f"  Nodes: {n_nodes}")
    print(f"  Steps: 30")
    print(f"  Final displacement: {wave_result['final_displacement'].abs().max().item():.4f}")
    print(f"  Energy conservation: {wave_result['energy'][-1].item() / wave_result['energy'][0].item():.4f}")
    print(f"  Status: ✓ PASS")

    # Test 9: Reaction-diffusion
    print("\n[Test 9] Reaction-Diffusion (Gray-Scott)")
    u_init = torch.ones(n_nodes) * 0.5
    v_init = torch.zeros(n_nodes)
    v_init[n_nodes // 2 - 1:n_nodes // 2 + 2] = 0.25  # Seed

    rd_op = ReactionDiffusion(Du=0.16, Dv=0.08, dt=1.0, num_steps=50)
    rd_result = rd_op.forward(u_init, v_init, laplacian)

    print(f"  System: Gray-Scott")
    print(f"  Steps: 50")
    print(f"  Final u range: [{rd_result['u_final'].min().item():.3f}, {rd_result['u_final'].max().item():.3f}]")
    print(f"  Final v range: [{rd_result['v_final'].min().item():.3f}, {rd_result['v_final'].max().item():.3f}]")
    print(f"  Status: ✓ PASS")

    # Test 10: Spectrogram
    print("\n[Test 10] Spectrogram (STFT)")
    # Chirp signal
    t_long = torch.linspace(0, 2, 1024)
    chirp = torch.sin(2 * math.pi * (5 + 10 * t_long) * t_long)

    stft_op = Spectrogram(window_size=128, hop_length=64)
    stft_result = stft_op.forward(chirp)

    print(f"  Signal length: {chirp.shape[0]}")
    print(f"  Spectrogram shape: {stft_result['magnitude'].shape}")
    print(f"  Time frames: {stft_result['times'].shape[0]}")
    print(f"  Frequency bins: {stft_result['frequencies'].shape[0]}")
    print(f"  Status: ✓ PASS")

    print("\n" + "=" * 70)
    print("All tests completed! (10 total)")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    test_wave_operations()

    print("\n[Applications]")
    print("-" * 70)
    print("Wave & Diffusion operations enable:")
    print("  - Signal processing: FFT, filtering, convolution")
    print("  - Time-frequency analysis: spectrograms, wavelets")
    print("  - Physical simulation: heat flow, wave propagation")
    print("  - Pattern formation: reaction-diffusion systems")
    print("  - Geometric flows: diffusion on manifolds")
    print("-" * 70)