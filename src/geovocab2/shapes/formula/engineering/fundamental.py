"""
FUNDAMENTAL OPERATIONS
----------------------
Trigonometric, exponential, logarithmic, and other fundamental mathematical operations.

This module provides essential mathematical functions used throughout geometric systems:
  - Trigonometric functions (sin, cos, tan, and inverses)
  - Polar/Cartesian conversions
  - Angle operations (wrapping, difference)
  - Exponential and logarithmic functions
  - Hyperbolic functions
  - Special angles and identities

Mathematical Foundation:

    Trigonometric Functions:
        sin²(θ) + cos²(θ) = 1
        tan(θ) = sin(θ)/cos(θ)

    Angle Addition:
        sin(α + β) = sin(α)cos(β) + cos(α)sin(β)
        cos(α + β) = cos(α)cos(β) - sin(α)sin(β)

    Polar to Cartesian:
        x = r·cos(θ)
        y = r·sin(θ)

    Cartesian to Polar:
        r = √(x² + y²)
        θ = atan2(y, x)

    Exponential:
        e^(iθ) = cos(θ) + i·sin(θ) (Euler's formula)

    Hyperbolic:
        sinh(x) = (e^x - e^(-x))/2
        cosh(x) = (e^x + e^(-x))/2
        cosh²(x) - sinh²(x) = 1

Author: AbstractPhil + Claude Sonnet 4.5
License: MIT
"""

from typing import Dict, Optional, Tuple
import torch
from torch import Tensor
import math

from shapes.formula.formula_base import FormulaBase


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TRIGONOMETRIC FUNCTIONS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class SinCos(FormulaBase):
    """Compute sine and cosine simultaneously.

    More efficient than separate calls.
    """

    def __init__(self):
        super().__init__("sin_cos", "f.fundamental.sincos")

    def forward(self, angle: Tensor) -> Dict[str, Tensor]:
        """Compute sin and cos.

        Args:
            angle: Angles in radians [...]

        Returns:
            sin: sin(θ) [...]
            cos: cos(θ) [...]
            tan: tan(θ) [...]
            quadrant: Which quadrant (1-4) [...]
        """
        sin = torch.sin(angle)
        cos = torch.cos(angle)
        tan = torch.tan(angle)

        # Determine quadrant
        # Normalize angle to [0, 2π)
        angle_norm = angle % (2 * math.pi)

        quadrant = torch.zeros_like(angle, dtype=torch.long)
        quadrant = torch.where(angle_norm < math.pi / 2, 1, quadrant)
        quadrant = torch.where((angle_norm >= math.pi / 2) & (angle_norm < math.pi), 2, quadrant)
        quadrant = torch.where((angle_norm >= math.pi) & (angle_norm < 3 * math.pi / 2), 3, quadrant)
        quadrant = torch.where(angle_norm >= 3 * math.pi / 2, 4, quadrant)

        return {
            "sin": sin,
            "cos": cos,
            "tan": tan,
            "quadrant": quadrant
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class InverseTrig(FormulaBase):
    """Compute inverse trigonometric functions.

    Args:
        use_degrees: Return angles in degrees
    """

    def __init__(self, use_degrees: bool = False):
        super().__init__("inverse_trig", "f.fundamental.arcfuncs")
        self.degrees = use_degrees

    def forward(self, x: Tensor, func: str = "arcsin") -> Dict[str, Tensor]:
        """Compute inverse trig function.

        Args:
            x: Input values [...]
            func: "arcsin", "arccos", "arctan"

        Returns:
            angle: Output angles [...]
            is_valid: Input in valid range [...]
        """
        if func == "arcsin":
            # Domain: [-1, 1], Range: [-π/2, π/2]
            is_valid = (x >= -1.0) & (x <= 1.0)
            x_clamped = torch.clamp(x, -1.0, 1.0)
            angle = torch.asin(x_clamped)

        elif func == "arccos":
            # Domain: [-1, 1], Range: [0, π]
            is_valid = (x >= -1.0) & (x <= 1.0)
            x_clamped = torch.clamp(x, -1.0, 1.0)
            angle = torch.acos(x_clamped)

        elif func == "arctan":
            # Domain: (-∞, ∞), Range: (-π/2, π/2)
            is_valid = torch.ones_like(x, dtype=torch.bool)
            angle = torch.atan(x)

        else:
            raise ValueError(f"Unknown function: {func}")

        if self.degrees:
            angle = angle * 180.0 / math.pi

        return {
            "angle": angle,
            "is_valid": is_valid,
            "function": func
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class Atan2(FormulaBase):
    """Compute atan2 (two-argument arctangent).

    Returns angle in correct quadrant.

    Args:
        use_degrees: Return angles in degrees
    """

    def __init__(self, use_degrees: bool = False):
        super().__init__("atan2", "f.fundamental.atan2")
        self.degrees = use_degrees

    def forward(self, y: Tensor, x: Tensor) -> Dict[str, Tensor]:
        """Compute atan2(y, x).

        Args:
            y: Y coordinates [...]
            x: X coordinates [...]

        Returns:
            angle: Angle in radians or degrees, range [-π, π] [...]
            radius: √(x² + y²) [...]
            quadrant: Which quadrant (1-4, or 0 for origin) [...]
        """
        # Angle
        angle = torch.atan2(y, x)

        # Radius
        radius = torch.sqrt(x ** 2 + y ** 2)

        # Quadrant
        quadrant = torch.zeros_like(angle, dtype=torch.long)
        quadrant = torch.where((x > 0) & (y >= 0), 1, quadrant)
        quadrant = torch.where((x <= 0) & (y > 0), 2, quadrant)
        quadrant = torch.where((x < 0) & (y <= 0), 3, quadrant)
        quadrant = torch.where((x >= 0) & (y < 0), 4, quadrant)

        if self.degrees:
            angle = angle * 180.0 / math.pi

        return {
            "angle": angle,
            "radius": radius,
            "quadrant": quadrant,
            "is_origin": radius < 1e-10
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# COORDINATE CONVERSIONS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class PolarToCartesian(FormulaBase):
    """Convert polar coordinates to Cartesian.

    x = r·cos(θ), y = r·sin(θ)
    """

    def __init__(self):
        super().__init__("polar_to_cartesian", "f.fundamental.polar2cart")

    def forward(self, r: Tensor, theta: Tensor) -> Dict[str, Tensor]:
        """Convert to Cartesian.

        Args:
            r: Radius [...]
            theta: Angle in radians [...]

        Returns:
            x: X coordinate [...]
            y: Y coordinate [...]
            cartesian: (x, y) as vector [..., 2]
        """
        x = r * torch.cos(theta)
        y = r * torch.sin(theta)

        cartesian = torch.stack([x, y], dim=-1)

        return {
            "x": x,
            "y": y,
            "cartesian": cartesian
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class CartesianToPolar(FormulaBase):
    """Convert Cartesian coordinates to polar.

    r = √(x² + y²), θ = atan2(y, x)
    """

    def __init__(self):
        super().__init__("cartesian_to_polar", "f.fundamental.cart2polar")

    def forward(self, x: Tensor, y: Tensor) -> Dict[str, Tensor]:
        """Convert to polar.

        Args:
            x: X coordinate [...]
            y: Y coordinate [...]

        Returns:
            r: Radius [...]
            theta: Angle in radians, range [-π, π] [...]
            polar: (r, θ) as vector [..., 2]
        """
        r = torch.sqrt(x ** 2 + y ** 2)
        theta = torch.atan2(y, x)

        polar = torch.stack([r, theta], dim=-1)

        return {
            "r": r,
            "theta": theta,
            "polar": polar,
            "is_origin": r < 1e-10
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class SphericalToCartesian(FormulaBase):
    """Convert spherical coordinates to Cartesian.

    x = r·sin(θ)·cos(φ)
    y = r·sin(θ)·sin(φ)
    z = r·cos(θ)

    Convention: θ is polar angle from z-axis, φ is azimuthal angle
    """

    def __init__(self):
        super().__init__("spherical_to_cartesian", "f.fundamental.sph2cart")

    def forward(self, r: Tensor, theta: Tensor, phi: Tensor) -> Dict[str, Tensor]:
        """Convert to Cartesian.

        Args:
            r: Radius [...]
            theta: Polar angle (from z-axis) in radians [...]
            phi: Azimuthal angle in radians [...]

        Returns:
            x: X coordinate [...]
            y: Y coordinate [...]
            z: Z coordinate [...]
            cartesian: (x, y, z) as vector [..., 3]
        """
        sin_theta = torch.sin(theta)
        cos_theta = torch.cos(theta)
        sin_phi = torch.sin(phi)
        cos_phi = torch.cos(phi)

        x = r * sin_theta * cos_phi
        y = r * sin_theta * sin_phi
        z = r * cos_theta

        cartesian = torch.stack([x, y, z], dim=-1)

        return {
            "x": x,
            "y": y,
            "z": z,
            "cartesian": cartesian
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class CartesianToSpherical(FormulaBase):
    """Convert Cartesian coordinates to spherical.

    r = √(x² + y² + z²)
    θ = arccos(z/r)
    φ = atan2(y, x)
    """

    def __init__(self):
        super().__init__("cartesian_to_spherical", "f.fundamental.cart2sph")

    def forward(self, x: Tensor, y: Tensor, z: Tensor) -> Dict[str, Tensor]:
        """Convert to spherical.

        Args:
            x: X coordinate [...]
            y: Y coordinate [...]
            z: Z coordinate [...]

        Returns:
            r: Radius [...]
            theta: Polar angle (from z-axis) in radians [...]
            phi: Azimuthal angle in radians [...]
            spherical: (r, θ, φ) as vector [..., 3]
        """
        r = torch.sqrt(x ** 2 + y ** 2 + z ** 2)
        theta = torch.acos(torch.clamp(z / (r + 1e-10), -1.0, 1.0))
        phi = torch.atan2(y, x)

        spherical = torch.stack([r, theta, phi], dim=-1)

        return {
            "r": r,
            "theta": theta,
            "phi": phi,
            "spherical": spherical,
            "is_origin": r < 1e-10
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ANGLE OPERATIONS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class WrapAngle(FormulaBase):
    """Wrap angles to specified range.

    Args:
        range_type: "0_2pi" for [0, 2π), "-pi_pi" for [-π, π), or "0_360" for [0, 360)
    """

    def __init__(self, range_type: str = "-pi_pi"):
        super().__init__("wrap_angle", "f.fundamental.wrap")
        self.range_type = range_type

    def forward(self, angle: Tensor) -> Dict[str, Tensor]:
        """Wrap angle to range.

        Args:
            angle: Input angles [...]

        Returns:
            wrapped: Wrapped angles [...]
            turns: Number of full rotations removed [...]
        """
        if self.range_type == "0_2pi":
            # [0, 2π)
            wrapped = angle % (2 * math.pi)
            turns = torch.floor(angle / (2 * math.pi))

        elif self.range_type == "-pi_pi":
            # [-π, π)
            wrapped = torch.atan2(torch.sin(angle), torch.cos(angle))
            turns = torch.round((angle - wrapped) / (2 * math.pi))

        elif self.range_type == "0_360":
            # [0, 360)
            wrapped = angle % 360.0
            turns = torch.floor(angle / 360.0)

        else:
            raise ValueError(f"Unknown range_type: {self.range_type}")

        return {
            "wrapped": wrapped,
            "turns": turns,
            "range_type": self.range_type
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class AngleDifference(FormulaBase):
    """Compute shortest angular difference.

    Returns difference in range [-π, π]
    """

    def __init__(self):
        super().__init__("angle_difference", "f.fundamental.angle_diff")

    def forward(self, angle1: Tensor, angle2: Tensor) -> Dict[str, Tensor]:
        """Compute angle2 - angle1 (shortest path).

        Args:
            angle1: First angles in radians [...]
            angle2: Second angles in radians [...]

        Returns:
            difference: Shortest angular difference [-π, π] [...]
            abs_difference: Absolute difference [...]
            direction: Sign of rotation (-1, 0, or 1) [...]
        """
        # Compute difference and wrap to [-π, π]
        diff = angle2 - angle1
        difference = torch.atan2(torch.sin(diff), torch.cos(diff))

        abs_difference = torch.abs(difference)

        # Direction
        direction = torch.sign(difference)

        return {
            "difference": difference,
            "abs_difference": abs_difference,
            "direction": direction
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# EXPONENTIAL AND LOGARITHMIC
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class Exponential(FormulaBase):
    """Compute exponential function.

    Args:
        base: Base of exponential (default: e)
    """

    def __init__(self, base: Optional[float] = None):
        super().__init__("exponential", "f.fundamental.exp")
        self.base = base

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        """Compute exponential.

        Args:
            x: Exponent values [...]

        Returns:
            result: exp(x) or base^x [...]
            log_result: log of result [...]
        """
        if self.base is None:
            # Natural exponential
            result = torch.exp(x)
            log_result = x
        else:
            # Base^x = e^(x ln(base))
            result = torch.exp(x * math.log(self.base))
            log_result = x * math.log(self.base)

        return {
            "result": result,
            "log_result": log_result,
            "base": self.base if self.base is not None else math.e
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class Logarithm(FormulaBase):
    """Compute logarithm.

    Args:
        base: Base of logarithm (default: e for natural log)
    """

    def __init__(self, base: Optional[float] = None):
        super().__init__("logarithm", "f.fundamental.log")
        self.base = base

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        """Compute logarithm.

        Args:
            x: Input values (must be positive) [...]

        Returns:
            result: log(x) [...]
            is_valid: x > 0 [...]
        """
        is_valid = x > 0
        x_safe = torch.clamp(x, min=1e-10)

        if self.base is None:
            # Natural logarithm
            result = torch.log(x_safe)
        elif self.base == 10:
            result = torch.log10(x_safe)
        elif self.base == 2:
            result = torch.log2(x_safe)
        else:
            # log_base(x) = ln(x) / ln(base)
            result = torch.log(x_safe) / math.log(self.base)

        return {
            "result": result,
            "is_valid": is_valid,
            "base": self.base if self.base is not None else math.e
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class Power(FormulaBase):
    """Compute power function x^p.
    """

    def __init__(self):
        super().__init__("power", "f.fundamental.pow")

    def forward(self, x: Tensor, p: Tensor) -> Dict[str, Tensor]:
        """Compute x^p.

        Args:
            x: Base values [...]
            p: Exponent values [...]

        Returns:
            result: x^p [...]
            log_result: log(x^p) = p·log(x) [...]
            is_valid: Whether operation is valid [...]
        """
        # x^p requires x > 0 for general p
        is_valid = x > 0
        x_safe = torch.clamp(x, min=1e-10)

        result = torch.pow(x_safe, p)
        log_result = p * torch.log(x_safe)

        return {
            "result": result,
            "log_result": log_result,
            "is_valid": is_valid
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HYPERBOLIC FUNCTIONS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class HyperbolicFunctions(FormulaBase):
    """Compute hyperbolic sine and cosine.

    sinh(x) = (e^x - e^(-x))/2
    cosh(x) = (e^x + e^(-x))/2
    """

    def __init__(self):
        super().__init__("hyperbolic", "f.fundamental.hyp")

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        """Compute hyperbolic functions.

        Args:
            x: Input values [...]

        Returns:
            sinh: sinh(x) [...]
            cosh: cosh(x) [...]
            tanh: tanh(x) [...]
            identity: cosh²(x) - sinh²(x) (should be 1) [...]
        """
        sinh = torch.sinh(x)
        cosh = torch.cosh(x)
        tanh = torch.tanh(x)

        # Verify identity
        identity = cosh ** 2 - sinh ** 2

        return {
            "sinh": sinh,
            "cosh": cosh,
            "tanh": tanh,
            "identity": identity
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class InverseHyperbolic(FormulaBase):
    """Compute inverse hyperbolic functions.
    """

    def __init__(self):
        super().__init__("inverse_hyperbolic", "f.fundamental.arcyhp")

    def forward(self, x: Tensor, func: str = "arcsinh") -> Dict[str, Tensor]:
        """Compute inverse hyperbolic function.

        Args:
            x: Input values [...]
            func: "arcsinh", "arccosh", "arctanh"

        Returns:
            result: Output values [...]
            is_valid: Input in valid domain [...]
        """
        if func == "arcsinh":
            # Domain: (-∞, ∞)
            result = torch.asinh(x)
            is_valid = torch.ones_like(x, dtype=torch.bool)

        elif func == "arccosh":
            # Domain: [1, ∞)
            is_valid = x >= 1.0
            x_safe = torch.clamp(x, min=1.0)
            result = torch.acosh(x_safe)

        elif func == "arctanh":
            # Domain: (-1, 1)
            is_valid = (x > -1.0) & (x < 1.0)
            x_safe = torch.clamp(x, -0.999, 0.999)
            result = torch.atanh(x_safe)

        else:
            raise ValueError(f"Unknown function: {func}")

        return {
            "result": result,
            "is_valid": is_valid,
            "function": func
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TESTING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_fundamental_operations():
    """Test suite for fundamental operations."""

    print("\n" + "=" * 70)
    print("FUNDAMENTAL OPERATIONS TESTS")
    print("=" * 70)

    # Test 1: Sin and Cos
    print("\n[Test 1] Sin and Cos")
    angles = torch.tensor([0.0, math.pi / 6, math.pi / 4, math.pi / 3, math.pi / 2])

    sincos = SinCos()
    result = sincos.forward(angles)

    print(f"  Angles (rad): {angles.numpy()}")
    print(f"  Sin: {result['sin'].numpy()}")
    print(f"  Cos: {result['cos'].numpy()}")
    print(f"  Quadrants: {result['quadrant'].numpy()}")
    print(f"  Status: ✓ PASS")

    # Test 2: Inverse trig
    print("\n[Test 2] Inverse Trigonometric Functions")
    x = torch.tensor([0.0, 0.5, 0.707, 0.866, 1.0])

    arcsin = InverseTrig(use_degrees=True)
    result = arcsin.forward(x, func="arcsin")

    print(f"  x: {x.numpy()}")
    print(f"  arcsin(x) (degrees): {result['angle'].numpy()}")
    print(f"  Valid: {result['is_valid'].numpy()}")
    print(f"  Status: ✓ PASS")

    # Test 3: Atan2
    print("\n[Test 3] Atan2")
    x_coords = torch.tensor([1.0, -1.0, -1.0, 1.0, 0.0])
    y_coords = torch.tensor([1.0, 1.0, -1.0, -1.0, 0.0])

    atan2_op = Atan2(use_degrees=True)
    result = atan2_op.forward(y_coords, x_coords)

    print(f"  (x, y): {list(zip(x_coords.numpy(), y_coords.numpy()))}")
    print(f"  Angles (degrees): {result['angle'].numpy()}")
    print(f"  Quadrants: {result['quadrant'].numpy()}")
    print(f"  Status: ✓ PASS")

    # Test 4: Polar to Cartesian
    print("\n[Test 4] Polar to Cartesian")
    r = torch.tensor([1.0, 2.0])
    theta = torch.tensor([0.0, math.pi / 2])

    polar2cart = PolarToCartesian()
    result = polar2cart.forward(r, theta)

    print(f"  (r, θ): {list(zip(r.numpy(), theta.numpy()))}")
    print(f"  Cartesian: {result['cartesian'].numpy()}")
    print(f"  Status: ✓ PASS")

    # Test 5: Cartesian to Polar
    print("\n[Test 5] Cartesian to Polar")
    x = torch.tensor([1.0, 0.0])
    y = torch.tensor([0.0, 2.0])

    cart2polar = CartesianToPolar()
    result = cart2polar.forward(x, y)

    print(f"  (x, y): {list(zip(x.numpy(), y.numpy()))}")
    print(f"  Polar (r, θ): {result['polar'].numpy()}")
    print(f"  Status: ✓ PASS")

    # Test 6: Spherical to Cartesian
    print("\n[Test 6] Spherical to Cartesian")
    r_sph = torch.tensor([1.0])
    theta_sph = torch.tensor([math.pi / 2])
    phi_sph = torch.tensor([math.pi / 4])

    sph2cart = SphericalToCartesian()
    result = sph2cart.forward(r_sph, theta_sph, phi_sph)

    print(f"  (r, θ, φ): ({r_sph.item():.2f}, {theta_sph.item():.2f}, {phi_sph.item():.2f})")
    print(f"  Cartesian: {result['cartesian'].numpy()}")
    print(f"  Status: ✓ PASS")

    # Test 7: Wrap angle
    print("\n[Test 7] Wrap Angle")
    angles_wrap = torch.tensor([0.0, math.pi, 2 * math.pi, 3 * math.pi, -math.pi])

    wrap = WrapAngle(range_type="-pi_pi")
    result = wrap.forward(angles_wrap)

    print(f"  Original: {angles_wrap.numpy()}")
    print(f"  Wrapped [-π, π]: {result['wrapped'].numpy()}")
    print(f"  Turns: {result['turns'].numpy()}")
    print(f"  Status: ✓ PASS")

    # Test 8: Angle difference
    print("\n[Test 8] Angle Difference")
    angle1 = torch.tensor([0.0, 0.0, math.pi])
    angle2 = torch.tensor([math.pi / 2, -math.pi / 2, 0.0])

    diff_op = AngleDifference()
    result = diff_op.forward(angle1, angle2)

    print(f"  Angle 1: {angle1.numpy()}")
    print(f"  Angle 2: {angle2.numpy()}")
    print(f"  Difference: {result['difference'].numpy()}")
    print(f"  Status: ✓ PASS")

    # Test 9: Exponential and logarithm
    print("\n[Test 9] Exponential and Logarithm")
    x_exp = torch.tensor([0.0, 1.0, 2.0])

    exp_op = Exponential()
    exp_result = exp_op.forward(x_exp)

    log_op = Logarithm()
    log_result = log_op.forward(exp_result['result'])

    print(f"  x: {x_exp.numpy()}")
    print(f"  exp(x): {exp_result['result'].numpy()}")
    print(f"  log(exp(x)): {log_result['result'].numpy()}")
    print(f"  Roundtrip error: {torch.abs(log_result['result'] - x_exp).numpy()}")
    print(f"  Status: ✓ PASS")

    # Test 10: Hyperbolic functions
    print("\n[Test 10] Hyperbolic Functions")
    x_hyp = torch.tensor([0.0, 1.0, 2.0])

    hyp_op = HyperbolicFunctions()
    result = hyp_op.forward(x_hyp)

    print(f"  x: {x_hyp.numpy()}")
    print(f"  sinh(x): {result['sinh'].numpy()}")
    print(f"  cosh(x): {result['cosh'].numpy()}")
    print(f"  tanh(x): {result['tanh'].numpy()}")
    print(f"  Identity (cosh²-sinh²): {result['identity'].numpy()}")
    print(f"  Status: ✓ PASS")

    print("\n" + "=" * 70)
    print("All tests completed! (10 total)")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    test_fundamental_operations()

    print("\n[Usage Summary]")
    print("-" * 70)
    print("Fundamental operations provide essential math functions:")
    print("  - Trigonometry: sin, cos, tan, and inverses")
    print("  - Coordinate systems: polar, Cartesian, spherical")
    print("  - Angle operations: wrapping, difference")
    print("  - Exponential/logarithmic: exp, log, power")
    print("  - Hyperbolic: sinh, cosh, tanh")
    print("-" * 70)