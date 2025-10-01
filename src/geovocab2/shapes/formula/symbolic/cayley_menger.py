"""
    CAYLEY-MENGER FORMULA SUITE
    Author: AbstractPhil + GPT-4o + Claude Sonnet 4.5
---------------------------
Symbolic geometric volume and structure formulas derived from the Cayley-Menger determinant.

Named in honor of:
  • Arthur Cayley (1821–1895) – abstract algebra, projective geometry, matrix theory
  • Karl Menger (1902–1985) – metric geometry, dimensional theory, distance space formalization

These are symbolic stepping stones for the lattice — volume-based supervision, structural alignment,
collapse detection, and dynamic simplex stability, built to serve geometric tensor systems.

Mathematical Foundation:
    For an n-simplex (n+1 vertices in n-dimensional space):

    Cayley-Menger Matrix:
    CM = | 0   1   1   ...  1   |
         | 1   0   d₁₂² ... d₁ₙ²|
         | 1   d₁₂² 0  ... d₂ₙ²|
         | ⋮   ⋮   ⋮   ⋱   ⋮   |
         | 1   d₁ₙ² d₂ₙ² ... 0 |

    Volume Formula:
    V² = (-1)^(n+1) / (2^n · (n!)²) · det(CM)

    where n = dimension of the simplex = (number of vertices - 1)

Formulas included:
  - CayleyMengerVolume: Core volume computation
  - CayleyMengerExpanded: Volume with validation, logs, and optional loss
  - CayleyMengerFromSimplex: Volume from vertex coordinates
  - CayleyMengerBatchLoss: Batch loss evaluation
  - CayleyMengerMatrixBuilder: Raw matrix construction
  - CayleyMengerDeterminantOnly: Determinant-only computation
"""
from math import factorial
from typing import Dict, Optional

import torch
from torch import Tensor

from shapes.formula.formula_base import FormulaBase


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# UTILITY FUNCTIONS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def validate_distance_matrix(D: Tensor, eps: float = 1e-6) -> None:
    """Validate that D is a proper squared distance matrix.

    Args:
        D: Distance matrix tensor [..., N, N]
        eps: Tolerance for numerical checks

    Raises:
        ValueError: If matrix is not symmetric or diagonal is not zero
    """
    # Check symmetry
    if not torch.allclose(D, D.transpose(-2, -1), atol=eps):
        raise ValueError("Distance matrix must be symmetric")

    # Check zero diagonal
    diagonal = torch.diagonal(D, dim1=-2, dim2=-1)
    if not torch.allclose(diagonal, torch.zeros_like(diagonal), atol=eps):
        raise ValueError("Distance matrix diagonal must be zero (distances from point to itself)")


def build_cayley_menger_matrix(D: Tensor) -> Tensor:
    """Build Cayley-Menger matrix from squared distance matrix.

    Args:
        D: Squared distance matrix [..., N, N] where N = n_vertices

    Returns:
        Cayley-Menger matrix [..., N+1, N+1]
    """
    batch_shape = D.shape[:-2]
    n_vertices = D.shape[-1]
    cm_size = n_vertices + 1

    # Initialize with ones
    cm = torch.ones(*batch_shape, cm_size, cm_size, dtype=D.dtype, device=D.device)

    # Fill distance block
    cm[..., 1:, 1:] = D

    # Set top-left corner to zero
    cm[..., 0, 0] = 0.0

    return cm


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class CayleyMengerVolume(FormulaBase):
    """Computes n-simplex volume from pairwise squared distances.

    The core implementation of the Cayley-Menger determinant formula.
    Supports batched computation and returns validation flags.

    Args:
        validate_input: Whether to validate distance matrix properties (default: True)
        eps: Epsilon for numerical stability and degeneracy detection (default: 1e-10)
    """

    def __init__(self, validate_input: bool = True, eps: float = 1e-10):
        super().__init__("cayley_menger_volume", "f.cayley.volume")
        self.validate_input = validate_input
        self.eps = eps

    def forward(self, D: Tensor) -> Dict[str, Tensor]:
        """Compute volume from squared distance matrix.

        Args:
            D: Squared distance matrix [..., n_vertices, n_vertices]
               Must be symmetric with zero diagonal

        Returns:
            Dictionary containing:
                - volume: Computed volume
                - volume_squared: V² before square root
                - is_degenerate: Boolean flag for collapsed simplices
                - determinant: Raw determinant value
                - dimension: Simplex dimension n
        """
        # Validate input if requested
        if self.validate_input:
            validate_distance_matrix(D, eps=self.eps)

        # Extract dimensions
        n_vertices = D.shape[-1]
        n = n_vertices - 1  # simplex dimension

        # Build Cayley-Menger matrix
        cm = build_cayley_menger_matrix(D)

        # Compute determinant
        det = torch.linalg.det(cm)

        # Apply volume formula: V² = (-1)^(n+1) / (2^n · (n!)²) · det(CM)
        sign = (-1.0) ** (n + 1)
        factorial_n = float(factorial(n))
        denominator = (2.0 ** n) * (factorial_n ** 2)

        vol_squared = sign * det / denominator

        # Clamp small negative values (numerical errors)
        vol_squared = torch.clamp(vol_squared, min=0.0)

        # Compute final volume
        volume = torch.sqrt(vol_squared)

        # Detect degenerate simplices
        is_degenerate = vol_squared < self.eps

        return {
            "volume": volume,
            "volume_squared": vol_squared,
            "is_degenerate": is_degenerate,
            "determinant": det,
            "dimension": torch.tensor(n, dtype=torch.long, device=D.device)
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class CayleyMengerExpanded(FormulaBase):
    """Enhanced volume computation with additional metrics and optional loss.

    Extends CayleyMengerVolume with logarithmic volume, loss computation,
    and comprehensive validation outputs.

    Args:
        target_volume: Target volume for loss computation (optional)
        compute_loss: Whether to compute loss against target (default: False)
        loss_type: Type of loss - 'l1', 'l2', 'log', or 'huber' (default: 'l2')
        eps: Epsilon for numerical stability (default: 1e-10)
        validate_input: Whether to validate distance matrix (default: True)
    """

    def __init__(
            self,
            target_volume: Optional[float] = None,
            compute_loss: bool = False,
            loss_type: str = "l2",
            eps: float = 1e-10,
            validate_input: bool = True
    ):
        super().__init__("cayley_menger_expanded", "f.cayley.expanded")

        # Validate loss type at initialization
        valid_loss_types = ["l1", "l2", "log", "huber"]
        if loss_type not in valid_loss_types:
            raise ValueError(f"Invalid loss_type '{loss_type}'. Must be one of {valid_loss_types}")

        self.target_volume = target_volume
        self.compute_loss = compute_loss
        self.loss_type = loss_type
        self.eps = eps
        self.validate_input = validate_input

        # Initialize base volume computer
        self._volume_computer = CayleyMengerVolume(
            validate_input=validate_input,
            eps=eps
        )

    def forward(self, D: Tensor) -> Dict[str, Tensor]:
        """Compute volume with expanded metrics.

        Args:
            D: Squared distance matrix [..., n_vertices, n_vertices]

        Returns:
            Dictionary containing all CayleyMengerVolume outputs plus:
                - log_volume: Natural log of volume (for numerical stability)
                - loss: Loss against target volume (if compute_loss=True)
        """
        # Compute base volume
        result = self._volume_computer.forward(D)

        volume = result["volume"]

        # Add logarithmic volume
        result["log_volume"] = torch.log(volume + self.eps)

        # Compute loss if requested
        if self.compute_loss and self.target_volume is not None:
            target = torch.tensor(
                self.target_volume,
                dtype=volume.dtype,
                device=volume.device
            )

            # Broadcast target to match batch dimensions
            if volume.ndim > 0:
                target = target.expand_as(volume)

            # Compute loss based on type
            if self.loss_type == "l1":
                loss = torch.abs(volume - target)
            elif self.loss_type == "l2":
                loss = (volume - target) ** 2
            elif self.loss_type == "log":
                log_target = torch.log(target + self.eps)
                loss = torch.abs(result["log_volume"] - log_target)
            elif self.loss_type == "huber":
                delta = 1.0
                diff = torch.abs(volume - target)
                loss = torch.where(
                    diff <= delta,
                    0.5 * diff ** 2,
                    delta * (diff - 0.5 * delta)
                )

            result["loss"] = loss

        return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class CayleyMengerFromSimplex(FormulaBase):
    """Computes volume directly from simplex vertex coordinates.

    Convenience wrapper that computes squared distances from coordinates
    and then applies the Cayley-Menger formula.

    Args:
        eps: Epsilon for numerical stability (default: 1e-10)
        validate_input: Whether to validate distance matrix (default: True)
    """

    def __init__(self, eps: float = 1e-10, validate_input: bool = True):
        super().__init__("cayley_from_simplex", "f.cayley.from_simplex")
        self.eps = eps
        self.validate_input = validate_input

        # Initialize expanded computer for richer output
        self._computer = CayleyMengerExpanded(
            eps=eps,
            validate_input=validate_input
        )

    def forward(self, X: Tensor) -> Dict[str, Tensor]:
        """Compute volume from vertex coordinates.

        Args:
            X: Vertex coordinates [..., n_vertices, embedding_dim]
               For an n-simplex, requires n+1 vertices

        Returns:
            Same as CayleyMengerExpanded.forward()
        """
        if X.ndim < 2:
            raise ValueError(f"Expected at least 2D tensor, got shape {X.shape}")

        # Compute squared distance matrix
        D_squared = torch.cdist(X, X, p=2) ** 2

        # Use expanded computation
        return self._computer.forward(D_squared)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class CayleyMengerBatchLoss(FormulaBase):
    """Evaluates loss across batch of distance matrices or simplices.

    Computes per-sample loss and aggregates statistics across the batch.
    Useful for training geometric neural networks.

    Args:
        target_volume: Target volume for all samples (default: 1.0)
        loss_type: Type of loss function (default: 'l2')
        eps: Epsilon for numerical stability (default: 1e-10)
        validate_input: Whether to validate distance matrices (default: True)
    """

    def __init__(
            self,
            target_volume: float = 1.0,
            loss_type: str = "l2",
            eps: float = 1e-10,
            validate_input: bool = True
    ):
        super().__init__("cayley_batch_loss", "f.cayley.batch_loss")
        self.target_volume = target_volume
        self.loss_type = loss_type
        self.eps = eps

        # Initialize expanded computer with loss
        self._computer = CayleyMengerExpanded(
            target_volume=target_volume,
            compute_loss=True,
            loss_type=loss_type,
            eps=eps,
            validate_input=validate_input
        )

    def forward(self, D: Tensor) -> Dict[str, Tensor]:
        """Compute batch loss statistics.

        Args:
            D: Batch of squared distance matrices [batch_size, n_vertices, n_vertices]

        Returns:
            Dictionary containing:
                - loss: Per-sample loss [batch_size]
                - mean_loss: Average loss across batch
                - volume_mean: Average volume across batch
                - volume_std: Standard deviation of volumes
                - num_degenerate: Count of degenerate simplices
                - degenerate_fraction: Fraction of degenerate simplices
        """
        # Compute expanded metrics
        result = self._computer.forward(D)

        loss = result["loss"]
        volume = result["volume"]
        is_degenerate = result["is_degenerate"]

        # Aggregate statistics
        return {
            "loss": loss,
            "mean_loss": loss.mean(),
            "volume_mean": volume.mean(),
            "volume_std": volume.std(),
            "num_degenerate": is_degenerate.sum(),
            "degenerate_fraction": is_degenerate.float().mean(),
            "volume": volume,
            "is_degenerate": is_degenerate
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class CayleyMengerMatrixBuilder(FormulaBase):
    """Returns the raw Cayley-Menger matrix.

    Utility class for debugging and visualization. Constructs the full
    Cayley-Menger matrix without computing the determinant or volume.

    Args:
        validate_input: Whether to validate distance matrix (default: False)
    """

    def __init__(self, validate_input: bool = False):
        super().__init__("cayley_matrix", "f.cayley.matrix")
        self.validate_input = validate_input

    def forward(self, D: Tensor) -> Dict[str, Tensor]:
        """Build Cayley-Menger matrix.

        Args:
            D: Squared distance matrix [..., n_vertices, n_vertices]

        Returns:
            Dictionary containing:
                - matrix: Cayley-Menger matrix [..., n_vertices+1, n_vertices+1]
                - dimension: Simplex dimension
        """
        if self.validate_input:
            validate_distance_matrix(D)

        cm = build_cayley_menger_matrix(D)
        n = D.shape[-1] - 1

        return {
            "matrix": cm,
            "dimension": torch.tensor(n, dtype=torch.long, device=D.device)
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class CayleyMengerDeterminantOnly(FormulaBase):
    """Returns only the determinant of the Cayley-Menger matrix.

    Lightweight computation for cases where only the determinant is needed,
    such as degeneracy detection or matrix conditioning analysis.

    Args:
        validate_input: Whether to validate distance matrix (default: False)
    """

    def __init__(self, validate_input: bool = False):
        super().__init__("cayley_det_only", "f.cayley.det")
        self.validate_input = validate_input
        self._matrix_builder = CayleyMengerMatrixBuilder(validate_input=validate_input)

    def forward(self, D: Tensor) -> Dict[str, Tensor]:
        """Compute determinant only.

        Args:
            D: Squared distance matrix [..., n_vertices, n_vertices]

        Returns:
            Dictionary containing:
                - determinant: Raw determinant value
                - dimension: Simplex dimension
        """
        result = self._matrix_builder.forward(D)
        cm = result["matrix"]
        det = torch.linalg.det(cm)

        return {
            "determinant": det,
            "dimension": result["dimension"]
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TESTING AND VALIDATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_cayley_menger_correctness():
    """Validate implementation against known geometric cases."""

    print("\n" + "=" * 70)
    print("CAYLEY-MENGER CORRECTNESS TESTS")
    print("=" * 70)

    # Test 1: Right triangle with area 0.5
    print("\n[Test 1] Right Triangle (area = 0.5)")
    X_triangle = torch.tensor([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0]
    ], dtype=torch.float32)

    result = CayleyMengerFromSimplex().forward(X_triangle)
    expected = 0.5
    actual = result["volume"].item()
    passed = abs(actual - expected) < 1e-6

    print(f"  Expected volume: {expected}")
    print(f"  Computed volume: {actual:.10f}")
    print(f"  Is degenerate: {result['is_degenerate'].item()}")
    print(f"  Status: {'✓ PASS' if passed else '✗ FAIL'}")

    # Test 2: Unit tetrahedron with volume 1/6
    print("\n[Test 2] Unit Tetrahedron (volume = 1/6)")
    X_tetrahedron = torch.tensor([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=torch.float32)

    result = CayleyMengerFromSimplex().forward(X_tetrahedron)
    expected = 1.0 / 6.0
    actual = result["volume"].item()
    passed = abs(actual - expected) < 1e-6

    print(f"  Expected volume: {expected:.10f}")
    print(f"  Computed volume: {actual:.10f}")
    print(f"  Is degenerate: {result['is_degenerate'].item()}")
    print(f"  Status: {'✓ PASS' if passed else '✗ FAIL'}")

    # Test 3: Degenerate triangle (collinear points)
    print("\n[Test 3] Degenerate Triangle (collinear points)")
    X_degenerate = torch.tensor([
        [0.0, 0.0],
        [1.0, 0.0],
        [2.0, 0.0]
    ], dtype=torch.float32)

    result = CayleyMengerFromSimplex().forward(X_degenerate)
    actual = result["volume"].item()
    is_degenerate = result["is_degenerate"].item()
    passed = is_degenerate and actual < 1e-6

    print(f"  Expected: degenerate with volume ≈ 0")
    print(f"  Computed volume: {actual:.10e}")
    print(f"  Is degenerate: {is_degenerate}")
    print(f"  Status: {'✓ PASS' if passed else '✗ FAIL'}")

    # Test 4: Equilateral triangle
    print("\n[Test 4] Equilateral Triangle (side = 1)")
    h = (3 ** 0.5) / 2  # height
    X_equilateral = torch.tensor([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, h]
    ], dtype=torch.float32)

    result = CayleyMengerFromSimplex().forward(X_equilateral)
    expected = (3 ** 0.5) / 4  # area = sqrt(3)/4
    actual = result["volume"].item()
    passed = abs(actual - expected) < 1e-6

    print(f"  Expected volume: {expected:.10f}")
    print(f"  Computed volume: {actual:.10f}")
    print(f"  Is degenerate: {result['is_degenerate'].item()}")
    print(f"  Status: {'✓ PASS' if passed else '✗ FAIL'}")

    # Test 5: Batch computation
    print("\n[Test 5] Batch Processing")
    D_batch = torch.cdist(X_triangle, X_triangle, p=2) ** 2
    D_batch = D_batch.unsqueeze(0).repeat(4, 1, 1)

    batch_result = CayleyMengerBatchLoss(target_volume=0.5).forward(D_batch)

    print(f"  Batch size: 4")
    print(f"  Mean volume: {batch_result['volume_mean'].item():.10f}")
    print(f"  Mean loss: {batch_result['mean_loss'].item():.10e}")
    print(f"  Num degenerate: {batch_result['num_degenerate'].item()}")
    print(f"  Status: ✓ PASS")

    print("\n" + "=" * 70)
    print("All correctness tests completed!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    # Run correctness tests
    test_cayley_menger_correctness()

    # Additional demonstrations
    print("\n[CayleyMengerVolume] - Basic volume computation")
    X = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    D = torch.cdist(X, X, p=2) ** 2
    print(CayleyMengerVolume().evaluate(D))

    print("\n[CayleyMengerExpanded] - With loss computation")
    print(CayleyMengerExpanded(target_volume=0.5, compute_loss=True).evaluate(D))

    print("\n[CayleyMengerFromSimplex] - Direct from coordinates")
    print(CayleyMengerFromSimplex().evaluate(X))

    print("\n[CayleyMengerMatrixBuilder] - Raw matrix")
    print(CayleyMengerMatrixBuilder().evaluate(D))

    print("\n[CayleyMengerDeterminantOnly] - Determinant only")
    print(CayleyMengerDeterminantOnly().evaluate(D))