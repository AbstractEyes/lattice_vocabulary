"""
FormulaBase
Author: AbstractPhil + GPT-4o + Claude Sonnet 4.5

Description: Dual-mode symbolic + TorchScript-safe formula base class.
- Use `evaluate()` during dev/runtime for symbolic dispatch.
- Use `forward()` for TorchScript or compiled execution.

TorchScript Compatibility Notes:
    - evaluate() is not available in TorchScript contexts (marked @torch.jit.ignore)
    - validate() and info() are runtime-only helpers
    - Subclass forward() signatures must use concrete types
    - Avoid *args/**kwargs in forward() when tracing/scripting is required

Design Philosophy:
    This base class enforces a contract for geometric formulas while maintaining
    flexibility for both interactive development and production deployment. The
    dual-mode approach allows formulas to be:
    1. Debugged and validated during development via evaluate()
    2. Compiled and optimized for production via forward()
    3. Cataloged and discovered via metadata (name, uid, info)

License: MIT

Specifically, MIT because of the abstract and flexible nature of this class.
"""

import torch
from torch import Tensor
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple


class FormulaBase(torch.nn.Module, ABC):
    """
    Dual-mode symbolic + TorchScript-safe formula base class.

    This class provides a foundation for implementing geometric formulas that can
    operate in both development (flexible, introspectable) and production
    (compiled, optimized) contexts.

    Attributes:
        name: Human-readable formula name (e.g., "cayley_menger_volume")
        uid: Unique identifier for registry/catalog systems (e.g., "f.cayley.volume")

    Methods:
        evaluate(): Runtime-flexible evaluation (development mode)
        forward(): TorchScript-safe computation (production mode)
        validate(): Optional input validation with diagnostics
        info(): Formula metadata for introspection

    Example:
        >>> class MyFormula(FormulaBase):
        ...     def __init__(self):
        ...         super().__init__("my_formula", "f.my.formula")
        ...
        ...     def forward(self, x: Tensor) -> Dict[str, Tensor]:
        ...         return {"result": x ** 2}
        ...
        >>> formula = MyFormula()
        >>> result = formula.evaluate(torch.tensor([1.0, 2.0, 3.0]))
        >>> print(result["result"])
        tensor([1., 4., 9.])
    """

    def __init__(self, name: str, uid: str):
        """
        Initialize the formula with identifying metadata.

        Args:
            name: Human-readable formula name
            uid: Unique identifier (recommend hierarchical format like "f.category.name")
        """
        super().__init__()
        self.name = name
        self.uid = uid

    @torch.jit.ignore
    def evaluate(self, *args, **kwargs) -> Dict[str, Tensor]:
        """
        Runtime dispatchable version of the formula.

        This method is ignored by TorchScript and delegates to forward().
        Use this during development for flexible argument passing and debugging.

        Args:
            *args: Positional arguments passed to forward()
            **kwargs: Keyword arguments passed to forward()

        Returns:
            Dictionary mapping output names to tensors

        Note:
            This method is NOT available in TorchScript contexts. For compiled
            execution, call forward() directly with concrete types.
        """
        return self.forward(*args, **kwargs)

    @abstractmethod
    def forward(self, *args, **kwargs) -> Dict[str, Tensor]:
        """
        TorchScript-safe forward pass.

        Subclasses MUST implement this method with a concrete signature.
        Avoid using *args/**kwargs in the actual implementation if you plan
        to use TorchScript tracing or scripting.

        Args:
            *args: Formula-specific positional arguments (use concrete types in implementation)
            **kwargs: Formula-specific keyword arguments (use concrete types in implementation)

        Returns:
            Dictionary mapping output names to result tensors

        Example:
            def forward(self, x: Tensor, y: Tensor) -> Dict[str, Tensor]:
                return {"sum": x + y, "product": x * y}
        """
        pass

    @torch.jit.ignore
    def validate(self, *args, **kwargs) -> Tuple[bool, str]:
        """
        Optional validation method to check input shapes/types.

        Subclasses can override this to provide input validation with
        helpful error messages. This is runtime-only and ignored by TorchScript.

        Args:
            *args: Positional arguments to validate
            **kwargs: Keyword arguments to validate

        Returns:
            Tuple of (is_valid, error_message)
            - is_valid: True if inputs are valid, False otherwise
            - error_message: Empty string if valid, descriptive error if invalid

        Example:
            def validate(self, x: Tensor) -> Tuple[bool, str]:
                if x.ndim != 2:
                    return False, f"Expected 2D tensor, got {x.ndim}D"
                return True, ""
        """
        return True, ""

    @torch.jit.ignore
    def info(self) -> Dict[str, Any]:
        """
        Metadata about the formula for introspection and cataloging.

        Subclasses can override this to provide rich metadata about the formula's
        purpose, mathematical foundation, expected inputs/outputs, etc.

        Returns:
            Dictionary containing formula metadata with at least:
                - name: Formula name
                - uid: Unique identifier
                - description: Human-readable description

        Example:
            def info(self) -> Dict[str, Any]:
                return {
                    "name": self.name,
                    "uid": self.uid,
                    "description": "Computes Cayley-Menger determinant",
                    "dimension": "n-simplex",
                    "inputs": ["squared_distance_matrix"],
                    "outputs": ["volume", "determinant"],
                    "reference": "https://en.wikipedia.org/wiki/Cayley-Menger_determinant"
                }
        """
        return {
            "name": self.name,
            "uid": self.uid,
            "description": "No description provided"
        }

    def __repr__(self) -> str:
        """
        String representation for debugging and logging.

        Returns:
            Human-readable string identifying the formula instance
        """
        return f"{self.__class__.__name__}(name='{self.name}', uid='{self.uid}')"

    def __str__(self) -> str:
        """
        User-friendly string representation.

        Returns:
            Formula name and class
        """
        return f"Formula[{self.name}] ({self.__class__.__name__})"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# EXAMPLE USAGE AND TESTING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    # Example: Simple quadratic formula
    class QuadraticFormula(FormulaBase):
        """Example formula that computes x² + bx + c"""

        def __init__(self, b: float = 1.0, c: float = 0.0):
            super().__init__("quadratic", "f.example.quadratic")
            self.b = b
            self.c = c

        def forward(self, x: Tensor) -> Dict[str, Tensor]:
            """Compute quadratic function."""
            result = x ** 2 + self.b * x + self.c
            return {
                "result": result,
                "derivative": 2 * x + self.b
            }

        def validate(self, x: Tensor) -> Tuple[bool, str]:
            """Validate input tensor."""
            if not isinstance(x, Tensor):
                return False, f"Expected Tensor, got {type(x)}"
            if x.numel() == 0:
                return False, "Input tensor is empty"
            return True, ""

        def info(self) -> Dict[str, Any]:
            """Rich metadata about the formula."""
            return {
                "name": self.name,
                "uid": self.uid,
                "description": f"Quadratic formula: f(x) = x² + {self.b}x + {self.c}",
                "parameters": {"b": self.b, "c": self.c},
                "inputs": ["x: Tensor of any shape"],
                "outputs": ["result: f(x)", "derivative: f'(x)"]
            }

    # Test the example formula
    print("="*70)
    print("FormulaBase Example: QuadraticFormula")
    print("="*70)

    formula = QuadraticFormula(b=2.0, c=1.0)
    print(f"\n{formula}")
    print(f"repr: {repr(formula)}")

    # Test validation
    x = torch.tensor([1.0, 2.0, 3.0])
    is_valid, error_msg = formula.validate(x)
    print(f"\nValidation: {'✓ PASS' if is_valid else f'✗ FAIL - {error_msg}'}")

    # Test evaluation
    result = formula.evaluate(x)
    print(f"\nInput: {x}")
    print(f"Result: {result['result']}")
    print(f"Derivative: {result['derivative']}")

    # Test metadata
    print("\nFormula Info:")
    info = formula.info()
    for key, value in info.items():
        print(f"  {key}: {value}")

    # Test TorchScript compilation
    print("\n" + "-"*70)
    print("TorchScript Compilation Test")
    print("-"*70)
    try:
        scripted_formula = torch.jit.script(formula)
        scripted_result = scripted_formula.forward(x)
        print("✓ TorchScript compilation successful")
        print(f"Scripted result: {scripted_result['result']}")
    except Exception as e:
        print(f"✗ TorchScript compilation failed: {e}")

    print("\n" + "="*70)
    print("Example completed successfully!")
    print("="*70)