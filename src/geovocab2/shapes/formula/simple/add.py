"""
    AddFormula
    Author: AbstractPhil

    Description: A simple formula meant to demonstrate the FormulaBase class.
    It adds two tensors element-wise and returns the result in a dictionary.

"""

from torch import Tensor
from typing import Dict

from ..formula_base import FormulaBase


class AddFormula(FormulaBase):
    def __init__(self):
        super().__init__(name="add", uid="f.add")

    def forward(self, x: Tensor, y: Tensor) -> Dict[str, Tensor]:
        return {"sum": x + y}

    def info(self):
        return {
            "name": self.name,
            "uid": self.uid,
            "description": "Adds two tensors element-wise."
        }
