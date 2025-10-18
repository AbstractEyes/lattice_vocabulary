import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Optional, Any, Union
import torch
import torch.nn as nn

from geovocab2.train.config.config_base import BaseConfig


class LossBase(ABC):
    """
    Base class for all loss functions in the geometric vocabulary framework.


    """

    def __init__(
        self,
        name: str = "loss_base",
        uid: str = "loss.loss_base"
    ):
        super().__init__()
        self.name = name
        self.uid = uid

    @abstractmethod
    def forward(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Compute the loss.

        Must be implemented by subclasses.

        Returns:
            Dictionary containing any relevant loss information
        """
        pass


if __name__ == "__main__":
    # Example subclass implementation
    class ExampleLoss(LossBase):
        def __init__(self, name: str = "example_loss", uid: str = "loss.example"):
            super().__init__(name, uid)
            self.mse_loss = nn.MSELoss()

        def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, torch.Tensor]:
            loss_value = self.mse_loss(predictions, targets)
            return {"mse_loss": loss_value}

    # Test the ExampleLoss
    loss_fn = ExampleLoss()
    preds = torch.randn(10, 5)
    targets = torch.randn(10, 5)
    loss_output = loss_fn.forward(preds, targets)
    print(loss_output)
    print(f"Loss Name: {loss_fn.name}, UID: {loss_fn.uid}")
#     print(f"Config: {FormulaConfig()}")
#     print(f"Config as dict: {json.dumps(FormulaConfig().to_dict(),
#                                   indent=2)}")
#     print(f"\nKey parameters:")
#     print(f"  target_volume: {FormulaConfig().target_volume}")