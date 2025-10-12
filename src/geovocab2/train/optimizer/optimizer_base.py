from abc import ABC


class OptimizerBase(ABC):

    def __init__(self,
                 name: str = "optimizer_base",
                 uid: str = "optimizer.optimizer_base"):
        self.name = name
        self.uid = uid


    def step(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement this method.")

    def zero_grad(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement this method.")

