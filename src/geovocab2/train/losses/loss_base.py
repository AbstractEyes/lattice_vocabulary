from abc import ABC, abstractmethod


# Not really necessary but keeps things consistent and useful for type checking
class LossBase(ABC):

    def __init__(self, name: str = "loss_base", uid: str = "loss.loss_base"):
        super().__init__()
        self.name = name
        self.uid = uid

    @abstractmethod
    def forward(self, *args, **kwargs):
        raise NotImplementedError("Must implement forward method in subclass.")

    @abstractmethod
    def log(self, *args, **kwargs):
        raise NotImplementedError("Must implement log method in subclass.")


