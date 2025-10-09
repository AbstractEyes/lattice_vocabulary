from abc import ABC


class TrainBase(ABC):
    def __init__(self, config):
        self.config = config

    def validate(self):
        raise NotImplementedError("Validate method must be implemented by subclasses.")

    def test(self):
        raise NotImplementedError("Test method must be implemented by subclasses.")

    def train_epoch(self):
        raise NotImplementedError("Train epoch method must be implemented by subclasses.")

    def train(self):
        raise NotImplementedError("Train method must be implemented by subclasses.")

    def freeze_targets(self, params: list[str]):
        raise NotImplementedError("Freeze targets method must be implemented by subclasses.")