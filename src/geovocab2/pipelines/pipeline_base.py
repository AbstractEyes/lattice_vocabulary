# pipelines/pipeline_base.py
from abc import ABC, abstractmethod


class PipelineBase(ABC):
    def __init__(self, name: str, uid: str):
        self.name = name
        self.uid = uid

    @abstractmethod
    def setup(self, **kwargs):
        """Initialize pipeline components."""
        pass

    @abstractmethod
    def __iter__(self):
        """Return iterator for batch generation."""
        pass

    def info(self):
        return {'name': self.name, 'uid': self.uid}