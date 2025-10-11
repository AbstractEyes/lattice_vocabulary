from abc import ABC, abstractmethod


class LoggerBase(ABC):
    @abstractmethod
    def log(self, *args, **kwargs):
        pass

    @abstractmethod
    def close(self):
        pass