from abc import ABC


class BaseSchedule(ABC):
    def __init__(self):
        pass

    def get_lr(self, epoch):
        raise NotImplementedError("This method should be overridden by subclasses.")


ScheduleBase = BaseSchedule