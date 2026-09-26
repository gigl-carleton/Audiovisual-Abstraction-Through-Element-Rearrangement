import abc
import typing


class BaseKeyframeSelector(abc.ABC):
    def __init__(self, interpolation_time: float = 5):
        super().__init__()

        self.interpolation_time = interpolation_time

    @abc.abstractmethod
    def execute(self, total_frames: int) -> typing.List[int]:
        pass
