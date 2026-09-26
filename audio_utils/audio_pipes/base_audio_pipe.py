import abc
import typing


class BaseAudioPipe(abc.ABC):
    def __init__(self):
        super().__init__()
        self.output_file = "audio_tmp.wav"

    @abc.abstractmethod
    def execute(self, key_times: typing.List[float], audio_path: str, interpolation_time: float):
        pass
