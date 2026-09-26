import abc
import typing

if typing.TYPE_CHECKING:
    import render_av


class BaseVideoPipe(abc.ABC):
    def __init__(self, framerate: float = 24, upscale: float = 3, input_video_resize: typing.Tuple[int, int] = None, max_distance: float = 50):
        super().__init__()
        self.framerate = framerate
        self.upscale = upscale
        self.input_video_resize = input_video_resize
        self.output_file = "video_tmp.avi"
        self.max_distance = max_distance

    @abc.abstractmethod
    def execute(self, keyframes: typing.List[int], exemplars: 'render_av.Exemplars', interpolation_time: float):
        pass
