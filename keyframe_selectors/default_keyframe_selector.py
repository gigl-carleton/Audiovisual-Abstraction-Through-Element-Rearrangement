import math
import typing

import numpy as np

import keyframe_selectors.base_keyframe_selector as base_keyframe_selector


class DefaultKeyframeSelector(base_keyframe_selector.BaseKeyframeSelector):
    def __init__(
            self, min_duration: float, max_duration: float = None,
            interpolation_time: float = 5, random_start: bool = True
    ):
        super().__init__(interpolation_time)
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.random_start = random_start

        if self.max_duration is None:
            self.max_duration = self.min_duration

    def get_step_sizes(self, total_frames: int) -> typing.Tuple[int, int]:
        if self.max_duration is None:
            max_step = None
        else:
            max_step = int(math.floor(total_frames / (self.max_duration / self.interpolation_time)))
        min_step = math.ceil(total_frames / (self.min_duration / self.interpolation_time))

        if self.max_duration == self.min_duration:
            return int(min_step), int(min_step)

        return int(min_step), max_step

    def execute(self, total_frames: int) -> typing.List[int]:
        min_step, max_step = self.get_step_sizes(total_frames)
        if self.random_start:
            if max_step == min_step:
                current_index = np.random.randint(0, min_step)
            else:
                current_index = np.random.randint(0, max_step)

        else:
            current_index = 0

        path = [current_index]
        frame_count = total_frames - 1

        while current_index < total_frames:
            if max_step == min_step:
                current_index += min_step
            else:
                current_index += np.random.randint(min(min_step, max_step), max(min_step, max_step))

            path.append(current_index)

        if path[-1] >= frame_count:
            if frame_count - path[-2] > min_step / 2:
                path[-1] = frame_count
            else:
                del path[-1]

        path[-1] -= 2
        return path
