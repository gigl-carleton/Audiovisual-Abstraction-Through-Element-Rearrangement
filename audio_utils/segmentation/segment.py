import abc

import numpy as np


class BaseSegmentation(abc.ABC):
    @abc.abstractmethod
    def __call__(self, distance_curve: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
        pass


class MaximumDuration(BaseSegmentation):
    def __init__(self, minimum_duration: float):
        super().__init__()
        self.minimum_duration = minimum_duration

    def __call__(self, distance_curve: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
        split_indices = []

        def recursive_split(start_idx, end_idx):
            if end_idx - start_idx <= 1:
                return

            start_time = timestamps[start_idx]
            end_time = timestamps[end_idx]
            duration = end_time - start_time

            if duration <= self.minimum_duration:
                return

            segment_distances = distance_curve[start_idx:end_idx]
            if len(segment_distances) == 0:
                return

            max_dist_rel_idx = np.argmax(segment_distances)
            split_idx = start_idx + max_dist_rel_idx + 1

            if split_idx == start_idx or split_idx == end_idx:
                split_idx = (start_idx + end_idx) // 2
                if split_idx == start_idx or split_idx == end_idx:
                    return  # still invalid; stop

            split_indices.append(split_idx)

            recursive_split(start_idx, split_idx)
            recursive_split(split_idx, end_idx)

        recursive_split(0, len(timestamps) - 1)
        split_indices.sort()
        return np.array(split_indices)
