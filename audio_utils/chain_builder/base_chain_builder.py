import abc

import numpy as np


class BaseChainBuilder(abc.ABC):
    @abc.abstractmethod
    def execute(
            self, feature_boundaries: np.ndarray, timestamp_boundaries: np.ndarray,
            start_timestamp: float, end_timestamp: float, interpolation_time: float, overlap: float = 0.125,
            last_feature: np.ndarray = None
    ):
        pass
