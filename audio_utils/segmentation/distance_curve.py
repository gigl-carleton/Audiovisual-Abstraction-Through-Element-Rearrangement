import abc

import numpy as np


class BaseDistanceCurve(abc.ABC):
    def get_name(self) -> str:
        return self.__class__.__name__

    @abc.abstractmethod
    def __call__(self, features: np.ndarray) -> np.ndarray:
        """
        Creates a distance curve from extracted features
        :param features: the features extracted from the audio clip
        :return: the distance curve
        """


class EuclideanDistanceCurve(BaseDistanceCurve):
    """
    Creates a distance curve by computing Euclidean distance between neighbouring windows
    """
    def __call__(self, features: np.ndarray) -> np.ndarray:
        """
        Creates a distance curve from extracted features using Euclidean distance
        :param features: the features extracted from the audio clip in the shape (w, f) where w is the number of windows, and f is the length of the feature vector per window
        :return: the distance curve in the shape (w - 1, )
        """
        diffs = features[1:] - features[:-1]
        return np.linalg.norm(diffs, axis=1)
