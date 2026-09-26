import abc
import typing

import librosa
import numpy as np


class BaseFeatureExtraction(abc.ABC):
    def __init__(self, frame_length_ms: float = 25, overlap_percent: float = 0.5):
        self.frame_length = frame_length_ms
        self.overlap_percent = overlap_percent

    def _get_frame_hop_length(self, sample_rate: float) -> typing.Tuple[int, int]:
        frame_length = int(sample_rate * self.frame_length / 1000)
        hop_length = int(frame_length * (1 - self.overlap_percent))
        hop_length = max(1, hop_length)

        return frame_length, hop_length

    def _get_timestamps(self, features, hop_length, sample_rate):
        num_frames = features.shape[1]
        return np.arange(num_frames) * hop_length / sample_rate

    def get_name(self) -> str:
        return self.__class__.__name__

    @abc.abstractmethod
    def __call__(self, filename: str, sample_rate: float = None) -> typing.Tuple[np.ndarray, np.ndarray]:
        pass


class MelFrequencyCepstralCoefficients(BaseFeatureExtraction):
    def __init__(self, coefficient_count: int = 13, frame_length: float = 25, overlap_percent: float = 0.5):
        super().__init__(frame_length, overlap_percent)
        self.coefficient_count = coefficient_count

    def __call__(self, filename: str, sample_rate: float = None) -> typing.Tuple[np.ndarray, np.ndarray]:
        audio_time_series, sample_rate = librosa.load(filename, sr=sample_rate)
        frame_length, hop_length = self._get_frame_hop_length(sample_rate)

        mfccs = librosa.feature.mfcc(
            y=audio_time_series,
            sr=sample_rate,
            n_mfcc=self.coefficient_count,
            n_fft=frame_length,
            hop_length=hop_length,
        )

        return mfccs.T, self._get_timestamps(mfccs, hop_length, sample_rate)
