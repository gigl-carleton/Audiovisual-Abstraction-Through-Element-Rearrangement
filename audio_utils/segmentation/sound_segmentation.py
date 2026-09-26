import datetime
import logging
import math
import os
import shutil
import typing
import soundfile as sf

import librosa
import numpy as np

import audio_utils.segmentation.feature_extraction as feature_extraction
import audio_utils.segmentation.distance_curve as distance_curve
import audio_utils.segmentation.segment as segment


def save_segments(filename: str, save_path: str, segments: np.ndarray, file_format: str = ".wav", sample_rate=None):
    y, sr = librosa.load(filename, sr=sample_rate)  # sr=None preserves original sample rate
    file_format = file_format.lstrip(".")

    for i, (start_time, end_time) in enumerate(segments):
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        segment = y[start_sample:end_sample]
        output_filename = os.path.join(save_path, "{:05d}.{}".format(i, file_format))
        sf.write(output_filename, segment, sr)


def get_invalid_length_segment_mask(
        timestamp_segments: np.ndarray, min_duration: float = 0, max_duration: float = math.inf
) -> np.ndarray:
    durations = timestamp_segments[:, 1] - timestamp_segments[:, 0]
    return (durations <= max_duration) & (durations >= min_duration)


def indices_to_timestamps(indices, timestamps: np.ndarray) -> np.ndarray:
    time_result = [
        [timestamps[0], timestamps[indices[0]]]
    ]
    for i, current_index in enumerate(indices):
        if i == 0:
            continue

        last_index = indices[i - 1]

        time_result.append([timestamps[last_index + 1], timestamps[current_index]])

    time_result.append([timestamps[indices[-1] + 1], timestamps[-1]])

    return np.array(time_result)


class SoundSegmentation:
    def __init__(self):
        self.max_duration = 1
        self.min_duration = 0.125
        self.sound_segments_path = "sound_segments"

        self.extractor = feature_extraction.MelFrequencyCepstralCoefficients()
        self.curve = distance_curve.EuclideanDistanceCurve()
        self.segmentation = segment.MaximumDuration(self.max_duration)

    def execute(self, sound_filename) -> typing.Tuple[np.ndarray, np.ndarray]:
        logging.info("Using {} feature extractor...".format(self.extractor.get_name()))
        features, timestamps = self.extractor(sound_filename)

        logging.info("Computing {} from feature vectors...".format(self.curve.get_name()))
        distance = self.curve(features)

        logging.info("Segmenting distance curve...")
        indices = self.segmentation(distance, timestamps)

        logging.info("Found {} segments".format(len(indices)))

        logging.info("Resolving segment timestamp and feature boundaries")
        timestamp_boundaries = indices_to_timestamps(indices, timestamps)
        feature_boundaries = indices_to_timestamps(indices, features)

        logging.info("Removing segments less than {}".format(datetime.timedelta(seconds=self.min_duration)))
        mask = get_invalid_length_segment_mask(timestamp_boundaries, self.min_duration)
        timestamp_boundaries = timestamp_boundaries[mask]
        feature_boundaries = feature_boundaries[mask]

        logging.info("{} segments remain".format(len(timestamp_boundaries)))

        logging.info("Saving segments to '{}'".format(os.path.abspath(self.sound_segments_path)))
        shutil.rmtree(self.sound_segments_path, ignore_errors=True)
        os.makedirs(self.sound_segments_path)
        save_segments(sound_filename, self.sound_segments_path, timestamp_boundaries)

        return feature_boundaries, timestamp_boundaries
