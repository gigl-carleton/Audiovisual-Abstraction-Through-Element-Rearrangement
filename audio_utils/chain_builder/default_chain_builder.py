import datetime
import logging

import numpy as np

import audio_utils.chain_builder.base_chain_builder as base_chain_builder


def softmin(d, temperature=10.0):
    d = d / temperature
    d = d - d.min(axis=1, keepdims=True)
    exp_neg = np.exp(-d)
    return exp_neg / exp_neg.sum(axis=1, keepdims=True)


class DefaultChainBuilder(base_chain_builder.BaseChainBuilder):
    def __init__(self, reselect_decay: float = 0.6):
        super().__init__()
        self.reselect_decay = reselect_decay

    def execute(
            self, feature_boundaries: np.ndarray, timestamp_boundaries: np.ndarray, start_timestamp: float,
            end_timestamp: float, interpolation_time: float, overlap: float = 0.125,
            last_feature: np.ndarray = None
    ):
        logging.info("Building sound chain for segments between {} and {}".format(
            datetime.timedelta(seconds=start_timestamp), datetime.timedelta(seconds=end_timestamp)
        ))
        segment_mask = (timestamp_boundaries[:, 1] > start_timestamp) & (timestamp_boundaries[:, 0] < end_timestamp)
        segment_ids = np.where(segment_mask)[0]

        logging.info("{} segments exist between these timestamps".format(len(segment_ids)))

        feature_boundaries = feature_boundaries[segment_mask]
        timestamp_boundaries = timestamp_boundaries[segment_mask]

        segment_durations = timestamp_boundaries[:, 1] - timestamp_boundaries[:, 0]

        logging.info("Building asymmetric distance matrix...")
        starts = feature_boundaries[:, 0]
        ends = feature_boundaries[:, 1]
        diff = ends[:, None, :] - starts[None, :, :]
        distance_matrix = np.linalg.norm(diff, axis=-1)

        logging.info("Converting matrix to Markov chain")
        markov_chain = softmin(distance_matrix)

        if last_feature is None:
            current_vertex = np.random.randint(0, len(segment_ids) - 1)
        else:
            diff = starts - last_feature
            dists = np.linalg.norm(diff, axis=1)
            current_vertex = np.argmin(dists)

        current_duration = segment_durations[current_vertex] - overlap
        chain = [segment_ids[current_vertex]]

        logging.info("Building sound chain...")
        while current_duration < interpolation_time:
            distro = markov_chain[current_vertex]
            next_vertex = np.random.choice(len(distro), p=distro)

            markov_chain[current_vertex, next_vertex] *= self.reselect_decay
            markov_chain[current_vertex] /= markov_chain[current_vertex].sum()

            current_vertex = next_vertex
            chain.append(segment_ids[current_vertex])
            current_duration += segment_durations[current_vertex] - overlap

        logging.info("Chain segment is {} (minimum duration is {})".format(
            datetime.timedelta(seconds=current_duration), datetime.timedelta(seconds=interpolation_time)
        ))

        return chain
