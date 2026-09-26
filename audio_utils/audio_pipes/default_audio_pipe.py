import datetime
import logging
import os
import typing
import soundfile as sf

import audio_utils.audio_pipes.base_audio_pipe as base_audio_pipe
import audio_utils.segmentation.sound_segmentation as sound_segmentation
import audio_utils.chain_builder.default_chain_builder as default_chain_builder
import audio_utils.sound_stitch.default_sound_stitch as default_sound_stitch


class DefaultAudioPipe(base_audio_pipe.BaseAudioPipe):
    def __init__(self):
        super().__init__()
        self.overlap: float = 0.125
        self.segment_path = "sound_segments"

        self.sound_segmenter = sound_segmentation.SoundSegmentation()
        self.chain_builder = default_chain_builder.DefaultChainBuilder()
        self.sound_stitch = default_sound_stitch.DefaultSoundStitch()

    def execute(self, key_times: typing.List[float], audio_path: str, interpolation_time: float):
        start = datetime.datetime.now()
        self.chain_builder.overlap = self.overlap
        self.sound_stitch.overlap = self.overlap

        feature_boundaries, timestamp_boundaries = self.sound_segmenter.execute(audio_path)

        if len(feature_boundaries) < 3:
            logging.warning("Input video appears to have no sound! Skipping audio creation!")
            return None

        sound_chains = []
        last_feature = None
        for frame_index in range(len(key_times) - 1):
            logging.info("Building chunk {} of {}...".format(frame_index + 1, len(key_times) - 1))

            timestamp_start = key_times[frame_index]
            timestamp_end = key_times[frame_index + 1]

            chain = self.chain_builder.execute(
                feature_boundaries, timestamp_boundaries, timestamp_start, timestamp_end,
                interpolation_time=interpolation_time, overlap=self.overlap,
                last_feature=last_feature
            )

            last_feature = feature_boundaries[chain[-1], 1]
            sound_chains.append(chain)

        filenames = []
        for chain in sound_chains:
            for i in chain:
                filenames.append(os.path.join(self.segment_path, "{:05d}.wav".format(i)))

        audio, sr = self.sound_stitch.execute(filenames)
        sf.write(self.output_file, audio.T, sr)

        logging.info("Raw audio synthesis complete! Took {}".format(datetime.datetime.now() - start))
        return self.output_file
