import datetime
import logging

import numpy as np
import soundfile as sf
import tqdm


def load_audio(fname):
    audio, sr = sf.read(fname, dtype="float32", always_2d=True)
    # audio shape: (samples, channels)
    return audio.T, sr  # -> (channels, samples)


class DefaultSoundStitch:
    def __init__(self, crossfade_time: float = 0.125):
        self.crossfade_time = crossfade_time

    def crossfade(self, a, b, fade_samples):
        fade_out = np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)
        fade_in  = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)

        a_end = a[:, -fade_samples:] * fade_out
        b_start = b[:, :fade_samples] * fade_in

        middle = a_end + b_start

        return np.concatenate([
            a[:, :-fade_samples],
            middle,
            b[:, fade_samples:]
        ], axis=1)

    def execute(self, files):
        logging.info("Saving sound chain (length = {}) to disk with {}s crossfade overlap".format(
            len(files), datetime.timedelta(seconds=self.crossfade_time)
        ))

        audio, sr = load_audio(files[0])
        fade_samples = int(self.crossfade_time * sr)

        for fname in tqdm.tqdm(files[1:], initial=1, total=len(files)):
            next_audio, next_sr = load_audio(fname)

            if next_sr != sr:
                raise ValueError("All files must have the same sample rate")

            if next_audio.shape[0] != audio.shape[0]:
                raise ValueError("All files must have the same number of channels")

            if audio.shape[1] < fade_samples or next_audio.shape[1] < fade_samples:
                raise ValueError("Audio too short for requested crossfade")

            audio = self.crossfade(audio, next_audio, fade_samples)

        logging.info("Sound stitch complete. Final duration is {}".format(datetime.timedelta(seconds=audio.shape[1] / sr)))
        return audio, sr
