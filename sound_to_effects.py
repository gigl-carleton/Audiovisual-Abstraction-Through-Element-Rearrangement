import logging
import os

import pedalboard
import tqdm

import common


class SoundToEffectsConfig(common.SavableObject):
    def __init__(self, input_path: str, output_directory: str, effects_chain: pedalboard.Pedalboard):
        self.input_path = input_path
        self.output_directory = output_directory
        self.effects_chain = effects_chain

    @classmethod
    def from_argv(cls):
        parser = common.argv.get_argparse(
            "Applies a sound effects chain to an audio clip. The audio clip can only be modified programmatically"
        )
        parser.add_argument("result", help="The name of the result to render with the effects chain")

        args = parser.parse_args()

        return cls(
            os.path.join("output", args.exemplar, "raw_audio.wav"), os.path.join("output", args.exemplar),
            pedalboard.Pedalboard([
                pedalboard.Chorus(),
                pedalboard.Invert(),

                pedalboard.Gain(gain_db=10),
                pedalboard.Compressor(threshold_db=-24, ratio=4, attack_ms=10),
                pedalboard.Gain(gain_db=6)
            ])
        )


def next_greater_multiple(n, m):
    return n + (m - (n % m))


if __name__ == '__main__':
    common.logger.configure_logger()

    config = SoundToEffectsConfig.from_argv()

    output_path = os.path.join(config.output_directory, "effects_audio.wav")

    logging.info("Reading input file '{}'".format(os.path.abspath(config.input_path)))
    with pedalboard.io.AudioFile(config.input_path) as f:

        logging.info("Creating file stream for output file '{}' (Sample rate={}, channels={})".format(
            os.path.abspath(output_path), f.samplerate, f.num_channels
        ))
        with pedalboard.io.AudioFile(output_path, 'w', f.samplerate, f.num_channels) as o:

            logging.info("Applying effect chain containing {} effects...".format(len(config.effects_chain)))
            progress = tqdm.trange(next_greater_multiple(f.frames, f.samplerate))
            while f.tell() < f.frames:
                chunk = f.read(f.samplerate)
                o.write(config.effects_chain(chunk, f.samplerate, reset=False))
                progress.update(f.samplerate)

            progress.close()

    logging.info("Finished applying effect chain")
