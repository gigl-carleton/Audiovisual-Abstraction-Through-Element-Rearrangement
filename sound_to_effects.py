import os
import logging
import common
import pedalboard


class SoundToEffectsConfig:
    def __init__(self, input_path: str, output_directory: str, effects_chain: pedalboard.Pedalboard = None):
        self.input_path = input_path
        self.output_directory = output_directory
        self.effects_chain = effects_chain

    @classmethod
    def from_argv(cls):
        return cls(
            "output/tmp/raw_audio.wav", "output/tmp", pedalboard.Pedalboard([
                pedalboard.Chorus(),
                pedalboard.Invert(),

                pedalboard.Gain(gain_db=10),
                pedalboard.Compressor(threshold_db=-24, ratio=4, attack_ms=10),
                pedalboard.Gain(gain_db=6)
            ])
        )


if __name__ == '__main__':
    common.logger.configure_logger()

    config = SoundToEffectsConfig.from_argv()

    input_path = os.path.join(config.input_path)
    output_path = os.path.join(config.output_directory, "effects_audio.wav")

    logging.info("Applying effects...")
    with pedalboard.io.AudioFile(input_path) as f:
        with pedalboard.io.AudioFile(output_path, 'w', f.samplerate, f.num_channels) as o:
            while f.tell() < f.frames:
                chunk = f.read(f.samplerate)
                o.write(config.effects_chain(chunk, f.samplerate, reset=False))
