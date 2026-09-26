import logging

import common.logger
import video_utils.audio_video as audio_video


class AVMergeConfig:
    def __init__(self, video_path: str, audio_path: str, output_path: str):
        self.video_path = video_path
        self.audio_path = audio_path
        self.output_path = output_path

    @classmethod
    def from_argv(cls):
        return cls(
            "output/tmp/raw_video.avi", "output/tmp/music_audio_effects.wav", "output/tmp/result.mp4"
        )


if __name__ == '__main__':
    common.logger.configure_logger()

    config = AVMergeConfig.from_argv()
    logging.info("Merging video and sound into final artifact...")

    audio_video.merge_audio_video(config.video_path, config.audio_path, config.output_path)
