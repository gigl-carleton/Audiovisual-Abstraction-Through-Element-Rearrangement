import logging
import os.path

import common.logger
import video_utils.audio_video as audio_video


class AVMergeConfig(common.SavableObject):
    def __init__(self, video_path: str, audio_path: str, output_path: str):
        self.video_path = video_path
        self.audio_path = audio_path
        self.output_path = output_path

    @classmethod
    def from_argv(cls):
        parser = common.argv.get_argparse(
            "Merges the result video with one of the audio renderings. "
            "Note that a midi file produced by sound_to_music.py is not a valid selection."
        )
        parser.add_argument("result", help="The name of the result merge the result of")
        parser.add_argument(
            "sound_file",
            help="The sound file to merge (name only, not the path. Must be in the output directory of \"result\")"
        )
        parser.add_argument("--name", default="result.mp4", help="The filename of the merged audio and video")

        args = parser.parse_args()

        return cls(
            os.path.join("output", args.result, "raw_video.avi"),
            os.path.join("output", args.result, args.sound_file),
            os.path.join("output", args.result, args.name)
        )


if __name__ == '__main__':
    common.logger.configure_logger()

    config = AVMergeConfig.from_argv()
    logging.info("Merging video and sound into final artifact...")

    audio_video.merge_audio_video(config.video_path, config.audio_path, config.output_path)
