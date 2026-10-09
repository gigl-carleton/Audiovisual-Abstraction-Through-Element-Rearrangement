import datetime
import logging
import math
import os
import typing

import cv2

import common.logger
import video_utils
import audio_utils
import keyframe_selectors


class Exemplars:
    def __init__(
            self, shape_video: str, flow_video: str = None, color_video: str = None,
            background_color_video: str = None, audio: str = None, overwrite_audio: bool = False
    ):
        self.default_path = shape_video
        self.shape_video = self.load_video(shape_video, resample=False)

        self.temp_files = []

        self.framerate = self.shape_video.get(cv2.CAP_PROP_FPS)

        self.video_resampler = video_utils.Resample(
            self.framerate, (
                int(self.shape_video.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(self.shape_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            )
        )

        self.flow_video = self.load_video(
            flow_video, resampled_name="tmp/flow_video.avi"
        )
        self.color_video = self.load_video(
            color_video, resampled_name="tmp/color_video.avi"
        )
        self.background_color_video = self.load_video(
            background_color_video, resampled_name="tmp/background_color_video.avi"
        )

        if audio is None:
            audio = self.default_path

        if audio.endswith((".avi", ".mp4")):
            _, audio = video_utils.split_audio_video(audio, overwrite_audio=overwrite_audio)

        self.audio = audio

        if self.flow_video is None:
            self.flow_video = self.shape_video

    def release(self, clear_temp_files: bool = True):
        if self.shape_video is not None:
            self.shape_video.release()

        if self.flow_video is not None:
            self.flow_video.release()

        if self.color_video is not None:
            self.color_video.release()

        if self.background_color_video is not None:
            self.background_color_video.release()

        if clear_temp_files:
            self.clear_temp_files()

    def load_video(
            self, filename: typing.Union[str, cv2.VideoCapture, None] = None,
            resample: bool = True, resampled_name: str = None
    ) -> typing.Union[cv2.VideoCapture, None]:

        if filename is None:
            return None

        if isinstance(filename, str):
            cap = cv2.VideoCapture(filename)
        else:
            cap = filename

        if resample:
            temp_filename = "tmp/tmp.avi" if resampled_name is None else resampled_name
            self.temp_files.append(temp_filename)
            return self.video_resampler.execute(cap, temp_filename)

        return cap

    def clear_temp_files(self):
        for path in self.temp_files:
            os.remove(path)

    def get_total_frames(self) -> int:
        frames = math.inf

        if self.shape_video is not None:
            frames = min(self.shape_video.get(cv2.CAP_PROP_FRAME_COUNT), frames)

        if self.flow_video is not None:
            frames = min(self.flow_video.get(cv2.CAP_PROP_FRAME_COUNT), frames)

        if self.color_video is not None:
            frames = min(self.color_video.get(cv2.CAP_PROP_FRAME_COUNT), frames)

        if self.background_color_video is not None:
            frames = min(self.background_color_video.get(cv2.CAP_PROP_FRAME_COUNT), frames)

        return int(frames)


class RenderAVConfig:
    def __init__(
            self,
            exemplars: Exemplars,
            save_name: str,
            keyframe_selector: keyframe_selectors.BaseKeyframeSelector,
            video_pipe: video_utils.video_pipes.BaseVideoPipe,
            audio_pipe: audio_utils.audio_pipes.BaseAudioPipe,
    ):
        self.exemplars = exemplars
        self.save_name = save_name

        self.keyframe_selector = keyframe_selector
        self.video_pipe = video_pipe
        self.audio_pipe = audio_pipe

        os.makedirs(os.path.join("output", save_name), exist_ok=True)

    @classmethod
    def from_argv(cls):
        parser = common.argv.get_argparse(
            "Renders the video and its soundtrack in an abstract way. "
            "Note that the resulting audio file will need to be further processed by "
            "'sound_to_effects.py' or 'sound_to_music.py' to complete the pipeline."
        )
        parser.add_argument(
            "exemplar",
            help="The input video. This is where the segment shapes will be extracted from, "
                 "and the default video for all other video types."
        )
        parser.add_argument(
            "output_name",
            help="The directory to save the resulting files in under 'output/'"
        )

        parser.add_argument(
            "duration", help="The duration of the result in seconds", type=float
        )

        # region ExemplarConfig
        parser.add_argument(
            "--flow_video",
            help="The path to the flow_video if it should be different than the exemplar", default=None
        )
        parser.add_argument(
            "--color_video",
            help="The path to the color_video if it should be different than the exemplar", default=None
        )
        parser.add_argument(
            "--background_color_video",
            help="The path to the background_color_video if it should be different than the exemplar", default=None
        )
        parser.add_argument(
            "--audio",
            help="The path to the audio if it should be different than the exemplar", default=None
        )
        # endregion

        # region KeyframeSelectConfig
        parser.add_argument(
            "--max_duration",
            help="If set, the result will be between 'duration' and 'max_duration' in seconds. "
                 "Otherwise, the result will be 'duration' seconds long", default=None, type=float
        )
        parser.add_argument(
            "--interpolation_time",
            help="The amount of time, in seconds, to space each keyframe in the result", default=5, type=float
        )
        parser.add_argument(
            "--deterministic_start", action='store_true',
            help="If enabled, the first keyframe will be the first frame. Otherwise, this will be random."
        )
        # endregion

        # region VideoConfig
        parser.add_argument(
            "--framerate", help="The framerate to render the final video at", default=24, type=float
        )
        parser.add_argument(
            "--upscale",
            help="The amount to upscale the final frames (vector upscale to avoid artifacts). "
                 "It is recommended to generate at a lower resolution (e.g. 360p) "
                 "and then upscale to higher resolution (e.g. 1080p)",
            default=3, type=float
        )
        parser.add_argument(
            "--input_video_resize",
            help="The size to rescale all frames to before being used. "
                 "It is recommended to generate at a lower resolution (e.g. 360p) "
                 "and then upscale to higher resolution (e.g. 1080p). "
                 "Default is to use the size of the 'shape' video",
            default=None, type=int, nargs=2
        )
        parser.add_argument(
            "--max_distance",
            help="The maximum distance a texton can move per frame",
            default=50, type=float
        )
        parser.add_argument(
            "--min_colors",
            help="The minimum colors that need to be present in a frame to be selected as a keyframe. "
                 "Otherwise, we will use a nearby frame.",
            default=2, type=int
        )
        parser.add_argument(
            "--min_textons",
            help="The minimum textons that need to be present in a frame to be selected as a keyframe. "
                 "Otherwise, we will use a nearby frame.",
            default=50, type=int
        )
        # endregion

        # region AudioConfig
        parser.add_argument(
            "--segment_overlap", help="The time (in seconds) for audio segments to overlap in the final chain",
            default=0.125, type=float
        )
        parser.add_argument(
            "--segment_path", help="The temporary directory to store audio segments",
            default="sound_segments"
        )
        # endregion

        args = parser.parse_args()

        return cls(
            Exemplars(
                shape_video=args.exemplar,
                flow_video=args.flow_video,
                color_video=args.color_video,
                background_color_video=args.background_color_video,
                audio=args.audio
            ),
            args.output_name,
            keyframe_selectors.DefaultKeyframeSelector(
                args.duration,
                args.max_duration,
                interpolation_time=args.interpolation_time,
                random_start=not args.deterministic_start
            ),
            video_utils.video_pipes.DefaultVideoPipe(
                framerate=args.framerate,
                upscale=args.upscale,
                input_video_resize=args.input_video_resize,
                max_distance=args.max_distance,
                min_colors=args.min_colors,
                min_textons=args.min_textons,
            ),
            audio_utils.audio_pipes.DefaultAudioPipe(
                segment_overlap=args.segment_overlap,
                segment_path=args.segment_path,
            )
        )


def render_av(config: RenderAVConfig, release_exemplars: bool = True):
    keyframes = config.keyframe_selector.execute(config.exemplars.get_total_frames())

    logging.info("Selected keyframes ({}) are {}".format(len(keyframes), keyframes))
    logging.info("Keyframe Timestamps: [{}]".format(
        ", ".join(str(datetime.timedelta(seconds=frame / config.exemplars.framerate)) for frame in keyframes)
    ))
    logging.info("Selection ranges from {} to {}".format(
        datetime.timedelta(seconds=keyframes[0] / config.exemplars.framerate),
        datetime.timedelta(seconds=keyframes[-1] / config.exemplars.framerate)
    ))
    logging.info(
        "Total length of result: {}".format(
            datetime.timedelta(seconds=(len(keyframes) - 1) * config.keyframe_selector.interpolation_time)
        )
    )

    key_times = [frame / config.exemplars.framerate for frame in keyframes]

    if config.exemplars.audio is not None:
        config.audio_pipe.output_file = os.path.join("output", config.save_name, "raw_audio.wav")
        config.audio_pipe.execute(key_times, config.exemplars.audio, config.keyframe_selector.interpolation_time)

    config.video_pipe.output_file = os.path.join("output", config.save_name, "raw_video.avi")
    config.video_pipe.execute(keyframes, config.exemplars, config.keyframe_selector.interpolation_time)

    if release_exemplars:
        cfg.exemplars.release()


if __name__ == '__main__':
    common.logger.configure_logger()

    cfg = RenderAVConfig.from_argv()

    render_av(cfg)
