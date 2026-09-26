import datetime
import logging
import math
import os
import os.path
import random
import typing

import cv2
import numpy as np
import tqdm

import models
import vector_textures.segmentation
import video_utils.frame_analysis.default_frame_analysis as default_frame_analysis
import video_utils.frame_synthesis.default_frame_synthesis as default_frame_synthesis
import video_utils.video_pipes.base_video_pipe as base_video_pipe
import video_utils.path_resolution.default_path_resolution as default_path_resolution
import video_utils.flow_field_extraction.default_flow_field_extraction as default_flow_field_extraction
import video_utils.render_frame.default_render_frame as default_render_frame

if typing.TYPE_CHECKING:
    import render_av


def get_frame(cap, frame: int, normalize: bool = True, set_size=None) -> np.ndarray:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ret, frame = cap.read()

    if set_size is not None:
        frame = cv2.resize(frame, dsize=set_size, interpolation=cv2.INTER_CUBIC)

    if normalize:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) / 255
    else:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def extract_textons(video, frame_index, generator: 'DefaultVideoPipe'):
    video_framerate = video.get(cv2.CAP_PROP_FPS)

    analyzed_frame = None
    while True:
        try:
            analyzed_frame = generator.frame_analysis.execute(
                get_frame(video, frame_index, set_size=generator.input_video_resize)
            )

        except cv2.error:
            logging.warning("Hit the end of the video!")
            if analyzed_frame is not None:
                logging.warning("Using last analyzed frame in place")

            return analyzed_frame

        if (
                len(analyzed_frame.primary_textons.primary_textons.children) >= generator.min_textons and
                len(analyzed_frame.gradient_field.colors) >= generator.min_colors
        ):
            logging.info("Extracted {} textons and {} colors".format(
                len(analyzed_frame.primary_textons.primary_textons.children),
                len(analyzed_frame.gradient_field.colors)
            ))
            return analyzed_frame

        frame_index += int(math.ceil(video_framerate))
        logging.warning("Not enough elements extracted from frame! Retrying with frame {}".format(frame_index))


class DefaultVideoPipe(base_video_pipe.BaseVideoPipe):
    def __init__(
            self, framerate: float = 24, upscale: float = 3, input_video_resize: typing.Tuple[int, int] = None,
            max_distance: float = 50, min_colors: int = 2, min_textons: int = 50
    ):
        super().__init__(framerate, upscale, input_video_resize, max_distance)
        self.min_colors = min_colors
        self.min_textons = min_textons
        self.frame_analysis = default_frame_analysis.DefaultFrameAnalysis()
        self.frame_synthesis = default_frame_synthesis.DefaultFrameSynthesis()
        self.path_resolution = default_path_resolution.DefaultPathResolution()
        self.flow_field_extraction = default_flow_field_extraction.DefaultFlowFieldExtraction()
        self.render_frame = default_render_frame.DefaultRenderFrame()

    def get_keyframes(self, keyframes, video, color_video=None, background_color_video=None) -> typing.List[
        models.AnalyzedFrame
    ]:
        logging.info("Analyzing keyframes...")
        analyzed_frames = []
        for frame_index in keyframes:
            logging.info("Analyzing keyframe {}...".format(frame_index))

            analyzed_frame = extract_textons(video, frame_index, self)
            if analyzed_frame is not None:
                analyzed_frames.append(analyzed_frame)

        if color_video is not None:
            logging.info("Analyzing color video keyframes...")
            color_frames = []
            for frame_index in keyframes:
                logging.info("Analyzing keyframe {}...".format(frame_index))

                analyzed_frame = extract_textons(color_video, frame_index, self)
                if analyzed_frame is not None:
                    color_frames.append(analyzed_frame)

            logging.info("Copying color from color video to structure video...")
            for structure_frame, color_frame in zip(analyzed_frames, color_frames):     # type: models.AnalyzedFrame
                color_distro = [texton.color for texton in color_frame.primary_textons.primary_textons.children]
                for texton in structure_frame.primary_textons.primary_textons.children:
                    texton.color = random.choice(color_distro)

        if background_color_video is not None:
            logging.info("Analyzing background color video keyframes...")
            color_frames = []
            for frame_index in keyframes:
                logging.info("Analyzing keyframe {}...".format(frame_index))

                analyzed_frame = extract_textons(background_color_video, frame_index, self)
                if analyzed_frame is not None:
                    color_frames.append(analyzed_frame)

            logging.info("Copying color from color video to structure video...")
            for structure_frame, color_frame in zip(analyzed_frames, color_frames):     # type: models.AnalyzedFrame
                structure_frame.gradient_field = color_frame.gradient_field

        return analyzed_frames

    def execute(self, keyframes: typing.List[int], exemplars: 'render_av.Exemplars', interpolation_time: float):
        start = datetime.datetime.now()

        interpolation_frames = interpolation_time * self.framerate
        max_distance_per_frame = self.max_distance / self.framerate

        video_width = int(exemplars.shape_video.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(exemplars.shape_video.get(cv2.CAP_PROP_FRAME_HEIGHT))

        logging.info("Creating video {}x{} @ {}fps".format(
            video_width * self.upscale, video_height * self.upscale, self.framerate
        ))

        logging.info("Creating file stream to '{}'...".format(os.path.abspath(self.output_file)))
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        result_video = cv2.VideoWriter(
            self.output_file, fourcc, self.framerate,
            (int(video_width * self.upscale), int(video_height * self.upscale))
        )

        analyzed_frames = self.get_keyframes(
            keyframes, exemplars.shape_video, exemplars.color_video, exemplars.background_color_video
        )

        logging.info("Synthesizing first keyframe...")
        initial = self.frame_synthesis.execute(analyzed_frames[0], (video_width, video_height))
        generated_frames = 0

        for frame_index in range(len(keyframes) - 1):
            logging.info("Building chunk {} of {}...".format(frame_index + 1, len(keyframes) - 1))
            interval_start = keyframes[frame_index]
            interval_end = keyframes[frame_index + 1]

            logging.info("Extracting flow fields...")
            flow_fields = self.flow_field_extraction.execute(
                exemplars.flow_video, interval_start, interval_end, interpolation_frames
            )

            logging.info("Synthesizing next keyframe...")
            final = self.frame_synthesis.execute(analyzed_frames[frame_index + 1], (video_width, video_height))

            logging.info("Resolving texton centroids...")
            centroids_a = [texton.get_centroid() for texton in initial.primary_textons.children]
            centroids_b = [texton.get_centroid() for texton in final.primary_textons.children]

            logging.info("Resolving texton paths through flow fields...")
            paths_a, paths_b = self.path_resolution.compute_paths(
                centroids_a, centroids_b, flow_fields, max_distance=max_distance_per_frame
            )

            logging.info("Resolving texton path correspondences...")
            correspondences = self.path_resolution.get_texton_correspondences(paths_a, paths_b)

            logging.info("Finalizing texton paths...")
            interpolation_paths = self.path_resolution.finalize_paths(
                paths_a, paths_b, correspondences, max_distance=max_distance_per_frame
            )

            logging.info("Creating polygon vertex interpolation plan...")
            interpolations = []
            for interp in interpolation_paths:
                interpolations.append((
                    vector_textures.interpolation.TextonInterpolation(
                        initial.primary_textons.children[interp.texton_a],
                        final.primary_textons.children[interp.texton_b]
                    ), interp.path
                ))
                final.primary_textons.children[interp.texton_b].z_order = initial.primary_textons.children[
                    interp.texton_a
                ].z_order

            logging.info("Created {} interpolations".format(len(interpolations)))
            interpolations.sort(key=lambda x: x[0].texton_a.z_order)

            frames = len(flow_fields)

            logging.info("Computing gradient point assignment...")
            gradient_point_mapping = vector_textures.interpolation.gradient_point_assignment(
                initial.gradient_field, final.gradient_field
            )

            logging.info("Rendering each interpolated frame...")
            for f in tqdm.trange(frames):
                t = f / frames
                frame = self.render_frame.execute(
                    t, f, interpolations, (video_width, video_height), gradient_point_mapping, upscale=self.upscale
                )
                result_video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                generated_frames += 1

            initial = final

        logging.info("Releasing resulting video file...")
        result_video.release()

        logging.info("Video synthesis complete! Took {}".format(datetime.datetime.now() - start))
        return self.output_file
