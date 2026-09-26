import logging
import typing

import bridson
import numpy as np

import models
import vector_textures
import video_utils.frame_synthesis.base_frame_synthesis as base_frame_synthesis


def primary_texton_poisson_disk(
        textons: vector_textures.VectorNode, frame_shape, selection_probability_decay: float = 2, spacing_range=(8, 10)
) -> vector_textures.VectorNode:
    spacing = np.random.uniform(*spacing_range)
    # spacing = np.random.uniform(15, 16)

    query_points = []
    for _ in range(10):
        query_points = bridson.poisson_disc_samples(frame_shape[0], frame_shape[1], spacing)
        if len(query_points) > 1:
            break

    logging.info("Selected spacing {:.2f}, which created {} query points".format(spacing, len(query_points)))

    result = vector_textures.VectorNode.from_rectangle(frame_shape)

    texton_count = len(textons.children)
    selection_probabilities = np.ones(texton_count)
    selection_probabilities /= selection_probabilities.sum()

    for point in query_points:
        idx = np.random.choice(texton_count, p=selection_probabilities)
        selection_probabilities[idx] /= selection_probability_decay
        selection_probabilities /= selection_probabilities.sum()

        new_texton = textons.children[idx].copy()
        new_texton.set_centroid(point)

        result.add_child(new_texton)

    return result


def generate_gradient_field(colors: np.ndarray, density, size: typing.Tuple[int, int]):
    control_points = []
    for _ in range(10):
        control_points = bridson.poisson_disc_samples(*size, density, k=10)
        if len(control_points) > min(size) / 2:
            break

    control_points = np.array(control_points)

    logging.info(
        "Selected spacing {} in {}x{}, which created {} control points".format(density, *size, len(control_points))
    )

    control_colors = colors[np.random.randint(0, len(colors), len(control_points))]

    return control_points, control_colors


class DefaultFrameSynthesis(base_frame_synthesis.BaseFrameSynthesis):
    def __init__(self, selection_probability_decay: float = 2):
        self.selection_probability_decay = selection_probability_decay
        self.spacing_range = (8, 10)

    def execute(self, analyzed_frame: models.AnalyzedFrame, frame_shape) -> models.SynthesizedFrame:
        primary_textons = primary_texton_poisson_disk(
            analyzed_frame.primary_textons.primary_textons, frame_shape,
            selection_probability_decay=self.selection_probability_decay,
            spacing_range=self.spacing_range
        )

        primary_textons.smooth_boundary()
        for z_order, texton in enumerate(primary_textons.children):
            texton.z_order = z_order

        gradient_field_points, gradient_field_colors = generate_gradient_field(
            analyzed_frame.gradient_field.colors, analyzed_frame.gradient_field.query_point_spacing, frame_shape
        )

        gradient_field_result = vector_textures.analysis.result_objects.GradientFieldResult(
            gradient_field_points, gradient_field_colors, analyzed_frame.gradient_field.query_point_spacing,
            analyzed_frame.gradient_field.solid_color
        )

        return models.SynthesizedFrame(primary_textons, gradient_field_result, frame_shape)
