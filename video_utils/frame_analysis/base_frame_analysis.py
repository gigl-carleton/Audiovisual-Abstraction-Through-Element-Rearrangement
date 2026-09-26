import logging

import numpy as np
import tqdm

import models
import vector_textures


class BaseFrameAnalysis:
    def __init__(
            self, segmentation: vector_textures.segmentation.BaseSegmentation = None,
            background_query_point_spacing: float = 25
    ):
        self.segmentation = segmentation
        self.background_query_point_spacing = background_query_point_spacing

        if self.segmentation is None:
            self.segmentation = vector_textures.segmentation.FelzenszwalbSegmentation(100, min_area=10, max_area=500)

    def execute(self, frame: np.ndarray) -> models.AnalyzedFrame:
        height, width = frame.shape[:2]

        logging.info("Segmenting textons with {}...".format(self.segmentation.get_algorithm_name()))
        remaining_mask = np.ones((height, width)).astype(bool)
        segments = self.segmentation.segment(frame)

        parent = vector_textures.VectorNode.from_rectangle((width, height))

        logging.info("Converting {} segments to vector...".format(len(segments)))
        for segment in tqdm.tqdm(segments):
            remaining_mask = np.logical_and(remaining_mask, np.logical_not(segment))
            mask = vector_textures.hierarchy_node.MaskNode(segment)
            vector = vector_textures.VectorNode(mask.to_polygon(), np.median(frame[mask.mask], axis=0))
            parent.add_child(vector)

        gradient_field_colors, gradient_field_points, density = vector_textures.analysis.get_background_gradient_field(
            frame, remaining_mask, density=self.background_query_point_spacing
        )

        result = models.AnalyzedFrame(
            vector_textures.analysis.result_objects.PrimaryTextonResult(parent, None, None, None),
            vector_textures.analysis.result_objects.GradientFieldResult(gradient_field_points, gradient_field_colors, density, None)
        )

        return result
