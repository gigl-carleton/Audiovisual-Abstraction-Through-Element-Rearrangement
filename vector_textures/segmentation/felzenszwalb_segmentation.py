import math

import numpy as np
import skimage
from skimage.segmentation import felzenszwalb

import vector_textures.segmentation.base_segmentation as base_segmentation
import common


class FelzenszwalbSegmentation(base_segmentation.BaseSegmentation):
    def __init__(
            self, scale: float, sigma: float = 0.95, min_area: int = 0,
            max_area: int = math.inf, silent: bool = False
    ):
        super().__init__(0, max_area, silent)

        self.scale = scale
        self.sigma = sigma
        self.min_size = min_area

    def _segment(self, image: np.ndarray, mask: np.ndarray = None):
        segments = felzenszwalb(
            image, scale=self.scale, sigma=self.sigma, min_size=self.min_size
        ) + 1

        result = []
        for segment in np.unique(segments):
            if segment == 0:
                continue

            result.append(segments == segment)

        # print(np.mean([np.count_nonzero(i) for i in result]))
        return result
