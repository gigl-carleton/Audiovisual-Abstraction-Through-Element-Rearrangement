import numpy as np

import common.savable_object as savable_object
import vector_textures


class AnalyzedFrame(savable_object.SavableObject):
    def __init__(self, primary_textons, background):
        self.primary_textons: vector_textures.analysis.result_objects.PrimaryTextonResult = primary_textons
        self.gradient_field: vector_textures.analysis.result_objects.GradientFieldResult = background
