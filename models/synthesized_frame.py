import common
import vector_textures.hierarchy_node as hierarchy_node
import vector_textures.analysis as analysis
import typing


class SynthesizedFrame(common.SavableObject):
    def __init__(self, primary_textons, background, frame_size):
        self.primary_textons: hierarchy_node.VectorNode = primary_textons
        self.gradient_field: analysis.result_objects.GradientFieldResult = background
        self.frame_size: typing.Tuple[int, int] = frame_size
