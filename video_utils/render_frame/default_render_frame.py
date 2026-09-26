import video_utils.render_frame.base_render_frame as base_render_frame
import PIL.Image
import PIL.ImageDraw
import numpy as np
import cv2
import vector_textures


class DefaultRenderFrame(base_render_frame.BaseRenderFrame):
    def __init__(self, show_background: bool = True):
        self.show_background = show_background

    def execute(
            self, t: float, step: int, interpolations, frame_size, gradient_point_mapping: dict, upscale: float = 1
    ) -> np.ndarray:
        # print(t, step)
        W, H = frame_size

        W *= upscale
        H *= upscale

        if self.show_background:
            colors = []
            positions = []

            for initial_polygon, final_polygons in gradient_point_mapping.items():
                for final_polygon in final_polygons:
                    positions.append((1 - t) * initial_polygon.position + t * final_polygon.position)
                    colors.append((1 - t) * initial_polygon.color + t * final_polygon.color)

            gradient_field = vector_textures.interpolation.rasterize_rbf(np.array(positions), np.array(colors),
                                                                         frame_size)
            gradient_field = cv2.resize(gradient_field, dsize=(W, H), interpolation=cv2.INTER_CUBIC)
            surface = PIL.Image.fromarray(np.clip(gradient_field * 255, 0, 255).astype(np.uint8))

        else:
            surface = PIL.Image.new("RGB", (W, H), (0, 0, 0))

        result_draw = PIL.ImageDraw.ImageDraw(surface, 'RGBA')

        # draw_textons = []
        for interp, path in interpolations:  # type: vector_textures.interpolation.TextonInterpolation, tuple
            texton = interp.get_interpolation(t, set_category=False, set_descriptor=False)
            texton.z_order = (1 - t) * interp.texton_a.z_order + t * interp.texton_b.z_order

            #     draw_textons.append((texton, path))
            #
            # draw_textons.sort(key=lambda x: x[0].z_order)
            #
            # for texton, path in draw_textons:
            texton.exterior += path[step]
            # texton.set_centroid(path[step])

            coordinates = [tuple(c) for c in (texton.get_polygon_coordinate_pairs() * upscale)]
            if len(coordinates) < 2:
                continue

            shifts = [
                (0, 0),
                (W, 0), (-W, 0),
                (0, H), (0, -H),
                (W, H), (W, -H),
                (-W, H), (-W, -H),
            ]

            for dx, dy in shifts:
                shifted = [(int(x + dx), int(y + dy)) for x, y in coordinates]
                result_draw.polygon(shifted, fill=tuple(texton.color_to_int()))

        return np.asarray(surface)
