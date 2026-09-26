import logging
import typing

import numpy as np
import scipy.optimize as optimize
import scipy.spatial as spatial
import shapely
from shapely.geometry import LineString

import vector_textures.hierarchy_node as hierarchy_node
from scipy.interpolate import Rbf


def assignment_from_cost_matrix(a_textons, b_textons, cost_matrix) -> dict:
    vertex_assignment = {}
    unassigned_a = a_textons.copy()
    unassigned_b = b_textons.copy()

    for assigned_a, assigned_b in zip(*optimize.linear_sum_assignment(cost_matrix)):
        if a_textons[assigned_a] not in vertex_assignment:
            vertex_assignment[a_textons[assigned_a]] = []

        vertex_assignment[a_textons[assigned_a]].append(b_textons[assigned_b])
        unassigned_a.remove(a_textons[assigned_a])
        unassigned_b.remove(b_textons[assigned_b])

    for unassigned in unassigned_a:
        index = a_textons.index(unassigned)
        assigned_index = np.argmin(cost_matrix[index])

        if a_textons[index] not in vertex_assignment:
            vertex_assignment[a_textons[index]] = []

        vertex_assignment[a_textons[index]].append(b_textons[assigned_index])

    for unassigned in unassigned_b:
        index = b_textons.index(unassigned)
        assigned_index = np.argmin(cost_matrix[:, index])
        vertex_assignment[a_textons[assigned_index]].append(unassigned)

    return vertex_assignment


def polygon_area(poly):
    return 0.5 * np.sum(
        poly[:-1, 0] * poly[1:, 1] - poly[1:, 0] * poly[:-1, 1]
    )


def insert_random_edge_points(poly: np.ndarray, num_points: int) -> np.ndarray:
    n = len(poly)
    if num_points == 0:
        return poly

    # Treat polygon as closed (connect last point to first)
    closed_poly = np.vstack([poly, poly[0]])
    line = LineString(closed_poly)
    total_length = line.length

    # Generate random distances along the perimeter
    insert_distances = np.sort(np.random.rand(num_points) * total_length)
    inserted = 0
    new_coords = []

    next_dist = insert_distances[inserted] if inserted < num_points else None
    seg_start = 0
    seg_end = 1
    accumulated = 0.0

    while seg_start < len(poly):
        p0 = closed_poly[seg_start]
        p1 = closed_poly[seg_end]
        edge = LineString([p0, p1])
        edge_len = edge.length
        new_coords.append(p0)

        # Insert any points along this edge
        while next_dist is not None and accumulated <= next_dist < accumulated + edge_len:
            local_t = (next_dist - accumulated) / edge_len
            new_point = (1 - local_t) * p0 + local_t * p1
            new_coords.append(new_point)
            inserted += 1
            next_dist = insert_distances[inserted] if inserted < num_points else None

        accumulated += edge_len
        seg_start += 1
        seg_end = (seg_end + 1) % len(closed_poly)

    return np.array(new_coords, dtype=np.float32)


def equalize_polygon_points(poly1: np.ndarray, poly2: np.ndarray):
    len1, len2 = len(poly1), len(poly2)
    if len1 == len2:
        return poly1, poly2
    elif len1 < len2:
        poly1 = insert_random_edge_points(poly1, len2 - len1)
    else:
        poly2 = insert_random_edge_points(poly2, len1 - len2)
    return poly1, poly2


def close_polygon(polygon: np.ndarray) -> np.ndarray:
    if not np.allclose(polygon[0], polygon[-1]):
        polygon = np.vstack([polygon, polygon[0]])

    return polygon


def is_clockwise(poly):
    if not np.allclose(poly[0], poly[-1]):
        poly = np.vstack([poly, poly[0]])
    return polygon_area(poly) < 0


class PolygonInterpolation:
    def __init__(self, polygon_a: np.ndarray, polygon_b: np.ndarray):
        if is_clockwise(polygon_a) != is_clockwise(polygon_b):
            polygon_b = polygon_b[::-1]

        polygon_a, polygon_b = equalize_polygon_points(polygon_a, polygon_b)

        polygon_a = close_polygon(polygon_a)
        polygon_b = close_polygon(polygon_b)

        self.polygon_a = polygon_a
        self.polygon_b = polygon_b

        a_point_count = len(self.polygon_a)
        b_point_count = len(self.polygon_b)

        if a_point_count > b_point_count:
            pass

        self.centroid_a = np.array(shapely.Polygon(self.polygon_a).centroid.xy).flatten()
        self.centroid_b = np.array(shapely.Polygon(self.polygon_b).centroid.xy).flatten()

        self.polygon_a_offset = self.polygon_a - self.centroid_a
        self.polygon_b_offset = self.polygon_b - self.centroid_b

        self.initial_color: np.ndarray = None
        self.final_color: np.ndarray = None

    def interpolate(self, t: float) -> np.ndarray:
        relative_points = (1 - t) * self.polygon_a_offset + t * self.polygon_b_offset
        offset = (1 - t) * self.centroid_a + t * self.centroid_b

        return relative_points + offset

    def __call__(self, t: float) -> np.ndarray:
        return self.interpolate(t)


def interpolation_assignment(a_textons, b_textons, silent: bool = False) -> typing.Tuple[
    typing.List[PolygonInterpolation],
    typing.Dict[hierarchy_node.VectorNode, typing.List[hierarchy_node.VectorNode]]
]:
    if silent:
        def log(*_):
            pass

    else:
        def log(*args):
            logging.info(*args)

    log("Building cost matrix")
    a_centroids = np.array([obj.get_centroid() for obj in a_textons])
    b_centroids = np.array([obj.get_centroid() for obj in b_textons])

    cost_matrix = spatial.distance_matrix(a_centroids, b_centroids)

    log("Computing polygon assignment")
    vertex_assignment = assignment_from_cost_matrix(a_textons, b_textons, cost_matrix)

    log("Computing interpolation plan...")
    interpolations = []
    for initial_polygon, final_polygons in vertex_assignment.items():
        for final_polygon in final_polygons:
            interpolations.append(PolygonInterpolation(
                initial_polygon.exterior,
                final_polygon.exterior
            ))
            interpolations[-1].initial_color = initial_polygon.color
            interpolations[-1].final_color = final_polygon.color

    return interpolations, vertex_assignment


class TextonInterpolation:
    def __init__(self, texton_a: hierarchy_node.VectorNode, texton_b: hierarchy_node.VectorNode):
        self.texton_a = texton_a.copy(deep_copy=True)
        self.texton_b = texton_b.copy(deep_copy=True)

        # self.texton_a.set_exterior(self.texton_a.as_shapely().buffer(0))
        # self.texton_b.set_exterior(self.texton_b.as_shapely().buffer(0))

        self.texton_a.set_centroid((0, 0))
        self.texton_b.set_centroid((0, 0))

        self.interpolation = PolygonInterpolation(self.texton_a.exterior, self.texton_b.exterior)
        self.interpolation.initial_color = self.texton_a.color
        self.interpolation.final_color = self.texton_b.color

    def get_interpolation(self, t: float, color_t: float = None, set_category: bool = True, set_descriptor: bool = True) -> hierarchy_node.VectorNode:
        if color_t is None:
            color_t = t

        if len(self.texton_a.children) == 0 and len(self.texton_b.children) == 0:
            child_interpolations = []
        else:
            child_interpolations, _ = interpolation_assignment(
                self.texton_a.children, self.texton_b.children, silent=True
            )

        interpolated = self.interpolation(t)
        new_texton = hierarchy_node.VectorNode(
            interpolated, (1 - t) * self.interpolation.initial_color + t * self.interpolation.final_color
        )

        # new_texton_shapely = new_texton.as_shapely().buffer(0)
        for child in child_interpolations:
            child_interp = child(t)
            new_child = hierarchy_node.VectorNode(
                child_interp, (1 - color_t) * child.initial_color + color_t * child.final_color
            )
            new_child.color = np.clip(new_child.color, 0, 1)

            new_texton.add_child(new_child)

        if np.random.random() < t:
            inherit_texton = self.texton_a
        else:
            inherit_texton = self.texton_b

        if set_category:
            new_texton.category = inherit_texton.category

        if set_descriptor:
            new_texton.descriptor = inherit_texton.descriptor.copy()

        return new_texton


class ControlPoint:
    def __init__(self, position, color):
        self.position: np.ndarray = position
        self.color: np.ndarray = color


def gradient_point_assignment(source_texture, destination_texture) -> dict:
    control_a = [ControlPoint(point, color) for point, color in zip(source_texture.points, source_texture.colors)]
    control_b = [ControlPoint(point, color) for point, color in zip(destination_texture.points, destination_texture.colors)]

    cost_matrix = spatial.distance_matrix(source_texture.points, destination_texture.points)

    return assignment_from_cost_matrix(control_a, control_b, cost_matrix)


def rasterize_rbf(points, colors, size: tuple, function='linear', smooth: int = 15):
    width, height = size
    width = int(width)
    height = int(height)
    out_image = np.zeros((height, width, 3), dtype=np.float32)

    for c in [0, 1, 2]:
        try:
            rbf = Rbf(
                points[:, 0], points[:, 1],
                colors[:, c], function=function, smooth=smooth
            )

            grid_x, grid_y = np.meshgrid(np.linspace(0, width - 1, width), np.linspace(0, height - 1, height))
            grid_intensity = rbf(grid_x, grid_y)

        except ZeroDivisionError:
            grid_intensity = np.median(colors[:, c])

        out_image[:, :, c] = grid_intensity

    return np.clip(out_image, 0, 1)
