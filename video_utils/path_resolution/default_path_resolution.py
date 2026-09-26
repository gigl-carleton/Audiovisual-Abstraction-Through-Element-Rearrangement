import typing

import numpy as np
import scipy.optimize as optimize
import scipy.spatial as spatial
import tqdm

import models
import video_utils.path_resolution.base_path_resolution as base_path_resolution


def correct_location(location, width, height):
    x, y = location

    while x < 0:
        x += width
    while y < 0:
        y += height
    while x >= width:
        x -= width
    while y >= height:
        y -= height

    return np.array([x, y])


class DefaultPathResolution(base_path_resolution.BasePathResolution):
    def compute_paths(self, centroids_a, centroids_b, flow_fields, max_distance) -> typing.Tuple[
        np.ndarray, np.ndarray
    ]:
        centroids_a = np.asarray(centroids_a).astype(np.float32)
        centroids_b = np.asarray(centroids_b).astype(np.float32)

        height, width, _ = flow_fields[0].shape

        field_count = len(flow_fields)
        a_move_frames = field_count // 2
        b_move_frames = field_count - a_move_frames

        path_a = np.zeros((len(centroids_a), a_move_frames, 2))
        path_b = np.zeros((len(centroids_b), b_move_frames, 2))

        for c, centroid in enumerate(tqdm.tqdm(centroids_a)):
            path_a[c, 0] = centroid

            for f in range(1, a_move_frames):
                field = flow_fields[f]

                location = path_a[c, f - 1]
                int_location = location.astype(int)
                int_location = correct_location(int_location, width, height)
                delta = field[int_location[1], int_location[0]]

                length = np.linalg.norm(delta)
                if length > max_distance:
                    delta *= (max_distance / length)

                path_a[c, f] = correct_location(location + delta, width, height)

            # centroids_b[0] = path_a[c, -1].copy()

        for c, centroid in enumerate(tqdm.tqdm(centroids_b)):
            path_b[c, -1] = centroid

            for f in range(1, b_move_frames):
                field = flow_fields[field_count - f]

                location = path_b[c, b_move_frames - f]
                int_location = location.astype(int)
                int_location = correct_location(int_location, width, height)
                delta = field[int_location[1], int_location[0]]

                length = np.linalg.norm(delta)
                if length > max_distance:
                    delta *= (max_distance / length)

                path_b[c, b_move_frames - f - 1] = correct_location(location - delta, width, height)

        return path_a, path_b

    def get_texton_correspondences(self, path_a: np.ndarray, path_b: np.ndarray) -> typing.Dict[int, typing.List[int]]:
        a_midpoint = path_a[:, -1]
        b_midpoint = path_b[:, 0]

        cost_matrix = spatial.distance_matrix(a_midpoint, b_midpoint)

        vertex_assignment = {}

        a_textons = list(range(len(path_a)))
        b_textons = list(range(len(path_b)))

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

    def finalize_paths(
            self, path_a: np.ndarray, path_b: np.ndarray, correspondences: typing.Dict[int, typing.List[int]],
            max_distance: float = 2
    ) -> typing.List[models.Interpolation]:
        result = []

        for texton_a, partner in correspondences.items():
            for texton_b in partner:
                texton_a_path = path_a[texton_a]
                texton_b_path = path_b[texton_b]

                gap = texton_b_path[0] - texton_a_path[-1]
                dist = np.linalg.norm(gap)

                if dist <= max_distance:
                    combined = np.vstack([texton_a_path, texton_b_path])

                else:
                    n_bridge = int(np.ceil(dist / max_distance))
                    nA = min(len(texton_a_path), n_bridge)
                    nB = min(len(texton_b_path), n_bridge)

                    wA = np.linspace(0, 1, nA)[:, None]
                    wB = np.linspace(1, 0, nB)[:, None]

                    A_adj = texton_a_path.copy()
                    B_adj = texton_b_path.copy()

                    A_adj[-nA:] += 0.5 * wA * gap
                    B_adj[:nB] -= 0.5 * wB * gap

                    combined = np.vstack([A_adj, B_adj])

                interp = models.Interpolation(texton_a, texton_b, combined)

                result.append(interp)

        return result

