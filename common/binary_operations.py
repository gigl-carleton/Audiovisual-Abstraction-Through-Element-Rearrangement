import numba
import numpy as np


@numba.jit(nopython=True)
def extract_segments(padded):
    pr, pc = padded.shape
    # We use a fixed-size array for segments; if needed, we overestimate and then trim.
    seg_arr = np.empty((pr * pc * 4, 4), dtype=np.int32)
    count = 0

    # Loop over pixels (skip border due to padding)
    for i in range(1, pr - 1):
        for j in range(1, pc - 1):
            if not padded[i, j]:
                continue

            # Top edge
            if not padded[i - 1, j]:
                # segment from (j - 1, i - 1) to (j, i - 1)
                seg_arr[count, 0] = j - 1
                seg_arr[count, 1] = i - 1
                seg_arr[count, 2] = j
                seg_arr[count, 3] = i - 1
                count += 1

            # Right edge
            if not padded[i, j + 1]:
                # segment from (j, i - 1) to (j, i)
                seg_arr[count, 0] = j
                seg_arr[count, 1] = i - 1
                seg_arr[count, 2] = j
                seg_arr[count, 3] = i
                count += 1

            # Bottom edge
            if not padded[i + 1, j]:
                # segment from (j - 1, i) to (j, i)
                seg_arr[count, 0] = j - 1
                seg_arr[count, 1] = i
                seg_arr[count, 2] = j
                seg_arr[count, 3] = i
                count += 1

            # Left edge
            if not padded[i, j - 1]:
                # segment from (j - 1, i - 1) to (j - 1, i)
                seg_arr[count, 0] = j - 1
                seg_arr[count, 1] = i - 1
                seg_arr[count, 2] = j - 1
                seg_arr[count, 3] = i
                count += 1

    return seg_arr[:count]


def build_edge_map(segments):
    # Use a dictionary to map each point to its connected segments.
    edge_map = {}
    for seg in segments:
        a = (seg[0], seg[1])
        b = (seg[2], seg[3])
        for pt in (a, b):
            if pt not in edge_map:
                edge_map[pt] = []
        edge_map[a].append((a, b))
        edge_map[b].append((a, b))
    return edge_map


def trace_polygon(edge_map):
    # Find the starting point: the one with minimal (y, x)
    all_points = list(edge_map.keys())
    start_point = min(all_points, key=lambda pt: (pt[1], pt[0]))
    polygon = [start_point]
    current_point = start_point
    segments_used = set()

    while True:
        found = False
        for seg in edge_map[current_point]:
            if seg in segments_used:
                continue
            a, b = seg
            next_point = b if a == current_point else a
            segments_used.add(seg)
            polygon.append(next_point)
            current_point = next_point
            found = True
            break
        if not found or current_point == start_point:
            break
    return polygon


def mask_to_polygon(mask, pad: int = 1):
    mask = mask.astype(np.bool_)
    padded = np.pad(mask, pad_width=pad, mode='constant', constant_values=False)
    segments = extract_segments(padded)
    if segments.shape[0] == 0:
        return []
    edge_map = build_edge_map(segments)
    polygon = trace_polygon(edge_map)
    return polygon
