import video_utils.flow_field_extraction.base_flow_field_extraction as base_flow_field_extraction
import cv2
import numpy as np
import tqdm


def get_flow_fields(cap: cv2.VideoCapture, start_frame: int, end_frame: int) -> np.ndarray:
    fields = []

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    _, current_frame = cap.read()
    current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

    for _ in range(end_frame - start_frame):
        status, next_frame = cap.read()

        if not status:
            break

        next_frame = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(current_frame, next_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        fields.append(flow)

        current_frame = next_frame

    return np.array(fields)


class DefaultFlowFieldExtraction(base_flow_field_extraction.BaseFlowFieldExtraction):
    def aggregate_fields(self, flow: np.ndarray) -> np.ndarray:
        # magnitude max
        mag = np.linalg.norm(flow, axis=-1)
        idx = np.argmax(mag, axis=0)
        return np.take_along_axis(flow, idx[None, ..., None], axis=0)[0]

    def execute(self, cap: cv2.VideoCapture, interval_start, interval_end, interpolation_frames) -> np.ndarray:
        flow_fields = []
        interval_frames = np.round(np.linspace(interval_start, interval_end, interpolation_frames)).astype(int)
        for start_frame, end_frame in tqdm.tqdm(
                zip(interval_frames[:-1], interval_frames[1:]), total=interpolation_frames - 1
        ):
            if start_frame == end_frame:
                end_frame += 1

            flow = get_flow_fields(cap, start_frame, end_frame)

            if len(flow) == 0:
                continue

            flow_fields.append(self.aggregate_fields(flow))

        return np.array(flow_fields)
