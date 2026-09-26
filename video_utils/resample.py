import logging
import os.path
import typing

import cv2
import tqdm


class Resample:
    def __init__(self, framerate: float, resolution: typing.Tuple[int, int]):
        self.framerate = framerate
        self.resolution = resolution

    def execute(
            self, cap: typing.Union[cv2.VideoCapture, str], tmp_name: str, release_cap: bool = True
    ) -> cv2.VideoCapture:

        logging.info("Resampling video to {}x{} @ {}fps...".format(*self.resolution, self.framerate))

        directory = os.path.split(tmp_name)[:-1]
        if len(directory) != 0:
            os.makedirs(os.path.join(*directory), exist_ok=True)

        if isinstance(cap, str):
            cap = cv2.VideoCapture(cap)
            release_cap = True

        old_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / old_fps

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(tmp_name, fourcc, self.framerate, self.resolution)

        progress = tqdm.trange(frame_count)

        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.resize(frame, self.resolution))
            progress.update()

        progress.close()

        new_frame_count = int(duration * self.framerate)

        for i in tqdm.trange(new_frame_count):
            t = i / self.framerate
            old_index = t * old_fps
            idx = int(round(old_index))
            idx = min(idx, len(frames) - 1)
            out.write(frames[idx])

        out.release()

        if release_cap:
            cap.release()

        logging.info("Resample complete")
        return cv2.VideoCapture(tmp_name)
