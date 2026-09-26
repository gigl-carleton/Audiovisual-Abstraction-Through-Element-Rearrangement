import numpy as np


class Interpolation:
    def __init__(self, texton_a: int, texton_b: int, path: np.ndarray):
        self.texton_a = texton_a
        self.texton_b = texton_b
        self.path = path
