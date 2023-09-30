from .BaseMeasure import BaseMeasure

import numpy as np

class DILCA(BaseMeasure):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        super().__init__(X, y)