from .MeasureManager import MeasureManager

import numpy as np
import os
import json


class BaseMeasure:
    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = X
        self.y = y

    def LoaddistMatrixAuto(self):
        if (
            MeasureManager.IS_LOAD_AUTO == False
            or MeasureManager.CURRENT_DATASET == "None"
        ):
            print(
                f"SKIP LOADING distMatrix because: IS_LOAD_AUTO={MeasureManager.IS_LOAD_AUTO} CURRENT_DATASET={MeasureManager.CURRENT_DATASET}"
            )
            return False
        path = f"saved_dist_matrices/json/{self.name}_{MeasureManager.CURRENT_DATASET}.json"
        if os.path.isfile(path):
            with open(path, "r") as fp:
                self.distMatrix = json.load(fp)
        else:
            print(f"CANNOT OPEN FILE: {path}")
            return False

        print(f"Loaded dist matrix: {path}")
        return True

    def GeneratesimMatrix():
        raise NotImplementedError
