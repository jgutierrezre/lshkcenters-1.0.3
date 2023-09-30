from .MeasureManager import MeasureManager

import numpy as np
import os
import json


class BaseMeasure:
    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = X
        self.y = y
        
        self.d = len(self.X[0])
        self.D = [len(np.unique(self.X[:, i])) for i in range(self.d)]
        
        self.distMatrix = []
        self.simMatrix = []

    def GenerateSimilarityMatrix(self) -> None:
        for di in range(self.d):
            matrix2D = []  # 2D array for 1 dimension
            for i in range(self.D[di]):
                matrix1D = []  # 1D array for 1 dimension
                for j in range(self.D[di]):
                    # matrix_tmp = 1-self.distMatrix[di][i][j]
                    if self.distMatrix[di][i][j] == 0:
                        matrix_tmp = 10000
                    else:
                        matrix_tmp = 1 / self.distMatrix[di][i][j]

                    matrix1D.append(matrix_tmp)
                matrix2D.append(matrix1D)
            self.simMatrix.append(matrix2D)
