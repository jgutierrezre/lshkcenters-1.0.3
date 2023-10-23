from .MeasureManager import MeasureManager

from typing import Union

import numpy as np
import os
import json


class BaseMeasure:
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n: Union[None, int] = None,
        d: Union[None, int] = None,
        D: Union[None, np.ndarray] = None,
    ) -> None:
        self._check_args(X, y)

        self.X = X
        self.y = y

        if n is None:
            self.n = self.X.shape[0]
        else:
            self.n = n

        if d is None:
            self.d = self.X.shape[1]
        else:
            self.d = d

        if D is None:
            self.D = np.apply_along_axis(lambda x: len(np.unique(x)), 0, self.X)
        else:
            self.D = D

    def _check_args(self, X: np.ndarray, y: np.ndarray) -> None:
        if X.size == 0:
            raise ValueError("X must not be empty.")

        if y.size == 0:
            raise ValueError("y must not be empty.")

        if len(X) != len(y):
            raise ValueError("X and y must have the same length.")

        if y.ndim != 1:
            raise ValueError("y must have only one row.")

    def generate_dist_matrices(self) -> np.ndarray:
        raise NotImplementedError()

    def generate_similarity_matrices(self) -> list:
        dist_matrix = self.generate_dist_matrices()
        sim_matrix = []

        for di in range(self.d):
            matrix2D = []  # 2D array for 1 dimension
            for i in range(self.D[di]):
                matrix1D = []  # 1D array for 1 dimension
                for j in range(self.D[di]):
                    if dist_matrix[di][i][j] == 0:
                        matrix_tmp = 10000
                    else:
                        matrix_tmp = 1 / dist_matrix[di][i][j]

                    matrix1D.append(matrix_tmp)
                matrix2D.append(matrix1D)
            sim_matrix.append(matrix2D)

        return sim_matrix
