from .Defaults import Defaults

from typing import Union

import numpy as np


class BaseFuzzyClustering:
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_init: Union[None, int] = None,
        n_iter: Union[None, int] = None,
        k: Union[None, int] = None,
        alpha: float = 1.1,
    ) -> None:
        self._check_args(X, y, n_init, k)

        self.X = X
        self.y = y

        self.n = X.shape[0]
        self.d = X.shape[1]
        self.D = [len(np.unique(X[:, i])) for i in range(self.d)]
        
        self.alpha = alpha
        self.power = float(1 / (self.alpha - 1))
        
        if n_iter is None:
            self.n_iter = Defaults.n_iter
        else:
            self.n_iter = n_iter

        if n_init is None:
            self.n_init = Defaults.n_init
        else:
            self.n_init = n_init

        if k is None:
            self.k = len(np.unique(y))
        else:
            self.k = k

    def _check_args(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_init: Union[None, int],
        k: Union[None, int],
    ) -> None:
        if X.size == 0:
            raise ValueError("X must not be empty.")

        if y.size == 0:
            raise ValueError("y must not be empty.")

        if y.ndim != 1:
            raise ValueError("y must have only one row.")

        if n_init is not None and n_init <= 0:
            raise ValueError("n_init must be greater than 0.")

        if k is not None and k <= 0:
            raise ValueError("k must be greater than 0.")

    def overlap_k_representatives(self, point, representative) -> float:
        sum = 0
        for i in range(self.d):
            for vj in range(self.D[i]):
                if point[i] != vj:
                    sum += representative[i][vj]
        return sum

    def distance_representative_to_representative(self, r1, r2):
        sum = 0
        for k in range(self.d):
            sum += np.linalg.norm(np.array(r1[k]) - np.array(r2[k]))
        return sum

    def minus_x_to_v_rep(self, X, v) -> np.ndarray:
        d = X.shape[1]
        m = np.ones((X.shape[0], d))
        for i, xi in enumerate(X):
            for di in range(d):
                m[i][di] = 1 - v[di][xi[di]]
        return m

    def squared_distances_rep(self, x, v) -> np.ndarray:
        m = np.zeros((self.n, self.k))
        for i in range(self.n):
            for j in range(self.k):
                m[i][j] = self.overlap_k_representatives(x[i], v[j]) ** 2
        return m

    def squared_distances_v_rep(self, v1, v2) -> np.ndarray:
        m = np.zeros((self.k, self.k))
        for i in range(self.k):
            for j in range(self.k):
                m[i][j] = self.distance_representative_to_representative(v1[i], v2[j]) ** 2
        return m
