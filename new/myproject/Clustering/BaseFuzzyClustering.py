from .Defaults import Defaults

from typing import Union

import numpy as np


class BaseFuzzyClustering:
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_init: Union[None, int] = None,
        k: Union[None, int] = None,
    ) -> None:
        self._check_args(X, y, n_init, k)

        self.X = X
        self.y = y

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
