from ..Measures.MeasureManager import MeasureManager

import importlib
import numpy as np
import math


class BaseHashing:
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_init: int = 10,
        hbits: int = None,
        k: int = None,
        measure_name: str = "DILCA",
    ) -> None:
        self._check_args(X, y, n_init, hbits, k, measure_name)

        self.X = X
        self.y = y
        self.n_init = n_init
        self.hbits = hbits
        self.measure = self._load_measure(measure_name)

        self.n = self.X.shape[0]
        self.d = self.X.shape[1]
        self.D = [np.unique(self.X[:, i]) for i in range(self.d)]

        if not k:
            self.k = len(np.unique(y))
        else:
            self.k = k

        if not hbits:
            self.hbits = math.ceil(math.log2(self.k))
            if self.hbits >= self.d:
                self.hbits = self.d - 1
        else:
            if 2**self.hbits <= self.k:
                print(
                    f"WARNING: BAD HBITS: hbits={self.hbits} d={self.d} nbucket={2**self.hbits} k={self.k} n={self.n}"
                )
            self.hbits = hbits

    def _check_args(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_init: int,
        hbits: int,
        k: int,
        measure_name: str,
    ) -> None:
        if X.size == 0:
            raise ValueError("X must not be empty.")

        if y.size == 0:
            raise ValueError("y must not be empty.")

        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of rows.")

        if y.shape[1] != 1:
            raise ValueError("y must have only one column.")

        if n_init <= 0:
            raise ValueError("n_init must be greater than 0.")

        if hbits and hbits <= 0:
            raise ValueError("hbits must be greater than 0.")

        if k and k <= 0:
            raise ValueError("k must be greater than 0.")

        if measure_name not in MeasureManager.MEASURE_LIST:
            raise ValueError(f"Measure '{measure_name}' not found!")

    def _load_measure(self, measure_name: str) -> None:
        try:
            module = importlib.import_module(f"..Measures.{measure_name}", __package__)
            measure_class = getattr(module, measure_name)
            return measure_class(self.X, self.y)
        except ImportError:
            raise ValueError(f"Measure '{measure_name}' not found!")
        except AttributeError:
            raise ValueError(
                f"Class '{measure_name}' not found in module '{measure_name}'!"
            )

    def DoHash(self) -> None:
        raise NotImplementedError
