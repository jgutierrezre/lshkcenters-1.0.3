from ..Measures.MeasureManager import MeasureManager
from ..Measures.BaseMeasure import BaseMeasure

from typing import Union

import importlib
import numpy as np


class BaseHashing:
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n: Union[None, int] = None,
        d: Union[None, int] = None,
        D: Union[None, np.ndarray] = None,
        n_init: int = 10,
        hbits: Union[None, int] = None,
        k: Union[None, int] = None,
        measure_name: str = "DILCA",
    ) -> None:
        self._check_args(X, y, n_init, hbits, k, measure_name)

        self.X = X
        self.y = y
        self.n_init = n_init

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

        if k is None:
            self.k = len(np.unique(y))
        else:
            self.k = k

        if hbits is None:
            self.hbits = int(np.ceil(np.log2(self.k + 1)))
            if self.hbits >= self.d:
                self.hbits = self.d - 1
        else:
            if 2**self.hbits <= self.k:
                print(
                    f"WARNING: BAD HBITS: hbits={self.hbits} d={self.d} nbucket={2**self.hbits} k={self.k} n={self.n}"
                )
            self.hbits = hbits

        self.measure = self._load_measure(measure_name)

        self._hash_table: Union[dict, None] = None
        self._hash_values: Union[list, None] = None

        self._init_hash()

    def _check_args(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_init: int,
        hbits: Union[None, int],
        k: Union[None, int],
        measure_name: str,
    ) -> None:
        if X.size == 0:
            raise ValueError("X must not be empty.")

        if y.size == 0:
            raise ValueError("y must not be empty.")

        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of rows.")

        if y.ndim != 1:
            raise ValueError("y must have only one row.")

        if n_init <= 0:
            raise ValueError("n_init must be greater than 0.")

        if hbits is not None and hbits <= 0:
            raise ValueError("hbits must be greater than 0.")

        if k is not None and k <= 0:
            raise ValueError("k must be greater than 0.")

        if measure_name not in MeasureManager.MEASURE_LIST:
            raise ValueError(f"Measure '{measure_name}' not found!")

    def _load_measure(self, measure_name: str) -> BaseMeasure:
        try:
            module = importlib.import_module(f"..Measures.{measure_name}", __package__)
            measure_class = getattr(module, measure_name)
            return measure_class(X=self.X, y=self.y)
        except ImportError:
            raise ValueError(f"Measure '{measure_name}' not found!")
        except AttributeError:
            raise ValueError(
                f"Class '{measure_name}' not found in module '{measure_name}'!"
            )

    def _init_hash(self) -> None:
        raise NotImplementedError()

    def hamming_distance(self, x, y) -> float:
        ans = 0
        for i in range(31, -1, -1):
            b1 = x >> i & 1
            b2 = y >> i & 1
            ans += not (b1 == b2)
        return ans

    def get_hash_table(self) -> dict:
        if self._hash_table is None:
            raise ValueError("Hash table not generated yet!")
        return self._hash_table

    def get_hash_values(self) -> list:
        if self._hash_values is None:
            raise ValueError("Hash values not generated yet!")
        return self._hash_values
