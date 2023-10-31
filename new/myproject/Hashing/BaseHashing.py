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
        n_init: int = 10,
        hbits: Union[None, int] = None,
        k: Union[None, int] = None,
        measure_name: str = "DILCA",
    ) -> None:
        """
        Initializes the BaseHashing object.

        Args:
            X (np.ndarray): Data matrix.
            y (np.ndarray): Labels.
            n_init (int): Number of iterations.
            hbits (Union[None, int]): Number of hash bits.
            k (Union[None, int]): Number of clusters.
            measure_name (str): Name of the measure.
        """
        self._check_args(X, y, n_init, hbits, k, measure_name)

        self.X = X
        self.y = y
        self.n_init = n_init

        self.n = self.X.shape[0]
        self.d = self.X.shape[1]
        self.D = np.apply_along_axis(lambda x: len(np.unique(x)), 0, self.X)

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
        """
        Check if the arguments are valid.

        Raises:
            ValueError: If X is empty.
            ValueError: If y is empty.
            ValueError: If X and y have different number of rows.
            ValueError: If y has more than one row.
            ValueError: If n_init is less than or equal to 0.
            ValueError: If hbits is less than or equal to 0.
            ValueError: If k is less than or equal to 0.
            ValueError: If the measure is not found.
        """
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
        """
        Load the measure class from the Measures package.

        Args:
            measure_name (str): Name of the measure.

        Returns:
            BaseMeasure: Measure class.

        Raises:
            ValueError: If the measure is not found.
            ValueError: If the class is not found in the module.
        """
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
        """
        Abstract method to initialize the hash table.

        Raises:
            NotImplementedError: If the method is not implemented in the subclass.
        """
        raise NotImplementedError()

    def hamming_distance(self, x: int, y: int) -> int:
        """
        Calculate the Hamming distance between two integers.

        Args:
            x (int): First integer.
            y (int): Second integer.

        Returns:
            int: Hamming distance between x and y.
        """
        xor_result = x ^ y
        distance = 0

        while xor_result:
            distance += xor_result & 1
            xor_result >>= 1

        return distance

    def get_hash_table(self) -> dict:
        """
        Returns the hash table.

        Returns:
            dict: Hash table.

        Raises:
            ValueError: If the hash table is not generated yet.
        """
        if self._hash_table is None:
            raise ValueError("Hash table not generated yet!")
        return self._hash_table
