from .BaseFuzzyClustering import BaseFuzzyClustering
from ..Hashing.LSH import LSH

import numpy as np
import timeit


class LSHkCenters(BaseFuzzyClustering):
    def __init__(
        self, X: np.ndarray, y: np.ndarray, n_init: int = None, k: int = None
    ) -> None:
        self.X = X
        self.y = y
        self.n_init = n_init
        self.k = k

        self.time_lsh = None

    def SetupLSH(self, hbits: int = None, measure_name: str = "DILCA"):
        # start = timeit.default_timer()

        self.lsh = LSH(self.X, self.y, hbits=hbits, measure_name=measure_name)
        # self.lsh.DoHash()

        # self.time_lsh = timeit.default_timer() - start

        # self.AddVariableToPrint("Time_lsh", self.time_lsh)
        # return self.time_lsh
