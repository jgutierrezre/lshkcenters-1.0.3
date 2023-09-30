from .BaseFuzzyClustering import BaseFuzzyClustering
from ..Hashing.LSH import LSH

from typing import Union

import numpy as np
import timeit


class LSHkCenters(BaseFuzzyClustering):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def SetupLSH(self, hbits: Union[None, int] = None, measure_name: str = "DILCA") -> None:
        # start = timeit.default_timer()

        self.lsh = LSH(self.X, self.y, hbits=hbits, measure_name=measure_name)
        self.lsh.DoHash()

        # self.time_lsh = timeit.default_timer() - start

        # self.AddVariableToPrint("Time_lsh", self.time_lsh)
        # return self.time_lsh
