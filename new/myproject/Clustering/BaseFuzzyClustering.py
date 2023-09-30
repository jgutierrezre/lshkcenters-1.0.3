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
        self.X = X
        self.y = y
        self.n_init = n_init
        self.k = k

        self.time_lsh = None
