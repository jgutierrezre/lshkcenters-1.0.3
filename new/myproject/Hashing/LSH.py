from .BaseHashing import BaseHashing

import numpy as np


class LSH(BaseHashing):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def DoHash(self):
        self.measure.GeneratesimMatrix()
        self.GenerateSimilarityMatrix(self.measure.simMatrix)
        self.bit_indexes = np.argpartition(self.cut_values_normal, self.hbits)[
            : self.hbits
        ]
        self.GenerateHashTable()
        print(self.hash_values)
        return -1
