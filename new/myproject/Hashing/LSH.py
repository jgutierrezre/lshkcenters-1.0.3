from .BaseHashing import BaseHashing

import numpy as np
from collections import defaultdict


class LSH(BaseHashing):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def compute_hash_value(self, x):
        val = 0
        for i in range(self.hbits):
            partitions = self.partitions[self.bit_indexes[i]]
            val <<= 1
            if x[self.bit_indexes[i]] in partitions[1]:
                val += 1
        return val

    def generate_hash_table(self) -> None:
        print(
            f"Generating LSH hash table: hbits: {self.hbits}({2**self.hbits}) k {self.k} d {self.d} n={self.n}"
        )
        self.hash_values = [self.compute_hash_value(x) for x in self.X]
        self.hashTable = defaultdict(list)
        for i in range(self.n):
            self.hashTable[self.hash_values[i]].append(i)

    def do_hash(self) -> None:
        self.measure.generate_similarity_matrix()
        self.generate_similarity_matrix(self.measure.simMatrix)
        self.bit_indexes = np.argpartition(self.cut_values_normal, self.hbits)[
            : self.hbits
        ]
        self.generate_hash_table()
        print(self.hash_values)
