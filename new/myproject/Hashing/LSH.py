from .BaseHashing import BaseHashing

import numpy as np
from collections import defaultdict


class LSH(BaseHashing):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def compute_hash_value(self, x, partitions: list, bit_indexes: np.ndarray) -> int:
        val = 0
        for i in range(self.hbits):
            val <<= 1
            if x[bit_indexes[i]] in partitions[bit_indexes[i]][1]:
                val += 1
        return val

    def generate_hash_table(
        self, partitions: list, bit_indexes: np.ndarray
    ) -> tuple[dict, list]:
        print(
            f"Generating LSH hash table: hbits: {self.hbits}({2**self.hbits}) k {self.k} d {self.d} n={self.n}"
        )
        hash_values = [
            self.compute_hash_value(x, partitions, bit_indexes) for x in self.X
        ]
        hash_table = defaultdict(list)
        for i in range(self.n):
            hash_table[hash_values[i]].append(i)

        return hash_table, hash_values

    def do_hash(self) -> None:
        similarity_matrices = self.measure.generate_similarity_matrices()

        # partitions, cut_values, cut_values_normal = self.generate_similarity_matrix(
        #     similarity_matrices
        # )

        # bit_indexes = np.argpartition(cut_values_normal, self.hbits)[: self.hbits]
        # hash_table, hash_values = self.generate_hash_table(partitions, bit_indexes)
        # print(hash_values)
