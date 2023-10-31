from .BaseHashing import BaseHashing

from collections import defaultdict
import networkx as nx
import numpy as np


class LSH(BaseHashing):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def _generate_similarity_graph(self, similarity_matrices: list) -> tuple[list, np.ndarray, np.ndarray]:
        """
        Generate similarity graph using provided similarity matrices. 
        Uses Stoer-Wagner algorithm to find the maximum cuts on the graphs.
        """
        
        partitions = []
        
        cut_values = np.full((self.d), np.inf, dtype=float)
        cut_values_normal = np.full((self.d), np.inf, dtype=float)

        # Iterate through each matrix in simMatrix using enumerate
        for i, matrix in enumerate(similarity_matrices):
            # Create graph G
            G = nx.from_numpy_array(np.triu(matrix, 1))
            
            # If there's more than 1 node, compute the cut
            if len(G.nodes) > 1:
                ut_value, partition = nx.stoer_wagner(G)
                partitions.append(partition)
                cut_values[i] = ut_value
                cut_values_normal[i] = ut_value / self.D[i]
            else:
                partitions.append([[0], [0]])

        return partitions, cut_values, cut_values_normal

    def _generate_hash_table(
        self, partitions: list, bit_indexes: np.ndarray
    ) -> tuple[dict, list]:
        print(
            f"Generating LSH hash table: hbits: {self.hbits}({2**self.hbits}) k {self.k} d {self.d} n={self.n}"
        )
        hash_values = [
            self._compute_hash_value(x, partitions, bit_indexes) for x in self.X
        ]
        hash_table = defaultdict(list)
        for i in range(self.n):
            hash_table[hash_values[i]].append(i)

        return hash_table, hash_values

    def _compute_hash_value(self, x, partitions: list, bit_indexes: np.ndarray) -> int:
        val = 0
        for i in range(self.hbits):
            val <<= 1
            if x[bit_indexes[i]] in partitions[bit_indexes[i]][1]:
                val += 1
        return val

    def _init_hash(self) -> None:
        similarity_matrices = self.measure.get_similarity_matrices()
        print("similarity_matrices\n", similarity_matrices)

        partitions, cut_values, cut_values_normal = self._generate_similarity_graph(
            similarity_matrices
        )
        print("partitions\n", partitions)
        print("cut_values\n", cut_values)
        print("cut_values_normal\n", cut_values_normal)

        # bit_indexes = np.argpartition(cut_values_normal, self.hbits)[: self.hbits]
        # hash_table, hash_values = self._generate_hash_table(partitions, bit_indexes)
        # print(hash_values)
