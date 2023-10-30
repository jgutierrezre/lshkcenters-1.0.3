from .BaseMeasure import BaseMeasure

from typing import Dict, Tuple

import numpy as np
import scipy.stats as st


class DILCA(BaseMeasure):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # Dictionaries to store probabilities.
        self.x_probabilities: Dict[Tuple[int, int], np.ndarray]  = {}
        self.y_probabilities: Dict[Tuple[int, int], np.ndarray]  = {}
        self.conditional_probabilities: Dict[Tuple[int, int], np.ndarray]  = {}

        self._init_probabilities()

    def _init_probabilities(self) -> None:
        d = self.X.shape[1]

        for i in range(d):
            # start from i+1 to get upper triangle without diagonal
            for j in range(i + 1, d):
                (
                    x_probabilities,
                    y_probabilities,
                    cond_probabilities,
                ) = self._compute_probabilities(i, j)

                # Store probabilities for (i, j) pair
                self.x_probabilities[(i, j)] = x_probabilities
                self.y_probabilities[(i, j)] = y_probabilities
                self.conditional_probabilities[(i, j)] = cond_probabilities

                # Store probabilities for (j, i) pair due to symmetry
                self.x_probabilities[(j, i)] = y_probabilities
                self.y_probabilities[(j, i)] = x_probabilities
                # transpose for symmetric conditional probabilities
                self.conditional_probabilities[(j, i)] = cond_probabilities.T

    def _compute_probabilities(
        self, i: int, j: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        X, Y = self.X[:, i], self.X[:, j]
        joint_counts = st.contingency.crosstab(X, Y).count

        x_probabilities = joint_counts.sum(axis=1) / joint_counts.sum()
        y_probabilities = joint_counts.sum(axis=0) / joint_counts.sum()
        conditional_probabilities = joint_counts / joint_counts.sum()

        return (x_probabilities, y_probabilities, conditional_probabilities)
    
    def _compute_correlation_matrix(self) -> np.ndarray:
        d = self.X.shape[1]
        SU_matrix = np.ones((d, d))

        for i in range(d):
            for j in range(d):
                if i != j:
                    SU_matrix[i, j] = self._compute_symmetrical_uncertainty(i, j)

        return SU_matrix

    def _compute_symmetrical_uncertainty(self, X_i: int, Y_i: int) -> float:
        # Probabilities
        p_x = self.x_probabilities[(X_i, Y_i)]
        p_y = self.y_probabilities[(X_i, Y_i)]
        p_xy = self.conditional_probabilities[(X_i, Y_i)]
    
        # Compute Entropies
        H_X = st.entropy(p_x, base=2)
        H_Y = st.entropy(p_y, base=2)
        H_Y_given_X = -np.sum(p_x * np.nansum(p_xy * np.log2(p_xy), axis=1))

        # Compute Information Gain
        IG_Y_given_X = H_Y - H_Y_given_X

        # Compute Symmetric Uncertainty
        SU_Y_X = 2 * IG_Y_given_X / (H_X + H_Y)
        
        return SU_Y_X

    def compute_distance(
        self, i: int, j: int, Y_i: int, context_Y_i: np.ndarray
    ) -> float:
        upper = sum(
            np.sum(
                np.square(
                    self.conditional_probabilities[(Y_i, X_i)][i]
                    - self.conditional_probabilities[(Y_i, X_i)][j]
                )
            )
            for X_i in range(context_Y_i.shape[0])
            if context_Y_i[X_i]
        )

        distance = np.sqrt(upper / np.sum(context_Y_i))

        return distance

    def compute_distance_matrix(
        self, Y: np.ndarray, Y_i: int, context_Y_i: np.ndarray
    ) -> np.ndarray:
        d = np.max(Y) + 1
        dist_matrix = np.zeros((d, d))
        for i in range(d):
            for j in range(d):
                if i != j:
                    dist_matrix[i][j] = self.compute_distance(i, j, Y_i, context_Y_i)
        return dist_matrix

    def DILCA_RR(self) -> None:
        raise NotImplementedError()

    def DILCA_M_helper(
        self, D: np.ndarray, SU_matrix: np.ndarray, Y_i: int, sigma: float
    ) -> np.ndarray:
        SU_vector_Y = SU_matrix[:, Y_i]
        Y = D[:, Y_i]

        mean = sigma * np.nanmean(SU_vector_Y)

        context_Y_i = SU_vector_Y >= mean

        return self.compute_distance_matrix(Y, Y_i, context_Y_i)

    def DILCA_M(self, D: np.ndarray, SU_matrix: np.ndarray, sigma: float = 1.0) -> list:
        # calculate DILCA_M for each attribute
        dist_matrices = []
        for i in range(D.shape[1]):
            dist_matrices.append(self.DILCA_M_helper(D, SU_matrix, i, sigma))
        return dist_matrices
    
    def generate_dist_matrices(self) -> list:
        print("X\n", self.X)

        SU_matrix = self._compute_correlation_matrix()
        print("SU_matrix\n", SU_matrix)

        dist_matrices = self.DILCA_M(self.X, SU_matrix)
        print("dist_matrices\n", dist_matrices)

        return dist_matrices
