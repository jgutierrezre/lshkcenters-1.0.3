from .BaseMeasure import BaseMeasure

import numpy as np
import scipy.stats as st


class DILCA(BaseMeasure):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.max = np.amax(self.X, axis=0)

        self._init_probabilities(self.X)

    def _init_probabilities(self, D: np.ndarray) -> None:
        self.x_probabilities = {}
        self.y_probabilities = {}
        self.conditional_probabilities = {}

        d = D.shape[1]

        for i in range(d):
            for j in range(d):
                if i != j:
                    (
                        x_probabilities,
                        y_probabilities,
                        conditional_probabilities,
                    ) = self.compute_probabilities(D[:, i], D[:, j])

                    self.x_probabilities[(i, j)] = x_probabilities
                    self.y_probabilities[(i, j)] = y_probabilities
                    self.conditional_probabilities[(i, j)] = conditional_probabilities

    def generate_dist_matrices(self) -> list:
        print("X\n", self.X)

        SU_matrix = self.compute_correlation_matrix(self.X)
        print("SU_matrix\n", SU_matrix)

        dist_matrices = self.DILCA_M(self.X, SU_matrix)
        print("dist_matrices\n", dist_matrices)

        return dist_matrices

    def compute_probabilities(
        self, X: np.ndarray, Y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        joint_counts = st.contingency.crosstab(X, Y).count

        x_probabilities = joint_counts.sum(axis=1) / joint_counts.sum()
        y_probabilities = joint_counts.sum(axis=0) / joint_counts.sum()
        conditional_probabilities = joint_counts / joint_counts.sum()

        return (x_probabilities, y_probabilities, conditional_probabilities)

    def compute_symmetrical_uncertainty(self, X_i: int, Y_i: int) -> float:
        # Compute entropies
        H_X = st.entropy(self.x_probabilities[(X_i, Y_i)], base=2)
        H_Y = st.entropy(self.y_probabilities[(X_i, Y_i)], base=2)
        H_Y_given_X = -np.sum(
            self.x_probabilities[(X_i, Y_i)]
            * np.sum(
                self.conditional_probabilities[(X_i, Y_i)]
                * np.log2(
                    self.conditional_probabilities[(X_i, Y_i)],
                    where=(self.conditional_probabilities[(X_i, Y_i)] > 0),
                ),
                axis=1,
            )
        )

        # Compute Information Gain
        IG_Y_given_X = H_Y - H_Y_given_X

        # Compute Symmetric Uncertainty
        SU_Y_X = 2 * IG_Y_given_X / (H_X + H_Y)
        return SU_Y_X

    def compute_correlation_matrix(self, D: np.ndarray) -> np.ndarray:
        d = D.shape[1]
        SU_matrix = np.zeros((d, d))

        for i in range(d):
            for j in range(d):
                if i != j:
                    SU_matrix[i][j] = self.compute_symmetrical_uncertainty(i, j)

        return SU_matrix

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
        d = Y.shape[0]
        dist_matrix = np.zeros((d, d))
        for i in range(d):
            for j in range(d):
                if i != j:
                    dist_matrix[i][j] = self.compute_distance(
                        Y[i], Y[j], Y_i, context_Y_i
                    )
        return dist_matrix

    def DILCA_RR(self) -> None:
        raise NotImplementedError()

    def DILCA_M_helper(
        self, D: np.ndarray, SU_matrix: np.ndarray, Y_i: int, sigma: float
    ) -> np.ndarray:
        SU_vector_Y = SU_matrix[:, Y_i]
        Y = D[:, Y_i]

        mean = sigma * np.mean(SU_vector_Y)

        context_Y_i = SU_vector_Y >= mean

        return self.compute_distance_matrix(Y, Y_i, context_Y_i)

    def DILCA_M(self, D: np.ndarray, SU_matrix: np.ndarray, sigma: float = 1.0) -> list:
        # calculate DILCA_M for each attribute
        dist_matrices = []
        for i in range(D.shape[1]):
            dist_matrices.append(self.DILCA_M_helper(D, SU_matrix, i, sigma))
        return dist_matrices
