from .BaseMeasure import BaseMeasure

from typing import Union

import math
import numpy as np
import scipy.stats as st

from collections import defaultdict


class DILCA(BaseMeasure):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.max = np.amax(self.X, axis=0)

    def generate_dist_matrix(self) -> np.ndarray:
        print("X", self.X)

        SU_matrix = self.compute_correlation_matrix(self.X)
        print("SU_matrix", SU_matrix)

        return self.DILCA_M(self.X, SU_matrix, 0)

    def compute_probabilities(
        self, X: np.ndarray, Y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        joint_counts = st.contingency.crosstab(X, Y).count

        x_probabilities = joint_counts.sum(axis=1) / joint_counts.sum()
        y_probabilities = joint_counts.sum(axis=0) / joint_counts.sum()
        conditional_probabilities = joint_counts / joint_counts.sum()

        return (x_probabilities, y_probabilities, conditional_probabilities)

    def compute_symmetrical_uncertainty(self, X: np.ndarray, Y: np.ndarray) -> float:
        (
            x_probabilities,
            y_probabilities,
            conditional_probabilities,
        ) = self.compute_probabilities(X, Y)

        # Compute entropies
        H_X = st.entropy(x_probabilities, base=2)
        H_Y = st.entropy(y_probabilities, base=2)
        H_Y_given_X = -np.sum(
            x_probabilities
            * np.sum(
                conditional_probabilities
                * np.log2(
                    conditional_probabilities, where=(conditional_probabilities > 0)
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
                    SU_matrix[i][j] = self.compute_symmetrical_uncertainty(
                        D[:, i], D[:, j]
                    )

        return SU_matrix
    
    def compute_context_probabilities(self, context_Y: np.ndarray, Y: np.ndarray) -> list:
        d = context_Y.shape[1]
        res = []
        for i in range(d):
            # TODO (this can be optimized, I think.)
            joint_counts = st.contingency.crosstab(context_Y[i], Y).count
            conditional_probabilities = joint_counts / joint_counts.sum()
            res.append(conditional_probabilities)
        return res

    def compute_distance(
        self, i: int, j: int, Y: np.ndarray, context_Y: np.ndarray
    ) -> float:
        d = context_Y.shape[1]

        # upper = sum(sum(math.pow(conditional_probabilities[i][k]-)))
        lower = sum(context_Y)

    def compute_distance_matrix(
        self, Y: np.ndarray, context_Y: np.ndarray
    ) -> np.ndarray:
        # TODO deal with the probability calculations.

        context_probabilities = self.compute_context_probabilities(context_Y, Y)


        d = Y.shape[1]
        dist_matrix = np.zeros((d, d))
        for i in range(d):
            for j in range(d):
                if i != j:
                    self.compute_distance(i, j, Y, context_Y, context_probabilities)
        pass

    def DILCA_RR(self) -> None:
        pass

    def DILCA_M(
        self, D: np.ndarray, SU_matrix: np.ndarray, Y_i: int, sigma: float = 1.0
    ) -> np.ndarray:
        SU_vector_Y = SU_matrix[:, Y_i]
        Y = D[:, Y_i]

        mean = sigma * np.mean(SU_vector_Y)

        context_Y = D[:, SU_vector_Y >= mean]

        return self.compute_distance_matrix(Y, context_Y)

    # def procondition_matrix_YX(
    #     self, Y: np.ndarray, X: np.ndarray, Y_unique: np.ndarray, X_unique: np.ndarray
    # ) -> np.ndarray:

    #     lenData = len(Y)

    #     lenX = len(X_unique)
    #     lenY = len(Y_unique)

    #     MATRIX = np.zeros((lenY, lenX))
    #     for x in range(lenX):
    #         for y in range(lenY):
    #             count_x = 0
    #             count_y = 0
    #             for i in range(lenData):
    #                 if X[i] == x:
    #                     count_x = count_x + 1
    #                     if Y[i] == y:
    #                         count_y = count_y + 1
    #             MATRIX[y][x] = count_y / count_x if count_x > 0 else 0
    #     return MATRIX

    # def compute_entropy(self, X: np.ndarray) -> None:

    #     N = len(X) # number of instances
    #     D = len(X[0]) # number of attributes

    #     self.contextMatrix = []
    #     self.conditionProMatrix = {}
    #     self.probabilityMatrix = []

    #     # Compute cross contextMatrix
    #     for i in range(D):
    #         Y_ = X[:, i] # values of attribute i

    #         Y_unique, Y_freq = np.unique(Y_, return_counts=True) # unique values and their frequencies
    #         X_probability = [ii / len(Y_) for ii in Y_freq] # probability of each value

    #         self.probabilityMatrix.append(X_probability) # probability of each value of each attribute

    #         HY = st.entropy(X_probability, base=2) # entropy of attribute i

    #         SUY_dict = {}
    #         for j in range(D):
    #             if i != j:
    #                 X_ = X[:, j] # values of attribute j

    #                 X_unique, X_freq = np.unique(X_, return_counts=True) # unique values and their frequencies
    #                 X_property = [ii / len(X_) for ii in X_freq] # probability of each value

    #                 HX = st.entropy(X_property, base=2) # entropy of attribute j

    #                 conditionMatrix = self.procondition_matrix_YX(
    #                     Y_, X_, Y_unique, X_unique
    #                 )

    #                 self.conditionProMatrix[(i, j)] = conditionMatrix

    #                 HYX = 0
    #                 for k in range(len(X_unique)):
    #                     sum_tmp = 0
    #                     for k2 in range(len(Y_unique)):
    #                         if conditionMatrix[k2][k] != 0:
    #                             sum_tmp = sum_tmp + conditionMatrix[k2][k] * math.log2(
    #                                 conditionMatrix[k2][k]
    #                             )
    #                     HYX = HYX + sum_tmp * X_property[k]

    #                 HYX = -HYX
    #                 IGYX = HY - HYX

    #                 if HX + HY == 0:
    #                     SUYX = 0
    #                 else:
    #                     SUYX = 2 * IGYX / (HY + HX)

    #                 SUY_dict[j] = SUYX

    #         print(self.conditionProMatrix)

    #         values = list(SUY_dict.values())
    #         o = 1
    #         mean = np.mean(values)
    #         context_Y = [key for (key, value) in SUY_dict.items() if value >= o * mean]
    #         self.contextMatrix.append(context_Y)

    #     # Compute distMatrix
    #     self.distMatrix = []
    #     for d in range(D):
    #         matrix = []  # 2D array for 1 dimension
    #         for i in range(self.max[d] + 1):
    #             matrix_tmp = []
    #             # 1D array for 1 values on the attribute d
    #             for j in range(self.max[d] + 1):
    #                 dist_sum_all = 0
    #                 dist_sum = 0
    #                 dist_sum2 = 0
    #                 for d2 in self.contextMatrix[d]:
    #                     dist_sum_tmp = 0
    #                     conditionMatrix = self.conditionProMatrix[(d, d2)]
    #                     for i_k in range(self.max[d2]):
    #                         dist_sum_tmp2 = math.pow(
    #                             conditionMatrix[i][i_k] - conditionMatrix[j][i_k], 2
    #                         )
    #                         dist_sum_tmp = (
    #                             dist_sum_tmp
    #                             + dist_sum_tmp2 * self.probabilityMatrix[d2][i_k]
    #                         )
    #                     dist_sum = dist_sum + dist_sum_tmp
    #                     dist_sum2 = dist_sum2 + self.max[d2] + 1
    #                 if dist_sum2 == 0:  # toanstt
    #                     dist_sum2 = 1
    #                 dist_sum_all = math.sqrt(dist_sum / dist_sum2)
    #                 matrix_tmp.append(dist_sum_all)
    #             matrix.append(matrix_tmp)
    #         self.distMatrix.append(matrix)
