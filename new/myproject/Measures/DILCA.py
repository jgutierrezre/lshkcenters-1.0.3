from .BaseMeasure import BaseMeasure

import numpy as np
import scipy.stats as st


class DILCA(BaseMeasure):
    def __init__(self, *args, **kwargs) -> None:
        """
        Initializes the DILCA object and computes initial probabilities.
        """
        super().__init__(*args, **kwargs)

    # ==========================
    # Distance Matrices
    # ==========================

    def _init_distance_matrices(self) -> None:
        """
        Initialize distance matrices for all attributes based on SU values.
        """
        print("X\n", self._X)

        # Dictionaries to store probabilities.
        self._x_probabilities: dict[tuple[int, int], np.ndarray] = {}
        self._y_probabilities: dict[tuple[int, int], np.ndarray] = {}
        self._conditional_probabilities: dict[tuple[int, int], np.ndarray] = {}
        self._init_probabilities()

        SU_matrix = self._compute_correlation_matrix()
        print("SU_matrix\n", SU_matrix)

        distance_matrices = self._DILCA_M(SU_matrix)
        print("distance_matrices\n", distance_matrices)

        self._distance_matrices = distance_matrices

    def _compute_correlation_matrix(self) -> np.ndarray:
        """
        Compute the symmetrical uncertainty matrix for all attributes.

        Returns:
            np.ndarray: Symmetrical Uncertainty matrix.
        """
        SU_matrix = np.ones((self._d, self._d))

        for i in range(self._d):
            for j in range(self._d):
                if i != j:
                    SU_matrix[i, j] = self._compute_symmetrical_uncertainty(i, j)

        return SU_matrix

    def _compute_symmetrical_uncertainty(self, X_i: int, Y_i: int) -> float:
        """
        Calculate symmetrical uncertainty between two attributes.

        Args:
            X_i (int): Index of the first attribute.
            Y_i (int): Index of the second attribute.

        Returns:
            float: Symmetrical Uncertainty value.
        """
        # Probabilities
        p_x = self._x_probabilities[(X_i, Y_i)]
        p_y = self._y_probabilities[(X_i, Y_i)]
        p_xy = self._conditional_probabilities[(X_i, Y_i)]

        # Compute Entropies
        H_X = st.entropy(p_x, base=2)
        H_Y = st.entropy(p_y, base=2)

        H_Y_given_X = -np.sum(
            p_x * np.nansum(p_xy * np.log2(p_xy, where=p_xy > 0), axis=1)
        )

        # Compute Information Gain
        IG_Y_given_X = H_Y - H_Y_given_X

        # Compute Symmetric Uncertainty
        if H_X + H_Y == 0:
            SU_Y_X = 0
        else:
            SU_Y_X = 2 * IG_Y_given_X / (H_X + H_Y)

        return SU_Y_X

    def _DILCA_RR(self) -> None:
        """
        Placeholder for DILCA_RR implementation.
        """
        raise NotImplementedError()

    def _DILCA_M(self, SU_matrix: np.ndarray, sigma: float = 1.0) -> list[np.ndarray]:
        """
        Compute DILCA_M distance matrices for all attributes.

        Args:
            SU_matrix (np.ndarray): Symmetrical Uncertainty matrix.
            sigma (float): Sigma multiplier for threshold computation.

        Returns:
            list: List of distance matrices.
        """
        # calculate DILCA_M for each attribute
        distance_matrices = [
            self._DILCA_M_helper(SU_matrix[:, i], i, sigma) for i in range(self._d)
        ]

        return distance_matrices

    def _DILCA_M_helper(
        self, SU_vector_Y: np.ndarray, Y_i: int, sigma: float
    ) -> np.ndarray:
        """
        Helper function for DILCA_M to compute the distance matrix for an attribute.

        Args:
            SU_vector_Y (np.ndarray): Symmetrical Uncertainty values for the target attribute.
            Y_i (int): Index of the target attribute.
            sigma (float): Sigma multiplier for threshold computation.

        Returns:
            np.ndarray: Distance matrix for the attribute.
        """
        # Mask to nullify the Y_ith element
        mask = np.ones_like(SU_vector_Y, dtype=bool)
        mask[Y_i] = False

        # Calculate mean without Y_ith value
        SU_mean_Y = sigma * np.mean(SU_vector_Y[mask])

        # Get indices that are >= mean (excluding Y_i itself)
        context_Y_i = np.where((SU_vector_Y >= SU_mean_Y) & mask)[0]

        return self._compute_distance_matrix(Y_i, context_Y_i)

    def _compute_distance_matrix(self, Y_i: int, context_Y_i: np.ndarray) -> np.ndarray:
        """
        Compute a distance matrix for all values of an attribute based on a context.

        Args:
            Y_i (int): Index of the target attribute.
            context_Y_i (np.ndarray): Indices of the context attributes. This context is such that
            the attributes belonging to this set have a high value of Symmetrical Uncertainty with
            respect to Y_i.

        Returns:
            np.ndarray: Distance matrix.
        """
        # number of unique values in Y_i
        d = self._D[Y_i]

        # initialize distance matrix
        dist_matrix = np.zeros((d, d))

        # compute distance matrix
        for i in range(d):
            # start from i+1 to get upper triangle without diagonal
            for j in range(i + 1, d):
                dist_matrix[i][j] = self._compute_distance(i, j, Y_i, context_Y_i)
                # due to symmetry
                dist_matrix[j][i] = dist_matrix[i][j]
        return dist_matrix

    def _compute_distance(
        self, i: int, j: int, Y_i: int, context_Y_i: np.ndarray
    ) -> float:
        """
        Compute distance between two values of an attribute based on a context.

        Args:
            i (int): First value.
            j (int): Second value.
            Y_i (int): Index of the target attribute.
            context_Y_i (np.ndarray): Indices of the context attributes.

        Returns:
            float: Computed distance.
        """
        diffs = np.array(
            [
                self._conditional_probabilities[(Y_i, X_i)][i]
                - self._conditional_probabilities[(Y_i, X_i)][j]
                for X_i in context_Y_i
            ]
        )
        upper = np.sum(np.square(diffs))

        lower = context_Y_i.shape[0]

        distance = np.sqrt(upper / lower)

        return distance

    # ==========================
    # Probabilities Computation
    # ==========================

    def _init_probabilities(self) -> None:
        """
        Initialize joint and conditional probabilities for attribute pairs.
        """
        for i in range(self._d):
            # start from i+1 to get upper triangle without diagonal
            for j in range(i + 1, self._d):
                (
                    x_probabilities,
                    y_probabilities,
                    cond_probabilities,
                ) = self._compute_probabilities(i, j)

                # Store probabilities for (i, j) pair
                self._x_probabilities[(i, j)] = x_probabilities
                self._y_probabilities[(i, j)] = y_probabilities
                self._conditional_probabilities[(i, j)] = cond_probabilities

                # Store probabilities for (j, i) pair due to symmetry
                self._x_probabilities[(j, i)] = y_probabilities
                self._y_probabilities[(j, i)] = x_probabilities
                # transpose for symmetric conditional probabilities
                self._conditional_probabilities[(j, i)] = cond_probabilities.T

    def _compute_probabilities(
        self, i: int, j: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute joint and conditional probabilities between two attributes.

        Args:
            i (int): Index of the first attribute.
            j (int): Index of the second attribute.

        Returns:
            tuple: Probabilities of X, Y, and their conditional probabilities.
        """
        X, Y = self._X[:, i], self._X[:, j]
        joint_counts = st.contingency.crosstab(X, Y).count

        x_probabilities = joint_counts.sum(axis=1) / joint_counts.sum()
        y_probabilities = joint_counts.sum(axis=0) / joint_counts.sum()
        conditional_probabilities = joint_counts / joint_counts.sum()

        return (x_probabilities, y_probabilities, conditional_probabilities)
