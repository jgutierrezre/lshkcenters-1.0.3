from .MeasureManager import MeasureManager

from typing import Union, List

import numpy as np


class BaseMeasure:
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> None:
        """
        Initializes the BaseMeasure object.

        Args:
            X (np.ndarray): Data matrix.
            y (np.ndarray): Labels.
        """
        self._check_args(X, y)

        self._X = X
        self._y = y

        self._n = self._X.shape[0]
        self._d = self._X.shape[1]
        self._D = np.apply_along_axis(lambda x: len(np.unique(x)), 0, self._X)

        # List to store distance matrices.
        self._distance_matrices: Union[List[np.ndarray], None] = None
        self._similarity_matrices: Union[List[np.ndarray], None] = None

        # Initialize similarity matrices.
        self._init_distance_matrices()
        self._init_similarity_matrices()

    def _check_args(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Check if the arguments are valid.

        Raises:
            ValueError: If X is empty.
            ValueError: If y is empty.
            ValueError: If X and y have different lengths.
            ValueError: If y has more than one row.
        """
        if X.size == 0:
            raise ValueError("X must not be empty.")

        if y.size == 0:
            raise ValueError("y must not be empty.")

        if len(X) != len(y):
            raise ValueError("X and y must have the same length.")

        if y.ndim != 1:
            raise ValueError("y must have only one row.")

    def _init_distance_matrices(self) -> None:
        """
        Abstract method to initialize the distance matrices.

        Raises:
            NotImplementedError: If the method is not implemented in the subclass.
        """
        raise NotImplementedError()

    def _init_similarity_matrices(self) -> None:
        """
        Initialize similarity matrices for all attributes based on distance matrices.

        Raises:
            ValueError: If distance matrices have not been generated yet.
        """
        if self._distance_matrices is None:
            raise ValueError("Distance matrices have not been generated yet.")

        self._similarity_matrices = []
        for dist_matrix in self._distance_matrices:
            # Create an array filled with large values
            sim_matrix = np.full_like(dist_matrix, np.inf, dtype=float)

            # Get a mask of non-zero distances
            nonzero_mask = dist_matrix != 0

            # Update the non-zero values with the inverse of the distance
            sim_matrix[nonzero_mask] = 1 / dist_matrix[nonzero_mask]

            self._similarity_matrices.append(sim_matrix)

    def get_similarity_matrices(self) -> List[np.ndarray]:
        """
        Returns the similarity matrices.

        Returns:
            List[np.ndarray]: List of similarity matrices.

        Raises:
            ValueError: If similarity matrices have not been generated yet.
        """
        if self._similarity_matrices is None:
            raise ValueError("Similarity matrices have not been generated yet.")
        return self._similarity_matrices
