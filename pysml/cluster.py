import numpy as np
from dataclasses import dataclass
from typing import Callable


@dataclass
class Estimator:
    """
    Generic cluster estimator

    Members
    -------

    centers : np.ndarray of shape (n_clusters, n_features)
        Coordinates of cluster centers.

    labels : np.ndarray of shape (len(X,))
        Labels for each point.

    inertia : float
        Sum of squared distances of samples to their closest cluster center.
    """
    centers: np.ndarray
    labels: np.ndarray
    inertia: float

    _predict_func: Callable
    _score_func: Callable

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Returns predicted labels for X"""
        return self._predict_func(X, self.centers, self.labels)

    def score(self, X: np.ndarray) -> np.ndarray:
        return self._score_func(X, self.centers, self.labels)
