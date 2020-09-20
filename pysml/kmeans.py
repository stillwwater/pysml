import numpy as np
import random

from .cluster import Estimator


def _k_init(X: np.ndarray, n_clusters: int, random_state=None) -> np.ndarray:
    """Initialize kmeans centers by picking a random point as the center"""
    return np.array([X[random.randint(0, len(X) - 1)]
                    for i in range(n_clusters)])


def _euclidean_sq_norms(X: np.ndarray,
                        centers: np.ndarray, sq_norms: np.ndarray):
    """Compute squared euclidean distances for each point with each center"""
    tmp = np.zeros_like(X[0])
    for i, x in enumerate(X):
        for j, center in enumerate(centers):
            np.subtract(x, center, out=tmp)
            sq_norms[i][j] = np.dot(tmp, tmp)


def _labels_inertia(X: np.ndarray, sq_norms: np.ndarray, labels: np.ndarray):
    """Compute labels (cluster index) for each point X
    returns sum of squared distances of samples to their closest center
    """
    inertia = 0.
    for i, x in enumerate(X):
        min_dist = np.amin(sq_norms[i])
        center = np.where(sq_norms[i] == min_dist)[0][0]
        labels[i] = center
        inertia += min_dist
    return inertia


def _kmeans_predict_lloyd(X: np.ndarray, centers: np.ndarray,
                          labels: np.ndarray) -> np.ndarray:
    """Runs the E step of lloyd's kmeans
    returns the labels for each point
    """
    sq_norms = np.zeros((len(X), len(centers)))
    labels = np.zeros((len(X,)), dtype=np.int32)

    _euclidean_sq_norms(X, centers, sq_norms)
    _labels_inertia(X, sq_norms, labels)
    return labels


def _kmeans_score_lloyd(X: np.ndarray, centers: np.ndarray,
                        labels: np.ndarray) -> float:
    """Opposite of the value of X on the kmeans objective"""
    sq_norms = np.zeros((len(X), len(centers)))
    labels = np.zeros((len(X,)), dtype=np.int32)

    _euclidean_sq_norms(X, centers, sq_norms)
    return -_labels_inertia(X, sq_norms, labels)


def _kmeans_single_lloyd(X: np.ndarray, centers_init: np.ndarray,
                         max_iter=300) -> Estimator:
    """Single run of lloyd's kmeans algorithm"""
    sq_norms = np.zeros((len(X), len(centers_init)))
    cluster_lengths = np.zeros((len(centers_init,)), np.int64)
    labels = np.zeros((len(X,)), dtype=np.int32)
    intertia = float('inf')
    centers_new = centers_init.copy()
    centers_old = np.zeros_like(centers_new)

    for i in range(max_iter):
        if np.allclose(centers_old, centers_new):
            break

        centers_old = centers_new.copy()

        # E step
        _euclidean_sq_norms(X, centers_new, sq_norms)
        inertia = _labels_inertia(X, sq_norms, labels)

        # M step
        centers_new.fill(0.)
        cluster_lengths.fill(0)

        for x, l in zip(X, labels):
            centers_new[l] += x
            cluster_lengths[l] += 1

        for i, center in enumerate(centers_new):
            centers_new[i] *= 1. / max(1, cluster_lengths[i])

    return Estimator(centers_new, labels, inertia,
                     _predict_func=_kmeans_predict_lloyd,
                     _score_func=_kmeans_score_lloyd)


def kmeans(X: np.ndarray, n_clusters=8, random_state=None,
           max_iter=300, n_init=10) -> Estimator:
    """Kmeans clustering.

    n_clusters : int
        Number of clusters

    random_state : int | None
        Initialize kmeans with a seed.

    max_iter : int
        The maximum number of iterations of the kmeans algorithm for a
        single run

    n_init : int
        Number of times the kmeans algorithm will be run with different
        centroid seeds. The final result will be the best output of all
        consecutive runs

    returns a fitted k-means estimator.
    """
    est = None

    if random_state:
        random.seed(random_state)

    for i in range(n_init):
        centers = _k_init(X, n_clusters, random_state)
        est_ = _kmeans_single_lloyd(X, centers, max_iter)

        if not est or est_.inertia < est.inertia:
            est = est_

        if random_state:
            random_state += random.randint(-random_state, random_state)

    return est
