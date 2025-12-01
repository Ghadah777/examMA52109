###
## cluster_maker
## James Foadi - University of Bath
## November 2025
###

from __future__ import annotations

from typing import Tuple, Optional

import numpy as np
from sklearn.cluster import KMeans


def init_centroids(
    X: np.ndarray,
    k: int,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """
    Initialise centroids by randomly sampling points from X without replacement.
    """
    if k <= 0:
        raise ValueError("k must be a positive integer.")

    n_samples = X.shape[0]
    # FIXED: compare k to number of samples (not number of features)
    if k > n_samples:
        raise ValueError("k cannot be larger than the number of samples.")

    rng = np.random.RandomState(random_state)

    # FIXED: choose exactly k centroids (not k+1)
    indices = rng.choice(n_samples, size=k, replace=False)
    return X[indices]


def assign_clusters(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """
    Assign each sample to the nearest centroid (Euclidean distance).
    """
    # Compute distances between each sample and each centroid
    distances = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)

    # Choose the nearest centroid
    labels = np.argmin(distances, axis=1)

    return labels


def update_centroids(
    X: np.ndarray,
    labels: np.ndarray,
    k: int,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """
    Update centroids by taking the mean of points in each cluster.
    If a cluster becomes empty, re-initialise its centroid randomly from X.
    """
    n_features = X.shape[1]
    new_centroids = np.zeros((k, n_features), dtype=float)
    rng = np.random.RandomState(random_state)

    for cluster_id in range(k):
        mask = labels == cluster_id
        if not np.any(mask):
            # Empty cluster: re-initialise randomly
            idx = rng.randint(0, X.shape[0])
            new_centroids[cluster_id] = X[idx]
        else:
            new_centroids[cluster_id] = X[mask].mean(axis=0)

    return new_centroids


def kmeans(
    X: np.ndarray,
    k: int,
    max_iter: int = 300,
    tol: float = 1e-4,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simple manual K-means implementation.
    """
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a NumPy array.")

    centroids = init_centroids(X, k, random_state=random_state)

    for _ in range(max_iter):
        labels = assign_clusters(X, centroids)
        new_centroids = update_centroids(X, labels, k, random_state=random_state)
        shift = np.linalg.norm(new_centroids - centroids)
        centroids = new_centroids

        if shift < tol:
            break

    labels = assign_clusters(X, centroids)
    return labels, centroids


def sklearn_kmeans(
    X: np.ndarray,
    k: int,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Thin wrapper around scikit-learn's KMeans.
    """
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a NumPy array.")

    model = KMeans(
        n_clusters=k,
        random_state=random_state,
        n_init=10,
    )
    model.fit(X)
    labels = model.labels_
    centroids = model.cluster_centers_
    return labels, centroids
