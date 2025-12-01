###
## agglomerative.py
## New module for hierarchical/agglomerative clustering
## MA52109 - Task 5
###

from __future__ import annotations

from typing import Tuple
import numpy as np
from sklearn.cluster import AgglomerativeClustering


def agglomerative_cluster(
    X: np.ndarray,
    n_clusters: int = 2,
    linkage: str = "ward"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform agglomerative hierarchical clustering and return labels + centroids.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Input data.
    n_clusters : int
        Number of clusters to form.
    linkage : {"ward", "complete", "average", "single"}
        Linkage method.

    Returns
    -------
    labels : ndarray of shape (n_samples,)
    centroids : ndarray of shape (n_clusters, n_features)
    """

    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a NumPy array.")

    # Fit model
    model = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage=linkage
    )

    labels = model.fit_predict(X)

    # Compute centroids
    centroids = np.zeros((n_clusters, X.shape[1]), dtype=float)
    for cluster_id in range(n_clusters):
        mask = (labels == cluster_id)
        if not np.any(mask):
            raise ValueError(f"Cluster {cluster_id} is empty.")
        centroids[cluster_id] = X[mask].mean(axis=0)

    return labels, centroids
