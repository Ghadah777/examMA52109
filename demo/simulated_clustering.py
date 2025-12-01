###
## simulated_clustering.py
## Task 4 - MA52109 Practical Exam
## Uses cluster_maker to analyse simulated_data.csv
###

from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from cluster_maker import (
    select_features,
    standardise_features,
    kmeans,
)
from cluster_maker.evaluation import elbow_curve
from cluster_maker.plotting_clustered import plot_clusters_2d, plot_elbow


OUTPUT_DIR = "demo_output"


def main():

    # Load dataset
    data_path = "data/simulated_data.csv"
    df = pd.read_csv(data_path)

    # Select numeric feature columns
    feature_cols = ["feature_1", "feature_2", "feature_3", "feature_4"]
    X_df = select_features(df, feature_cols)
    X = X_df.to_numpy()            # FIX: convert DataFrame → NumPy array
    X = standardise_features(X)

    # Try several values of k to find appropriate clustering
    k_values = [2, 3, 4, 5, 6]
    inertia_results = elbow_curve(X, k_values, random_state=42, use_sklearn=True)

    # Create elbow plot
    fig_elbow, ax_elbow = plot_elbow(list(inertia_results.keys()),
                                     list(inertia_results.values()))
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig_elbow.savefig(os.path.join(OUTPUT_DIR, "simulated_elbow.png"), dpi=150)
    plt.close(fig_elbow)

    # Choose best k (simple heuristic: lowest inertia elbow point)
    best_k = min(inertia_results, key=inertia_results.get)
    print(f"\nChosen k for clustering (lowest inertia): {best_k}\n")

    # Run clustering using best_k
    labels, centroids = kmeans(X, k=best_k, random_state=42)

    # Plot clusters in 2D using first two features
    fig_cluster, ax = plot_clusters_2d(
        X[:, :2],  # only first two dims for 2D plotting
        labels,
        centroids=centroids[:, :2],
        title=f"Simulated Data Clustering (k={best_k})"
    )
    fig_cluster.savefig(os.path.join(OUTPUT_DIR, f"simulated_clusters_k{best_k}.png"), dpi=150)
    plt.close(fig_cluster)

    print("Clustering completed. Plots saved in demo_output/")


if __name__ == "__main__":
    main()
