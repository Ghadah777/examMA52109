###
## demo_agglomerative.py
## Demonstration of agglomerative clustering on difficult_dataset.csv
## MA52109 - Task 5
###

from __future__ import annotations

import os
import pandas as pd
import matplotlib.pyplot as plt

from cluster_maker.preprocessing import select_features, standardise_features
from cluster_maker.agglomerative import agglomerative_cluster
from cluster_maker.plotting_clustered import plot_clusters_2d


OUTPUT_DIR = "demo_output"


def main():

    data_path = "data/difficult_dataset.csv"
    df = pd.read_csv(data_path)

    # Use all numeric columns
    feature_cols = [col for col in df.columns if df[col].dtype != "object"]

    X_df = select_features(df, feature_cols)
    X = X_df.to_numpy()
    X = standardise_features(X)

    # Try hierarchical clustering for k = 2, 3, 4
    ks = [2, 3, 4]
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for k in ks:
        print(f"\n=== Agglomerative clustering with k = {k} ===")
        labels, centroids = agglomerative_cluster(X, n_clusters=k)

        # Plot clusters (using first 2 features)
        fig, ax = plot_clusters_2d(
            X[:, :2],
            labels,
            centroids=centroids[:, :2],
            title=f"Agglomerative Clustering (k={k})"
        )

        fig.savefig(os.path.join(OUTPUT_DIR, f"difficult_clusters_k{k}.png"), dpi=150)
        plt.close(fig)

    print("\nAgglomerative clustering completed. Plots saved in demo_output/")


if __name__ == "__main__":
    main()
