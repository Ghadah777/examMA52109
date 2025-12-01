# EXPLANATION.md

## 1. What was wrong with the original script?
The demo script `cluster_plot.py` was intended to run K-means clustering for  
k = 2, 3, 4, and 5. However, the script incorrectly replaced the value of k with:

    k = min(k, 3)

This meant that for k = 4 and k = 5 the clustering was actually performed  
with k = 3. As a result, the metrics (inertia and silhouette score) and  
all generated plots were incorrect for higher values of k.  
The script therefore *ran without crashing*, but did not produce the  
intended clustering behaviour.

## 2. How the issue was fixed
The fix was simply to replace:

    k = min(k, 3)

with:

    k = k

so that the clustering is truly performed with k = 2, 3, 4, and 5 as intended.  
After this correction, each run produces distinct metrics and correct plots  
for each value of k.

## 3. What the corrected script now does
The corrected script:
- reads a CSV file containing 2D numerical data,
- runs K-means clustering for k = 2, 3, 4, and 5,
- saves a cluster plot for each value of k,
- saves a CSV file containing the labelled data,
- prints inertia and silhouette metrics for each k,
- stores all outputs in the `demo_output/` folder.

The script now behaves exactly as designed.

## 4. Overview of the `cluster_maker` package
The package provides a complete workflow for clustering analysis:

- **preprocessing**: selecting numerical features and standardising them  
- **algorithms**: manual K-means implementation and a wrapper around scikit-learn KMeans  
- **evaluation**: inertia, silhouette score, and elbow curve computation  
- **plotting**: 2D cluster plots and elbow plots  
- **interface**: a high-level `run_clustering` function that combines preprocessing,  
  clustering, evaluation, and visualisation into one coherent pipeline.

The package enables clean, modular, and reproducible clustering analyses.
