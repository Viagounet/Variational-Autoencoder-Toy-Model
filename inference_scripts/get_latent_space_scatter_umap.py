import pandas as pd
import matplotlib.pyplot as plt
import json
import umap
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load the data from the JSON file
with open(
    "inference_scripts/results/lv/decoded_results_from_env_truth.json",
    "r",
    encoding="utf-8",
) as f:
    data = json.load(f)

df = pd.DataFrame(data)

# Assuming 'ls_mean' contains vectors of dimension N, let's convert it to a numpy array
ls_mean_values = np.array(df["ls_mean"].tolist())

# Standardize the data for better clustering performance
scaler = StandardScaler()
ls_mean_values_scaled = scaler.fit_transform(ls_mean_values)

# Apply UMAP to reduce the dimensionality to 2D
reducer = umap.UMAP(n_components=2, random_state=42)
ls_mean_2d = reducer.fit_transform(ls_mean_values_scaled)

# Apply KMeans clustering on the UMAP-reduced data
n_clusters = 10  # You can change the number of clusters
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(ls_mean_2d)

# Convert the 2D UMAP results into a DataFrame for easy plotting
ls_mean_2d_df = pd.DataFrame(ls_mean_2d, columns=["UMAP_1", "UMAP_2"])
ls_mean_2d_df["Cluster"] = cluster_labels

# Retrieve the filenames from the "file" column
filenames = df["file"].tolist()

# Create a list of lists where each sublist contains the filenames of elements in each cluster
clusters_with_filenames = [[] for _ in range(n_clusters)]
for idx, cluster in enumerate(cluster_labels):
    clusters_with_filenames[cluster].append(filenames[idx])

# Save the clusters with filenames as a JSON file
with open(
    "inference_scripts/results/lv/kmeans_clusters_with_filenames.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(clusters_with_filenames, f, indent=4)

# Alternatively, print the clusters with filenames to the console
print("Cluster filenames:")
print(clusters_with_filenames)

# Plotting the 2D scatter plot with clusters
plt.figure(figsize=(10, 6))
scatter = plt.scatter(
    ls_mean_2d_df["UMAP_1"],
    ls_mean_2d_df["UMAP_2"],
    c=ls_mean_2d_df["Cluster"],
    cmap="viridis",
    s=50,
)
plt.title("2D Scatter Plot with KMeans Clusters after UMAP Dimensionality Reduction")
plt.xlabel("UMAP Dimension 1")
plt.ylabel("UMAP Dimension 2")
plt.grid(True)

# Add a color bar to indicate clusters
plt.colorbar(scatter, label="Cluster Label")

plt.savefig(
    "inference_scripts/results/lv/ls_mean_umap_kmeans_clusters_with_filenames.png"
)

# Show plot (optional)
plt.show()
