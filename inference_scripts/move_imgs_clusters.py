import os
import shutil
import json

# Define the paths
json_path = "inference_scripts/results/lv/kmeans_clusters_with_filenames.json"
original_images_path = "imgs/l"
output_base_path = "inference_scripts/results/lv/clusters"

# Create the output base directory if it doesn't exist
if not os.path.exists(output_base_path):
    os.makedirs(output_base_path)

# Load the clusters with filenames from the JSON file
with open(json_path, "r", encoding="utf-8") as f:
    clusters_with_filenames = json.load(f)

# Iterate over each cluster and move the images to corresponding folders
for cluster_idx, filenames in enumerate(clusters_with_filenames):
    # Create a new directory for this cluster
    cluster_folder_path = os.path.join(output_base_path, f"cluster_{cluster_idx}")
    if not os.path.exists(cluster_folder_path):
        os.makedirs(cluster_folder_path)

    # Move each image in the cluster to the cluster folder
    for filename in filenames:
        source_path = os.path.join(original_images_path, filename)
        destination_path = os.path.join(cluster_folder_path, filename)

        # Check if the source file exists
        if os.path.exists(source_path):
            shutil.move(source_path, destination_path)
            print(f"Moved {filename} to {cluster_folder_path}")
        else:
            print(f"File {filename} not found in {original_images_path}")

print("All images have been moved to their respective cluster folders.")
