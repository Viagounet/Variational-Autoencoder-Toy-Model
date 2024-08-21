import pandas as pd
import matplotlib.pyplot as plt
import json

with open(
    "inference_scripts/results/lv/decoded_results_from_env_truth.json",
    "r",
    encoding="utf-8",
) as f:
    data = json.load(f)

df = pd.DataFrame(data)

# Extracting ls_mean values for the scatter plot
ls_mean_values = df["ls_mean"].apply(pd.Series)
ls_mean_values.columns = ["ls_mean_x", "ls_mean_y"]

# Plotting the scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(ls_mean_values["ls_mean_x"], ls_mean_values["ls_mean_y"], color="blue")
plt.title("Scatter Plot of ls_mean")
plt.xlabel("ls_mean_x")
plt.ylabel("ls_mean_y")
plt.grid(True)
plt.savefig("inference_scripts/results/lv/ls_mean_scatter_plot.png")

# Show plot (optional)
plt.show()
