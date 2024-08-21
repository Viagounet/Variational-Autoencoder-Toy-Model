import os
import shutil
import subprocess

# Directory containing the images
image_dir = "inference_scripts/results/lv/clusters/cluster_3"
output_dir = "inference_scripts/results/lv/clusters/cluster_3_reordered"
output_video = "inference_scripts/results/lv/clusters/cluster_3/output.mp4"
fps = 30

# Step 1: Create output directory
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Step 2: Get and sort image filenames
image_files = sorted(
    [f for f in os.listdir(image_dir) if f.endswith(".png")],
    key=lambda x: int(x.split("_")[-1].split(".")[0]),
)

# Step 3: Rename/Copy images to sequential filenames in the output directory
for idx, filename in enumerate(image_files):
    new_filename = f"{idx+1:04d}.png"  # Sequential filenames (0001.png, 0002.png, etc.)
    src = os.path.join(image_dir, filename)
    dst = os.path.join(output_dir, new_filename)
    shutil.copy(src, dst)  # Copying the file to avoid modifying the original

# Step 4: Call ffmpeg to create the video
ffmpeg_command = [
    "ffmpeg",
    "-r",
    str(fps),  # Input frame rate
    "-f",
    "image2",  # Input format
    "-i",
    os.path.join(output_dir, "%04d.png"),  # Input files pattern
    "-vf",
    "format=yuv420p",  # Video format
    output_video,  # Output video file
]

subprocess.run(ffmpeg_command)

# Step 5: Clean up the copied files in the output directory
shutil.rmtree(output_dir)

print(f"Video created successfully: {output_video}")
