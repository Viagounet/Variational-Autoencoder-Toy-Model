import os
import sys
from typing import TypedDict

current_working_directory = os.getcwd()
module_path = os.path.abspath(current_working_directory)
sys.path.append(module_path)

import torch
import pandas as pd
import json

from training_scripts.vae_torch import VAE
from pathlib import Path
from PIL import Image
from torchvision import transforms


# Load the trained VAE model
MODEL = "models/sim1_ls2.pth"
latent_dim = int(MODEL.replace("models/sim1_ls", "").split(".")[0])
vae = VAE(latent_dim)
vae.load_state_dict(torch.load(MODEL, weights_only=True))
vae.eval()

encoder = vae.encoder

transform = transforms.Compose(
    [transforms.Resize((200, 200)), transforms.ToTensor()]  # Resize to 200x200
)

path = Path("imgs")
images = [file for file in path.iterdir() if file.is_file()]


class VAEData(TypedDict):
    file: list[str]
    ls_mean: list[float]
    ls_var: list[float]


data: VAEData = {"file": [], "ls_mean": [], "ls_var": []}

for img_path in images:
    image = Image.open(img_path).convert("RGB")  # Convert image to RGB
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    z_mean, z_var = encoder.forward(image)

    data["file"].append(img_path.name)
    data["ls_mean"].append(z_mean.tolist()[0])
    data["ls_var"].append(z_var.tolist()[0])

with open(
    "inference_scripts/results/decoded_results_from_env_truth.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(data, f, indent=4, ensure_ascii=False)
