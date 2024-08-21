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

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("-m", "--model")
parser.add_argument("-o", "--output_folder")
parser.add_argument("-d", "--latent_dim", type=int)

if __name__ == "__main__":
    # Load the trained VAE model
    IMG_SIZE = (256, 192)
    SHAPE = (3, 256, 192)

    args = parser.parse_args()
    MODEL = args.model
    latent_dim = args.latent_dim
    output = args.output_folder

    vae = VAE(latent_dim, SHAPE)
    vae.load_state_dict(torch.load(MODEL, weights_only=True))
    vae.eval()

    encoder = vae.encoder

    transform = transforms.Compose(
        [transforms.Resize(IMG_SIZE), transforms.ToTensor()]  # Resize to 200x200
    )

    path = Path("imgs/l")
    images = [
        file for file in path.iterdir() if (file.is_file() and ".png" in file.name)
    ]

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
        f"{output}/decoded_results_from_env_truth.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
