import argparse
from random import randrange
import torch
import os
import sys
import numpy as np
import json
import glob
from copy import copy

current_working_directory = os.getcwd()
module_path = os.path.abspath(current_working_directory)
sys.path.append(module_path)

from training_scripts.vae_torch import VAE
from PIL import Image
from torchvision import transforms

parser = argparse.ArgumentParser()
parser.add_argument("--initial_image", type=str)
parser.add_argument("--model_name", type=str)
parser.add_argument("--iterations", type=int, default=100)
parser.add_argument("--output_path", type=str, required=True)


if __name__ == "__main__":
    args = parser.parse_args()
    initial_image = args.initial_image
    model_name = args.model_name
    iterations = args.iterations

    with open(f"models/{model_name}/{model_name}.json", "r", encoding="utf-8") as f:
        model_data = json.load(f)

    latent_dim = model_data["ls_dim"]
    shape1, shape2 = model_data["images_size"]

    transform = transforms.Compose(
        [
            transforms.Resize((shape1, shape2)),
            transforms.ToTensor(),
        ]  # Resize to 200x200
    )

    vae = VAE(latent_dim, input_shape=[3, shape1, shape2])
    vae.load_state_dict(
        torch.load(f"models/{model_name}/{model_name}.pth", weights_only=True)
    )
    encoder = vae.encoder
    decoder = vae.decoder
    vae.eval()

    image = Image.open(initial_image).convert("RGB")  # Convert image to RGB
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    for dim in range(32):
        z_mean, z_var = encoder.forward(image)
        z_mean_original = z_mean.clone().detach()
        os.mkdir(f"{args.output_path}/{dim}")
        with torch.no_grad():
            for i in range(iterations):
                z_mean[:, dim] += 0.1
                distance = torch.norm(z_mean - z_mean_original)
                print(distance)
                image_tensor = decoder.forward(z_mean)
                image_tensor = image_tensor[0, :, :, :]
                np_array = (
                    image_tensor.detach().numpy()
                )  # Detach tensor before converting to numpy
                np_array = np.transpose(np_array, (1, 2, 0))
                new_image = Image.fromarray((np_array * 255).astype(np.uint8))
                new_image.save(f"{args.output_path}/{dim}/frame_{i}.png")
