import argparse
import torch
import os
import sys
import numpy as np
import json
import glob

current_working_directory = os.getcwd()
module_path = os.path.abspath(current_working_directory)
sys.path.append(module_path)

from training_scripts.vae_torch import VAE
from PIL import Image
from torchvision import transforms

parser = argparse.ArgumentParser()
parser.add_argument("--frames_path", type=str)
parser.add_argument("--model_name", type=str)
parser.add_argument("--output_path", type=str, required=True)


if __name__ == "__main__":
    args = parser.parse_args()
    frames_path = args.frames_path
    model_name = args.model_name

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
    vae.eval()

    imgs = glob.glob(f"{frames_path}/*")
    for image_path in imgs:
        img_name = image_path.split("/")[-1].split(".")[0]
        image = Image.open(image_path).convert("RGB")  # Convert image to RGB
        image = transform(image)
        image = image.unsqueeze(0)  # Add batch dimension

        with torch.no_grad():
            generated_image = vae(image)

        image_tensor = generated_image[0].squeeze(0)
        np_array = (
            image_tensor.detach().numpy()
        )  # Detach tensor before converting to numpy
        np_array = np.transpose(np_array, (1, 2, 0))
        image = Image.fromarray((np_array * 255).astype(np.uint8))
        image.save(f"{args.output_path}/{img_name}.png")
