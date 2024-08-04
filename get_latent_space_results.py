from vae_torch import VAE
from pathlib import Path
from PIL import Image

import torch
from torchvision import transforms

# Load the trained VAE model
latent_dim = 8
vae = VAE(latent_dim)
vae.load_state_dict(torch.load("models/sim1_ls8.pth"))
vae.eval()
encoder = vae.encoder

transform = transforms.Compose(
    [transforms.Resize((200, 200)), transforms.ToTensor()]  # Resize to 200x200
)

path = Path("imgs")
images = [file for file in path.iterdir() if file.is_file()]
for img_path in images[872:873]:
    print(img_path)
    image = Image.open(img_path).convert("RGB")  # Convert image to RGB
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    print(image.size())
    z_mean, z_var = encoder.forward(image)
    print(f"Latent space: {z_mean}")
    print(f"Var: {z_var}")
