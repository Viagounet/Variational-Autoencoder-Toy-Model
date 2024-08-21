import os
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import torchvision
import json
from glob import glob

from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset, random_split
from vae_torch import VAE

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("-d", "--dim_ls", type=int)
parser.add_argument("-n", "--name")
parser.add_argument("-e", "--epochs", type=int)
parser.add_argument("-s", "--scaling", type=float, default=1)
parser.add_argument("-i", "--input_folder", type=str)

import os
from PIL import Image
from torch.utils.data import Dataset


class CustomImageDataset(Dataset):
    def __init__(self, img_dir, transform=None, placeholder_image=None):
        self.img_dir = img_dir
        self.transform = transform
        self.img_files = [f for f in os.listdir(img_dir) if f.endswith(".png")]
        self.placeholder_image = (
            placeholder_image  # You can pass a placeholder image if needed
        )

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        while idx < len(self.img_files):
            img_path = os.path.join(self.img_dir, self.img_files[idx])
            try:
                image = Image.open(img_path).convert("RGB")  # Convert image to RGB
                if self.transform:
                    image = self.transform(image)
                return image
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
                idx += 1
        if self.placeholder_image is not None:
            if self.transform:
                return self.transform(self.placeholder_image)
            return self.placeholder_image
        raise IndexError("All images in the dataset are invalid or not accessible.")


# Define the loss function
def loss_function(x, x_reconstructed, z_mean, z_log_var):
    recon_loss = nn.functional.mse_loss(x_reconstructed, x, reduction="sum")
    kl_loss = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())
    return recon_loss + kl_loss


if __name__ == "__main__":
    args = parser.parse_args()
    name = args.name
    num_epochs = args.epochs
    img_dir = args.input_folder
    scaling = args.scaling

    first_image_path = glob(f"{img_dir}/*.png")[0]
    first_image = Image.open(first_image_path).convert("RGB")
    width, height = first_image.size

    width = int(width * scaling)
    height = int(height * scaling)

    if os.path.exists(f"training_logs/{name}"):
        erase_previous_save = input(
            "Folder {folder} already exists, do you want to continue ? (y/n) : "
        )
        if erase_previous_save == "y":
            pass
        else:
            exit()
    else:
        os.mkdir(f"training_logs/{name}")

    latent_dim = args.dim_ls
    print(latent_dim, type(latent_dim))

    with open(f"models/{name}/{name}{latent_dim}.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "ls_dim": latent_dim,
                "max_epochs": num_epochs,
                "data_source": img_dir,
                "images_size": [height, width],
            },
            f,
            indent=4,
            ensure_ascii=False,
        )
    # Load and preprocess the dataset
    transform = transforms.Compose(
        [
            transforms.Resize((height, width)),
            transforms.ToTensor(),
        ]
    )

    placeholder_image_path = first_image_path
    placeholder_image = Image.open(placeholder_image_path).convert("RGB")

    # Initialize the dataset
    dataset = CustomImageDataset(
        img_dir=img_dir,
        transform=transform,
        placeholder_image=placeholder_image,
    )

    # Split the dataset into training and testing sets
    train_size = int(0.7 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    print(torch.cuda.is_available())
    # Train the VAE
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae = VAE(latent_dim=latent_dim, input_shape=(3, height, width)).to(device)
    optimizer = optim.Adam(vae.parameters(), lr=1e-3)

    epochs_train: list[int] = []
    epochs_test: list[int] = []
    loss_train_logs: list[float] = []
    loss_test_logs: list[float] = []

    for epoch in range(num_epochs):
        vae.train()
        train_loss = 0
        for x in train_loader:
            x = x.to(device)
            optimizer.zero_grad()
            x_reconstructed, z_mean, z_log_var = vae(x)
            loss = loss_function(x, x_reconstructed, z_mean, z_log_var)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            vae.eval()
            test_loss = 0
            with torch.no_grad():
                for x in test_loader:
                    x = x.to(device)
                    x_reconstructed, z_mean, z_log_var = vae(x)
                    loss = loss_function(x, x_reconstructed, z_mean, z_log_var)
                    test_loss += loss.item()
            epochs_test.append(epoch + 1)
            loss_test_logs.append(test_loss / len(test_loader.dataset))
            print(
                f"TEST ({epoch} / {num_epochs}): ", test_loss / len(test_loader.dataset)
            )
        print(
            f"TRAIN ({epoch} / {num_epochs}): ", train_loss / len(train_loader.dataset)
        )
        epochs_train.append(epoch + 1)
        loss_train_logs.append(train_loss / len(train_loader.dataset))

        plt.figure(figsize=(10, 6))
        plt.plot(epochs_train, loss_train_logs, label="Train loss")
        plt.plot(epochs_test, loss_test_logs, label="Test loss")

        # Add a title and labels
        plt.title("Training losses")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.ylim(0, 300)  # Here, 10 is the lower limit and 30 is the upper limit
        plt.legend()
        plt.savefig(f"training_logs/{name}/loss_graph.png")

    # Save the model state_dict
    torch.save(vae.state_dict(), f"models/lv/{name}{latent_dim}.pth")
