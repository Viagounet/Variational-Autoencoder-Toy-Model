import os
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset, random_split
from vae_torch import VAE

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("-d", "--dim_ls", type=int)
parser.add_argument("-n", "--name")
parser.add_argument("-e", "--epochs", type=int)


# Custom dataset class to load color images from a single folder
class CustomImageDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.img_files = [f for f in os.listdir(img_dir) if f.endswith(".png")]

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        image = Image.open(img_path).convert("RGB")  # Convert image to RGB
        if self.transform:
            image = self.transform(image)
        return image


# Define the loss function
def loss_function(x, x_reconstructed, z_mean, z_log_var):
    recon_loss = nn.functional.mse_loss(x_reconstructed, x, reduction="sum")
    kl_loss = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())
    return recon_loss + kl_loss


if __name__ == "__main__":
    args = parser.parse_args()
    name = args.name
    if os.path.exists(name):
        raise FileExistsError("The log folder already exists")
    else:
        os.mkdir(f"training_logs/{name}")
    latent_dim = args.dim_ls

    # Load and preprocess the dataset
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    img_dir = "imgs/"  # Replace with your images directory
    dataset = CustomImageDataset(img_dir, transform=transform)

    # Split the dataset into training and testing sets
    train_size = int(0.7 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # Train the VAE
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae = VAE(latent_dim=latent_dim).to(device)
    optimizer = optim.Adam(vae.parameters(), lr=1e-2)

    num_epochs = args.epochs

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

        if (epoch + 1) % 10 == 0:
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
    torch.save(vae.state_dict(), f"models/sim1_ls{latent_dim}.pth")
