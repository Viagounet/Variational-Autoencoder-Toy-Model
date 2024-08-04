import os
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset, random_split
from vae_torch import VAE

LATENT_DIM = 2


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
vae = VAE(latent_dim=LATENT_DIM).to(device)
optimizer = optim.Adam(vae.parameters(), lr=1e-3)

num_epochs = 100

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
        print(
            f"Epoch {epoch + 1}, Test Loss: {test_loss / len(test_loader.dataset):.4f}"
        )

    print(
        f"Epoch {epoch + 1}, Train Loss: {train_loss / len(train_loader.dataset):.4f}"
    )

# Save the model state_dict
torch.save(vae.state_dict(), f"models/sim1_ls{LATENT_DIM}.pth")
