import os
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset, random_split


# Custom dataset class to load images from a single folder
class CustomImageDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.img_files = [f for f in os.listdir(img_dir) if f.endswith(".png")]

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        image = Image.open(img_path).convert("L")  # Convert image to grayscale
        if self.transform:
            image = self.transform(image)
        return image


# Define the encoder
class Encoder(nn.Module):
    def __init__(self, latent_dim=8):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.fc1 = nn.Linear(64 * 13 * 13, 128)
        self.fc2_mean = nn.Linear(128, latent_dim)
        self.fc2_log_var = nn.Linear(128, latent_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        z_mean = self.fc2_mean(x)
        z_log_var = self.fc2_log_var(x)
        return z_mean, z_log_var


# Define the decoder
class Decoder(nn.Module):
    def __init__(self, latent_dim=8):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(latent_dim, 64 * 13 * 13)
        self.conv1 = nn.ConvTranspose2d(
            64, 32, kernel_size=3, stride=2, padding=1, output_padding=1
        )
        self.conv2 = nn.ConvTranspose2d(
            32, 1, kernel_size=3, stride=2, padding=1, output_padding=1
        )
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc(x))
        x = x.view(x.size(0), 64, 13, 13)
        x = self.relu(self.conv1(x))
        x = self.sigmoid(self.conv2(x))
        x = x[:, :, :50, :50]  # Crop the output to match the input size
        return x


# Define the VAE
class VAE(nn.Module):
    def __init__(self, latent_dim=8):
        super(VAE, self).__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def reparameterize(self, z_mean, z_log_var):
        std = torch.exp(0.5 * z_log_var)
        eps = torch.randn_like(std)
        return z_mean + eps * std

    def forward(self, x):
        z_mean, z_log_var = self.encoder(x)
        z = self.reparameterize(z_mean, z_log_var)
        x_reconstructed = self.decoder(z)
        return x_reconstructed, z_mean, z_log_var


# Define the loss function
def loss_function(x, x_reconstructed, z_mean, z_log_var):
    recon_loss = nn.functional.binary_cross_entropy(x_reconstructed, x, reduction="sum")
    kl_loss = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())
    return recon_loss + kl_loss


# Load and preprocess the dataset
transform = transforms.Compose([transforms.Resize((50, 50)), transforms.ToTensor()])

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
vae = VAE(latent_dim=8).to(device)
optimizer = optim.Adam(vae.parameters(), lr=1e-3)

num_epochs = 50

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
torch.save(vae.state_dict(), "vae_state_dict.pth")
