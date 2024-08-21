import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, latent_dim=8, input_shape=(3, 200, 200)):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU()

        # Calculate the size of the feature map after the conv layers
        self.feature_size = self._get_conv_output(input_shape)

        self.fc1 = nn.Linear(self.feature_size, 128)
        self.fc2_mean = nn.Linear(128, latent_dim)
        self.fc2_log_var = nn.Linear(128, latent_dim)

    def _get_conv_output(self, shape):
        with torch.no_grad():
            input = torch.autograd.Variable(torch.rand(1, *shape))
            output = self.conv1(input)
            output = self.conv2(output)
            output = output.view(1, -1)
            n_size = output.size(1)
        return n_size

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        z_mean = self.fc2_mean(x)
        z_log_var = self.fc2_log_var(x)
        return z_mean, z_log_var


class Decoder(nn.Module):
    def __init__(self, latent_dim=8, input_shape=(3, 256, 192)):
        super(Decoder, self).__init__()
        self.input_shape = input_shape
        self.latent_dim = latent_dim

        # Calculate the size of the feature map after the conv layers
        # Assuming the input_shape is (3, 256, 192) and two conv layers with stride 2
        self.feature_map_height = input_shape[1] // 4  # 256 // 4 = 64
        self.feature_map_width = input_shape[2] // 4  # 192 // 4 = 48
        self.feature_size = 64 * self.feature_map_height * self.feature_map_width

        self.fc = nn.Linear(latent_dim, self.feature_size)

        # Define activation function
        self.relu = nn.ReLU()

        # Now, build the deconvolutional layers
        self.conv_layers = nn.Sequential(
            nn.ConvTranspose2d(
                64, 32, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                32, 16, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                16, 3, kernel_size=3, padding=1
            ),  # Final layer to get back to 3 channels
            nn.Sigmoid(),  # Sigmoid to bring pixel values between 0 and 1
        )

    def forward(self, x):
        x = self.relu(self.fc(x))

        # Reshape into (batch_size, 64, 64, 48) based on computed dimensions
        x = x.view(x.size(0), 64, self.feature_map_height, self.feature_map_width)

        # Pass through the transposed conv layers
        x = self.conv_layers(x)

        # Ensure the output matches the input shape
        x = nn.functional.interpolate(
            x,
            size=(self.input_shape[1], self.input_shape[2]),  # (256, 192)
            mode="bilinear",
            align_corners=False,
        )
        return x


# Define the VAE
class VAE(nn.Module):
    def __init__(self, latent_dim=8, input_shape=(3, 200, 200)):
        super(VAE, self).__init__()
        self.encoder = Encoder(latent_dim, input_shape)
        self.decoder = Decoder(latent_dim, input_shape)

    def reparameterize(self, z_mean, z_log_var):
        std = torch.exp(0.5 * z_log_var)
        eps = torch.randn_like(std)
        return z_mean + eps * std

    def forward(self, x):
        z_mean, z_log_var = self.encoder(x)
        z = self.reparameterize(z_mean, z_log_var)
        x_reconstructed = self.decoder(z)
        return x_reconstructed, z_mean, z_log_var
