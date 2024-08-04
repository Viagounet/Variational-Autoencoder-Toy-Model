import torch
import torch.nn as nn
import dash
import dash_bootstrap_components as dbc
import plotly.express as px
import numpy as np
from dash import dcc, html
from dash.dependencies import Input, Output


# Define the encoder
class Encoder(nn.Module):
    def __init__(self, latent_dim=8):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.fc1 = nn.Linear(64 * 50 * 50, 128)  # Adjust the input size here
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
        self.fc = nn.Linear(
            latent_dim, 64 * 50 * 50
        )  # Adjusted to match encoder output
        self.conv1 = nn.ConvTranspose2d(
            64, 32, kernel_size=3, stride=2, padding=1, output_padding=1
        )
        self.conv2 = nn.ConvTranspose2d(
            32, 3, kernel_size=3, stride=2, padding=1, output_padding=1
        )
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc(x))
        x = x.view(x.size(0), 64, 50, 50)  # Reshape to match the feature map size
        x = self.relu(self.conv1(x))
        x = self.sigmoid(self.conv2(x))
        x = x[:, :, :200, :200]  # Crop the output to match the input size
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


# Load the trained VAE model
latent_dim = 8
vae = VAE(latent_dim)
vae.load_state_dict(torch.load("color_vae_state_dict.pth"))
vae.eval()


# Function to generate an image from a latent vector
def generate_image(latent_vector):
    with torch.no_grad():
        generated_image = vae.decoder(latent_vector)
        return generated_image


def slider(i: int) -> dcc.Slider:
    return dcc.Slider(
        id=f"slider-{i}",
        min=-3,
        max=3,
        step=0.1,
        value=0,
        marks={i: str(i) for i in range(-3, 4)},
    )


# Create a Dash app
app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
app.layout = html.Div(
    [
        html.H1("VAE Color Image Generator"),
        html.Div(
            [
                html.Div(
                    [
                        html.Div(
                            [html.Label(f"Latent Dimension {i}"), slider(i), html.Br()]
                        )
                        for i in range(latent_dim)
                    ],
                    style={"width": "50vw"},
                ),
                html.Div(id="output-image"),
            ],
            className="d-flex flex-row",
        ),
    ]
)


@app.callback(
    Output("output-image", "children"),
    [Input(f"slider-{i}", "value") for i in range(latent_dim)],
)
def update_image(*latent_values):
    latent_vector = torch.tensor(latent_values, dtype=torch.float32).unsqueeze(0)
    generated_image = generate_image(latent_vector).cpu().numpy()
    generated_image = np.transpose(
        np.squeeze(generated_image), (1, 2, 0)
    )  # Rearrange dimensions for RGB

    fig = px.imshow(generated_image)
    fig.update_layout(
        coloraxis_showscale=False,
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
    )

    return dcc.Graph(
        figure=fig,
        style={"width": "100%", "height": "100%"},
        config={"responsive": True},
    )


if __name__ == "__main__":
    app.run_server(debug=False)
