import torch
import torch.nn as nn
import dash
import dash_bootstrap_components as dbc
import plotly.express as px
import numpy as np
from dash import dcc, html
from dash.dependencies import Input, Output
from vae_torch import VAE


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
