# Variational-Autoencoder-Toy-Model
Just some tests around VAE, to then try and implement technique such as DreamerV3 &amp; others

## 1. Install the requierements
You'll need PyTorch, Dash (for interactive play with the model) and PyGame to simulate a "game" (not **really** a game tho lol)

```bash
pip install -r requirements.txt```

## 2. Explore the latent space
You can explore the 8-dimensional latent from an already trained model by starting the Dash app:

```bash
python vae_inference_color.py
```

## 3. Generate your own data
You can generate the data used to train the model (& modify it if you want) by starting the PyGame script. This will generate about 10k images in an /imgs folder.

```bash
python main.py
```

## 4. Retrain the model
Use the vae_color_training file to train the VAE on the generated data
```bash
python vae_color_training.py
```

# Images

## Original training frames (9 random examples out of 10k)

![](README_imgs\env_grid.png)

## Exploring the latent space & generated image

![](README_imgs\interface.png)