# Echocardiography
Analysis of PLAX echocardiography is conducted on the available dataset [Echonet-LVH](https://echonet.github.io/lvh/). The repository contains two modules: regression and diffusion. The regression module is designed for keypoints regression in the left ventricle, while the diffusion module is intended for unconditional/conditional generation using the Latent Diffusion Model.

## Install
Setting up a virtual environment and installing PyTorch following the [official guidelines](https://pytorch.org/get-started/locally/)

Download the repository with `git clone`.

```bash
git clone https://github.com/AngeloLasala/Echocardiography.git
```

and run the `setup.py` file

```bash
pip install -e .
```

## Usage

### Regression
To do...

### Diffusion

#### Autoencoder
The initial step entails training the Variational Autoencoder model with the objective of reconstructing images from the training set using a latent space. The model can be either a Variational Autoencoder (VAE) or a Vector Quantized Variational Autoencoder (VQVAE)
To train autoecoder model use the following comand:

```bash
python train_vae.py --data eco
```

`infer_vae.py` file enable you to visualize some recostruction and the latent space. The trained model is saved during the training in a `trial_n` folder.

```bash
python infer_vae.py --data eco --trail trail_n
```




