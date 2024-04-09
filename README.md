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

**Guideline for the developer**. 
- Any change in the tree structure of the repository affect the installation of the packages with `setup.py`. Afert any modification in the tree, you have to reinstall the packages, here same usefull command line:

    - `pip list`: list of installed packeges in the virtual env, the name of echocardiography is `echocardiography version=1.0 ../Echocardiography`
    - `pip uninstall echocargiography`: uninstall the package
    - `pip install -e .`: reinstall the packege

- GPU note for Lixus user. Use the comand `nvidia-smi --query-gpu=utilization.gpu --format=csv --loop=1` to chech the usage of GPU
## Usage

### Regression

### Diffusion

#### Autoencoder
The initial step entails training the *Variational Autoencoder* model with the objective of reconstructing images from the training set using a latent space. The model can be either a Variational Autoencoder (VAE) or a Vector Quantized Variational Autoencoder (VQVAE)
To train autoecoder model use the following comand:

```bash
python train_vae.py --data eco
```

`infer_vae.py` file enable you to visualize some recostruction and the latent space. The trained model is saved during the training in a `trial_n` folder.

```bash
python infer_vae.py --data eco --trail trail_n
```

#### Latent Diffusion Model

*The Latent Diffusion Model (LDM)* is a denoising model that operates on the latent space extracted from the autoencoder, rather than on the pixel space like DDPM. It's important to note that for training the LDM, the training set must be the **same** as that used for the autoencoder. Pay attention to the batch size during training.

Run the following code for the training

```bash
python train_ldm --data eco --vae_train train_#n
```
where `--vae_train` is related to the train_#n folder of trained VAE model.






