# Echocardiography
Analysis of PLAX echocardiography is conducted on the available dataset [Echonet-LVH](https://echonet.github.io/lvh/). The repository contains two modules: regression and diffusion. The regression module is designed for keypoints regression in the left ventricle, while the diffusion module is intended for unconditional/conditional generation using the Latent Diffusion Model.

The generation of a new PLAX view can be achieved by anatomically conditioning:
- Chamber dimensions: dimensions of IVS, LVID, and LVPW
- Classification of hypertrophy:
    - Concentric hypertrophy: [1,0,0,0]- (rwt>0.42  lvm<200 : color red)
    - Concentric remodeling:  [0,1,0,0]- (rwt>0.42  lvm<200 : color orange)
    - Eccentric hypertrophy:  [0,0,1,0] - (rwt<0.42  lvm>200 : color olive)
    - Normal geometry:        [0,0,0,1] - (rwt<0.42  lvm<200 : color green)

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
he initial step of analyzing PLAX images is the identification of the key points of IVS, LVID, and LVPW. To date, the repository has implemented three different approaches:

1) **Direct regression**: Regression of the coordinates. Models: ResNet50, Swinv2

2) **Heatmaps regression**: Regression of Gaussian heatmaps centered on the coordinates. Models: Unet (up-sampling and up-conv)

3) **Mask segmentation**: Segmentation of thresholded heatmaps. Models: Unet (up-sampling and up-conv)

### Diffusion

#### Autoencoder
The initial step entails training the *Variational Autoencoder* model with the objective of reconstructing images from the training set using a latent space. The model can be either a Variational Autoencoder (VAE) or a Vector Quantized Variational Autoencoder (VQVAE)
To train autoecoder model use the following comand:

```bash
python train_vae.py --data eco
```

`infer_vae.py` file enable you to visualize some recostruction and the latent space. The trained model is saved during the training in a `trial_n` folder.

```bash
python infer_vae.py --data eco --triall trial_#n
```

##### Evaluation
To evaluate how the VAE encode the information in the latent space, use this code:

```bash
python investigate_vae.py --data eco --trial trial_#8
```

#### Latent Diffusion Model

The *Latent Diffusion Model (LDM)* is a denoising model that operates on the latent space extracted from the autoencoder, rather than on the pixel space like DDPM. It's important to note that for training the LDM, the training set must be the **same** as that used for the autoencoder. Pay attention to the batch size during training.

Run the following code for the training

```bash
python train_ldm --data eco --vae_train train_#n
```
where `--vae_train` is related to the train_#n folder of trained VAE model.

To sample a set of synthetic images use the following comand line

```bash
python tools/sample_ldm.py --data eco --trial trial_#n --epoch #_epoch
```

The *condition Latent Diffusion Model (condLDM)* sample a condiotional distrubution given the heatmaps of LVPW, LVID and IVS keypoints. To train the condLDM run the following code

```bash
python tools/train_cond_ldm.py --data eco_image_cond --vae_train train_#n
```

for the sampling:

```bash
python tools/sample_cond_ldm.py --data eco_image_cond --trial trial_#n --epoch #epoch
```

##### Evaluation
The main score to evaluate the quality of generated images is Fréchet Inception Distance (FID). In this work we use the implementation in [official Pythorch implementation](https://github.com/mseitzer/pytorch-fid/tree/master?tab=readme-ov-file):

```bash
python -m pytorch_fid.fid_score path_real path_gen
```







