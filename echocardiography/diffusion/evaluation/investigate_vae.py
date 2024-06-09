"""
Invastigate haw the VAE encode the PLAX, in other words, is the latent space of the VAE able to capture the PLAX information?
"""
import argparse
import glob
import os
import pickle

import torch
import torchvision
import yaml
import numpy as np
from torch.utils.data.dataloader import DataLoader
from torchvision.utils import make_grid
from tqdm import tqdm
import matplotlib.pyplot as plt

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from echocardiography.diffusion.dataset.dataset import CelebDataset, MnistDataset, EcoDataset
from echocardiography.diffusion.models.unet_cond_base import Unet, get_config_value
from echocardiography.diffusion.models.vqvae import VQVAE
from echocardiography.diffusion.models.vae import VAE
from echocardiography.diffusion.models.cond_vae import condVAE
from echocardiography.diffusion.models.lpips import LPIPS


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_best_model(trial_folder):
    """
    Get the best model from the trial folder
    """
    best_model = 0

    # in the folder give me only the file with extention '.pth'
    for i in os.listdir(trial_folder):
        if '.pth' in i and i.split('_')[0] == 'vae':
            model = i.split('_')[-1].split('.')[0]
            if int(model) > best_model:
                best_model = int(model)
    return best_model


def plot_image_latent(image, latent):
    """
    Plot the latent space of the image
    """
    latent_dict = {}

    ## plot the latent space
    latent_original = latent[0,:,:,:].cpu().permute(1,2,0).numpy()
    print(latent_original.shape)
    fig, ax = plt.subplots(1, 1, figsize=(8, 8), tight_layout=True)
    ax.set_title('Image - latent space', fontsize=20)
    ax.axis('off')
    ax.imshow(latent_original, cmap='jet')

    ## plot only the original images
    fig, ax = plt.subplots(1, 1, figsize=(8, 8), tight_layout=True)
    ax.set_title('Image', fontsize=20)
    ax.axis('off')
    ax.imshow(image[0, 0, :, :].cpu().numpy(), cmap='gray')
    


    laten_img = torchvision.transforms.Resize((image.shape[2], image.shape[3]))(latent)
    
    fig, ax = plt.subplots(1, laten_img.shape[1], figsize=(21, 8), tight_layout=True)
    for i in range(laten_img.shape[1]):
        ax[i].set_title(f'Image - ch latent {i}', fontsize=20)
        ax[i].imshow(image[0, 0, :, :].cpu().numpy(), cmap='gray')
        ax[i].imshow(laten_img[0, i, :, :].cpu().numpy(), cmap='jet', alpha=0.2)

    for axis in ax:
        axis.axis('off')

    ## 
    if latent.shape[1] == 4:
        latent_1 = laten_img[0, 0, :, :].cpu().numpy() + laten_img[0, 2, :, :].cpu().numpy() 
        latent_2 = laten_img[0, 1, :, :].cpu().numpy() + laten_img[0, 3, :, :].cpu().numpy()
        #normalize the latent in the range 0-1
        latent_1 = (latent_1 - np.min(latent_1)) / (np.max(latent_1) - np.min(latent_1))
        latent_2 = (latent_2 - np.min(latent_2)) / (np.max(latent_2) - np.min(latent_2))

        fig, ax = plt.subplots(1, 2, figsize=(21, 8), tight_layout=True)
        ax[0].imshow(image[0, 0, :, :].cpu().numpy(), cmap='gray')
        ax[0].imshow(latent_1, cmap='jet', alpha=0.2)
        ax[0].set_title('Image - ch latent 0 + ch latent 2', fontsize=20)
        ax[0].axis('off')

        ax[1].imshow(image[0, 0, :, :].cpu().numpy(), cmap='gray')
        ax[1].imshow(latent_2, cmap='jet', alpha=0.2)
        ax[1].set_title('Image - ch latent 1 + ch latent 3', fontsize=20)
        ax[1].axis('off')

    plt.show()

def infer(par_dir, conf, trial, show_plot=False):
    ######## Read the config file #######
    with open(conf, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)
    
    autoencoder_model_config = config['autoencoder_params']
    train_config = config['train_params']
    diffusion_config = config['diffusion_params']
    dataset_config = config['dataset_params']
    diffusion_model_config = config['ldm_params']
    autoencoder_model_config = config['autoencoder_params']
    train_config = config['train_params']
    condition_config = get_config_value(autoencoder_model_config, key='condition_config', default_value=None)
    if condition_config is not None:
        assert 'condition_types' in condition_config, \
            "condition type missing in conditioning config"
        condition_types = condition_config['condition_types']
    
    #############################

    # Create the dataset
    im_dataset_cls = {
        'mnist': MnistDataset,
        'celebhq': CelebDataset,
        'eco': EcoDataset,
    }.get(dataset_config['name'])

    # Create the dataset and dataloader
    data = im_dataset_cls(split='train', size=(dataset_config['im_size'], dataset_config['im_size']), 
                              im_path=dataset_config['im_path'], dataset_batch=dataset_config['dataset_batch'], phase=dataset_config['phase'],
                              dataset_batch_regression=dataset_config['dataset_batch_regression'], trial=dataset_config['trial'],
                              condition_config=condition_config)
    data_loader = DataLoader(data, batch_size=1, shuffle=False, num_workers=4)

    num_images = train_config['num_samples']
    ngrid = train_config['num_grid_rows']
    
    trial_folder = os.path.join(par_dir, 'trained_model', dataset_config['name'], trial)
    assert os.listdir(trial_folder), f'No trained model found in trial folder {trial_folder}'
    
    if 'cond_vae' in os.listdir(trial_folder):
        type_model = 'cond_vae'
        print(f'Load trained {os.listdir(trial_folder)[0]} model')
        best_model = get_best_model(os.path.join(trial_folder,'cond_vae'))
        print(f'best model  epoch {best_model}')
        vae = condVAE(im_channels=dataset_config['im_channels'], model_config=autoencoder_model_config, condition_config=condition_config).to(device)
        vae.eval()
        vae.load_state_dict(torch.load(os.path.join(trial_folder, 'cond_vae', f'vae_best_{best_model}.pth'), map_location=device))

    if 'vae' in os.listdir(trial_folder):
        type_model = 'vae'
        print(f'Load trained {os.listdir(trial_folder)[0]} model')
        best_model = get_best_model(os.path.join(trial_folder,'vae'))
        print(f'best model  epoch {best_model}')
        vae = VAE(im_channels=dataset_config['im_channels'], model_config=autoencoder_model_config).to(device)
        vae.eval()
        vae.load_state_dict(torch.load(os.path.join(trial_folder, 'vae', f'vae_best_{best_model}.pth'), map_location=device))

    if 'vqvae' in os.listdir(trial_folder):
        print(f'Load trained {os.listdir(trial_folder)[0]} model')
        vae = VQVAE(im_channels=dataset_config['im_channels'], model_config=autoencoder_model_config).to(device)
        vae.eval()
        vae.load_state_dict(torch.load(os.path.join(trial_folder, 'vqvae', 'vqvae.pth'),map_location=device))

    ## evalaute the recon loss on test set
    recon_criterion = torch.nn.MSELoss()            # L1/L2 loss for Reconstruction
    lpips_model = LPIPS().eval().to(device) 
    encoded_output_list = []
    hypertrophy_list = []
    with torch.no_grad():
        test_recon_losses = []
        test_perceptual_losses = []
        for im, cond in tqdm(data_loader):
            im = im.float().to(device)
            for key in cond.keys(): ## for all the type of condition, we move the  tensor on the device
                cond[key] = cond[key].to(device)

            ## add the condition only for the conditional model
            if type_model == 'cond_vae': model_output = vae(im, cond['class'])
            else: model_output = vae(im)

            output, encoder_out = model_output
            mean, logvar = torch.chunk(encoder_out, 2, dim=1) 

            if type_model == 'cond_vae': encoded_output, _ = vae.encode(im, cond['class'])
            else: encoded_output, _ = vae.encode(im)
            recon_loss = recon_criterion(output, im)
            encoded_output = torch.clamp(encoded_output, -1., 1.)
            encoded_output = (encoded_output + 1) / 2
            
            if show_plot: plot_image_latent(im, encoded_output)
            
            encoded_output = encoded_output[0,:,:,:].flatten()
            encoded_output_list.append(encoded_output.cpu().numpy())
            hypertrophy_list.append(np.argmax(cond['class'].cpu().numpy()))
    
    encoded_output_list = np.array(encoded_output_list)

    # Reduce dimensionality with PCA
    print("PCA reduction...")
    pca = PCA(n_components=3)
    encoded_output_pca = pca.fit_transform(encoded_output_list)
    encoded_output_pca = (encoded_output_pca - np.min(encoded_output_pca)) / (np.max(encoded_output_pca) - np.min(encoded_output_pca))
    explained_variance_ratio = pca.explained_variance_ratio_
    print("Explained Variance Ratio PCA:", explained_variance_ratio)
    print()

    # Reduce dimensionality with Linear Discriminant Analysis
    print("LDA reduction...")
    lda = LinearDiscriminantAnalysis(n_components=3)
    encoded_output_lda = lda.fit_transform(encoded_output_list, hypertrophy_list)
    encoded_output_lda = (encoded_output_lda - np.min(encoded_output_lda)) / (np.max(encoded_output_lda) - np.min(encoded_output_lda))
    print("Explained Variance Ratio LDA:", lda.explained_variance_ratio_)
    print()

    ## reduce the dimensionality of the latent space with tsne
    print("TSNE reduction...")
    tsne = TSNE(n_components=2, random_state=0)
    encoded_output_tsne = tsne.fit_transform(encoded_output_list)
    encoded_output_tsne = (encoded_output_tsne - np.min(encoded_output_tsne)) / (np.max(encoded_output_tsne) - np.min(encoded_output_tsne))
    print(encoded_output_tsne.shape)
    print()



    ######################## PLOT #################################################################
    # plot the 3D scatter plot of each point
    fig = plt.figure(figsize=(10,10), num=f'{type_model} - 3D PCA of latent space of PLAX')
    ax = fig.add_subplot(111, projection='3d')
    color_dict = {0: 'red', 1: 'orange', 2: 'olive', 3: 'green'}
    color = [color_dict[i] for i in hypertrophy_list]
    ax.scatter(encoded_output_pca[:,0], encoded_output_pca[:,1], encoded_output_pca[:,2], c=color)
    ax.set_xlabel('PCA 1', fontsize=16)
    ax.set_ylabel('PCA 2', fontsize=16)
    ax.set_zlabel('PCA 3', fontsize=16)
    ax.tick_params(labelsize=14)


    # plot the 2D scatter plot of each point
    plt.figure(figsize=(8,8), num=f'{type_model} - PCA of latent space of PLAX')
    plt.scatter(encoded_output_pca[:,0], encoded_output_pca[:,1], c=color)
    plt.xlabel('PCA 1', fontsize=20)
    plt.ylabel('PCA 2', fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    # plt.show()

    ## plt the 3d scatter plot of LDA 
    fig = plt.figure(figsize=(8,8), num=f'{type_model} - 3D LDA of latent space of PLAX')
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(encoded_output_lda[:,0], encoded_output_lda[:,1], encoded_output_lda[:,2], c=color)
    ax.set_xlabel('LDA 1', fontsize=20)
    ax.set_ylabel('LDA 2', fontsize=20)
    ax.set_zlabel('LDA 3', fontsize=20)
    ax.tick_params(labelsize=18)

    # plot the 2D scatter plot of each point
    plt.figure(figsize=(8,8), num=f'{type_model} - LDA of latent space of PLAX')
    # color = [f'C{i}' for i in hypertrophy_list]
    plt.scatter(encoded_output_lda[:,0], encoded_output_lda[:,1], c=color)
    plt.xlabel('LDA 1', fontsize=20)
    plt.ylabel('LDA 2', fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    ## plt the 2d scatter plot of tsne
    plt.figure(figsize=(8,8), num=f'{type_model} - TSNE of latent space of PLAX')
    plt.scatter(encoded_output_tsne[:,0], encoded_output_tsne[:,1], c=color)
    plt.xlabel('TSNE 1', fontsize=20)
    plt.ylabel('TSNE 2', fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.show()


    ##################################################################################################

        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for vq vae inference')
    parser.add_argument('--data', type=str, default='eco', help='type of the data, mnist, celebhq, eco, eco_image_cond')  
    parser.add_argument('--trial', type=str, default='trial_1', help='trial name for saving the model')
    parser.add_argument('--show_plot', action='store_true', help="show the latent space imgs, default=False")
    
    args = parser.parse_args()

    current_directory = os.path.dirname(__file__)
    par_dir = os.path.dirname(current_directory)
    configuration = os.path.join(par_dir, 'conf', f'{args.data}.yaml')

    save_folder = os.path.join(par_dir, 'trained_model', args.trial)
    infer(par_dir = par_dir, conf = configuration, trial = args.trial, show_plot=args.show_plot)