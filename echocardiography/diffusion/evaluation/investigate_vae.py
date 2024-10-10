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

def get_hypertrophy_from_one_hot(one_hot):
    """
    Get the hypertrophy from the one hot encoding
    """
    one_hot = one_hot.cpu().numpy()
    hypertrophy = np.argmax(one_hot)
    if hypertrophy == 0: return 'Concentric hypertrophy'
    if hypertrophy == 1: return 'Concentric remodeling'
    if hypertrophy == 2: return 'Eccentric hypertrophy'
    if hypertrophy == 3: return 'Normal geometry'


def plot_im_cond_rec(im, rec, cond):
    """
    Plot the image, condition and the reconstructed image
    """
    im = (im + 1) / 2
    rec = (rec + 1) / 2
    fig, ax = plt.subplots(1, 3, figsize=(21, 8), num=get_hypertrophy_from_one_hot(cond), tight_layout=True)
    real_img = (im[0, 0, :, :].cpu().numpy() * 255) + 0.00003
    rec_img = (rec[0, 0, :, :].cpu().numpy() * 255) + 0.00003

    ax[0].imshow(real_img, cmap='gray')
    ax[0].set_title('Original image', fontsize=20)
    ax[0].axis('off')


    ax[1].imshow(rec_img, cmap='gray')
    ax[1].set_title('Reconstructed Image', fontsize=20)
    ax[1].axis('off')

    ax[2].imshow(np.abs(real_img - rec_img) / rec_img, cmap='hot')
    ax[2].set_title('Difference', fontsize=20)

def plot_difference_matrix(original, out_1, out_2, out_3, out_4):
    """
    Plot the difference matrix
    """
    original = (original + 1) / 2
    out_1 = (out_1 + 1) / 2
    out_2 = (out_2 + 1) / 2
    out_3 = (out_3 + 1) / 2
    out_4 = (out_4 + 1) / 2

    generations = [original.cpu().numpy(), out_1.cpu().numpy(), out_2.cpu().numpy(), out_3.cpu().numpy(), out_4.cpu().numpy()]
    generations = [(i * 255) + 0.00003 for i in generations]
    label = ['Original_cond', 'CH', 'CR', 'EC', 'Normal geometry']
    fig, ax = plt.subplots(5, 5, figsize=(30, 20), num='Difference between cond', tight_layout=True)
    for i,data_1 in enumerate(generations):
        for j, data_2 in enumerate(generations):
            ax[i, j].imshow(np.abs(generations[i][0,0,:,:] - generations[j][0,0,:,:])/ generations[0][0,0,:,:], cmap='hot')
            ax[i, j].set_title(f'{label[i]} - {label[j]}', fontsize=20)
            ax[i, j].axis('off')

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
    print('dataset', dataset_config['dataset_batch'])
    data_list = []
    for dataset_batch in dataset_config['dataset_batch']:
        data_batch = im_dataset_cls(split=dataset_config['split'], size=(dataset_config['im_size_h'], dataset_config['im_size_w']),
                            parent_dir=dataset_config['parent_dir'], im_path=dataset_config['im_path'], dataset_batch=dataset_batch , phase=dataset_config['phase'],
                            condition_config=condition_config)
        data_list.append(data_batch)

    data_img = torch.utils.data.ConcatDataset(data_list)
    print('len of the dataset', len(data_img))
    data_loader = DataLoader(data_img, batch_size=1, shuffle=False, num_workers=8)


    num_images = train_config['num_samples']
    ngrid = train_config['num_grid_rows']

    trial_folder = os.path.join(par_dir, dataset_config['name'], trial)
    assert os.listdir(trial_folder), f'No trained model found in trial folder {trial_folder}'
    print(os.listdir(trial_folder))

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
    ## type of condition
    cond_1 = torch.tensor([1, 0, 0, 0]).to(device)
    cond_2 = torch.tensor([0, 1, 0, 0]).to(device)
    cond_3 = torch.tensor([0, 0, 1, 0]).to(device)
    cond_4 = torch.tensor([0, 0, 0, 1]).to(device)

    with torch.no_grad():
        test_recon_losses = []
        test_perceptual_losses = []
        for im, cond in tqdm(data_loader):
            im = im.float().to(device)
            for key in cond.keys(): ## for all the type of condition, we move the  tensor on the device
                cond[key] = cond[key].to(device)

            ## Get the reconstracted image from the VAE/condVAE
            print(type_model)
            # if type_model == 'cond_vae': model_output = vae(im, cond['class'])
            # else: model_output = vae(im)
            # output, encoder_out = model_output

            ## Encoder - get the latent space
            if type_model == 'cond_vae': encoded_output, _ = vae.encode(im, cond['class'])
            else: encoded_output, _ = vae.encode(im)

            ## decoder - get the reconstructed image
            if type_model == 'cond_vae':
                ## output from original condition
                output = torch.clamp(vae.decode(encoded_output, cond['class']),-1., 1.)

                output_1 = torch.clamp(vae.decode(encoded_output, cond_1),-1., 1.)
                output_2 = torch.clamp(vae.decode(encoded_output, cond_2),-1., 1.)
                output_3 = torch.clamp(vae.decode(encoded_output, cond_3),-1., 1.)
                output_4 = torch.clamp(vae.decode(encoded_output, cond_4),-1., 1.)
            else:
                output = vae.decode(encoded_output)

            recon_loss = recon_criterion(output, im)
            encoded_output = torch.clamp(encoded_output, -1., 1.)
            encoded_output = (encoded_output + 1) / 2

            if show_plot:
                # plot_image_latent(im, encoded_output)
                plot_im_cond_rec(im, output, cond['class'])
                plot_im_cond_rec(im, output_1, cond_1)
                plot_im_cond_rec(im, output_2, cond_2)
                plot_im_cond_rec(im, output_3, cond_3)
                plot_im_cond_rec(im, output_4, cond_4)
                plot_difference_matrix(output, output_1, output_2, output_3, output_4)
                plt.show()

            ## calculate the perceptual loss

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
    parser = argparse.ArgumentParser(description='Invastigate the latent space')
    parser.add_argument('--save_folder', type=str, default='trained_model', help='folder to save the model, default = trained_model')
    parser.add_argument('--trial', type=str, default='trial_1', help='trial name for saving the model, it is the trial folde that contain the VAE model')
    parser.add_argument('--show_plot', action='store_true', help="show the latent space imgs, default=False")

    args = parser.parse_args()

    # current_directory = os.path.dirname(__file__)
    # par_dir = os.path.dirname(current_directory)
    # eco_dir = os.path.join(par_dir, 'trained_model', 'eco')

    # experiment_dir = os.path.join(eco_dir, args.trial)
    # if 'vae' in os.listdir(experiment_dir): configuration = os.path.join(experiment_dir, 'vae', 'config.yaml')
    # if 'vqvae' in os.listdir(experiment_dir): configuration = os.path.join(experiment_dir, 'vqvae', 'config.yaml')
    # if 'cond_vae' in os.listdir(experiment_dir): configuration = os.path.join(experiment_dir, 'cond_vae', 'config.yaml')

    # config = os.path.join(experiment_dir, 'config.yaml')
    # save_folder = os.path.join(par_dir, 'trained_model', args.trial)
    # infer(par_dir = par_dir, conf = configuration, trial = args.trial, show_plot=args.show_plot)

    ###########################
    args = parser.parse_args()

    experiment_dir = os.path.join(args.save_folder, 'eco', args.trial)
    if 'vae' in os.listdir(experiment_dir): config = os.path.join(experiment_dir, 'vae', 'config.yaml')
    if 'vqvae' in os.listdir(experiment_dir): config = os.path.join(experiment_dir, 'vqvae', 'config.yaml')
    if 'cond_vae' in os.listdir(experiment_dir): config = os.path.join(experiment_dir, 'cond_vae', 'config.yaml')


    infer(par_dir = args.save_folder, conf=config, trial=args.trial, show_plot=args.show_plot)
    plt.show()