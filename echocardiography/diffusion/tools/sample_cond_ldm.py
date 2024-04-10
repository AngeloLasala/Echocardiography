"""
Sample from trained conditional latent diffusion model
"""
import numpy as np
import torch
import random
import torchvision
import argparse
import yaml
import os
from torchvision.utils import make_grid
from PIL import Image
from tqdm import tqdm

from echocardiography.diffusion.models.unet_cond_base import Unet, get_config_value
from echocardiography.diffusion.models.vqvae import VQVAE
from echocardiography.diffusion.models.vae import VAE
from echocardiography.diffusion.scheduler import LinearNoiseScheduler
from echocardiography.diffusion.dataset.dataset import MnistDataset, EcoDataset, CelebDataset
from echocardiography.diffusion.tools.infer_vae import get_best_model
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def sample(model, scheduler, train_config, diffusion_model_config, condition_config,
           autoencoder_model_config, diffusion_config, dataset_config, vae, save_folder):
    """
    Sample stepwise by going backward one timestep at a time.
    We save the x0 predictions
    """
    im_size = dataset_config['im_size'] // 2 ** sum(autoencoder_model_config['down_sample'])
    
    ########### Sample random noise latent ##########
    # For now fixing generation with one sample
    xt = torch.randn((1,
                      autoencoder_model_config['z_channels'],
                      im_size,
                      im_size)).to(device)
    ###############################################
    
    ############ Create Conditional input ###############
    ## FUTURE WORK: Add text-like conditioning
    # Here put the text-like condiction with cross-attetion conditioning
    # ...
    # ...
    # ...   


    # Get the spatial conditional mask, i.e. the heatmaps
    im_dataset_cls = {
        'mnist': MnistDataset,
        'celebhq': CelebDataset,
        'eco': EcoDataset,
    }.get(dataset_config['name'])


    data_img = im_dataset_cls(split=dataset_config['split_val'], size=(dataset_config['im_size'], dataset_config['im_size']), 
                              im_path=dataset_config['im_path'], dataset_batch=dataset_config['dataset_batch'], phase=dataset_config['phase'],
                              dataset_batch_regression=dataset_config['dataset_batch_regression'], trial=dataset_config['trial'],
                              condition_config=condition_config)

    print('xt shape', xt.shape)
    mask_idx = random.randint(0, len(data_img))
    mask = data_img[0][1]['image'].unsqueeze(0).to(device)

    # plt image and mask
    plt.figure()
    # print the mask cobvertd in to the numpy
    plt.imshow(mask.squeeze().cpu().numpy()[0], cmap='jet')
    print(f'Using mask index {mask_idx}')
    plt.show()
    
    uncond_input = {
        # 'text': empty_text_embed,
        'image': torch.zeros_like(mask)
    }
    cond_input = {
        # 'text': text_prompt_embed,
        'image': mask
    }
    # ###############################################
    
    
    ################# Sampling Loop ########################
    for i in tqdm(reversed(range(diffusion_config['num_timesteps']))):
        # Get prediction of noise
        t = (torch.ones((xt.shape[0],)) * i).long().to(device)
        noise_pred_cond = model(xt, t, cond_input)
        
        noise_pred = noise_pred_cond
        
        # Use scheduler to get x0 and xt-1
        xt, x0_pred = scheduler.sample_prev_timestep(xt, noise_pred, torch.as_tensor(i).to(device))
        
        # Save x0
        if i == 0:
            # Decode ONLY the final image to save time
            ims = vae.decode(xt)
        else:
            ims = x0_pred
        
        ims = torch.clamp(ims, -1., 1.).detach().cpu()
        ims = (ims + 1) / 2
        grid = make_grid(ims, nrow=10)
        img = torchvision.transforms.ToPILImage()(grid)
        
        img.save(os.path.join(save_folder, 'x0_{}.png'.format(i)))
        img.close()
    #############################################################

def infer(par_dir, conf, trial, epoch):
    # Read the config file #
    with open(conf, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    
    diffusion_config = config['diffusion_params']
    dataset_config = config['dataset_params']
    diffusion_model_config = config['ldm_params']
    autoencoder_model_config = config['autoencoder_params']
    train_config = config['train_params']
    condition_config = get_config_value(diffusion_model_config, key='condition_config', default_value=None)
    if condition_config is not None:
        assert 'condition_types' in condition_config, \
            "condition type missing in conditioning config"
        condition_types = condition_config['condition_types']

    # Set the desired seed value #
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)
    #############################
    
    ########## Create the noise scheduler #############
    scheduler = LinearNoiseScheduler(num_timesteps=diffusion_config['num_timesteps'],
                                     beta_start=diffusion_config['beta_start'],
                                     beta_end=diffusion_config['beta_end'])
    ###############################################
    
    
    ############# Load tokenizer and text model #################
    # Here the section of cross-attention conditioning
    # ...
    # ...
    # ...
    ###############################################
    
    ########## Load Unet #############
    model = Unet(im_channels=autoencoder_model_config['z_channels'], model_config=diffusion_model_config).to(device)
    model.eval()
    model_dir = os.path.join(par_dir, 'trained_model', dataset_config['name'], trial, 'cond_ldm')
    model.load_state_dict(torch.load(os.path.join(model_dir, f'ldm_{epoch}.pth'),map_location=device), strict=False)
   
    #####################################
    
    ########## Load AUTOENCODER #############
    trial_folder = os.path.join(par_dir, 'trained_model', dataset_config['name'], trial)
    assert os.listdir(trial_folder), f'No trained model found in trial folder {trial_folder}'
    print(os.listdir(trial_folder))
    if 'vae' in os.listdir(trial_folder):
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
    #####################################

    ######### Create output directories #############
    save_folder = os.path.join(model_dir, f'samples_ep_{epoch}')
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    else:
        overwrite = input("The save folder already exists. Do you want to overwrite it? (y/n): ")
        if overwrite.lower() != 'y':
            print("Training aborted.")
            exit()
    
    ######## Sample from the model 
    with torch.no_grad():
        sample(model, scheduler, train_config, diffusion_model_config, condition_config,
               autoencoder_model_config, diffusion_config, dataset_config, vae, save_folder)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train unconditional LDM with VQVAE')
    parser.add_argument('--data', type=str, default='mnist', help='type of the data, mnist, celebhq, eco') 
    parser.add_argument('--trial', type=str, default='trial_1', help='trial name for saving the model')
    parser.add_argument('--epoch', type=int, default=49, help='epoch to sample')

    args = parser.parse_args()

    current_directory = os.path.dirname(__file__)
    par_dir = os.path.dirname(current_directory)
    configuration = os.path.join(par_dir, 'conf', f'{args.data}.yaml')
    save_folder = os.path.join(par_dir, 'trained_model', args.trial)
    infer(par_dir = par_dir, conf=configuration, trial=args.trial, epoch=args.epoch)
    plt.show()

