"""
Train latent diffusion model with VQ-VAE
"""
import torch
import yaml
import argparse
import os
import numpy as np
from tqdm import tqdm
from torch.optim import Adam
from echocardiography.diffusion.models.unet_base import Unet
from echocardiography.diffusion.scheduler import LinearNoiseScheduler
from echocardiography.diffusion.models.vqvae import VQVAE
from echocardiography.diffusion.dataset.dataset import MnistDataset, EcoDataset, CelebDataset
from torch.utils.data import DataLoader
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()

def train(conf):
   # Read the config file #
    with open(conf, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)
    
    diffusion_config = config['diffusion_params']
    dataset_config = config['dataset_params']
    diffusion_model_config = config['ldm_params']
    autoencoder_model_config = config['autoencoder_params']
    train_config = config['train_params']

    # Set the desired seed value #
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)
    #############################

    
    # Create the dataset
    im_dataset_cls = {
        'mnist': MnistDataset,
        'celebhq': CelebDataset,
        'eco': EcoDataset,
    }.get(dataset_config['name'])

    # Create the dataset and dataloader
    data = im_dataset_cls(split=dataset_config['split'], size=(dataset_config['im_size'], dataset_config['im_size']), im_path=dataset_config['im_path'])
    data_loader = DataLoader(data, batch_size=64, shuffle=True, num_workers=8)
 
    
    ## generate save folder
    save_folder = 'prova_ldm_vqvae_tools_sgsdf'
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    # Create the model and scheduler
    scheduler = LinearNoiseScheduler(num_timesteps=diffusion_config['num_timesteps'],
                                     beta_start=diffusion_config['beta_start'],
                                     beta_end=diffusion_config['beta_end'])

    model = Unet(im_channels=autoencoder_model_config['z_channels'], model_config=diffusion_model_config).to(device)
    model.train()

    print('Loading vqvae model as latents not present')
    vae = VQVAE(im_channels=dataset_config['im_channels'], model_config=autoencoder_model_config).to(device)
    vae.eval()
    vae.load_state_dict(torch.load('prova_vqvae_tools_sgsdf/vqvae_eco.pth',map_location=device))
    
    num_epochs = train_config['ldm_epochs']
    optimizer = Adam(model.parameters(), lr=train_config['ldm_lr'])
    criterion = torch.nn.MSELoss()
    
    # Run training
    print('Start training ...')
    for epoch_idx in range(num_epochs):
        losses = []
        for im in tqdm(data_loader):
            optimizer.zero_grad()
            im = im.float().to(device)
            with torch.no_grad():
                im, _ = vae.encode(im)
            
            # Sample random noise
            noise = torch.randn_like(im).to(device)
            
            # Sample timestep
            t = torch.randint(0, diffusion_config['num_timesteps'], (im.shape[0],)).to(device)
            
            # Add noise to images according to timestep
            noisy_im = scheduler.add_noise(im, noise, t)
            noise_pred = model(noisy_im, t)

            loss = criterion(noise_pred, noise)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
        print(f'Finished epoch:{epoch_idx+1} | Loss : {np.mean(losses):.4f}')
        torch.save(model.state_dict(), os.path.join(save_folder, f'ldm_vqvae_{epoch_idx}.pth'))
    
    print('Done Training ...')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train unconditional LDM with VQVAE')
    parser.add_argument('--data', type=str, default='mnist', help='type of the data, mnist, celebhq, eco')  
    args = parser.parse_args()

    current_directory = os.path.dirname(__file__)
    par_dir = os.path.dirname(current_directory)
    configuration = os.path.join(par_dir, 'conf', f'{args.data}.yaml')
    print(configuration)
    train(conf = configuration)