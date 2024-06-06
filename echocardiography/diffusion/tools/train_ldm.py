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
from echocardiography.diffusion.models.vae import VAE 
from echocardiography.diffusion.dataset.dataset import MnistDataset, EcoDataset, CelebDataset
from echocardiography.diffusion.tools.infer_vae import get_best_model
from torch.utils.data import DataLoader
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()

def train(par_dir, conf, trial):
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
    data = im_dataset_cls(split=dataset_config['split'], size=(dataset_config['im_size'], dataset_config['im_size']),
                         im_path=dataset_config['im_path'], dataset_batch=dataset_config['dataset_batch'] , phase=dataset_config['phase'])
    data_loader = DataLoader(data, batch_size=train_config['ldm_batch_size'], shuffle=True, num_workers=8)
 
    # Create the model and scheduler
    scheduler = LinearNoiseScheduler(num_timesteps=diffusion_config['num_timesteps'],
                                     beta_start=diffusion_config['beta_start'],
                                     beta_end=diffusion_config['beta_end'])

    model = Unet(im_channels=autoencoder_model_config['z_channels'], model_config=diffusion_model_config).to(device)
    model.train()

    ## get the trained encoder model 
     ## generate save folder
    # save_folder = 'prova_ldm_vqvae_tools_celebhq'
    # if not os.path.exists(save_folder):
    #     os.mkdir(save_folder)

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

    save_folder = os.path.join(trial_folder, 'ldm_1')
    if not os.path.exists(save_folder):
        save_folder = os.path.join(trial_folder, 'ldm_1')
        os.makedirs(save_folder)
    else:
        ## count how many folder start with ldm 
        count = 0
        for folder in os.listdir(trial_folder):
            if folder.startswith('ldm'):
                count += 1
        save_folder = os.path.join(trial_folder, f'ldm_{count+1}')
        os.makedirs(save_folder)

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

        # Save the model
        if (epoch_idx+1) % train_config['save_frequency'] == 0:
            torch.save(model.state_dict(), os.path.join(save_folder, f'ldm_{epoch_idx+1}.pth'))
    
    print('Done Training ...')
    ## save the config file
    with open(os.path.join(save_folder, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train unconditional LDM with VQVAE')
    parser.add_argument('--data', type=str, default='eco', help='type of the data, mnist, celebhq, eco')
    parser.add_argument('--trial', type=str, default='trial_1', help='trial name for the trained VAE  model')
    args = parser.parse_args()

    current_directory = os.path.dirname(__file__)
    par_dir = os.path.dirname(current_directory)
    configuration = os.path.join(par_dir, 'conf', f'{args.data}.yaml')

    save_folder = os.path.join(par_dir, 'trained_model', args.trial)
    train(par_dir = par_dir,
        conf = configuration, 
        trial = args.trial)