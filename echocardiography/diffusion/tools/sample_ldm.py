"""
Sample new data from the trained LDM-VQVAE model
"""
import torch
import torchvision
import argparse
import yaml
import os
from torchvision.utils import make_grid
from PIL import Image
from tqdm import tqdm
from echocardiography.diffusion.models.unet_base import Unet
from echocardiography.diffusion.scheduler import LinearNoiseScheduler
from echocardiography.diffusion.dataset.dataset import MnistDataset, EcoDataset, CelebDataset
from echocardiography.diffusion.models.vqvae import VQVAE
from echocardiography.diffusion.models.vae import VAE
from echocardiography.diffusion.tools.infer_vae import get_best_model
import cv2


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def sample(model, scheduler, train_config, diffusion_model_config,
               autoencoder_model_config, diffusion_config, dataset_config, vae, save_folder):
    """
    Sample stepwise by going backward one timestep at a time.
    We save the x0 predictions
    """
    im_size = dataset_config['im_size'] // 2**sum(autoencoder_model_config['down_sample'])

    # Get the spatial conditional mask, i.e. the heatmaps
    im_dataset_cls = {
        'mnist': MnistDataset,
        'celebhq': CelebDataset,
        'eco': EcoDataset,
    }.get(dataset_config['name'])


    data_img = im_dataset_cls(split=dataset_config['split_val'], size=(dataset_config['im_size'], dataset_config['im_size']), 
                              im_path=dataset_config['im_path'], dataset_batch=dataset_config['dataset_batch'], phase=dataset_config['phase'])
    
    for jj in range(len(data_img)):
        xt = torch.randn((1,
                        autoencoder_model_config['z_channels'],
                        im_size,
                        im_size)).to(device)
        for i in tqdm(reversed(range(diffusion_config['num_timesteps']))):
            # Get prediction of noise
            noise_pred = model(xt, torch.as_tensor(i).unsqueeze(0).to(device))
            
            # Use scheduler to get x0 and xt-1
            xt, x0_pred = scheduler.sample_prev_timestep(xt, noise_pred, torch.as_tensor(i).to(device))

            # Save x0
            #ims = torch.clamp(xt, -1., 1.).detach().cpu()
            if i == 0:
                # Decode ONLY the final iamge to save time
                ims = vae.decode(xt)
            else:
                ims = xt
            
            ims = torch.clamp(ims, -1., 1.).detach().cpu()
            ims = (ims + 1) / 2

        cv2.imwrite(os.path.join(save_folder, f'x0_{jj}.png'), ims[i].numpy()[0]*255)

    ## OLD CODE - EXAMPLE OF GENERATION FOR SINGLE IMAGE #####################################À
    if False: 
        xt = torch.randn((1,
                        autoencoder_model_config['z_channels'],
                        im_size,
                        im_size)).to(device)

        save_count = 0
        for i in tqdm(reversed(range(diffusion_config['num_timesteps']))):
            # Get prediction of noise
            noise_pred = model(xt, torch.as_tensor(i).unsqueeze(0).to(device))
            
            # Use scheduler to get x0 and xt-1
            xt, x0_pred = scheduler.sample_prev_timestep(xt, noise_pred, torch.as_tensor(i).to(device))

            # Save x0
            #ims = torch.clamp(xt, -1., 1.).detach().cpu()
            if i == 0:
                # Decode ONLY the final iamge to save time
                ims = vae.decode(xt)
            else:
                ims = xt
            
            ims = torch.clamp(ims, -1., 1.).detach().cpu()
            ims = (ims + 1) / 2
            grid = make_grid(ims, nrow=train_config['num_grid_rows'])
            img = torchvision.transforms.ToPILImage()(grid)
            
            img.save(os.path.join(save_folder, 'x0_{}.png'.format(i)))
            img.close()


def infer(par_dir, conf, trial, epoch):
    # Read the config file #
    with open(conf, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)
    ########################
    
    diffusion_config = config['diffusion_params']
    dataset_config = config['dataset_params']
    diffusion_model_config = config['ldm_params']
    autoencoder_model_config = config['autoencoder_params']
    train_config = config['train_params']
    
    # Create the noise scheduler
    scheduler = LinearNoiseScheduler(num_timesteps=diffusion_config['num_timesteps'],
                                     beta_start=diffusion_config['beta_start'],
                                     beta_end=diffusion_config['beta_end'])
    
    # Load the trained models
    model = Unet(im_channels=autoencoder_model_config['z_channels'],
                 model_config=diffusion_model_config).to(device)
    model.eval()
    model_dir = os.path.join(par_dir, 'trained_model', dataset_config['name'], trial, 'ldm')
    model.load_state_dict(torch.load(os.path.join(model_dir, f'ldm_{epoch}.pth'),map_location=device))
    
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
    

    # Create output directories
    save_folder = os.path.join(model_dir, f'samples_ep_{epoch}')
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    else:
        overwrite = input("The save folder already exists. Do you want to overwrite it? (y/n): ")
        if overwrite.lower() != 'y':
            print("Training aborted.")
            exit()


    with torch.no_grad():
        sample(model, scheduler, train_config, diffusion_model_config,
               autoencoder_model_config, diffusion_config, dataset_config, vae, save_folder)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train unconditional LDM with VQVAE')
    parser.add_argument('--data', type=str, default='mnist', help='type of the data, mnist, celebhq, eco') 
    parser.add_argument('--trial', type=str, default='trial_1', help='trial name for the trained model')
    parser.add_argument('--epoch', type=int, default=49, help='epoch of trained LDM model')

    args = parser.parse_args()

    current_directory = os.path.dirname(__file__)
    par_dir = os.path.dirname(current_directory)
    configuration = os.path.join(par_dir, 'conf', f'{args.data}.yaml')
    save_folder = os.path.join(par_dir, 'trained_model', args.trial)
    infer(par_dir = par_dir, conf=configuration, trial=args.trial, epoch=args.epoch)

