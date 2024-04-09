"""
Inferecnce script for Autoencoder for LDM. it help to understand the latent space of the data
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

from echocardiography.diffusion.dataset.dataset import CelebDataset, MnistDataset, EcoDataset
from echocardiography.diffusion.models.vqvae import VQVAE
from echocardiography.diffusion.models.vae import VAE
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

def infer(par_dir, conf, trial):
    ######## Read the config file #######
    with open(conf, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)
    
    dataset_config = config['dataset_params']
    autoencoder_model_config = config['autoencoder_params']
    train_config = config['train_params']
    
    # Create the dataset
    im_dataset_cls = {
        'mnist': MnistDataset,
        'celebhq': CelebDataset,
        'eco': EcoDataset,
    }.get(dataset_config['name'])
    
    # Create the dataset and dataloader
    ## updating, not using the train but the test set
    data = im_dataset_cls(split='test', size=(dataset_config['im_size'], dataset_config['im_size']), 
                          im_path=dataset_config['im_path'], dataset_batch=dataset_config['dataset_batch'] , phase=dataset_config['phase'])
    data_loader = DataLoader(data, batch_size=1, shuffle=False, num_workers=4)

    num_images = train_config['num_samples']
    ngrid = train_config['num_grid_rows']
    
    idxs = torch.randint(0, len(data) - 1, (num_images,))
    ims = torch.cat([data[idx][None, :] for idx in idxs]).float()
    ims = ims.to(device)

    trial_folder = os.path.join(par_dir, 'trained_model', dataset_config['name'], trial)
    assert os.listdir(trial_folder), f'No trained model found in trial folder {trial_folder}'
    
    

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

    ## evalaute the recon loss on test set
    recon_criterion = torch.nn.MSELoss()            # L1/L2 loss for Reconstruction
    lpips_model = LPIPS().eval().to(device) 
    with torch.no_grad():
        test_recon_losses = []
        test_perceptual_losses = []
        for im in tqdm(data_loader):
            im = im.float().to(device)
            model_output = vae(im)
            output, encoder_out = model_output
            mean, logvar = torch.chunk(encoder_out, 2, dim=1) 

            test_recon_loss = recon_criterion(output, im)
            test_recon_losses.append(test_recon_loss.item())

            test_lpips_loss = torch.mean(lpips_model(output, im)) 
            test_perceptual_losses.append(train_config['perceptual_weight'] * test_lpips_loss.item())
    import matplotlib.pyplot as plt


    ## plt the histogram of the losses
    fig, ax = plt.subplots(1, 2, figsize=(12, 8), num='test metrics', tight_layout=True)
    ax[0].hist(test_recon_losses, color='blue', alpha=0.7, label='MAE')
    ax[0].grid('dotted')
    ax[0].set_xlabel('MAE', fontsize=20)
    ax[0].set_title(f'MAE: mean={np.array(test_recon_losses).mean():.4f}, std={np.array(test_recon_losses).std():.4f}', fontsize=20)
    ## ad vertical line for the mean
    ax[0].axvline(np.array(test_recon_losses).mean(), color='C4', linestyle='dashed', linewidth=5, label=f'mean {np.array(test_recon_losses).mean():.4f}')
    ax[0].axvline(np.median(test_recon_losses), color='C5', linestyle='dashed', linewidth=5, label=f'median {np.median(test_recon_losses):.4f}')
    ax[0].legend(fontsize=16)
    ax[1].hist(test_perceptual_losses, color='red', alpha=0.7, label='Perceptual metric')
    ax[1].grid('dotted')
    ax[1].set_xlabel('Perceptual metric', fontsize=20)
    ax[1].set_title(f'Perceptual metric: mean={np.array(test_perceptual_losses).mean():.3f}, std={np.array(test_perceptual_losses).std():.3f}', fontsize=20)
    ax[1].axvline(np.array(test_perceptual_losses).mean(), color='C4', linestyle='dashed', linewidth=5, label=f'mean {np.array(test_perceptual_losses).mean():.3f}')
    ax[1].axvline(np.median(test_perceptual_losses), color='C5', linestyle='dashed', linewidth=5, label=f'median {np.median(test_perceptual_losses):.3f}')
    ax[1].legend(fontsize=18)
    ## change the label fontsize
    for a in ax:
        a.tick_params(axis='both', which='major', labelsize=18)
    plt.savefig(os.path.join(trial_folder, 'vae', 'samples', 'test_metrics.png'))
    plt.show()

    ## save this ijmage with a name
    save_folder = os.path.join(trial_folder, 'vae', 'samples')
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    with torch.no_grad():
        
        encoded_output, _ = vae.encode(ims)
        decoded_output = vae.decode(encoded_output)
        encoded_output = torch.clamp(encoded_output, -1., 1.)
        encoded_output = (encoded_output + 1) / 2
        decoded_output = torch.clamp(decoded_output, -1., 1.)
        decoded_output = (decoded_output + 1) / 2
        ims = (ims + 1) / 2

        encoder_grid = make_grid(encoded_output.cpu(), nrow=ngrid)
        decoder_grid = make_grid(decoded_output.cpu(), nrow=ngrid)
        input_grid = make_grid(ims.cpu(), nrow=ngrid)
        encoder_grid = torchvision.transforms.ToPILImage()(encoder_grid)
        decoder_grid = torchvision.transforms.ToPILImage()(decoder_grid)
        input_grid = torchvision.transforms.ToPILImage()(input_grid)
        
        
        input_grid.save(os.path.join(save_folder, 'input_samples.png'))
        encoder_grid.save(os.path.join(save_folder, 'encoded_samples.png'))
        decoder_grid.save(os.path.join(save_folder, 'reconstructed_samples.png'))

        input_grid.show()
        encoder_grid.show()
        decoder_grid.show()
            

        ## THIS PART IS FOR THE VQVAE   
        # input_grid.save(os.path.join(train_config['task_name'], 'input_samples.png'))
        # encoder_grid.save(os.path.join(train_config['task_name'], 'encoded_samples.png'))
        # decoder_grid.save(os.path.join(train_config['task_name'], 'reconstructed_samples.png'))
        
        # if train_config['save_latents']:
        #     # save Latents (but in a very unoptimized way)
        #     latent_path = os.path.join(train_config['task_name'], train_config['vqvae_latent_dir_name'])
        #     latent_fnames = glob.glob(os.path.join(train_config['task_name'], train_config['vqvae_latent_dir_name'],
        #                                            '*.pkl'))
        #     assert len(latent_fnames) == 0, 'Latents already present. Delete all latent files and re-run'
        #     if not os.path.exists(latent_path):
        #         os.mkdir(latent_path)
        #     print('Saving Latents for {}'.format(dataset_config['name']))
            
        #     fname_latent_map = {}
        #     part_count = 0
        #     count = 0
        #     for idx, im in enumerate(tqdm(data_loader)):
        #         encoded_output, _ = model.encode(im.float().to(device))
        #         fname_latent_map[data.images[idx]] = encoded_output.cpu()
        #         # Save latents every 1000 images
        #         if (count+1) % 1000 == 0:
        #             pickle.dump(fname_latent_map, open(os.path.join(latent_path,
        #                                                             '{}.pkl'.format(part_count)), 'wb'))
        #             part_count += 1
        #             fname_latent_map = {}
        #         count += 1
        #     if len(fname_latent_map) > 0:
        #         pickle.dump(fname_latent_map, open(os.path.join(latent_path,
        #                                            '{}.pkl'.format(part_count)), 'wb'))
        #     print('Done saving latents')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for vq vae inference')
    parser.add_argument('--data', type=str, default='eco', help='type of the data, mnist, celebhq, eco')  
    parser.add_argument('--trial', type=str, default='trial_1', help='trial name for saving the model')
    args = parser.parse_args()

    current_directory = os.path.dirname(__file__)
    par_dir = os.path.dirname(current_directory)
    configuration = os.path.join(par_dir, 'conf', f'{args.data}.yaml')

    save_folder = os.path.join(par_dir, 'trained_model', args.trial)
    infer(par_dir = par_dir, conf = configuration, trial = args.trial)