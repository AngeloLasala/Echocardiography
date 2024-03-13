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
from torch.utils.data.dataloader import DataLoader
from torchvision.utils import make_grid
from tqdm import tqdm

from echocardiography.diffusion.dataset.dataset import CelebDataset, MnistDataset, EcoDataset
from echocardiography.diffusion.models.vqvae import VQVAE
from echocardiography.diffusion.models.vae import VAE

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    data = im_dataset_cls(split=dataset_config['split'], size=(dataset_config['im_size'], dataset_config['im_size']), im_path=dataset_config['im_path'])
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
        vae = VAE(im_channels=dataset_config['im_channels'], model_config=autoencoder_model_config).to(device)
        vae.eval()
        vae.load_state_dict(torch.load(os.path.join(trial_folder, 'vae', 'vae.pth'), map_location=device))

    if 'vqvae' in os.listdir(trial_folder):
        print(f'Load trained {os.listdir(trial_folder)[0]} model')
        vae = VQVAE(im_channels=dataset_config['im_channels'], model_config=autoencoder_model_config).to(device)
        vae.eval()
        vae.load_state_dict(torch.load(os.path.join(trial_folder, 'vqvae', 'vqvae.pth'),map_location=device))

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
    parser.add_argument('--data', type=str, default='mnist', help='type of the data, mnist, celebhq, eco')  
    parser.add_argument('--trial', type=str, default='trial_1', help='trial name for saving the model')
    args = parser.parse_args()

    current_directory = os.path.dirname(__file__)
    par_dir = os.path.dirname(current_directory)
    configuration = os.path.join(par_dir, 'conf', f'{args.data}.yaml')

    save_folder = os.path.join(par_dir, 'trained_model', args.trial)
    infer(par_dir = par_dir, conf = configuration, trial = args.trial)