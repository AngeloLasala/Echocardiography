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
from echocardiography.diffusion.sheduler.scheduler import LinearNoiseScheduler
from echocardiography.diffusion.dataset.dataset import MnistDataset, EcoDataset, CelebDataset
from echocardiography.diffusion.models.vqvae import VQVAE
from echocardiography.diffusion.models.vae import VAE
from echocardiography.diffusion.tools.infer_vae import get_best_model
from torch.utils.data import DataLoader
import cv2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def sample(model, scheduler, train_config, diffusion_model_config,
               autoencoder_model_config, diffusion_config, dataset_config, vae, save_folder):
    """
    Sample stepwise by going backward one timestep at a time.
    We save the x0 predictions
    """
    im_size_h = dataset_config['im_size_h'] // 2**sum(autoencoder_model_config['down_sample'])
    im_size_w = dataset_config['im_size_w'] // 2**sum(autoencoder_model_config['down_sample'])
    print(f'Resolution of latent space [{im_size_h},{im_size_w}]')

    # Get the spatial conditional mask, i.e. the heatmaps
    im_dataset_cls = {
        'mnist': MnistDataset,
        'celebhq': CelebDataset,
        'eco': EcoDataset,
    }.get(dataset_config['name'])


    print('dataset', dataset_config['dataset_batch'])
    data_list = []
    for dataset_batch in dataset_config['dataset_batch']:
        data_batch = im_dataset_cls(split=dataset_config['split_val'], size=(dataset_config['im_size_h'], dataset_config['im_size_w']),
                            parent_dir=dataset_config['parent_dir'], im_path=dataset_config['im_path'], dataset_batch=dataset_batch , phase=dataset_config['phase'])
        data_list.append(data_batch)
    
    data_img = torch.utils.data.ConcatDataset(data_list)
    print('len of the dataset', len(data_img))
    data_loader = DataLoader(data_img, batch_size=train_config['ldm_batch_size_sample'], shuffle=False, num_workers=8)

    for btc, img in enumerate(data_loader):
        print(img.shape)
        xt = torch.randn((img.shape[0],
                        autoencoder_model_config['z_channels'],
                        im_size_h,
                        im_size_w)).to(device)

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
        
        for i in range(ims.shape[0]):
            cv2.imwrite(os.path.join(save_folder, f'x0_{btc * train_config["ldm_batch_size_sample"] + i}.png'), ims[i].numpy()[0]*255)

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


def infer(par_dir, conf, trial, experiment, epoch):
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
    model_dir = os.path.join(par_dir, dataset_config['name'], trial, experiment)
    model.load_state_dict(torch.load(os.path.join(model_dir, f'ldm_{epoch}.pth'),map_location=device))
    
    trial_folder = os.path.join(par_dir, dataset_config['name'], trial)
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
    save_folder = os.path.join(model_dir, f'w_-1.0', f'samples_ep_{epoch}')
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
    parser.add_argument('--data', type=str, default='eco', help='type of the data, mnist, celebhq, eco, eco_image_cond') 
    parser.add_argument('--save_folder', type=str, default='trained_model', help='folder to save the model, default = trained_model')
    parser.add_argument('--trial', type=str, default='trial_1', help='trial name for saving the model, it is the trial folde that contain the VAE model')
    parser.add_argument('--experiment', type=str, default='cond_ldm', help="""name of expermient, it is refed to the type of condition and in general to the 
                                                                              hyperparameters (file .yaml) that is used for the training, it can be cond_ldm, cond_ldm_2, """)
    parser.add_argument('--epoch', type=int, default=100, help='epoch to sample, this is the epoch of cond ldm model')
    # mp.set_start_method("spawn")

    args = parser.parse_args()

    # current_directory = os.path.dirname(__file__)
    # par_dir = os.path.dirname(current_directory)
    # configuration = os.path.join(par_dir, 'conf', f'{args.data}.yaml')

    experiment_dir = os.path.join(args.save_folder, 'eco', args.trial, args.experiment)
    config = os.path.join(experiment_dir, 'config.yaml')

    # save_folder = os.path.join(par_dir, 'trained_model', args.trial)
    infer(par_dir = args.save_folder, conf=config, trial=args.trial, experiment=args.experiment ,epoch=args.epoch, guide_w=args.guide_w)
    plt.show()