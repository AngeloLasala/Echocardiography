"""
Train Varaiational Autoencoder for LDM
"""
import argparse
import torch
import random
import torchvision
import os
import numpy as np
from tqdm import tqdm
import json
import yaml
from echocardiography.diffusion.models.vae import VAE
from echocardiography.diffusion.models.lpips import LPIPS
from echocardiography.diffusion.models.discriminator import Discriminator
from torch.utils.data.dataloader import DataLoader
from echocardiography.diffusion.dataset.dataset import MnistDataset, EcoDataset, CelebDataset
from torch.optim import Adam
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(conf, save_folder):

    with open(conf, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)
    dataset_config = config['dataset_params']
    autoencoder_config = config['autoencoder_params']
    train_config = config['train_params']

    # Set the desired seed value #
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)
    #############################

    # Create the model and dataset #
    model = VAE(im_channels=dataset_config['im_channels'],
                  model_config=autoencoder_config).to(device)

     # Create the dataset
    im_dataset_cls = {
        'mnist': MnistDataset,
        'celebhq': CelebDataset,
        'eco': EcoDataset,
    }.get(dataset_config['name'])

    # Create the dataset and dataloader
    data = im_dataset_cls(split=dataset_config['split'], size=(dataset_config['im_size'], dataset_config['im_size']),
                          im_path=dataset_config['im_path'], dataset_batch=dataset_config['dataset_batch'] , phase=dataset_config['phase'])
    data_loader = DataLoader(data, batch_size=train_config['autoencoder_batch_size'], shuffle=True, num_workers=4, timeout=10)
    
    ## generate save folder
    save_dir = os.path.join(save_folder, dataset_config['name'])
    if not os.path.exists(save_dir):
        save_dir = os.path.join(save_dir, 'trial_1', 'vae')
        os.makedirs(save_dir)
    else:
        current_trial = len(os.listdir(save_dir))
        save_dir = os.path.join(save_dir, f'trial_{current_trial + 1}', 'vae')
        os.makedirs(save_dir)

      
    # Create the loss and optimizer
    num_epochs = train_config['autoencoder_epochs']

    recon_criterion = torch.nn.MSELoss()            # L1/L2 loss for Reconstruction
    disc_criterion = torch.nn.MSELoss()            # Disc Loss can even be BCEWithLogits MSELoss()

    lpips_model = LPIPS().eval().to(device)         # Perceptual loss, No need to freeze lpips as lpips.py takes care of that
    discriminator = Discriminator(im_channels=dataset_config['im_channels']).to(device)

    optimizer_d = Adam(discriminator.parameters(), lr=train_config['autoencoder_lr'], betas=(0.5, 0.999))
    optimizer_g = Adam(model.parameters(), lr=train_config['autoencoder_lr'], betas=(0.5, 0.999))

    # Configuration of the training loop
    disc_step_start = train_config['disc_start'] * len(data) // train_config['autoencoder_batch_size']
    step_count = 0

    # This is for accumulating gradients incase the images are huge
    # And one cant afford higher batch sizes
    # acc_steps = train_config['autoencoder_acc_steps']
    # image_save_steps = train_config['autoencoder_img_save_steps']
    image_save_steps = len(data) // train_config['autoencoder_batch_size'] 
    losses_epoch = {'recon': [], 'kl': [], 'lpips': [], 'disc': [], 'gen': []}

    for epoch_idx in range(num_epochs):
        recon_losses = []
        kl_losses = []
        perceptual_losses = []
        disc_losses = []
        gen_losses = []
        losses = []
        
        optimizer_g.zero_grad()
        optimizer_d.zero_grad()
        
        for im in tqdm(data_loader):
            step_count += 1
            im = im.float().to(device)

            ##########################  Generator ################################
            model_output = model(im)
            output, encoder_out = model_output
            mean, logvar = torch.chunk(encoder_out, 2, dim=1) 

            # Image Saving Logic
            if step_count % image_save_steps == 0 or step_count == 1:
                sample_size = min(8, im.shape[0])
                save_output = torch.clamp(output[:sample_size], -1., 1.).detach().cpu()
                save_output = ((save_output + 1) / 2)
                save_input = ((im[:sample_size] + 1) / 2).detach().cpu()

                grid = make_grid(torch.cat([save_input, save_output], dim=0), nrow=sample_size)
                img = torchvision.transforms.ToPILImage()(grid)
                plt.figure(figsize=(20, 10), tight_layout=True)
                plt.imshow(img)
                plt.axis('off')
                plt.savefig(os.path.join(save_dir, f'output_{step_count}.png'))
                plt.close()

            recon_loss = recon_criterion(output, im)
            recon_losses.append(recon_loss.item())

            kl_loss = torch.mean(0.5 * torch.sum(torch.exp(logvar) + mean ** 2 - 1.0 - logvar, dim=-1))
            kl_losses.append(train_config['kl_weight'] * kl_loss.item())
            g_loss = recon_loss + train_config['kl_weight'] * kl_loss

            if step_count > disc_step_start:
                disc_fake_pred = discriminator(model_output[0])
                disc_fake_loss = disc_criterion(disc_fake_pred,
                                                torch.ones(disc_fake_pred.shape,device=disc_fake_pred.device))
                g_loss += train_config['disc_weight'] * disc_fake_loss 
                gen_losses.append(train_config['disc_weight'] * disc_fake_loss.item())


            lpips_loss = torch.mean(lpips_model(output, im)) 
            perceptual_losses.append(train_config['perceptual_weight'] * lpips_loss.item())

            g_loss += train_config['perceptual_weight'] * lpips_loss   
            g_loss.backward()
            losses.append(g_loss.item())            
            #############################################################################

            #########################  Discriminator ################################
            if step_count > disc_step_start:
                fake = output
                disc_fake_pred = discriminator(fake.detach())
                disc_real_pred = discriminator(im)
                disc_fake_loss = disc_criterion(disc_fake_pred,
                                                torch.zeros(disc_fake_pred.shape,device=disc_fake_pred.device))
                disc_real_loss = disc_criterion(disc_real_pred,
                                                torch.ones(disc_real_pred.shape,device=disc_real_pred.device))
                disc_loss = train_config['disc_weight'] * (disc_fake_loss + disc_real_loss) / 2
                disc_losses.append(disc_loss.item())
                disc_loss.backward()
                optimizer_d.step()
                optimizer_d.zero_grad()
            #############################################################################
            optimizer_g.step()
            optimizer_g.zero_grad()

        ## Print epoch
        if len(disc_losses) > 0:
            print(f'Epoch {epoch_idx+1}/{num_epochs}) Recon Loss: {np.mean(recon_losses):.4f}| KL Loss: {np.mean(kl_losses):.4f}| LPIPS Loss: {np.mean(perceptual_losses):.4f}| G Loss: {np.mean(gen_losses):.4f}| D Loss: {np.mean(disc_losses):.4f}')
            losses_epoch['recon'].append(np.mean(recon_losses))
            losses_epoch['kl'].append(np.mean(kl_losses))
            losses_epoch['lpips'].append(np.mean(perceptual_losses))
            losses_epoch['disc'].append(np.mean(disc_losses))
            losses_epoch['gen'].append(np.mean(gen_losses))
        else:
            print(f'Epoch {epoch_idx+1}/{num_epochs}) Recon Loss: {np.mean(recon_losses):.4f}| KL Loss: {np.mean(kl_losses):.4f}| LPIPS Loss: {np.mean(perceptual_losses):.4f})')
            losses_epoch['recon'].append(np.mean(recon_losses))
            losses_epoch['kl'].append(np.mean(kl_losses))
            losses_epoch['lpips'].append(np.mean(perceptual_losses))
            losses_epoch['disc'].append(0)
            losses_epoch['gen'].append(0)
        # plt.show()

    torch.save(model.state_dict(), os.path.join(save_dir, 'vae.pth'))                                            
    torch.save(discriminator.state_dict(), os.path.join(save_dir, 'discriminator.pth'))

    ## save json file of losses
    with open(os.path.join(save_dir, 'losses.json'), 'w') as f:
        json.dump(losses_epoch, f, indent=4)

    ## save the config file
    with open(os.path.join(save_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)

    ## plot the loss
    fig, ax = plt.subplots(nrows=5, ncols=1, figsize=(10, 6), num='Losses')
    ax[0].plot(np.array(losses_epoch['recon']), label='recon')
    ax[1].plot(np.array(losses_epoch['kl']), label='kl')
    ax[2].plot(np.array(losses_epoch['lpips']), label='lpips')
    ax[3].plot(np.array(losses_epoch['disc']), label='disc')
    ax[4].plot(np.array(losses_epoch['gen']), label='gen')
    for i in range(5):
        ax[i].set_xlabel('Epochs', fontsize=15)
        ax[i].set_ylabel('Loss', fontsize=15)
        ax[i].tick_params(axis='both', which='major', labelsize=15)
        ax[i].legend(fontsize=15)
    plt.savefig(os.path.join(save_dir, 'losses.png'))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train VAE on MNIST or CelebA-HQ')
    parser.add_argument('--data', type=str, default='mnist', help='type of the data, mnist, celebhq, eco')  
    parser.add_argument('--save_folder', type=str, default='trained_model', help='folder to save the model, default = trained_model')
    args = parser.parse_args()

    current_directory = os.path.dirname(__file__)
    par_dir = os.path.dirname(current_directory)

    configuration = os.path.join(par_dir, 'conf', f'{args.data}.yaml')
    save_folder = os.path.join(par_dir, args.save_folder)
    train(conf = configuration, save_folder = save_folder)