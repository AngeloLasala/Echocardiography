"""
Train  Vector Quantized Varaiational Autoencoder for LDM
"""
import argparse
import torch
import random
import torchvision
import os
import numpy as np
import json
import yaml
from tqdm import tqdm
from echocardiography.diffusion.models.vqvae import VQVAE
from echocardiography.diffusion.models.lpips import LPIPS
from echocardiography.diffusion.models.discriminator import Discriminator
from torch.utils.data.dataloader import DataLoader
from echocardiography.diffusion.dataset.dataset import MnistDataset, EcoDataset, CelebDataset
from torch.optim import Adam
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(conf):
    # autoencoder_config = {
    #     'z_channels': 3,
    #     'codebook_size' : 8192,
    #     'down_channels': [32, 64, 128, 256],
    #     'mid_channels': [256, 256],
    #     'down_sample': [True, True, True],
    #     'attn_down': [False, False, False], ## False, deactivate the attention meccanism of autoencoder
    #     'norm_channels': 32,
    #     'num_heads': 4,
    #     'num_down_layers': 2,
    #     'num_mid_layers': 2,
    #     'num_up_layers': 2
    # }
    # dataset_config = {
    #     'im_path': 'data/mnist/train/images',
    #     'im_channels': 1,
    #     'im_size': 256,
    # }
    # train_config = {'autoencoder_epochs': 20,
    #                 'autoencoder_lr': 0.00001,
    #                 'disc_start': 10, #epoch after how many step discriminator should start
    #                 'autoencoder_batch_size': 4,
    #                 'kl_weight': 0.000005,
    #                 'perceptual_weight': 1.,
    #                 'disc_weight': 0.5,
    #                 'codebook_weight': 0.25,
    #                 'commitment_beta': 0.25,

    # }

    # Read the config file #
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
    model = VQVAE(im_channels=dataset_config['im_channels'], model_config=autoencoder_config).to(device)
    x = torch.randn(1, 1, 32, 32).to(device)
    out = model(x)
    
    # Create the dataset
    im_dataset_cls = {
        'mnist': MnistDataset,
        'celebhq': CelebDataset,
        'eco': EcoDataset,
    }.get(dataset_config['name'])

    # Create the dataset and dataloader
    data = im_dataset_cls(split=dataset_config['split'], size=(dataset_config['im_size'], dataset_config['im_size']), im_path=dataset_config['im_path'])
    data_loader = DataLoader(data, batch_size=train_config['autoencoder_batch_size'], shuffle=True, num_workers=8)
 
    
    ## generate save folder
    save_folder = 'prova_vqvae_tools_sgsdf'
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

        
    # Create the loss and optimizer
    num_epochs = train_config['autoencoder_epochs']

    recon_criterion = torch.nn.MSELoss()            # L1/L2 loss for Reconstruction
    disc_criterion = torch.nn.MSELoss()             # Disc Loss can even be BCEWithLogits MSELoss()

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
    losses_epoch = {'recon': [], 'codebook': [], 'lpips': [], 'disc': [], 'gen': []}

    for epoch_idx in range(num_epochs):
        recon_losses = []
        codebook_losses = []
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
            output, z, quantize_losses = model_output

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
                plt.savefig(os.path.join(save_folder, f'output_{step_count}.png'))
                # plt.show()


            recon_loss = recon_criterion(output, im)
            recon_losses.append(recon_loss.item())

            g_loss = (recon_loss +
                      (train_config['codebook_weight'] * quantize_losses['codebook_loss']) +
                      (train_config['commitment_beta'] * quantize_losses['commitment_loss']))
            codebook_losses.append(train_config['codebook_weight'] * quantize_losses['codebook_loss'].item())
            
            
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
            optimizer_g.step()
            
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
        optimizer_g.zero_grad()

        ## Print epoch
        if len(disc_losses) > 0:
            print(f'Epoch {epoch_idx+1}/{num_epochs}) Recon Loss: {np.mean(recon_losses):.4f}| Codeb0ok Loss: {np.mean(codebook_losses):.4f}| LPIPS Loss: {np.mean(perceptual_losses):.4f}| G Loss: {np.mean(gen_losses):.4f}| D Loss: {np.mean(disc_losses):.4f}')
            losses_epoch['recon'].append(np.mean(recon_losses))
            losses_epoch['codebook'].append(np.mean(codebook_losses))
            losses_epoch['lpips'].append(np.mean(perceptual_losses))
            losses_epoch['disc'].append(np.mean(disc_losses))
            losses_epoch['gen'].append(np.mean(gen_losses))
        else:
            print(f'Epoch {epoch_idx+1}/{num_epochs}) Recon Loss: {np.mean(recon_losses):.4f}| Codebook Loss: {np.mean(codebook_losses):.4f}| LPIPS Loss: {np.mean(perceptual_losses):.4f})')
            losses_epoch['recon'].append(np.mean(recon_losses))
            losses_epoch['codebook'].append(np.mean(codebook_losses))
            losses_epoch['lpips'].append(np.mean(perceptual_losses))
            losses_epoch['disc'].append(0)
            losses_epoch['gen'].append(0)
        # plt.show()

       
    torch.save(model.state_dict(), os.path.join(save_folder, 'vqvae_eco.pth'))                                            
    torch.save(discriminator.state_dict(), os.path.join(save_folder, 'discriminator_vqvae_eco.pth'))

    ## save json file of losses
    with open(os.path.join(save_folder, 'losses.json'), 'w') as f:
        json.dump(losses_epoch, f, indent=4)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train VAE on MNIST or CelebA-HQ')
    parser.add_argument('--data', type=str, default='mnist', help='type of the data, mnist, celebhq, eco')  
    args = parser.parse_args()

    current_directory = os.path.dirname(__file__)
    par_dir = os.path.dirname(current_directory)
    configuration = os.path.join(par_dir, 'conf', f'{args.data}.yaml')
    print(configuration)
    train(conf = configuration)
