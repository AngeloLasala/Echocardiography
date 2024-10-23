"""
Train Conditional latent diffusion model with
"""
import torch
import yaml
import argparse
import os
import numpy as np
from tqdm import tqdm
from torch.optim import Adam
from echocardiography.diffusion.models.unet_cond_base import get_config_value
import echocardiography.diffusion.models.unet_cond_base as unet_cond_base
import echocardiography.diffusion.models.unet_base as unet_base
from echocardiography.diffusion.sheduler.scheduler import LinearNoiseScheduler
from echocardiography.diffusion.models.vqvae import VQVAE
from echocardiography.diffusion.models.cond_vae import condVAE
from echocardiography.diffusion.models.vae import VAE 
from echocardiography.diffusion.dataset.dataset import MnistDataset, EcoDataset, CelebDataset
from echocardiography.diffusion.tools.infer_vae import get_best_model
from torch.utils.data import DataLoader
import random
import multiprocessing as mp
import time

mp.set_start_method('spawn', force=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()

def drop_image_condition(image_condition, im, im_drop_prob):
    if im_drop_prob > 0:
        im_drop_mask = torch.zeros((im.shape[0], 1, 1, 1), device=im.device).float().uniform_(0,1) > im_drop_prob
        return image_condition * im_drop_mask
    else:
        return image_condition

def drop_class_condition(class_condition, class_drop_prob, im):
    if class_drop_prob > 0:
        class_drop_mask = torch.zeros((im.shape[0], 1), device=im.device).float().uniform_(0,1) > class_drop_prob
        # print(class_drop_mask)
        return class_condition * class_drop_mask
    else:
        return class_condition
def drop_keypoints_condition(keypoints_condition, keypoints_drop_prob, im):

    if keypoints_drop_prob > 0:
        keypoints_drop_mask = torch.zeros((im.shape[0], 1), device=im.device).float().uniform_(0,1) > keypoints_drop_prob
        return keypoints_condition * keypoints_drop_mask
    else:
        return keypoints_condition

def drop_eco_parameters_condition(eco_condition, eco_drop_prob, im):
    if eco_drop_prob > 0:
        eco_drop_mask = torch.zeros((im.shape[0], 1), device=im.device).float().uniform_(0,1) > eco_drop_prob
        return eco_condition * eco_drop_mask
    else:
        return eco_condition

def drop_text_condition(text_condition, text_drop_prob):
    if text_drop_prob > 0:
        text_drop_mask = torch.zeros((text_condition.shape[0], 1, 1, 1), device=text_condition.device).float().uniform_(0,1) > text_drop_prob
        return text_condition * text_drop_mask
    else:
        return text_condition

def get_text_embeddeing(text, model, device):
    """
    given the text-line condition extrapolate the text embedding
    """
    text = text.to(device)
    model = model.to(device)
    with torch.no_grad():
        text_embedding, prediction = model(text)
    return text_embedding


def train(par_dir, conf, trial, activate_cond_ldm=False):
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
    print(condition_config)
    print()
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

    # Create the dataset
    im_dataset_cls = {
        'mnist': MnistDataset,
        'celebhq': CelebDataset,
        'eco': EcoDataset,
    }.get(dataset_config['name'])

    # Create the dataset and dataloader
    # data_img = im_dataset_cls(split=dataset_config['split'], size=(dataset_config['im_size_h'], dataset_config['im_size_w']), 
    #                           im_path=dataset_config['im_path'], dataset_batch=dataset_config['dataset_batch'], phase=dataset_config['phase'],
    #                           dataset_batch_regression=dataset_config['dataset_batch_regression'], trial=dataset_config['trial'],
    #                           condition_config=condition_config)
    # data_loader = DataLoader(data_img, batch_size=train_config['ldm_batch_size'], shuffle=True, num_workers=8)

    print('dataset', dataset_config['dataset_batch'])
    data_list = []
    for dataset_batch in dataset_config['dataset_batch']:
        data_batch = im_dataset_cls(split=dataset_config['split'], size=(dataset_config['im_size_h'], dataset_config['im_size_w']),
                            parent_dir=dataset_config['parent_dir'], im_path=dataset_config['im_path'], dataset_batch=dataset_batch , phase=dataset_config['phase'],
                            condition_config=condition_config)
        data_list.append(data_batch)
    
    data_img = torch.utils.data.ConcatDataset(data_list)
    print('len of the dataset', len(data_img))
    data_loader = DataLoader(data_img, batch_size=train_config['ldm_batch_size'], shuffle=True, num_workers=8)

    # Create the model and scheduler
    scheduler = LinearNoiseScheduler(num_timesteps=diffusion_config['num_timesteps'],
                                     beta_start=diffusion_config['beta_start'],
                                     beta_end=diffusion_config['beta_end'])

    
    trial_folder = trial #os.path.join(par_dir, 'trained_model', dataset_config['name'], trial)
    assert os.listdir(trial_folder), f'No trained model found in trial folder {trial_folder}'
    print(os.listdir(trial_folder))

    if 'cond_vae' in os.listdir(trial_folder):
        ## Condition VAE + LDM
        type_model = 'cond_vae'
        print(f'type model {type_model}')
        print(f'Load trained {os.listdir(trial_folder)[0]} model')
        best_model = get_best_model(os.path.join(trial_folder,'cond_vae'))
        print(f'best model  epoch {best_model}')
        vae = condVAE(im_channels=dataset_config['im_channels'], model_config=autoencoder_model_config, condition_config=condition_config).to(device)
        vae.eval()
        vae.load_state_dict(torch.load(os.path.join(trial_folder, 'cond_vae', f'vae_best_{best_model}.pth'), map_location=device))

        ## unconditional ldm
        if activate_cond_ldm:
            model = unet_cond_base.Unet(im_channels=autoencoder_model_config['z_channels'], model_config=diffusion_model_config).to(device)
            model.train()
        else:
            model = unet_base.Unet(im_channels=autoencoder_model_config['z_channels'], model_config=diffusion_model_config).to(device)
            model.train()



    if 'vae' in os.listdir(trial_folder):
        ## VAE + conditional LDM
        type_model = 'vae'
        print(f'type model {type_model}')
        print(f'Load trained {os.listdir(trial_folder)[0]} model')
        best_model = get_best_model(os.path.join(trial_folder,'vae'))
        print(f'best model  epoch {best_model}')
        vae = VAE(im_channels=dataset_config['im_channels'], model_config=autoencoder_model_config).to(device)
        vae.eval()
        vae.load_state_dict(torch.load(os.path.join(trial_folder, 'vae', f'vae_best_{best_model}.pth'), map_location=device))

        # conditional ldm
        model = unet_cond_base.Unet(im_channels=autoencoder_model_config['z_channels'], model_config=diffusion_model_config).to(device)
        model.train()

    if 'vqvae' in os.listdir(trial_folder):
        print(f'Load trained {os.listdir(trial_folder)[0]} model')
        vae = VQVAE(im_channels=dataset_config['im_channels'], model_config=autoencoder_model_config).to(device)
        vae.eval()
        vae.load_state_dict(torch.load(os.path.join(trial_folder, 'vqvae', 'vqvae.pth'),map_location=device))

        model = unet_cond_base.Unet(im_channels=autoencoder_model_config['z_channels'], model_config=diffusion_model_config).to(device)
        model.train()

    ## if the condition is 'text' i have to load the text model
    if 'text' in condition_types:
        text_configuration = condition_config['text_condition_config']
        regression_model = data_img.get_model_embedding(text_configuration['text_embed_model'], text_configuration['text_embed_trial'])
        regression_model.eval()

    save_folder = os.path.join(trial_folder, 'cond_ldm_1')
    if not os.path.exists(save_folder):
        save_folder = os.path.join(trial_folder, 'cond_ldm_1')
        os.makedirs(save_folder)
    else:
        ## count how many folder start with cond_ldm
        count = 0
        for folder in os.listdir(trial_folder):
            if folder.startswith('cond_ldm'):
                count += 1
        save_folder = os.path.join(trial_folder, f'cond_ldm_{count+1}')
        os.makedirs(save_folder)

    num_epochs = train_config['ldm_epochs']
    optimizer = Adam(model.parameters(), lr=train_config['ldm_lr'])
    criterion = torch.nn.MSELoss()

    # Run training
    print('Start training ...')
    for epoch_idx in range(num_epochs):
        time_start = time.time()
        losses = []
        for data in data_loader:
            cond_input = None
            if condition_config is not None:
                im, cond_input = data
            else:
                im = data

            optimizer.zero_grad()
            im = im.float().to(device)
            with torch.no_grad():
                ## get the VAE/conditional_VAE latent space
                if type_model == 'cond_vae': 
                    for key in condition_types:  ## fake for loop., for now it is only one, get only one type of condition
                        key_vae = key # cond_input = cond_input[key].to(device)
                    im, _ = vae.encode(im, cond_input[key_vae].to(device))
                if type_model == 'vae': 
                    im, _ = vae.encode(im)
                

            #############  Handiling the condition input for cond LDM ########################################
            if 'image' in condition_types and (type_model == 'vae' or activate_cond_ldm):
                assert 'image' in cond_input, 'Conditioning Type Image but no image conditioning input present'
                cond_input_image = cond_input['image'].to(device) 
                # Drop condition
                im_drop_prob = get_config_value(condition_config['image_condition_config'], 'cond_drop_prob', 0.)
                cond_input['image'] = drop_image_condition(cond_input_image, im, im_drop_prob)

            if 'class' in condition_types and (type_model == 'vae' or activate_cond_ldm):
                assert 'class' in cond_input, 'Conditioning Type Class but no class conditioning input present'
                class_condition = cond_input['class'].to(device)
                class_drop_prob = get_config_value(condition_config['class_condition_config'],'cond_drop_prob', 0.)
                # Drop condition
                cond_input['class'] = drop_class_condition(class_condition, class_drop_prob, im)

            if 'keypoints' in condition_types and (type_model == 'vae' or activate_cond_ldm):
                assert 'keypoints' in cond_input, 'Conditioning Type Keypoints but no keypoints conditioning input present'
                keypoints_condition = cond_input['keypoints'].to(device)
                keypoints_drop_prob = get_config_value(condition_config['keypoints_condition_config'], 'cond_drop_prob', 0.)
                keypoints_condition = drop_keypoints_condition(keypoints_condition, keypoints_drop_prob, im)
                cond_input['keypoints'] = keypoints_condition

            if 'eco_parameters' in condition_types and (type_model == 'vae' or activate_cond_ldm):
                assert 'eco_parameters' in cond_input, 'Conditioning Type Eco Parameters but no eco parameters conditioning input present'
                eco_condition = cond_input['eco_parameters'].to(device)
                eco_drop_prob = get_config_value(condition_config['eco_parameters_condition_config'], 'cond_drop_prob', 0.)
                eco_condition = drop_eco_parameters_condition(eco_condition, eco_drop_prob, im)
                cond_input['eco_parameters'] = eco_condition

            if 'text' in condition_types and (type_model == 'vae' or activate_cond_ldm):
                assert 'text' in cond_input, 'Conditioning Type Text but no text conditioning input present'
                text_condition_input = cond_input['text'].to(device)
                text_embedding = get_text_embeddeing(text_condition_input, regression_model, device).to(device)
                text_drop_prob = get_config_value(condition_config['text_condition_config'], 'cond_drop_prob', 0.)
                text_condition = drop_text_condition(text_embedding, text_drop_prob)
                cond_input['text'] = text_condition
            #########################################################################################
                
            # Sample random noise
            noise = torch.randn_like(im).to(device)

            # Sample timestep
            t = torch.randint(0, diffusion_config['num_timesteps'], (im.shape[0],)).to(device)
            
            # Add noise to images according to timestep
            noisy_im = scheduler.add_noise(im, noise, t)

            if type_model == 'cond_vae': 
                if activate_cond_ldm:
                    noise_pred = model(noisy_im, t, cond_input=cond_input)
                else:
                    noise_pred = model(noisy_im, t)
            if type_model == 'vae': 
                noise_pred = model(noisy_im, t, cond_input=cond_input)

            loss = criterion(noise_pred, noise)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
        # end of the epoch
        
        ## Validation - computation of the FID score between real images (train) and generated images (validation)
        # Real images: from the datasete loader of the training set
        # Generated images: from the dataset loader of the validation set on wich i apply the diffusion and the decoder
        time_end = time.time()
        total_time = time_end - time_start
        print(f'Finished epoch:{epoch_idx+1} | Loss : {np.mean(losses):.4f} | Time: {total_time:.4f} sec')

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
    parser.add_argument('--save_folder', type=str, default='trained_model', help='folder to save the model, default = trained_model')
    parser.add_argument('--trial', type=str, default='trial_1', help='trial name, here you select the trained VAE to compute the latent space')
    parser.add_argument('--cond_ldm', action='store_true', help="""Choose whether or not activate the conditional ldm. Id activate enable the combo condVAE + condLDM
                                                                     Default=False that means
                                                                     'cond_vae' -> cond VAE + unconditional LDM
                                                                     'vae' -> VAE + conditional LDM""")
    args = parser.parse_args()

    current_directory = os.path.dirname(__file__)
    par_dir = os.path.dirname(current_directory)
    configuration = os.path.join(par_dir, 'conf', f'{args.data}.yaml')
    # save_folder = os.path.join(args.save_folder, args.trial)
    train(par_dir = par_dir,
        conf = configuration, 
        trial = os.path.join(args.save_folder, args.trial),
        activate_cond_ldm=args.cond_ldm)