"""
Test file for conditional LDM.
Here i want to test the variability of the model to generete diverse image get the same randm noise
and a litlle variatio of condition. The underlying reasoning is
input noise = set the image
condition = set the hypertrophy
little modification = domain shift mimiking different hyoertrophy condiction
"""
"""
Sample from trained conditional latent diffusion model. the sampling follow the classifier-free guidance

w = -1 [unconditional] = the learned conditional model completely ignores the conditioner and learns an unconditional diffusion model
w = 0 [vanilla conditional] =  the model explicitly learns the vanilla conditional distribution without guidance
w > 0 [guided conditional] =  the diffusion model not only prioritizes the conditional score function, but also moves in the direction away from the unconditional score function
"""
import numpy as np
import torch
import random
import torchvision
import argparse
import yaml
import os
from torchvision.utils import make_grid
from torchvision import transforms
from PIL import Image

from echocardiography.diffusion.models.unet_cond_base import get_config_value
import echocardiography.diffusion.models.unet_cond_base as unet_cond_base
import echocardiography.diffusion.models.unet_base as unet_base
from echocardiography.diffusion.models.vqvae import VQVAE
from echocardiography.diffusion.models.cond_vae import condVAE
from echocardiography.diffusion.models.vae import VAE 
from echocardiography.diffusion.sheduler.scheduler import LinearNoiseScheduler
from echocardiography.diffusion.dataset.dataset import MnistDataset, EcoDataset, CelebDataset
from echocardiography.diffusion.tools.infer_vae import get_best_model
from torch.utils.data import DataLoader
from echocardiography.diffusion.tools.train_cond_ldm import get_text_embeddeing
from echocardiography.regression.utils import get_corrdinate_from_heatmap, get_corrdinate_from_heatmap_ellipses
import torch.multiprocessing as mp
import math
from scipy.stats import multivariate_normal


import matplotlib.pyplot as plt
import cv2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def line_give_points(p1, p2, x_max, y_max):
    """
    Given two points p1 and p2 return the line equation
    """
    x1, y1 = p1[0], y_max - p1[1]
    x2, y2 = p2[0], y_max - p2[1]
    dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    m = (y1 - y2) / (x1 - x2)
    q = y1 - m * x1
    return m, q, dist

def distance_between_two_points(p1, p2, x_max, y_max):
    """
    Given two points p1 and p2 return the distance between them
    """
    x1, y1 = p1[0], y_max - p1[1]
    x2, y2 = p2[0], y_max - p2[1]
    return


def get_heatmap(labels, w=320, h=240):
        """
        given a index of the patient return the 6D heatmap of the keypoints
        """

        #get the percentace of the label w.r.t the image size
        # print('labels', labels)
        # converter = np.tile([w, h], 6)
        # labels = labels / converter
        # labels = labels * converter
        # print('labels', labels)

        x, y = np.meshgrid(np.arange(0, w), np.arange(0, h))
        pos = np.dstack((x, y))

        std_dev = int(w * 0.05)
        covariance = np.array([[std_dev * 20, 0.], [0., std_dev]])

        # Initialize an empty 6-channel heatmap vector
        heatmaps_label= np.zeros((h, w, 6), dtype=np.float32)
        # print('heatmaps_label', heatmaps_label.shape)
        for hp, heart_part in enumerate([labels[0:4], labels[4:8], labels[8:12]]): ## LVIDd, IVSd, LVPWd
            ## compute the angle of the heart part
            x_diff = heart_part[0:2][0] - heart_part[2:4][0]
            y_diff = heart_part[2:4][1] - heart_part[0:2][1]
            angle = math.degrees(math.atan2(y_diff, x_diff))

            for i in range(2): ## each heart part has got two keypoints with the same angle
                mean = (int(heart_part[i*2]), int(heart_part[(i*2)+1]))

                gaussian = multivariate_normal(mean=mean, cov=covariance)
                base_heatmap = gaussian.pdf(pos)

                rotation_matrix = cv2.getRotationMatrix2D(mean, angle + 90, 1.0)
                base_heatmap = cv2.warpAffine(base_heatmap, rotation_matrix, (base_heatmap.shape[1], base_heatmap.shape[0]))
                base_heatmap = base_heatmap / np.max(base_heatmap)
                channel_index = hp * 2 + i
                heatmaps_label[:, :, channel_index] = base_heatmap

        return heatmaps_label


def augumenting_heatmap(heatmap, delta):
    """
    Given a heatmap retern several augumented images chenges the RWT and RST
    """
    ## augementation steps
    number_of_step = np.arange(-delta, delta + 5 , 5) / 100
    label_list = get_corrdinate_from_heatmap(heatmap[0])

    ## get the line equation for the pw points
    m_pw, q_pw, dist_pw = line_give_points(label_list[0:2], label_list[2:4], heatmap.shape[3], heatmap.shape[2])
    m_ivs, q_ivs, dist_ivs = line_give_points(label_list[8:10], label_list[10:12], heatmap.shape[3], heatmap.shape[2])

    ## augumentation of the LVPWd 
    heatmap_pw = []
    for step in number_of_step:
        displacement = np.abs(label_list[0]-label_list[3]) * step
        new_x1 = label_list[0] + displacement
        new_y1 = heatmap.shape[2] - (m_pw * new_x1 + q_pw)
        new_label = label_list.copy()
        new_label[0], new_label[1] = int(new_x1), int(new_y1)
        new_heatmap = get_heatmap(new_label)
        heatmap_pw.append(new_heatmap)

    heatmap_ivs = []
    for step in number_of_step:
        displacement = np.abs(label_list[10]-label_list[11]) * step
        new_x1 = label_list[10] + displacement
        new_y1 = heatmap.shape[2] - (m_ivs * new_x1 + q_ivs)
        new_label = label_list.copy()
        new_label[10], new_label[11] = int(new_x1), int(new_y1)
        new_heatmap = get_heatmap(new_label)
        heatmap_ivs.append(new_heatmap)
    heatmaps_pw = np.array(heatmap_pw)
    heatmaps_ivs = np.array(heatmap_ivs)
    heatmaps_label = np.concatenate((heatmaps_pw, heatmaps_ivs), axis=0)
    heatmaps_label = np.array(heatmaps_label)
    
    ## reshape the heatmaps_label in batch, channel, h, w
    label = torch.tensor(heatmaps_label).permute(0, 3, 1, 2)
    return label



def sample(model, scheduler, train_config, diffusion_model_config, condition_config,
           autoencoder_model_config, diffusion_config, dataset_config, type_model, vae, save_folder, guide_w, activate_cond_ldm):
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

    print(condition_config)
    condition_config = get_config_value(diffusion_model_config, key='condition_config', default_value=None)
    if condition_config is not None:
        assert 'condition_types' in condition_config, \
            "condition type missing in conditioning config"
        condition_types = condition_config['condition_types']

    print('DIMENSION OF THE LATENT SPACE: ', autoencoder_model_config['z_channels'])

    print('dataset', dataset_config['dataset_batch'])
    data_list = []
    for dataset_batch in dataset_config['dataset_batch']:
        data_batch = im_dataset_cls(split=dataset_config['split_test'], size=(dataset_config['im_size_h'], dataset_config['im_size_w']),
                            parent_dir=dataset_config['parent_dir'], im_path=dataset_config['im_path'], dataset_batch=dataset_batch , phase=dataset_config['phase'],
                            condition_config=condition_config)
        data_list.append(data_batch)

    data_img = torch.utils.data.ConcatDataset(data_list)
    print('len of the dataset', len(data_img))
    batch_size_sample = 1 ## in this case the variability is on condition
    data_loader = DataLoader(data_img, batch_size=batch_size_sample, shuffle=False, num_workers=8)

    ## if the condition is 'text' i have to load the text model
    if 'text' in condition_types:
        text_configuration = condition_config['text_condition_config']
        regression_model = data_img.get_model_embedding(text_configuration['text_embed_model'], text_configuration['text_embed_trial'])
        regression_model.eval()

    for btc, data in enumerate(data_loader):
        cond_input = None
        uncond_input = {}
        if condition_config is not None:
            im, cond_input = data  # im is the image (batch_size=8), cond_input is the conditional input ['image for the mask']
            for key in cond_input.keys(): ## for all the type of condition, we move the  tensor on the device
                cond_input[key] = cond_input[key].to(device)
                uncond_input[key] = torch.zeros_like(cond_input[key])
        else:
            im = data


        ## convert the cond and uncond with augumented heatmaps
        new_heatmap = augumenting_heatmap(cond_input[key].cpu().numpy(), delta = 10).to(device)
        xt = torch.randn((1, autoencoder_model_config['z_channels'], im_size_h, im_size_w)).repeat(new_heatmap.shape[0],1,1,1).to(device) 
        cond_input[key] = new_heatmap
        uncond_input[key] = torch.zeros_like(cond_input[key])

  
        ################ Sampling Loop ########################
        for i in reversed(range(diffusion_config['num_timesteps'])):
            # Get prediction of noise
            t = (torch.ones((xt.shape[0],)) * i).long().to(device)

            if type_model == 'vae':
                noise_pred_cond = model(xt, t, cond_input)
                noise_pred_uncond = model(xt, t, uncond_input)
                noise_pred = (1 + guide_w) * noise_pred_cond - guide_w * noise_pred_uncond

            if type_model == 'cond_vae':
                if activate_cond_ldm:
                    noise_pred_cond = model(xt, t, cond_input)
                    noise_pred_uncond = model(xt, t, uncond_input)
                    noise_pred = (1 + guide_w) * noise_pred_cond - guide_w * noise_pred_uncond
                else:
                    noise_pred = model(xt, t)

            # Use scheduler to get x0 and xt-1
            xt, x0_pred = scheduler.sample_prev_timestep(xt, noise_pred, torch.as_tensor(i).to(device))

            # Save x0
            if i == 0:
                # Decode ONLY the final image to save time
                if type_model == 'vae':
                    ims = vae.decode(xt)
                if type_model == 'cond_vae':
                    for key in condition_types:  ## fake for loop., for now it is only one, get only one type of condition
                        cond_input = cond_input[key].to(device)
                    ims = vae.decode(xt, cond_input)
            else:
                ims = x0_pred

            ims = torch.clamp(ims, -1., 1.).detach().cpu()
            ims = (ims + 1) / 2

        for i in range(ims.shape[0]):
            cv2.imwrite(os.path.join(save_folder, f'x0_{btc}_{i}.png'), ims[i].numpy()[0]*255)
        
        ## save the new heatmap as a npy file
        np.save(os.path.join(save_folder, f'heatmap_{btc}.npy'), new_heatmap.cpu().numpy())


def infer(par_dir, conf, trial, experiment, epoch, guide_w, activate_cond_ldm):
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


   
    ########## Load AUTOENCODER #############
    trial_folder = os.path.join(par_dir, dataset_config['name'], trial)
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

        
        if activate_cond_ldm:
            ## conditional ldm
            model = unet_cond_base.Unet(im_channels=autoencoder_model_config['z_channels'], model_config=diffusion_model_config).to(device)
            model.eval()
            model_dir = os.path.join(par_dir, dataset_config['name'], trial, experiment)
            model.load_state_dict(torch.load(os.path.join(model_dir, f'ldm_{epoch}.pth'),map_location=device), strict=False)
        
        else:
            ## unconditional ldm
            model = unet_base.Unet(im_channels=autoencoder_model_config['z_channels'], model_config=diffusion_model_config).to(device)
            model.eval()
            model_dir = os.path.join(par_dir, dataset_config['name'], trial, experiment)
            model.load_state_dict(torch.load(os.path.join(model_dir, f'ldm_{epoch}.pth'),map_location=device), strict=False)

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
        model.eval()
        model_dir = os.path.join(par_dir, dataset_config['name'], trial, experiment)
        model.load_state_dict(torch.load(os.path.join(model_dir, f'ldm_{epoch}.pth'),map_location=device), strict=False)

    if 'vqvae' in os.listdir(trial_folder):
        print(f'Load trained {os.listdir(trial_folder)[0]} model')
        vae = VQVAE(im_channels=dataset_config['im_channels'], model_config=autoencoder_model_config).to(device)
        vae.eval()
        vae.load_state_dict(torch.load(os.path.join(trial_folder, 'vqvae', 'vqvae.pth'),map_location=device))
   #####################################

    ######### Create output directories #############
    save_folder = os.path.join(model_dir, 'test', f'w_{guide_w}', f'samples_ep_{epoch}')
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
   

    ######## Sample from the model
    with torch.no_grad():
        sample(model, scheduler, train_config, diffusion_model_config, condition_config,
               autoencoder_model_config, diffusion_config, dataset_config, type_model, vae, save_folder, guide_w, activate_cond_ldm)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train unconditional LDM with VQVAE')
    parser.add_argument('--save_folder', type=str, default='trained_model', help='folder to save the model, default = trained_model')
    parser.add_argument('--trial', type=str, default='trial_1', help='trial name for saving the model, it is the trial folde that contain the VAE model')
    parser.add_argument('--experiment', type=str, default='cond_ldm', help="""name of expermient, it is refed to the type of condition and in general to the
                                                                              hyperparameters (file .yaml) that is used for the training, it can be cond_ldm, cond_ldm_2, """)
    parser.add_argument('--epoch', type=int, default=100, help='epoch to sample, this is the epoch of cond ldm model')
    parser.add_argument('--guide_w', type=float, default=0.0, help='guide_w for the conditional model, w=-1 [unconditional], w=0 [vanilla conditioning], w>0 [guided conditional]')
    parser.add_argument('--cond_ldm', action='store_true', help="""Choose whether or not activate the conditional ldm. Id activate enable the combo condVAE + condLDM
                                                                     Default=False that means
                                                                     'cond_vae' -> cond VAE + unconditional LDM
                                                                     'vae' -> VAE + conditional LDM""")

    args = parser.parse_args()

    experiment_dir = os.path.join(args.save_folder, 'eco', args.trial, args.experiment)
    config = os.path.join(experiment_dir, 'config.yaml')

    # save_folder = os.path.join(par_dir, 'trained_model', args.trial)
    infer(par_dir = args.save_folder, conf=config, trial=args.trial, experiment=args.experiment ,epoch=args.epoch, guide_w=args.guide_w, activate_cond_ldm=args.cond_ldm)
    plt.show()

