"""
Qualitative comparison between real image (training) and generated image 
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
from echocardiography.diffusion.evaluation.investigate_vae import get_config_value, get_best_model
from echocardiography.diffusion.evaluation.hypertropy_eval import GenerateDataset
from echocardiography.diffusion.dataset.dataset import CelebDataset, MnistDataset, EcoDataset
from echocardiography.diffusion.models.vae import VAE
from echocardiography.diffusion.models.cond_vae import condVAE

from sklearn.metrics import silhouette_score


from sklearn.manifold import TSNE

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def infer(par_dir, conf, trial, experiment, epoch, guide_w, compute_real, compute_gen):
    ######## Read the config file #######
    with open(conf, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    autoencoder_model_config = config['autoencoder_params']
    train_config = config['train_params']
    diffusion_config = config['diffusion_params']
    dataset_config = config['dataset_params']
    diffusion_model_config = config['ldm_params']
    autoencoder_model_config = config['autoencoder_params']
    train_config = config['train_params']
    condition_config = get_config_value(autoencoder_model_config, key='condition_config', default_value=None)
    if condition_config is not None:
        assert 'condition_types' in condition_config, \
            "condition type missing in conditioning config"
        condition_types = condition_config['condition_types']

    #############################

    # Create the dataset
    im_dataset_cls = {
        'eco': EcoDataset,
    }.get(dataset_config['name'])

    # Create the dataset and dataloader of real images
    data_list = []
    for dataset_batch in dataset_config['dataset_batch']:
        data_batch = im_dataset_cls(split=dataset_config['split'], size=(dataset_config['im_size_h'], dataset_config['im_size_w']),
                            parent_dir=dataset_config['parent_dir'], im_path=dataset_config['im_path'], dataset_batch=dataset_batch , phase=dataset_config['phase'],
                            condition_config=condition_config)
        data_list.append(data_batch)

    data_img = torch.utils.data.ConcatDataset(data_list)
    print('len of the dataset', len(data_img))
    data_loader = DataLoader(data_img, batch_size=1, shuffle=False, num_workers=8)

    # create the dataset and the dataloader of the generated images
     ## load the generated image from path
    data_gen = GenerateDataset(par_dir=os.path.join(par_dir,'eco'), trial=trial, experiment=experiment, epoch=epoch, guide_w=guide_w,
                               size=(dataset_config['im_size_h'], dataset_config['im_size_w']), input_channels=dataset_config['im_channels'])
    print('len of the generated dataset', len(data_gen))
    data_loader_gen = DataLoader(data_gen, batch_size=1, shuffle=False, num_workers=8)



    num_images = train_config['num_samples']
    ngrid = train_config['num_grid_rows']

    trial_folder = os.path.join(par_dir, dataset_config['name'], trial)
    assert os.listdir(trial_folder), f'No trained model found in trial folder {trial_folder}'
    print(os.listdir(trial_folder))

    if 'cond_vae' in os.listdir(trial_folder):
        type_model = 'cond_vae'
        print(f'Load trained {os.listdir(trial_folder)[0]} model')
        best_model = get_best_model(os.path.join(trial_folder,'cond_vae'))
        print(f'best model  epoch {best_model}')
        vae = condVAE(im_channels=dataset_config['im_channels'], model_config=autoencoder_model_config, condition_config=condition_config).to(device)
        vae.eval()
        vae.load_state_dict(torch.load(os.path.join(trial_folder, 'cond_vae', f'vae_best_{best_model}.pth'), map_location=device))

    if 'vae' in os.listdir(trial_folder):
        type_model = 'vae'
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
    encoded_output_list, gen_encoded_output_list = [], []

    if compute_real:
        with torch.no_grad():
            ## Real data
            for im, cond in tqdm(data_loader):
                im = im.float().to(device)
                for key in cond.keys(): ## for all the type of condition, we move the  tensor on the device
                    cond[key] = cond[key].to(device)

                ## Encoder - get the latent space
                if type_model == 'cond_vae': encoded_output, _ = vae.encode(im, cond['class'])
                else: encoded_output, _ = vae.encode(im)

                encoded_output = encoded_output[0,:,:,:].flatten()
                encoded_output_list.append(encoded_output.cpu().numpy())
       
        encoded_output_list = np.array(encoded_output_list)

        ## save the encoded_output_list and gen_encoded_output_list with np.save
        ## make dir tnse
        for experiments in ['cond_ldm_1', 'cond_ldm_2', 'cond_ldm_3', 'cond_ldm_4', 'cond_ldm_5', 'ldm_1']:
            tsne_folder = os.path.join(trial_folder, experiments, 'tsne')
            if not os.path.exists(tsne_folder):
                os.makedirs(tsne_folder)
            np.save(os.path.join(tsne_folder, 'encoded_output_list.npy'), encoded_output_list)
    
    else:
        tsne_folder = os.path.join(trial_folder, experiment, 'tsne')        
        encoded_output_list = np.load(os.path.join(tsne_folder, 'encoded_output_list.npy'))

    if compute_gen:    
        with torch.no_grad():
            for gen_img in tqdm(data_loader_gen):
                gen_img = gen_img.float().to(device)

                ## Encoder - get the latent space
                if type_model == 'cond_vae': gen_encoded_output, _ = vae.encode(gen_img, cond['class'])
                else: gen_encoded_output, _ = vae.encode(gen_img)

                gen_encoded_output = gen_encoded_output[0,:,:,:].flatten()
                gen_encoded_output_list.append(gen_encoded_output.cpu().numpy())
            gen_encoded_output_list = np.array(gen_encoded_output_list)

            tsne_folder = os.path.join(trial_folder, experiment, 'tsne')
            if not os.path.exists(tsne_folder):
                os.makedirs(tsne_folder)
            np.save(os.path.join(tsne_folder, 'gen_encoded_output_list.npy'), gen_encoded_output_list)

    
    else:
        tsne_folder = os.path.join(trial_folder, experiment, 'tsne')        
        gen_encoded_output_list = np.load(os.path.join(tsne_folder, 'gen_encoded_output_list.npy'))

 
    print('TSNE embedding...')
    real_gen_stack = np.vstack((encoded_output_list, gen_encoded_output_list))
    real_gen_label_stack = np.concatenate((np.ones(encoded_output_list.shape[0]), np.zeros(gen_encoded_output_list.shape[0])))
   
    tsne = TSNE(n_components=2, random_state=42)
    encoded_output_tsne = tsne.fit_transform(real_gen_stack)
    real_gen_stack_tsne = encoded_output_tsne.copy()
    print('encoded_output_tsne', encoded_output_tsne.shape)

    # real is where real_gen_label_stack == 1
    real_gen_stack_tsne = (real_gen_stack_tsne - np.min(real_gen_stack_tsne, axis=0)) / (np.max(real_gen_stack_tsne, axis=0) - np.min(real_gen_stack_tsne, axis=0))
    real_tsne = real_gen_stack_tsne[real_gen_label_stack==1]
    gen_tsne = real_gen_stack_tsne[real_gen_label_stack==0]
    print('real_tsne', real_tsne.shape)
    print('gen_tsne', gen_tsne.shape)

   
  
    ## compute the silouhette score between the real and generated data
    real_gen_stack_tsne = (real_gen_stack_tsne - np.min(real_gen_stack_tsne, axis=0)) / (np.max(real_gen_stack_tsne, axis=0) - np.min(real_gen_stack_tsne, axis=0))
    silhouette_score_all = silhouette_score(real_gen_stack_tsne, real_gen_label_stack)
    print(f"Silhouette score all data: {silhouette_score_all}")
    print()
    
    # compute the DB values for the real and generated data
    from sklearn.metrics import davies_bouldin_score
    davies_bouldin_score_all = davies_bouldin_score(real_gen_stack_tsne, real_gen_label_stack)
    print(f"Davies Bouldin score all data: {davies_bouldin_score_all}")
    print()


    ## compute the centroids of the real and generated data
    centroids = np.array([np.mean(real_tsne, axis=0), np.mean(gen_tsne, axis=0)])
    centroid_distance = np.linalg.norm(centroids[0] - centroids[1])
    print('centroid_distance', centroid_distance)
    ()
    
    # Inter-cluster distances (minimum, maximum, average)
    from scipy.spatial.distance import cdist
    distances = cdist(real_tsne , gen_tsne, metric='euclidean')
    min_distance = np.min(distances)
    max_distance = np.max(distances)
    avg_distance = np.mean(distances)
    print("Minimum Distance:", min_distance)
    print("Maximum Distance:", max_distance)
    print("Average Distance:", avg_distance)



    ######################## PLOT #################################################################
    #
    

    ## plt the 2d scatter plot of tsne
    plt.figure(figsize=(8,8), num=f'{type_model} - TSNE of latent space of PLAX')

    plt.scatter(real_tsne[:,0], real_tsne[:,1], c='lightblue', label='Real data')
    plt.scatter(gen_tsne[:,0], gen_tsne[:,1], c='lightgreen', label='Generated data')

    # plt.scatter(real_gen_stack_tsne[real_gen_stack_lable==0,0], real_gen_stack_tsne[real_gen_stack_lable==0,1], c='lightgreen', label='Generated data')
    # scttaer of centroids
    # plt.scatter(centroids[0,0], centroids[0,1], c='blue', label='Real centroid')
    # plt.scatter(centroids[1,0], centroids[1,1], c='green', label='Generated centroid')
    plt.xlabel('TSNE 1', fontsize=20)
    plt.ylabel('TSNE 2', fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(fontsize=20)
    plt.show()


    ##################################################################################################


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Invastigate the latent space')
    parser.add_argument('--save_folder', type=str, default='trained_model', help='folder to save the model, default = trained_model')
    parser.add_argument('--trial', type=str, default='trial_1', help='trial name for saving the model, it is the trial folde that contain the VAE model')
    parser.add_argument('--experiment', type=str, default='cond_ldm', help="""name of expermient, it is refed to the type of condition and in general to the 
                                                                              hyperparameters (file .yaml) that is used for the training, it can be cond_ldm, cond_ldm_2, """)
    parser.add_argument('--guide_w', type=float, default=0.0, help='guide_w for the conditional model, w=-1 [unconditional], w=0 [vanilla conditioning], w>0 [guided conditional]')
    parser.add_argument('--epoch', type=int, default=99, help='epoch to sample, this is the epoch of cond ldm model') 
    parser.add_argument('--compute_encodings_real', action='store_true', help="compite the embeddings, default=False")
    parser.add_argument('--compute_encodings_gen', action='store_true', help="compite the embeddings, default=False")

    args = parser.parse_args()

    experiment_dir = os.path.join(args.save_folder, 'eco', args.trial)
    if 'vae' in os.listdir(experiment_dir): config = os.path.join(experiment_dir, 'vae', 'config.yaml')
    if 'vqvae' in os.listdir(experiment_dir): config = os.path.join(experiment_dir, 'vqvae', 'config.yaml')
    if 'cond_vae' in os.listdir(experiment_dir): config = os.path.join(experiment_dir, 'cond_vae', 'config.yaml')


    infer(par_dir = args.save_folder, conf=config, trial=args.trial, 
         experiment=args.experiment, epoch=args.epoch, guide_w=args.guide_w, compute_real=args.compute_encodings_real, compute_gen=args.compute_encodings_gen)
    plt.show()