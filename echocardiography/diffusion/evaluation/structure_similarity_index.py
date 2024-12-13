"""
Stracture similarity indeces fro validation datsets
"""
import torch
import os
import argparse
import yaml
import numpy as np
from PIL import Image
from PIL import Image
from scipy import linalg
import matplotlib.pyplot as plt

from image_similarity_measures.evaluate import evaluation
from echocardiography.diffusion.dataset.dataset import MnistDataset, EcoDataset, CelebDataset


def structure_similarity(par_dir, conf, experiment_dir_w):
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

 
    # Create the dataset
    im_dataset_cls = {
 
        'eco': EcoDataset,
    }.get(dataset_config['name'])

    print('dataset', dataset_config['dataset_batch'])
    data_list = []
    for dataset_batch in dataset_config['dataset_batch']:
        data_batch = im_dataset_cls(split='val', size=(dataset_config['im_size_h'], dataset_config['im_size_w']),
                            parent_dir=dataset_config['parent_dir'], im_path=dataset_config['im_path'], dataset_batch=dataset_batch , phase=dataset_config['phase'])
        ## add the patient files to the data_list
        for patient_hash in data_batch.patient_files:
            img_path = os.path.join(data_batch.data_dir, f'{patient_hash}.' + data_batch.im_ext)
            data_list.append(img_path)

    real_images = data_list
    gen_images = [os.path.join(experiment_dir_w, f'x0_{i}.png') for i in range(len(os.listdir(experiment_dir_w)))]

    ## compute the structure similarity index
    ssim_list, ms_ssim_list, rmse_list = [], [], []
    for i,j in zip(real_images, gen_images):
        metrics = evaluation(org_img_path=i, 
                            pred_img_path=j, 
                            metrics=["rmse", "ssim", 'fsim'])
        im1 = Image.open(i)
        im2 = Image.open(j)

        ssim = metrics['ssim']
        ms_ssim = metrics['fsim']
        rmse = metrics['rmse']
        print(ssim, rmse)

        ssim_list.append(ssim)
        ms_ssim_list.append(ms_ssim)
        rmse_list.append(rmse)

    ## print mean and std
    print(f'ssim {np.array(ssim_list).mean():.4f} +- {np.array(ssim_list).std(ddof=1):.4f}')
    print(f'rmse {np.array(rmse_list).mean():.4f} +- {np.array(rmse_list).std(ddof=1):.4f}')

    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compute structure similarity indeces")
    parser.add_argument('--par_dir', type=str, default='/home/angelo/Documents/Echocardiography/echocardiography/diffusion/trained_model/eco',
                         help="""parent directory of the trained model
                        local: /home/angelo/Documents/Echocardiography/echocardiography/diffusion/trained_model/eco
                        cluster: /leonardo_work/IscrC_Med-LMGM/Angelo/trained_model/diffusion/eco""")
    parser.add_argument('--trial', type=str, default='trial_1', help='trial name for saving the model, it is the trial folde that contain the VAE model')
    parser.add_argument('--experiment', type=str, default='cond_ldm', help="""name of expermient, it is refed to the type of condition and in general to the 
                                                                              hyperparameters (file .yaml) that is used for the training, it can be cond_ldm, cond_ldm_2, """)
    parser.add_argument('--guide_w', type=float, default=0.0, help='guide_w for the conditional model, w=-1 [unconditional], w=0 [vanilla conditioning], w>0 [guided conditional]')
    parser.add_argument('--epoch', type=int, default=99, help='epoch to sample, this is the epoch of cond ldm model') 


    # parser.add_argument('--epoch', type=int, default=99, help='epoch to sample, this is the epoch of cond ldm model')
    args = parser.parse_args()

    experiment_dir = os.path.join(args.par_dir, args.trial, args.experiment)
    print(os.listdir(experiment_dir))
    config = os.path.join(experiment_dir, 'config.yaml')
    print(config)
    experiment_dir_w = os.path.join(experiment_dir, f'w_{args.guide_w}', f'samples_ep_{args.epoch}')
    
    structure_similarity(args.par_dir, config, experiment_dir_w)

   
    # mse_path  = 'GAN/'+ args.attribute + '/' + args.clas +  '/' + key 
    # smart_makedir(mse_path +'/comparison_real_generate')
    # main_path_real = 'GAN/'+ args.attribute + '/' + args.clas +  '/' + key + f'/checkpoint_{best_check[key]}/fid/real'
    # main_path_gen = 'GAN/'+ args.attribute + '/' + args.clas +  '/' + key + f'/checkpoint_{best_check[key]}/fid/synthetic'

    # real_images = [main_path_real + '/'+ sample for sample in os.listdir(main_path_real)]
    # synthetic_images = [main_path_gen + '/'+ sample for sample in os.listdir(main_path_gen)]

    # real_images.sort()
    # synthetic_images.sort() 
    # ssim_list, ms_ssim_list, rmse_list = [], [], []
    # for i,j in zip(real_images, synthetic_images):
    #     if from_image_sm: 
    #         metrics = evaluation(org_img_path=i, 
    #                             pred_img_path=j, 
    #                             metrics=["rmse", "ssim", 'fsim'])
    #         im1 = cv.imread(i, cv.IMREAD_GRAYSCALE)/255.
    #         im2 = cv.imread(j, cv.IMREAD_GRAYSCALE)/255.
    #         fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(14,3), num=f"{i.split('/')[-1]} vs {j.split('/')[-1]}", tight_layout=True)
    #         ax[0].set_title('Real image')
    #         ax[0].imshow(im1, cmap='gray')
    #         ax[0].axis('off')

    #         ax[1].set_title('Generated image')
    #         ax[1].imshow(im2, cmap='gray')
    #         ax[1].axis('off')

    #         ax[2].set_title('DIFFERENCE')
    #         pos = ax[2].imshow(np.abs(im1-im2)**2, cmap='jet', vmin=0., vmax=1.)
    #         divider = make_axes_locatable(ax[2])
    #         cax = divider.append_axes("right", size="5%", pad=0.05)
    #         fig.colorbar(pos, cax=cax)
    #         ax[2].axis('off')
    #         plt.savefig(mse_path +'/comparison_real_generate' + '/'+ f"{i.split('/')[-1]} vs {j.split('/')[-1]}")
    #         # plt.show()
    #         plt.close()

    #         ssim = metrics['ssim']
    #         ms_ssim = metrics['fsim']
    #         rmse = metrics['rmse']

    #         print(ssim, rmse, ms_ssim)
    #         # plt.show()

    #     ssim_list.append(ssim)
    #     ms_ssim_list.append(ms_ssim)
    #     rmse_list.append(rmse)

    # structure_index_dict[key] = [ssim_list, ms_ssim_list, rmse_list]

    # ssim_list, ms_ssim_list, rmse_list = [], [], []
    # for key in structure_index_dict.keys():
    #     ssim = np.array(structure_index_dict[key][0])
    #     ms_ssim = np.array(structure_index_dict[key][1])
    #     rmse = np.array(structure_index_dict[key][2])

    #     print(key)
    #     ssim_list.append(ssim.mean())
    #     ms_ssim_list.append(ms_ssim.mean())
    #     rmse_list.append(rmse.mean())

    #     print(f'ssim {ssim.mean():.4f} +- {ssim.std(ddof=1):.4f}')
    #     print(f'ms-ssim {ms_ssim.mean():.4f} +- {ms_ssim.std(ddof=1):.4f}')
    #     print(f'rmse {rmse.mean():.4f} +- {rmse.std(ddof=1):.4f}')

    #     print('==========================================')

    # names = list(structure_index_dict.keys())
    # X_axis = np.arange(len(names))
    # plt.bar(X_axis, ssim_list, 0.3, tick_label=names)
    # plt.bar(X_axis - 0.3, ms_ssim_list, 0.3)
    # plt.bar(X_axis + 0.3, rmse_list, 0.3)

    # plt.show()