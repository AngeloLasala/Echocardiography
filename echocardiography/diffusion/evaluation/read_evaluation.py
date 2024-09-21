"""
Read the evaluation files for FID and hypertrophy consistency
"""
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def read_hypertrophy_consistency(hypertrophy_path):
    """
    Read the hypertrophy consistency file
    """

    eval_dict = {}
    for guide_w in os.listdir(hypertrophy_path):
        epoch_list = []
        for eval_path in os.listdir(os.path.join(hypertrophy_path, guide_w)):
            epoch_list.append(eval_path.split('.')[0].split('_')[-1])

        epoch_list = np.unique(np.array(epoch_list))
        epoch_dict = {}
        for epoch in epoch_list:

            # read the file rwt and rst real and fake
            if f'rwt_real_{epoch}.npy' in os.listdir(os.path.join(hypertrophy_path, guide_w)):
                rwt_real = np.load(os.path.join(hypertrophy_path, guide_w, f'rwt_real_{epoch}.npy'))
            else:
                print(f'rwt_real_{epoch}.npy is not in the folder {os.path.join(hypertrophy_path, guide_w)}')
                rwt_real = None

            if f'rwt_gen_{epoch}.npy' in os.listdir(os.path.join(hypertrophy_path, guide_w)):
                rwt_gen = np.load(os.path.join(hypertrophy_path, guide_w, f'rwt_gen_{epoch}.npy'))
            else:
                print(f'rwt_gen_{epoch}.npy is not in the folder {os.path.join(hypertrophy_path, guide_w)}')
                rwt_gen = None  

            if f'rst_real_{epoch}.npy' in os.listdir(os.path.join(hypertrophy_path, guide_w)):
                rst_real = np.load(os.path.join(hypertrophy_path, guide_w, f'rst_real_{epoch}.npy'))
            else:
                print(f'rst_real_{epoch}.npy is not in the folder {os.path.join(hypertrophy_path, guide_w)}')
                rst_real = None
            
            if f'rst_gen_{epoch}.npy' in os.listdir(os.path.join(hypertrophy_path, guide_w)):
                rst_gen = np.load(os.path.join(hypertrophy_path, guide_w, f'rst_gen_{epoch}.npy'))
            else:
                print(f'rst_gen_{epoch}.npy is not in the folder {os.path.join(hypertrophy_path, guide_w)}')
                rst_gen = None

            
            # read the file eco list
            if f'eco_list_real_{epoch}.npy' in os.listdir(os.path.join(hypertrophy_path, guide_w)):
                eco_list_real = np.load(os.path.join(hypertrophy_path, guide_w, f'eco_list_real_{epoch}.npy'))
            else:
                print(f'eco_list_real_{epoch}.npy is not in the folder {os.path.join(hypertrophy_path, guide_w)}')
                eco_list_real = None

            if f'eco_list_gen_{epoch}.npy' in os.listdir(os.path.join(hypertrophy_path, guide_w)):
                eco_list_gen = np.load(os.path.join(hypertrophy_path, guide_w, f'eco_list_gen_{epoch}.npy'))
            else:
                print(f'eco_list_gen_{epoch}.npy is not in the folder {os.path.join(hypertrophy_path, guide_w)}')
                eco_list_gen = None

            epoch_dict[epoch] = {'rwt_real': rwt_real, 'rwt_gen': rwt_gen, 'rst_real': rst_real, 'rst_gen': rst_gen, 'eco_list_real': eco_list_real, 'eco_list_gen': eco_list_gen}
        
        eval_dict[guide_w.split('_')[-1]] = epoch_dict
    return eval_dict

def read_fid_value(experiment):
    """
    Read the FID value
    """
    # find all folders that start with 'w_'
    fid_dict = {}
    for folder in os.listdir(experiment):
        if folder.startswith('w_'):
            if 'FID_score.txt' in os.listdir(os.path.join(experiment, folder)):
                fid_file = os.path.join(experiment, folder, 'FID_score.txt')
                guide_w = folder.split('_')[1]
                with open(fid_file, 'r') as f:
                    fid_value = f.read().split('\n')
                    epoch = folder.split(',')[0]
                
                epoch_list, fid_list = [], []
                for line in fid_value:
                    if line == '':
                        continue
                    epoch = float(line.split(',')[0].split(':')[1])
                    fid = float(line.split(',')[1].split(':')[1])
                    fid_list.append(fid)
                    epoch_list.append(epoch)

                fid_dict[guide_w] = {'epoch': np.array(epoch_list), 'fid': np.array(fid_list)}     
    return fid_dict


def read_eval_fid_dict(eval_dict, fid_dict):
    """
    read and plot the results in the eval and fid dictionary
    """
    ## read the eval_dict
    for guide_w in eval_dict.keys():
        for epoch in eval_dict[guide_w].keys():
            
            pass


    ## read the fid_dict
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 8), tight_layout=True)
    for guide_w in fid_dict.keys():
        ax.plot(fid_dict[guide_w]['epoch'], fid_dict[guide_w]['fid'], label=guide_w, lw=4)
        ax.set_xlabel('Epoch', fontsize=20)
        ax.set_ylabel('FID', fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=18)
        ax.grid('dashed')
        ax.legend(fontsize=20)
    plt.show()


def main(args_parser):
    experiment_dir = os.path.join(args_parser.par_dir, args_parser.trial, args_parser.experiment)

    # get hypertrophy consistency path
    hypertrophy_path = os.path.join(experiment_dir, 'hypertrophy_evaluation')
    eval_dict = read_hypertrophy_consistency(hypertrophy_path)

    # get FID path
    fid_dict = read_fid_value(experiment_dir)
    

    # read the eval and the fid dict
    read_eval_fid_dict(eval_dict, fid_dict)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for reading evaluation files')
    parser.add_argument('--par_dir', type=str, default='/home/angelo/Documents/Echocardiography/echocardiography/diffusion/trained_model/eco',
                         help="""parent directory of the folder with the evaluation file, it is the same as the trained model
                        local: /home/angelo/Documents/Echocardiography/echocardiography/diffusion/trained_model/eco
                        cluster: /leonardo_work/IscrC_Med-LMGM/Angelo/trained_model/diffusion/eco""")
    parser.add_argument('--trial', type=str, default='trial_1', help='trial name for saving the model, it is the trial folde that contain the VAE model')
    parser.add_argument('--experiment', type=str, default='cond_ldm', help="""name of expermient, it is refed to the type of condition and in general to the 
                                                                              hyperparameters (file .yaml) that is used for the training, it can be cond_ldm, cond_ldm_2, """)
    args = parser.parse_args()


    main(args_parser=args)
