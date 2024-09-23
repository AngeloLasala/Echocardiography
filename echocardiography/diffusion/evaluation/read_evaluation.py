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
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 8), tight_layout=True)
    fig1, ax1 = plt.subplots(nrows=2, ncols=1, figsize=(10, 8), tight_layout=True)
    label_dict = {'-1.0': 'uncond LDM', '0.0': 'vanilla cond LDM ', '0.2': 'cond LDM - w=0.2',
                 '0.4': 'cond LDM - w=0.4', '0.6': 'cond LDM - w=0.6', '0.8': 'cond LDM - w=0.8',  '1.0': 'cond LDM - w=1.0', '2.0': 'cond LDM - w=2.0'}
    for guide_w in eval_dict.keys():
        mpe_pw, mpe_lvid, mpe_ivs = {}, {}, {}
        mae_rwt, mae_rst = {}, {}
        for epoch in eval_dict[guide_w].keys():
            rwt_real = eval_dict[guide_w][epoch]['rwt_real']
            rwt_gen = eval_dict[guide_w][epoch]['rwt_gen']
            rst_real = eval_dict[guide_w][epoch]['rst_real']
            rst_gen = eval_dict[guide_w][epoch]['rst_gen']
            eco_list_real = eval_dict[guide_w][epoch]['eco_list_real']
            eco_list_gen = eval_dict[guide_w][epoch]['eco_list_gen']

            ## absolute diffrece eco_list
            eco_list_diff = np.abs(eco_list_real - eco_list_gen)
            eco_percentages_error = np.abs(eco_list_diff / eco_list_real) * 100
            mean_absolute_diff, std_absolute_diff = np.mean(eco_list_diff, axis=0), np.std(eco_list_diff, ddof=1, axis=0)
            mpe_pw[float(epoch)] = np.mean(eco_percentages_error[0]) #mean_absolute_diff[0]
            mpe_lvid[float(epoch)] = np.mean(eco_percentages_error[1]) #mean_absolute_diff[1]
            mpe_ivs[float(epoch)] = np.mean(eco_percentages_error[2]) #mean_absolute_diff[2]

            ## median avarenge error rwt and rst
            mae_rwt[float(epoch)] = np.median(np.abs(rwt_real - rwt_gen))
            mae_rst[float(epoch)] = np.median(np.abs(rst_real - rst_gen))

            ## classification error of rwt and rst
            rwt_real_class = np.where(rwt_real > 0.42, 1, 0) # 1 > 0.42, 0 < 0.42
            rwt_gen_class = np.where(rwt_gen > 0.42, 1, 0)

            rst_real_class = np.where(rst_real > 0.42, 1, 0)
            rst_gen_class = np.where(rst_gen > 0.42, 1, 0)
            print('real:', rwt_real_class[:10])
            print('gen:',rwt_gen_class[:10])

        epoch_pw = list(mpe_pw.keys())
        epoch_pw.sort()
        mpe_pw = [mpe_pw[epoch] for epoch in epoch_pw]

        epoch_lvid = list(mpe_lvid.keys())
        epoch_lvid.sort()
        mpe_lvid = [mpe_lvid[epoch] for epoch in epoch_lvid]

        epoch_ivs = list(mpe_ivs.keys())
        epoch_ivs.sort()
        mpe_ivs = [mpe_ivs[epoch] for epoch in epoch_ivs]

        epoch_rwt = list(mae_rwt.keys())
        epoch_rwt.sort()
        mae_rwt = [mae_rwt[epoch] for epoch in epoch_rwt]

        epoch_rst = list(mae_rst.keys())
        epoch_rst.sort()
        mae_rst = [mae_rst[epoch] for epoch in epoch_rst]


        ax[0].set_title('PW', fontsize=20)
        ax[0].plot(epoch_pw, mpe_pw, label=label_dict[guide_w], lw=2, marker='o', markersize=10)
        ax[0].legend(fontsize=20)
        ax[1].set_title('LVID', fontsize=20)
        ax[1].plot(epoch_lvid, mpe_lvid, label=label_dict[guide_w], lw=2, marker='o', markersize=10)
        ax[2].set_title('IVS', fontsize=20)
        ax[2].plot(epoch_ivs, mpe_ivs, label=label_dict[guide_w], lw=2, marker='o', markersize=10)
        for aa in ax:
            aa.set_xlabel('Epoch', fontsize=20)
            aa.set_ylabel('Mean Percentage error', fontsize=20)
            aa.tick_params(axis='both', which='major', labelsize=18)
            aa.grid('dashed')

        ax1[0].set_title('RWT', fontsize=20)
        ax1[0].plot(epoch_rwt, mae_rwt, label=label_dict[guide_w], lw=2, marker='o', markersize=10)
        ax1[0].legend(fontsize=20)
        ax1[1].set_title('RST', fontsize=20)
        ax1[1].plot(epoch_rst, mae_rst, label=label_dict[guide_w], lw=2, marker='o', markersize=10)
        for aa in ax1:
            aa.set_xlabel('Epoch', fontsize=20)
            aa.set_ylabel('Median Absolute Error', fontsize=20)
            aa.tick_params(axis='both', which='major', labelsize=18)
            aa.grid('dashed')
            

        print(f'Epoch: {epoch}')
        print(f'PW: {mean_absolute_diff[0]:.4f} +- {std_absolute_diff[0]:.4f} - percentage error: {np.mean(eco_percentages_error[0]):.4f} ')
        print(f'LVID: {mean_absolute_diff[1]:.4f} +- {std_absolute_diff[1]:.4f} - percentage error: {np.mean(eco_percentages_error[1]):.4f} ')
        print(f'IVS: {mean_absolute_diff[2]:.4f} +- {std_absolute_diff[2]:.4f} - percentage error: {np.mean(eco_percentages_error[2]):.4f}\n')
    
        print('----------------------------------------------')

    ## read the fid_dict
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 8), tight_layout=True)
    label_dict = {'-1.0': 'uncond LDM', '0.0': 'vanilla cond LDM ', '0.2': 'cond LDM - w=0.2',
                 '0.4': 'cond LDM - w=0.4', '0.6': 'cond LDM - w=0.6', '0.8': 'cond LDM - w=0.8',  '1.0': 'cond LDM - w=1.0', '2.0': 'cond LDM - w=2.0'}
    for guide_w in fid_dict.keys():
        ax.plot(fid_dict[guide_w]['epoch'], fid_dict[guide_w]['fid'], label=label_dict[guide_w], lw=2, marker='o', markersize=10)
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
