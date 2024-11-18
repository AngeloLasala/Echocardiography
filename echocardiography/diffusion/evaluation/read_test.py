"""
Read the file of shape guided experiment for synthetic data augumentation
"""
import argparse
import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from echocardiography.regression.utils import get_corrdinate_from_heatmap, echocardiografic_parameters
import json

def original_keypoints(test_path, number_of_heatmaps):
    """
    List of real keypoints
    """

    keypoints_list = []
    rwt_list, rst_list = [], []
    for n in number_of_heatmaps:
        heat = np.load(os.path.join(test_path, f'heatmap_{n}.npy'))[5]
        keypoints = get_corrdinate_from_heatmap(heat)
        rwt, rst = echocardiografic_parameters(keypoints)
        keypoints_list.append(keypoints)
        rwt_list.append(rwt)
        rst_list.append(rst)
    return keypoints_list, rwt_list, rst_list

def plot_real_labels(rwt_list, rst_list):
    fig1, ax1 = plt.subplots(nrows=1, ncols=1, num='Real keypoints', figsize=(8,8), tight_layout=True)
    ax1.scatter(rwt_list, rst_list, s=100, color='grey', alpha=0.7)
    ax1.set_xlabel('RWT', fontsize=20)
    ax1.set_ylabel('RST', fontsize=20)
    ax1.grid(linestyle='--')
    ## set the font of ticks
    ax1.tick_params(axis='both', which='major', labelsize=18)
    # set the lim of x and y axis
    ax1.set_xlim(0.0, 1.8)
    ax1.set_ylim(0.0, 1.8)
    plt.show()

def plot_genereted_and_label(image_shape_guide, heat, i):
    image = cv2.imread(image_shape_guide)
    keypoints = get_corrdinate_from_heatmap(heat[i])
    rwt, rst = echocardiografic_parameters(keypoints)
    print(rwt, rst)

    fig, ax = plt.subplots(nrows=1, ncols=1, num=image_shape_guide.split('/')[-1], figsize=(12,8), tight_layout=True)
    ax.imshow(image)

    for j in range(6):
        ax.scatter(keypoints[j*2], keypoints[(j*2) + 1], s=300, color='yellow', marker='x')

    ax.axis('off')
    plt.show()

def main(args):
    """
    Read the heatmaps and the img from the test
    """
    test_path = os.path.join(args.par_dir, args.trial, args.experiment, 'test', f'w_{args.guide_w}', f'samples_ep_{args.epoch}')

    # return the list of file that satst with heatmaps
    heatmaps_list = [f for f in os.listdir(test_path) if f.startswith('heatmap')]
    number_of_heatmaps = [int(f.split('_')[-1].split('.')[0]) for f in heatmaps_list]

    real_keypoints, rwt_list, rst_list = original_keypoints(test_path, number_of_heatmaps)
    if args.show_plot: plot_real_labels(rwt_list, rst_list)

    label_dict = {}
    for n in number_of_heatmaps:
        print(f'Numeber of image: {n}')
        ## read heatmap
        heat = np.load(os.path.join(test_path, f'heatmap_{n}.npy'))
        
        label_heart = {'LVIDd': None, 'IVSd': None, 'LVPWd': None}
        for i in np.arange(5-3, 5+4, 1): # select the heatmap atounf the real keyponts, position 5th
            ## get image
            image_shape_guide = os.path.join(test_path, f'x0_{n}_{i}.png')
            
            
            image = cv2.imread(image_shape_guide)
            keypoints = get_corrdinate_from_heatmap(heat[i])
            rwt, rst = echocardiografic_parameters(keypoints)

            if rwt > 0.20 and rwt < 1.4:

                for j, heart_part in enumerate(['LVPWd', 'LVIDd', 'IVSd']):
                    x1 = keypoints[j*4]
                    y1 = keypoints[(j*4) + 1]
                    x2 = keypoints[(j*4) + 2]
                    y2 = keypoints[(j*4) + 3]
                    w, h = image.shape[1], image.shape[0]
                    label_heart[heart_part] = {'x1': int(x1), 'x2': int(x2), 'y1': int(y1), 'y2': int(y2), 'calc_value': int(0), 'width':int(w), 'height':int(h)}
                label_heart['split'] = 'generated_train'
                if args.show_plot: plot_genereted_and_label(image_shape_guide, heat, i)


                label_dict[f'x0_{n}_{i}'] = label_heart


    #save the label dict as json
    with open(os.path.join(test_path, 'label.json'), 'w') as f:
        json.dump(label_dict, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Read shape-guided experiments ')
    parser.add_argument('--par_dir', type=str, default='/home/angelo/Documents/Echocardiography/echocardiography/diffusion/trained_model/eco',
                         help="""parent directory of the folder with the evaluation file, it is the same as the trained model
                        local: /home/angelo/Documents/Echocardiography/echocardiography/diffusion/trained_model/eco
                        cluster: /leonardo_work/IscrC_Med-LMGM/Angelo/trained_model/diffusion/eco""")
    parser.add_argument('--trial', type=str, default='trial_1', help='trial name for saving the model, it is the trial folde that contain the VAE model')
    parser.add_argument('--experiment', type=str, default='cond_ldm', help="""name of expermient, it is refed to the type of condition and in general to the 
                                                                              hyperparameters (file .yaml) that is used for the training, it can be cond_ldm, cond_ldm_2, """)
    parser.add_argument('--guide_w', type=float, default=0.0, help='guide_w for the conditional model, w=-1 [unconditional], w=0 [vanilla conditioning], w>0 [guided conditional]')
    parser.add_argument('--epoch', type=int, default=99, help='epoch to sample, this is the epoch of cond ldm model') 
    parser.add_argument('--show_plot', action='store_true', help="show the prediction, default=False")
    args = parser.parse_args()

    
    main(args)