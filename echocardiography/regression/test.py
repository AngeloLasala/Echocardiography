"""
Main file to test the PLAX regression model
"""
import os
import argparse

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from datetime import datetime
import time
import tqdm
# from cfg import train_config
from echocardiography.regression.cfg import train_config

import json
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tqdm
from scipy.stats import multivariate_normal
from scipy import ndimage
import cv2
import math
import numpy as np

from echocardiography.regression.utils import get_corrdinate_from_heatmap, echocardiografic_parameters
from echocardiography.regression.dataset import EchoNetDataset, convert_to_serializable

def get_best_model(train_dir):
    """
    Function to find the best model in the directory
    """
    best_model = 0
    for i in os.listdir(train_dir):
        if 'model' in i:
            model = i.split('_')[-1].split('.')[0]
            if int(model) > best_model:
                best_model = int(model)
    return best_model

def show_prediction(image, label, output, target):
    """
    Show the prediction of PLAX keypoits based on the type of the target

    Parameters
    ----------
    image: np.array
        image in the range -1, 1
    """
    image = (image * 0.5) + 0.5

    if target == 'keypoints':
        plt.figure(figsize=(14,14), num='Example')
        plt.imshow(image, cmap='gray', alpha=1.0)

        #labels
        plt.scatter(label[0] * image.shape[1], label[1] * image.shape[0], color='green', marker='o',s=150, alpha=0.7)
        plt.scatter(label[2] * image.shape[1], label[3] * image.shape[0], color='green', marker='o',s=150, alpha=0.7)

        plt.scatter(label[4] * image.shape[1], label[5] * image.shape[0], color='red', marker='o',s=150, alpha=0.7)
        plt.scatter(label[6] * image.shape[1], label[7] * image.shape[0], color='red', marker='o',s=150, alpha=0.7)

        plt.scatter(label[8] * image.shape[1], label[9] * image.shape[0], color='blue', marker='o',s=150, alpha=0.7)
        plt.scatter(label[10] * image.shape[1], label[11] * image.shape[0], color='blue', marker='o',s=150, alpha=0.7)

        #predictions
        plt.scatter(output[0] * image.shape[1], output[1] * image.shape[0], color='green', marker='*',s=150, alpha=0.7)
        plt.scatter(output[2] * image.shape[1], output[3] * image.shape[0], color='green', marker='*',s=150, alpha=0.7)

        plt.scatter(output[4] * image.shape[1], output[5] * image.shape[0], color='red', marker='*',s=150, alpha=0.7)
        plt.scatter(output[6] * image.shape[1], output[7] * image.shape[0], color='red', marker='*',s=150, alpha=0.7)

        plt.scatter(output[8] * image.shape[1], output[9] * image.shape[0], color='blue', marker='*',s=150, alpha=0.7)
        plt.scatter(output[10] * image.shape[1], output[11] * image.shape[0], color='blue', marker='*',s=150, alpha=0.7)
        plt.axis('off')

    else:
        ## for 'segmentation' and 'heatmaps' the output shape is (num_classes, h, w)
        # put the chaneel on the last shape
        label = label.transpose((1, 2, 0))
        output = output.transpose((1, 2, 0))

        fig, axes = plt.subplots(nrows=2, ncols=4, num='example', figsize=(26,14), tight_layout=True)
        num_classes = [0,1,3,5] # for the visualizazion i aviod the superimpose classes

        ## sum the channels to have a single label and prediction
        label_single = np.zeros((256,256))
        output_single = np.zeros((256,256))
        for i in num_classes:
            label_single += label[:,:,i]
            output_single += output[:,:,i]

        # real image on each spot
        for ax in axes: ## 2 elements == label and prediction
            for i in range(len(num_classes)): ## 4 eleminets, num of classes
                ax[i].imshow(image, cmap='gray', alpha=1.0)
                ax[i].axis('off')

        # plot the label
        for i, ch in enumerate(num_classes):
            axes[0,i].imshow(label[:,:,ch], cmap='jet', alpha=0.5)

        # plot the ouput
        if target == 'segmentation':
            output = (output > 0.5).astype(np.float32)

        for i, ch in enumerate(num_classes):
            axes[1,i].imshow(output[:,:,ch], cmap='jet', alpha=0.5)

        fig1, ax1 = plt.subplots(nrows=1, ncols=2, num='Example', figsize=(26,14), tight_layout=True)
        ax1[0].imshow(image, cmap='gray', alpha=1.0)
        ax1[0].imshow(label_single, cmap='jet', alpha=0.3)
        ax1[1].imshow(image, cmap='gray', alpha=1.0)
        ax1[1].imshow(output_single, cmap='jet', alpha=0.3)
        ax1[0].axis('off')
        ax1[1].axis('off')
        plt.show()



def percentage_error(label, output, target):
    """
    Compute the percentage error between the distance of 'LVPW', 'LVID', 'IVS'
    """
    if target == 'keypoints':
        label, output = label * 256., output * 256.    
        distances_label, distances_output = [], []
        for i in range(3):
            x1, y1 = label[(i*4)], label[(i*4)+1]
            x2, y2 = label[(i*4)+2], label[(i*4)+3]
            distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            distances_label.append(distance)

            x1, y1 = output[(i*4)], output[(i*4)+1]
            x2, y2 = output[(i*4)+2], output[(i*4)+3]
            distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            distances_output.append(distance)

    if target == 'heatmaps':
        ## compute the coordinate of the max value in the heatmaps
        label = get_corrdinate_from_heatmap(label)
        output = get_corrdinate_from_heatmap(output)

        distances_label, distances_output = [], []
        for i in range(3):
            x1, y1 = label[(i*4)], label[(i*4)+1]
            x2, y2 = label[(i*4)+2], label[(i*4)+3]
            distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            distances_label.append(distance)

            x1, y1 = output[(i*4)], output[(i*4)+1]
            x2, y2 = output[(i*4)+2], output[(i*4)+3]
            distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            distances_output.append(distance)

    if target == 'segmentation':
        ## compute the coordinate of the max value in the heatmaps
        label = get_corrdinate_from_heatmap(label)
        output = get_corrdinate_from_heatmap(output)

        distances_label, distances_output = [], []
        for i in range(3):
            x1, y1 = label[(i*4)], label[(i*4)+1]
            x2, y2 = label[(i*4)+2], label[(i*4)+3]
            distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            distances_label.append(distance)

            x1, y1 = output[(i*4)], output[(i*4)+1]
            x2, y2 = output[(i*4)+2], output[(i*4)+3]
            distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            distances_output.append(distance)

    return distances_label, distances_output

def keypoints_error(label, output, target):
    """
    Compute the error of the position of the keypoints
    """
    if target == 'keypoints':
        label, output = label * 256., output * 256.
        label, output = np.array(label), np.array(output)
        error = label - output

    if target == 'heatmaps':
        ## compute the coordinate of the max value in the heatmaps
        label = get_corrdinate_from_heatmap(label)
        output = get_corrdinate_from_heatmap(output)
        label, output = np.array(label), np.array(output)
        error = label - output

    if target == 'segmentation':
        ## compute the coordinate of the max value in the heatmapsù
        ## to do.. modify to fet the ellipses and get the coordinate
        label = get_corrdinate_from_heatmap(label)
        output = get_corrdinate_from_heatmap(output)
        label, output = np.array(label), np.array(output)
        error = label - output
    return error

def echo_parameter_error(label, output, target):
    """
    Compute the error of the echocardiografic parameters
    """
    if target == 'keypoints':
        label, output = label * 256., output * 256.    

        rwt_label, rst_label = echocardiografic_parameters(label)
        rwt_output, rst_output = echocardiografic_parameters(output)

        parameter_label = [rwt_label, rst_label]
        parameter_out = [rwt_output, rst_output]


    if target == 'heatmaps':
        ## compute the coordinate of the max value in the heatmaps
        label = get_corrdinate_from_heatmap(label)
        output = get_corrdinate_from_heatmap(output)

        rwt_label, rst_label = echocardiografic_parameters(label)
        rwt_output, rst_output = echocardiografic_parameters(output)

        parameter_label = [rwt_label, rst_label]
        parameter_out = [rwt_output, rst_output]

    if target == 'segmentation':
        ## compute the coordinate of the max value in the heatmaps
        label = get_corrdinate_from_heatmap(label)
        output = get_corrdinate_from_heatmap(output)

        rwt_label, rst_label = echocardiografic_parameters(label)
        rwt_output, rst_output = echocardiografic_parameters(output)

        parameter_label = [rwt_label, rst_label]
        parameter_out = [rwt_output, rst_output]

    return parameter_label, parameter_out

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Read the dataset')
    parser.add_argument('--data_dir', type=str, default="/media/angelo/OS/Users/lasal/Desktop/Phd notes/Echocardiografy/EchoNet-LVH", help='Directory of the dataset')
    parser.add_argument('--batch', type=str, default='Batch2', help='Batch number of video folder, e.g. Batch1, Batch2, Batch3, Batch4')
    parser.add_argument('--phase', type=str, default='diastole', help='select the phase of the heart, diastole or systole')
    parser.add_argument('--trial', type=str, default='trial_1', help='trial number to analyse')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ## retrive the information
    train_dir = os.path.join('TRAINED_MODEL', args.batch, args.phase, args.trial)
    with open(os.path.join(train_dir, 'losses.json')) as json_file:
        losses = json.load(json_file)
    with open(os.path.join(train_dir, 'args.json')) as json_file:
        trained_args = json.load(json_file)
    cfg = train_config(trained_args['target'], 
                       threshold_wloss=trained_args['threshold_wloss'], 
                       model=trained_args['model'],
                       device=device)


    test_set = EchoNetDataset(batch=args.batch, split='test', phase=args.phase, label_directory=None,
                              target=trained_args['target'], augmentation=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)

    ## load the model
    best_model = get_best_model(train_dir)
    print(f"Model: {trained_args['model']}, Best model: {best_model}")
    model_dir = os.path.join('TRAINED_MODEL', args.batch, args.phase, args.trial)
    model = cfg['model'].to(device)
    print(model)
    model.load_state_dict(torch.load(os.path.join(train_dir, f'model_{best_model}')))
    model.to(device)

    ## test the model
    model.eval()
    distances_label_list , distances_output_list = [], []
    keypoints_error_list = []
    parameters_label_list, parameters_output_list = [], []
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images).to(device)

            ## convert images in numpy
            images = images.cpu().numpy().transpose((0, 2, 3, 1))
            outputs = outputs.cpu().numpy()
            labels = labels.cpu().numpy()

            for i in range(images.shape[0]):
                image = images[i]
                label = labels[i]
                output = outputs[i]
                # show_prediction(image, label, output, target=trained_args['target'])
                
                
                dist_label, dist_output = percentage_error(label, output, target=trained_args['target'])
                err = keypoints_error(label, output, target=trained_args['target'])
                parameter_label, parameter_out = echo_parameter_error(label, output, target=trained_args['target'])
                # plt.show()
                
                distances_label_list.append(dist_label)
                distances_output_list.append(dist_output)
                keypoints_error_list.append(err)
                parameters_label_list.append(parameter_label)
                parameters_output_list.append(parameter_out)

    distances_label_list = np.array(distances_label_list)
    distances_output_list = np.array(distances_output_list)
    keypoints_error_list = np.array(keypoints_error_list)
    parameters_label_list = np.array(parameters_label_list)
    parameters_output_list = np.array(parameters_output_list)

    ## echo parameters error
    rwt_error = np.abs(parameters_label_list[:,0] - parameters_output_list[:,0])
    rst_error = np.abs(parameters_label_list[:,1] - parameters_output_list[:,1])
    print(f'RWT error: mean={np.mean(rwt_error):.4f},  median={np.median(rwt_error):.4f} - 1 quintile {np.quantile(rwt_error, 0.25):.4f} - 3 quintile {np.quantile(rwt_error, 0.75):.4f}')
    print(f'RST error: mean={np.mean(rst_error):.4f},  median={np.median(rst_error):.4f} - 1 quintile {np.quantile(rst_error, 0.25):.4f} - 3 quintile {np.quantile(rst_error, 0.75):.4f}')

    ## Mean Percentage error and Positional error
    mpe = np.abs(distances_label_list - distances_output_list) / distances_label_list
    mpe = np.mean(mpe, axis=0)
    positional_error = np.mean(keypoints_error_list, axis=0)
    import matplotlib.pyplot as plt

    print(f'Mean Percantace Error: {mpe}')
    print(f'Positional_error: {positional_error}')

    ## Regression plt label vs output
    plt.figure(figsize=(8,8), num=f'{trained_args["target"]} - Regression plot', tight_layout=True)
    plt.scatter(parameters_label_list[:,0], parameters_output_list[:,0], s=100, c='C0', label='RWT')
    plt.plot(parameters_label_list[:,0],parameters_label_list[:,0], c='black')
    plt.scatter(parameters_label_list[:,1], parameters_output_list[:,1], s=100, c='C1',label='RSD')
    plt.axvline(x=0.42, color='r', linestyle='--')
    plt.grid('dotted')
    plt.legend()
    plt.xlabel('Label', fontsize=20)
    plt.ylabel('Prediction', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(fontsize=20)
    


    ## Plot some plots to visualize the error
    title_name = {0:'LVPW', 1:'LVID', 2:'IVS'}
    for point in range(3):
        plt.figure(figsize=(14,14), num=f'Positional error histogram {title_name[point]}', tight_layout=True)
        plt.title(f'Error of the keypoints {title_name[point]}', fontsize=20)
        plt.hist(keypoints_error_list[:,point*4], alpha=0.5, label='x1')
        plt.hist(keypoints_error_list[:,(point*4)+1], alpha=0.5, label='y1')
        plt.hist(keypoints_error_list[:,(point*4)+2], alpha=0.5, label='x2')
        plt.hist(keypoints_error_list[:,(point*4)+3], bins=50, alpha=0.5, label='y2')
        plt.xlabel('Error', fontsize=20)
        plt.ylabel('Frequency', fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.legend(fontsize=20)
        plt.grid()
    plt.show()