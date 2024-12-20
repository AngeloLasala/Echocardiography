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
import scipy.stats as stats
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from thop import profile, clever_format



from echocardiography.regression.utils import get_corrdinate_from_heatmap, get_corrdinate_from_heatmap_ellipses, echocardiografic_parameters
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

def show_prediction(image, label, output, target, size):
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
        
        ## plot only the heatmap
        for ch in range(label.shape[2]):
            fig3, ax3 = plt.subplots(nrows=1, ncols=2, num=f'Label and Prediction {ch}', figsize=(10,8), tight_layout=True)
            ax3[0].imshow(label[:,:,ch], cmap='jet', alpha=1.0)
            ax3[1].imshow(output[:,:,ch], cmap='jet', alpha=1.0)
            ax3[0].axis('off')
            ax3[1].axis('off')
        
        ## plot only the image
        fig2, ax2 = plt.subplots(nrows=1, ncols=1, num='Real Images', figsize=(10,8), tight_layout=True)
        ax2.imshow(image, cmap='gray', alpha=1.0)
        ax2.imshow(image, cmap='gray', alpha=1.0)
        ax2.axis('off')



        ## sum the channels to have a single label and prediction
        label_single = np.zeros(size)
        output_single = np.zeros(size)
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



def percentage_error(label, output, target, size, calc_value, original_shape, method='max_value'):
    """
    Compute the percentage error between the distance of 'LVPW', 'LVID', 'IVS'
    """
    label = label.copy()
    output = output.copy()

    if target == 'keypoints':
        label[0::2] = label[0::2] * size[1]
        label[1::2] = label[1::2] * size[0]
        output[0::2] = output[0::2] * size[1]
        output[1::2] = output[1::2] * size[0]

        distances_label, distances_output = [], []
        for i in range(3):
            x1, y1 = label_in[(i*4)], label_in[(i*4)+1]
            x2, y2 = label_in[(i*4)+2], label_in[(i*4)+3]
            distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            distances_label.append(distance)

            x1, y1 = output_in[(i*4)], output_in[(i*4)+1]
            x2, y2 = output_in[(i*4)+2], output_in[(i*4)+3]
            distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            distances_output.append(distance)

        ## compute the distance in cm
        cm_px_value = calc_value / np.array(distances_label) 
        distances_label_cm = cm_px_value * np.array(distances_label)
        distances_output_cm = cm_px_value * np.array(distances_output)

    if target == 'heatmaps':
        ## compute the coordinate of the max value in the heatmaps
        # #resize the label and output to the original shape noting that the shape is (chann, h,w) and the original shape is (h,w)
        resize = transforms.Resize(size=(int(original_shape[0]), int(original_shape[1])))
        label_res = [Image.fromarray(label[ch,:,:] * 255) for ch in range(label.shape[0])]
        label_res = [np.array(resize(ch)) / 255 for ch in label_res]

        output_res = [Image.fromarray(output[ch,:,:] * 255) for ch in range(output.shape[0])]
        output_res = [np.array(resize(ch)) / 255 for ch in output_res]

        label = np.array(label_res)
        output = np.array(output_res)

        label = get_corrdinate_from_heatmap(label)
        
        if method == 'max_value': output = get_corrdinate_from_heatmap(output)
        if method == 'ellipses': output = get_corrdinate_from_heatmap_ellipses(output)


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

        ## compute the distance in cm
        cm_px_value = calc_value / np.array(distances_label) 
        distances_label_cm = cm_px_value * np.array(distances_label)
        distances_output_cm = cm_px_value * np.array(distances_output)

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

        ## compute the distance in cm
        cm_px_value = calc_value / np.array(distances_label) 
        distances_label_cm = cm_px_value * np.array(distances_label)
        distances_output_cm = cm_px_value * np.array(distances_output)

    return distances_label, distances_output, distances_label_cm, distances_output_cm

def keypoints_error(label, output, target, size, method='max_value'):
    """
    Compute the error of the position of the keypoints
    """
    label = label.copy()
    output = output.copy()
    if target == 'keypoints':
        label[0::2] = label[0::2] * size[1]
        label[1::2] = label[1::2] * size[0]
        output[0::2] = output[0::2] * size[1]
        output[1::2] = output[1::2] * size[0]
        label, output = np.array(label), np.array(output)
        error = label - output

    if target == 'heatmaps':
        label = get_corrdinate_from_heatmap(label)
        if method == 'max_value': output = get_corrdinate_from_heatmap(output)
        if method == 'ellipses': output = get_corrdinate_from_heatmap_ellipses(output)

        ## compute the coordinate of the max value in the heatmaps
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

def echo_parameter_error(label, output, target, size, method='max_value'):
    """
    Compute the error of the echocardiografic parameters
    """
    label = label.copy()
    output = output.copy()
    if target == 'keypoints':
        label[0::2] = label[0::2] * size[1]
        label[1::2] = label[1::2] * size[0]
        output[0::2] = output[0::2] * size[1]
        output[1::2] = output[1::2] * size[0]  

        rwt_label, rst_label = echocardiografic_parameters(label)
        rwt_output, rst_output = echocardiografic_parameters(output)

        parameter_label = [rwt_label, rst_label]
        parameter_out = [rwt_output, rst_output]


    if target == 'heatmaps':
        ## compute the coordinate of the max value in the heatmaps
        label = get_corrdinate_from_heatmap(label)
        if method == 'max_value': output = get_corrdinate_from_heatmap(output)
        if method == 'ellipses': output = get_corrdinate_from_heatmap_ellipses(output)

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

def linear_fit(label, output, num='Regression plot'):
    """
    Compute the linear regression of the label vs output
    """
    x = label
    y = output

    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

    # Calculate R-squared
    r_squared = r_value**2

    # Calculate y_fit
    y_fit = slope * x + intercept

    # Calculate Chi-squared
    chi_squared = np.sum(((y - y_fit) ** 2) / y_fit)

    return slope, intercept, r_squared, chi_squared

def fit_linear(label, output):
    """
    Fit a linear regression between the label and the output
    """
    x = label
    y = output

    model = LinearRegression()
    model.fit(x.reshape(-1, 1), y.reshape(-1, 1))
    y_fit = model.predict(x.reshape(-1, 1))

    ## get the slope intersectio and r-squared
    slope = model.coef_[0][0]
    intercept = model.intercept_[0]
    r_squared = model.score(x.reshape(-1, 1), y.reshape(-1, 1))
    return slope, intercept, r_squared
   

    
def get_macs_parms(model, config, size):
    """
    Compute the MACs and parameters of the model
    """
    x_try = torch.randn(1, config['input_channels'], size[1], size[0]).to(device)
    macs, params = profile(model, inputs=(x_try,))
    macs, params = clever_format([macs, params], "%.3f")
    return macs, params
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Read the dataset')
    parser.add_argument('--data_path', type=str, default="/media/angelo/OS/Users/lasal/Desktop/DATA_h", help='Directory of the dataset, in general usinf DATA_h')
    parser.add_argument('--model_path', type=str, default='TRAINED_MODEL', help='Directory of models, i.e. trained_model')
    parser.add_argument('--batch', type=str, default='Batch2', help='Batch number of video folder, e.g. Batch1, Batch2, Batch3, Batch4')
    parser.add_argument('--phase', type=str, default='diastole', help='select the phase of the heart, diastole or systole. Default=diastole')
    parser.add_argument('--trial', type=str, default='trial_1', help='trial number to analyse')
    parser.add_argument('--method_center', type=str, default='max_value', help='how the keypoits are computed: max_value or ellipses')
    parser.add_argument('--split', type=str, default='test', help='select split: val or test, default = test')
    parser.add_argument('--show_plot', action='store_true', help="show the prediction, default=False")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ## retrive the information
    train_dir = os.path.join(args.model_path, args.batch, args.phase, args.trial)
    with open(os.path.join(train_dir, 'losses.json')) as json_file:
        losses = json.load(json_file)
    with open(os.path.join(train_dir, 'args.json')) as json_file:
        trained_args = json.load(json_file)
    cfg = train_config(trained_args['target'], 
                       threshold_wloss=trained_args['threshold_wloss'], 
                       model=trained_args['model'],
                       input_channels=trained_args['input_channels'],                       
                       device=device)
    size = tuple(trained_args['size'])


    test_set = EchoNetDataset(batch=args.batch, split=args.split, phase=args.phase, label_directory=None, data_path=args.data_path,
                              target=trained_args['target'], input_channels=cfg['input_channels'], size=size , augmentation=False,
                              original_shape=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)

    # for idx, i in enumerate(test_set):
    #     print(idx, len(i))
    
    ## load the model
    best_model = get_best_model(train_dir)
    print(f"Model: {trained_args['model']}, Best model: {best_model}")
    model_dir = os.path.join(args.model_path, args.batch, args.phase, args.trial)
    model = cfg['model'].to(device)
    

    # print(model)
    model.load_state_dict(torch.load(os.path.join(train_dir, f'model_{best_model}')))
    model.to(device)

    ## test the model
    model.eval()
    distances_label_list , distances_output_list = [], []
    distances_label_cm_list, distances_output_cm_list = [], []
    keypoints_error_list = []
    parameters_label_list, parameters_output_list = [], []
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            images, labels, calc_values, original_shapes = data
    
            images = images.to(device)
            labels = labels.to(device)
            calc_values = calc_values.to(device)
            original_shapes = original_shapes.to(device)
            outputs = model(images)
            if len(outputs) == 2: outputs = outputs[-1]
            outputs = outputs.to(device)

            ## convert images in numpy
            images = images.cpu().numpy().transpose((0, 2, 3, 1))
            outputs = outputs.cpu().numpy()
            labels = labels.cpu().numpy()
            original_shapes = original_shapes.cpu().numpy()
            calc_values = calc_values.cpu().numpy()

            for i in range(images.shape[0]):
                image = images[i]
                label = labels[i]
                output = outputs[i]
                calc_value = calc_values[i]
                original_shape = original_shapes[i]

                dist_label, dist_output, dist_label_cm, dist_output_cm = percentage_error(label=label, output=output, target=trained_args['target'], size=size, 
                                                            calc_value=calc_value, original_shape=original_shape, method=args.method_center)
                err = keypoints_error(label, output, target=trained_args['target'], size=size, method=args.method_center)
                parameter_label, parameter_out = echo_parameter_error(label, output, target=trained_args['target'], size=size, method=args.method_center)
                if args.show_plot:
                    show_prediction(image, label, output, target=trained_args['target'], size=size)
                    plt.show()
                
                distances_label_list.append(dist_label)
                distances_output_list.append(dist_output)
                distances_label_cm_list.append(dist_label_cm)
                distances_output_cm_list.append(dist_output_cm)
                keypoints_error_list.append(err)
                parameters_label_list.append(parameter_label)
                parameters_output_list.append(parameter_out)

    distances_label_list = np.array(distances_label_list)     ## LVWP, LVID, IVS annotation
    distances_output_list = np.array(distances_output_list)   ## LVWP, LVID, IVS prediction
    distances_label_cm_list = np.array(distances_label_cm_list) ## LVWP, LVID, IVS annotation in cm
    distances_output_cm_list = np.array(distances_output_cm_list) ## LVWP, LVID, IVS prediction in cm
    keypoints_error_list = np.array(keypoints_error_list)
    parameters_label_list = np.array(parameters_label_list)   ## RWT, RST annotation
    parameters_output_list = np.array(parameters_output_list) ## RWT, RST prediction

    plt.figure()
    plt.hist(distances_label_cm_list[:,1] - distances_output_cm_list[:,1], alpha=0.5, label='LVID')


    ## echo parameters error
    rwt_error = np.abs(parameters_label_list[:,0] - parameters_output_list[:,0])
    rst_error = np.abs(parameters_label_list[:,1] - parameters_output_list[:,1])

    ## save the parameters_output_list and parameters_label_list in np.array
    np.save(os.path.join(train_dir, f'{trained_args["target"]}_{args.method_center}_parameters_output_list_{args.split}.npy'), parameters_output_list)
    np.save(os.path.join(train_dir, f'{trained_args["target"]}_{args.method_center}_parameters_label_list_{args.split}.npy'), parameters_label_list)

    print('ECHOCARDIOGRAPHY PARAMETERS: RWT, RST')
    print(f'RWT error: mean={np.mean(rwt_error):.4f},  median={np.median(rwt_error):.4f} - 1 quintile {np.quantile(rwt_error, 0.25):.4f} - 3 quintile {np.quantile(rwt_error, 0.75):.4f}')
    print(f'RST error: mean={np.mean(rst_error):.4f},  median={np.median(rst_error):.4f} - 1 quintile {np.quantile(rst_error, 0.25):.4f} - 3 quintile {np.quantile(rst_error, 0.75):.4f}')
    print()

    ## Mean Percentage error and Positional error
    mpe = np.abs(distances_label_list - distances_output_list) / distances_label_list
    mpe = np.mean(mpe, axis=0)
    mae_cm = np.abs(distances_label_cm_list - distances_output_cm_list)
    positional_error = np.mean(keypoints_error_list, axis=0)

    slope_lvpw, intercept_lvpw, r_squared_lvpw, chi_squared_lvpw = linear_fit(distances_label_cm_list[:,0], distances_output_cm_list[:,0])
    slope_lvid, intercept_lvid, r_squared_lvid, chi_squared_lvid = linear_fit(distances_label_cm_list[:,1], distances_output_cm_list[:,1])
    slope_ivs, intercept_ivs, r_squared_ivs, chi_squared_ivs = linear_fit(distances_label_cm_list[:,2], distances_output_cm_list[:,2])
    print(f'Mean Percantace Error:  LVPW={mpe[0]:.4f}, LVID={mpe[1]:.4f}, IVS={mpe[2]:.4f}')
    print(f'Mean Absolute Error in cm:  LVPW={mae_cm[:,0].mean():.4f}, LVID={mae_cm[:,1].mean():.4f}, IVS={mae_cm[:,2].mean():.4f}')
    print(f'LVPW: slope={slope_lvpw:.4f}, intercept={intercept_lvpw:.4f}, R-squared={r_squared_lvpw:.4f}, Chi-squared={chi_squared_lvpw:.4f}')
    print(f'LVID: slope={slope_lvid:.4f}, intercept={intercept_lvid:.4f}, R-squared={r_squared_lvid:.4f}, Chi-squared={chi_squared_lvid:.4f}')
    print(f'IVS: slope={slope_ivs:.4f}, intercept={intercept_ivs:.4f}, R-squared={r_squared_ivs:.4f}, Chi-squared={chi_squared_ivs:.4f}')
    print()
    print(f'Positional_error: {positional_error}')
    print()

    ## regression plt distance label vs output
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(21,8), num=f'{trained_args["target"]}_{args.method_center} - Regression plot heart - {args.split}', tight_layout=True)
    
    # Scatter plot for LVPW
    
    ax[0].scatter(distances_label_cm_list[:,0], distances_output_cm_list[:,0], s=100, c='C3', label='LVPW', alpha=0.5)
    ax[0].plot(distances_label_cm_list[:,0], slope_lvpw * distances_label_cm_list[:,0] + intercept_lvpw, c='C3', label=f'fit LVPW',)
    ax[0].plot(distances_label_cm_list[:,0],distances_label_cm_list[:,0], c='black', linewidth=2)
    ax[0].grid('dotted')
    ax[0].set_xlabel('Real measure (cm)', fontsize=20)
    ax[0].set_ylabel('Predicted measure (cm)', fontsize=20)
    ax[0].legend(fontsize=20)
    
    
    # Scatter plot for LVID
    ax[1].scatter(distances_label_cm_list[:,1], distances_output_cm_list[:,1], s=100, c='C4',label='LVID', alpha=0.5)
    ax[1].plot(distances_label_cm_list[:,1], slope_lvid * distances_label_cm_list[:,1] + intercept_lvid, c='C4', label=f'fit LVID',)
    ax[1].plot(distances_label_cm_list[:,1],distances_label_cm_list[:,1], c='black', linewidth=2)
    ax[1].grid('dotted')
    ax[1].set_xlabel('Real measure (cm)', fontsize=20)
    ax[1].set_ylabel('Predicted measure (cm)', fontsize=20)
    ax[1].legend(fontsize=20)
    
    # Scatter plot for IVS
    ax[2].scatter(distances_label_cm_list[:,2], distances_output_cm_list[:,2], s=100, c='C5',label='IVS', alpha=0.5)
    ax[2].plot(distances_label_cm_list[:,2], slope_ivs * distances_label_cm_list[:,2] + intercept_ivs, c='C5', label=f'fit IVS',)
    ax[2].plot(distances_label_cm_list[:,2],distances_label_cm_list[:,2], c='black', linewidth=2)
    ax[2].grid('dotted')
    ax[2].set_xlabel('Real measure (cm)', fontsize=20)
    ax[2].set_ylabel('Predicted measure (cm)', fontsize=20)
    ax[2].legend(fontsize=20)
    for a in ax:
        a.tick_params(axis='both', which='major', labelsize=18)
    plt.savefig(os.path.join(train_dir, f'{trained_args["target"]}_{args.method_center} - Regression plot heart - {args.split}.png'))
    
    ## Regression plt label vs output
    # compute the linear regression
    slope_RWT, intercept_RWT, r_squared_RWT, chi_squared_RWT = linear_fit(parameters_label_list[:,0], parameters_output_list[:,0])
    slope_RST, intercept_RST, r_squared_RST, chi_squared_RST = linear_fit(parameters_label_list[:,1], parameters_output_list[:,1])
    print('Linear regression of the echocardiografic parameters:')
    print(f'RWT: slope={slope_RWT:.4f}, intercept={intercept_RWT:.4f}, R-squared={r_squared_RWT:.4f}, Chi-squared={chi_squared_RWT:.4f}')
    print(f'RST: slope={slope_RST:.4f}, intercept={intercept_RST:.4f}, R-squared={r_squared_RST:.4f}, Chi-squared={chi_squared_RST:.4f}')
    print()

    plt.figure(figsize=(8,8), num=f'{trained_args["target"]}_{args.method_center}- Regression plot - {args.split}', tight_layout=True)
    plt.scatter(parameters_label_list[:,0], parameters_output_list[:,0], s=100, c='C0', label='RWT', alpha=0.5)
    plt.plot(parameters_label_list[:,0], slope_RWT * parameters_label_list[:,0] + intercept_RWT, c='C0', label=f'fit RWT',)
    plt.plot(parameters_label_list[:,0],parameters_label_list[:,0], c='black', linewidth=2)
    plt.scatter(parameters_label_list[:,1], parameters_output_list[:,1], s=100, c='C1',label='RSD', alpha=0.5)
    plt.plot(parameters_label_list[:,1], slope_RST * parameters_label_list[:,1] + intercept_RST, c='C1', label=f'fit RST',)
    plt.axvline(x=0.42, color='r', linestyle='--')
    plt.grid('dotted')
    plt.legend()
    plt.xlabel('Label', fontsize=20)
    plt.ylabel('Prediction', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(fontsize=20)
    plt.savefig(os.path.join(train_dir, f'{trained_args["target"]}_{args.method_center}- Regression plot - {args.split}.png'))


    macs, params = get_macs_parms(model, cfg, size)
    print(f'MACS: {macs}, Parameters: {params}')
    

    ## create a file txt with all the printed string
    with open(os.path.join(train_dir, f'{trained_args["target"]}_results_{args.method_center}_{args.split}.txt'), 'w') as f:
        f.write(f'Batch: {args.batch}, Phase: {args.phase}, Trial: {args.trial}, Split: {args.split}\n')
        f.write(f'Model: {trained_args["model"]}, Best model: {best_model}\n')
        f.write(f'MACS: {macs}, Parameters: {params}\n')
        f.write('==================================================================================\n')
        f.write(f'Mean Percantace Error:  LVPW={mpe[0]:.4f}, LVID={mpe[1]:.4f}, IVS={mpe[2]:.4f}\n')
        f.write(f'Mean Absolute Error in cm:  LVPW={mae_cm[:,0].mean():.4f} +- {np.std(mae_cm[:,0], ddof=1):.4f}, LVID={mae_cm[:,1].mean():.4f} +- {np.std(mae_cm[:,1], ddof=1):.4f}, IVS={mae_cm[:,2].mean():.4f} +- {np.std(mae_cm[:,2], ddof=1):.4f}\n')
        f.write('\n')
        f.write(f'LVPW: slope={slope_lvpw:.4f}, intercept={intercept_lvpw:.4f}, R-squared={r_squared_lvpw:.4f}, Chi-squared={chi_squared_lvpw:.4f}\n')
        f.write(f'LVID: slope={slope_lvid:.4f}, intercept={intercept_lvid:.4f}, R-squared={r_squared_lvid:.4f}, Chi-squared={chi_squared_lvid:.4f}\n')
        f.write(f'IVS: slope={slope_ivs:.4f}, intercept={intercept_ivs:.4f}, R-squared={r_squared_ivs:.4f}, Chi-squared={chi_squared_ivs:.4f}\n')
        f.write('\n')
        f.write(f'RWT error: mean={np.mean(rwt_error):.4f},  median={np.median(rwt_error):.4f} - 1 quintile {np.quantile(rwt_error, 0.25):.4f} - 3 quintile {np.quantile(rwt_error, 0.75):.4f}\n')
        f.write(f'RST error: mean={np.mean(rst_error):.4f},  median={np.median(rst_error):.4f} - 1 quintile {np.quantile(rst_error, 0.25):.4f} - 3 quintile {np.quantile(rst_error, 0.75):.4f}\n')
        f.write(f'RWT: slope={slope_RWT:.4f}, intercept={intercept_RWT:.4f}, R-squared={r_squared_RWT:.4f}, Chi-squared={chi_squared_RWT:.4f}\n')
        f.write(f'RST: slope={slope_RST:.4f}, intercept={intercept_RST:.4f}, R-squared={r_squared_RST:.4f}, Chi-squared={chi_squared_RST:.4f}\n')
    


    ## Plot some plots to visualize the error
    plot_additional = False
    if plot_additional:
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