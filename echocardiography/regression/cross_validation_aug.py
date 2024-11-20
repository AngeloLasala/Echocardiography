"""
Implementing 5 fold cross validatio for data augm experiment
"""
import os
import argparse

import torch
from torchvision import transforms
import time
import tqdm
import random
from PIL import Image

import json
import numpy as np
import matplotlib.pyplot as plt

from echocardiography.regression.dataset import EchoNetDataset, convert_to_serializable, EchoNetGeneretedDataset, EchoNetConcatenate
from echocardiography.regression.models import ResNet50Regression, PlaxModel, UNet, UNet_up
from echocardiography.regression.losses import RMSELoss, WeightedRMSELoss, WeightedMSELoss
from echocardiography.regression.cfg import train_config
from echocardiography.regression.utils import get_corrdinate_from_heatmap, get_corrdinate_from_heatmap_ellipses, echocardiografic_parameters
from echocardiography.regression.test import linear_fit, percentage_error, keypoints_error, echo_parameter_error, show_prediction, get_best_model, show_prediction

## deactivate the warning of the torch
import warnings
warnings.filterwarnings("ignore")

def train_one_epoch(training_loader, model, loss, optimizer, device, tb_writer = None):
    """
    Funtion that performe the training of the model for one epoch
    """
    running_loss = 0. #torch.tensor(0.).to(device)
    loss = 0.           ## this have to be update with the last_loss
    time_load_start = time.time()
    for i, (inputs, labels, _, _) in enumerate(training_loader):
        # print(f' START BATCH {i}')
        ## load data
        # time_load_end = time.time()
        # time_load_tot = time_load_end - time_load_start
        # print(f'time loading data {i}: {time_load_tot:.5f}')

        # if i == 0: print(f'time loading data {i}: {time.time() - time_start}')
        # time_move_to_device = time.time()
        inputs, labels = inputs.to(device), labels.to(device)       # Every data instance is an input + label pair
        # time_move_to_device_end = time.time()
        # time_move = time_move_to_device_end - time_move_to_device
        # print(f'time move to device {i}: {time_move:.5f}')

        # time_loss = time.time()
        optimizer.zero_grad()                           # Zero your gradients for every batch!
        outputs = model(inputs)                         # Make predictions for this batch
        if len(outputs) == 2: outputs = outputs[-1]
        loss = loss_fn(outputs.float(), labels.float()) # Compute the loss and its gradientsù
        loss.backward()
        
        optimizer.step() # Adjust learning weights
        # time_loss_end = time.time()
        # time_loss_tot = time_loss_end - time_loss
        # print(f'time loss {i}: {time_loss_tot:.5f}')

        # Gather data and report
        # time_report = time.time()
        running_loss += loss.item()
        # print(f'running loss {i}: {type(running_loss)}') 
        # print(f'running loss {i}: {running_loss.device}') 

        # if i == len(training_loader) - 1:
        #      #torch.tensor((i + 1)).to(device) 
        #     # print(f'last batch loss: {last_loss}')
        #     # tb_x = epoch_index * len(training_loader) + i + 1     # add time step to the tensorboard
        #     # tb_writer.add_scalar('Loss/train', last_loss, tb_x)   # 
        #     running_loss = 0.
        # time_report_end = time.time()
        # time_report_tot = time_report_end - time_report
        # print(f'time report {i}: {time_report_tot:.5f}')
        # print(f'TOT {i}: {time_move + time_loss_tot + time_report_tot + time_load_tot:.5f}')
        # time_load_start = time.time()
        # print(f'END BATCH {i}: time {time_load_start - time_load_end:.5f}\n')
    last_loss = running_loss / len(training_loader)
        # time_end = time.time()

    return last_loss

def fit(training_loader, validation_loader,
        model, loss_fn, optimizer, 
        epochs=5, device='cpu', save_dir='./'):
    """
    Fit function to train the model

    Parameters
    ----------
    training_loader : torch.utils.data.DataLoader
        DataLoader object that contains the training dataset

    validation_loader : torch.utils.data.DataLoader
        DataLoader object that contains the validation dataset

    model : torch.nn.Module
        Model to train

    loss_fn : torch.nn.Module
        Loss function to use

    optimizer : torch.optim.Optimizer
        Optimizer to use

    epochs : int
        Number of epochs to train the model
    
    device : torch.device
        Device to use for training
    """
    EPOCHS = epochs
    best_vloss = 1_000_000.     # initialize the current best validation loss with a large value

    losses = {'train': [], 'valid': []}
    for epoch in range(EPOCHS):
        start = time.time()
        epoch += 1
        # print(f'EPOCH {epoch}')

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        print(f'Starting epoch {epoch}/{EPOCHS}')
        start_one_epoch = time.time()
        avg_loss = train_one_epoch(training_loader, model, loss_fn, optimizer, device=device)
        print(f'Epoch {epoch}/{EPOCHS} | Time: {time.time() - start_one_epoch:.2f}s')

        running_vloss = 0.0 
        model.eval() # Set the model to evaluation mode, disabling dropout and using population statistics for batch normalization.
        
        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for i, vdata in enumerate(validation_loader):
                vinputs, vlabels, _, _ = vdata
                vinputs = vinputs.to(device)
                vlabels = vlabels.to(device)

                voutputs = model(vinputs)
                if len(voutputs) == 2: voutputs = voutputs[-1]
                voutputs = voutputs.to(device)
                vloss = loss_fn(voutputs, vlabels).item()
                running_vloss += vloss

        avg_vloss = running_vloss / (i + 1)
        #convert the torch tensor  to float

        losses['train'].append(avg_loss)
        losses['valid'].append(avg_vloss)

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            # print('best model found')
            best_vloss = avg_vloss
            model_path = f'model_{epoch}'
            torch.save(model.state_dict(), os.path.join(save_dir, model_path))
        print(f'Epoch {epoch}/{EPOCHS} | Train Loss: {avg_loss:.6f} | Validation Loss: {avg_vloss:.6f} | Time: {time.time() - start:.2f}s\n')
    return losses

def testing_cross_validation(train_dir, test_loader, device):
    """
    Testing best model performance on test folder of the cross validation
    """
    ## retrive the cfg file and the best model
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

    best_model = get_best_model(train_dir)
    print(f'Best model: {best_model}')
    model = cfg['model'].to(device)
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
                                                            calc_value=calc_value, original_shape=original_shape, method='ellipses')
                err = keypoints_error(label, output, target=trained_args['target'], size=size, method='ellipses')
                parameter_label, parameter_out = echo_parameter_error(label, output, target=trained_args['target'], size=size, method='ellipses')
                # if True:
                #     show_prediction(image, label, output, target=trained_args['target'], size=size)
                #     plt.show()
                
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

    ## COMPUTE THE TEST METRICS
    ## echo parameters error
    rwt_error = np.abs(parameters_label_list[:,0] - parameters_output_list[:,0])
    rst_error = np.abs(parameters_label_list[:,1] - parameters_output_list[:,1])

    np.save(os.path.join(train_dir, f'{trained_args["target"]}_ellipses_parameters_output_list.npy'), parameters_output_list)
    np.save(os.path.join(train_dir, f'{trained_args["target"]}_ellipses_parameters_label_list.npy'), parameters_label_list)

    print('ECHOCARDIOGRAPHY PARAMETERS: RWT, RST')
    print(f'RWT error: mean={np.mean(rwt_error):.4f},  median={np.median(rwt_error):.4f} - 1 quintile {np.quantile(rwt_error, 0.25):.4f} - 3 quintile {np.quantile(rwt_error, 0.75):.4f}')
    print(f'RST error: mean={np.mean(rst_error):.4f},  median={np.median(rst_error):.4f} - 1 quintile {np.quantile(rst_error, 0.25):.4f} - 3 quintile {np.quantile(rst_error, 0.75):.4f}')
    print()

    ## Mean Percentage error and Positional error
    mpe = np.abs(distances_label_list - distances_output_list) / distances_label_list
    mpe = np.mean(mpe, axis=0)
    mae_cm = np.abs(distances_label_cm_list - distances_output_cm_list)
    positional_error = np.mean(keypoints_error_list, axis=0)

    np.save(os.path.join(train_dir, f'{trained_args["target"]}_ellipses_distances_output_list.npy'), distances_output_list)
    np.save(os.path.join(train_dir, f'{trained_args["target"]}_ellipses_distances_label_list.npy'), distances_label_list)
    np.save(os.path.join(train_dir, f'{trained_args["target"]}_ellipses_distances_output_cm_list.npy'), distances_output_cm_list)
    np.save(os.path.join(train_dir, f'{trained_args["target"]}_ellipses_distances_label_cm_list.npy'), distances_label_cm_list)

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

    # compute the linear regression
    slope_RWT, intercept_RWT, r_squared_RWT, chi_squared_RWT = linear_fit(parameters_label_list[:,0], parameters_output_list[:,0])
    slope_RST, intercept_RST, r_squared_RST, chi_squared_RST = linear_fit(parameters_label_list[:,1], parameters_output_list[:,1])
    print('Linear regression of the echocardiografic parameters:')
    print(f'RWT: slope={slope_RWT:.4f}, intercept={intercept_RWT:.4f}, R-squared={r_squared_RWT:.4f}, Chi-squared={chi_squared_RWT:.4f}')
    print(f'RST: slope={slope_RST:.4f}, intercept={intercept_RST:.4f}, R-squared={r_squared_RST:.4f}, Chi-squared={chi_squared_RST:.4f}')
    print()
    
    ## create a file txt with all the printed string
    with open(os.path.join(train_dir, f'{trained_args["target"]}_results_ellipses.txt'), 'w') as f:
        f.write(f'Model: {trained_args["model"]}, Best model: {best_model}\n')
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
    
def reset_model_weights(model):
    """
    Reset the model weights to the initial state before the training of new fold
    """
    for m in model.children():
        if hasattr(m, 'reset_parameters'):
            m.reset_parameters()
        else:
            reset_model_weights(m)
    
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Read the dataset')
    parser.add_argument('--data_path', type=str, default="/media/angelo/OS/Users/lasal/OneDrive - Scuola Superiore Sant'Anna/PhD_notes/Echocardiografy/DATA/regression/DATA", help='Directory of the dataset')
    parser.add_argument('--batch_dir', type=str, default='Batch2', help='Batch number of video folder, e.g. Batch1, Batch2, Batch3, Batch4')
    parser.add_argument('--phase', type=str, default='diastole', help='select the phase of the heart, diastole or systole')
    parser.add_argument('--target', type=str, default='keypoints', help='select the target to predict, e.g. keypoints, heatmaps, segmentation')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs to train the model')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for the dataloader')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for the optimizer')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='L2 regularization for the optimizer, default=0 that means no regularization')
    parser.add_argument('--threshold_wloss', type=float, default=0.5, help='Threshold for the weighted loss, if 0. all the weights are 1 and the lass fall back to regular ones')
    parser.add_argument('--save_dir', type=str, default='TRAINED_MODEL', help='Directory to save the model')
    parser.add_argument('--model', type=str, default=None, help='model architecture to use, e.g. resnet50, unet, plaxmodel')
    parser.add_argument('--input_channels', type=int, default=1, help='Number of input channels, default=1 for grayscale images, 3 for the RGB')
    parser.add_argument('--size', nargs='+', type=int, default= [256, 256] , help='Size of image, default is (256, 256), aspect ratio (240, 320)')
    parser.add_argument('--seed', type=int, default=42, help='Seed for reproducibility')
    parser.add_argument('--workers', type=int, default=4, help='Number of workers for the dataloader')
    parser.add_argument('--prefetch_factor', type=int, default=2, help='Number of workers for the dataloader')
    ##
    parser.add_argument('--par_dir_generate', type=str, default="/media/angelo/OS/Users/lasal/OneDrive - Scuola Superiore Sant'Anna/PhD_notes/Echocardiografy/trained_model/diffusion/eco",
                         help="""parent directory of the folder with the evaluation file, it is the same as the trained model
                        local: /home/angelo/Documents/Echocardiography/echocardiography/diffusion/trained_model/eco
                        cluster: /leonardo_work/IscrC_Med-LMGM/Angelo/trained_model/diffusion/eco""")
    parser.add_argument('--trial_generate', type=str, default='trial_2', help='trial name for saving the model, it is the trial folde that contain the VAE model')
    parser.add_argument('--experiment_generate', type=str, default='cond_ldm_1', help="""name of expermient, it is refed to the type of condition and in general to the 
                                                                              hyperparameters (file .yaml) that is used for the training, it can be cond_ldm, cond_ldm_2, """)
    parser.add_argument('--guide_w_generate', type=float, default=0.6, help='guide_w for the conditional model, w=-1 [unconditional], w=0 [vanilla conditioning], w>0 [guided conditional]')
    parser.add_argument('--epoch_generate', type=int, default=100, help='epoch to sample, this is the epoch of cond ldm model')
    parser.add_argument('--percentace', type=float, default=1.0, help='percentace of the data to use for the generated dataset')
    args = parser.parse_args()
    args.size = tuple(args.size)
    
    ## device and reproducibility    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(args.seed)
    cfg = train_config(args.target, threshold_wloss=args.threshold_wloss, model=args.model, input_channels=args.input_channels, device=device)
    print(f'Using device: {device}')
    
    print('start creating the dataset...')
    real_dataset = EchoNetConcatenate(data_path=args.data_path, batch=args.batch_dir, split=['train', 'val', 'test'], phase=args.phase,
                                        target=args.target, input_channels=args.input_channels, size=args.size, augmentation=False, 
                                        range_cv=None, original_shape=True)
    

    train_gen_set = EchoNetGeneretedDataset(par_dir=args.par_dir_generate, trial=args.trial_generate, experiment=args.experiment_generate, guide_w=args.guide_w_generate, epoch=args.epoch_generate,
                                     phase=args.phase, target=args.target, input_channels=args.input_channels, size=args.size, augmentation=True,
                                     percentace=args.percentace)


    #check if the save directory exist, if not create it - then create the subfolder for fold i
    save_dir_main = os.path.join(args.save_dir, args.batch_dir, args.phase, 'data_augumentation')
    if not os.path.exists(save_dir_main):
        save_dir_main = os.path.join(save_dir_main, 'trial_1')
        os.makedirs(os.path.join(save_dir_main))
    else:
        current_trial = len(os.listdir(save_dir_main))
        save_dir_main = os.path.join(save_dir_main, f'trial_{current_trial + 1}')
        os.makedirs(os.path.join(save_dir_main))

    ## 5 fold cross validation
    k = 5
    len_real_dataset = len(real_dataset)
    fold_size = len_real_dataset // k
    for i in range(k):
        ## use 1 fold for validation, 1 fold for test and the rest for training
        print(f'Fold_{i+1}')
        validation_set = EchoNetConcatenate(data_path=args.data_path, batch=args.batch_dir, split=['train', 'val', 'test'], phase=args.phase, 
                                    target=args.target, input_channels=args.input_channels, size=args.size, augmentation=False,
                                    range_cv=[i*fold_size, (i+1)*fold_size], original_shape=True)
        # validation_set = torch.utils.data.Subset(real_dataset, range(i*fold_size, (i+1)*fold_size))
        # print(range(i*fold_size, (i+1)*fold_size))
        if (i+2)*fold_size <= len_real_dataset: 
            test_set = EchoNetConcatenate(data_path=args.data_path, batch=args.batch_dir, split=['train', 'val', 'test'], phase=args.phase,
                                    target=args.target, input_channels=args.input_channels, size=args.size, augmentation=False,
                                    range_cv=[(i+1)*fold_size, (i+2)*fold_size], original_shape=True)
            train_set = EchoNetConcatenate(data_path=args.data_path, batch=args.batch_dir, split=['train', 'val', 'test'], phase=args.phase,
                                    target=args.target, input_channels=args.input_channels, size=args.size, augmentation=True,
                                    range_cv=[0, i*fold_size], original_shape=True) + EchoNetConcatenate(data_path=args.data_path, batch=args.batch_dir, split=['train', 'val', 'test'], phase=args.phase,
                                    target=args.target, input_channels=args.input_channels, size=args.size, augmentation=True,
                                    range_cv=[(i+2)*fold_size, len_real_dataset], original_shape=True)

            # train_set = torch.utils.data.Subset(real_dataset, range(0, i*fold_size)) + torch.utils.data.Subset(real_dataset, range((i+2)*fold_size, len_real_dataset))
            # test_set = torch.utils.data.Subset(real_dataset, range((i+1)*fold_size, (i+2)*fold_size))
            # print(range(0, i*fold_size), range((i+2)*fold_size, len_real_dataset))
            # print(range((i+1)*fold_size, (i+2)*fold_size))
        else:
            test_set = EchoNetConcatenate(data_path=args.data_path, batch=args.batch_dir, split=['train', 'val', 'test'], phase=args.phase,
                                    target=args.target, input_channels=args.input_channels, size=args.size, augmentation=False,
                                    range_cv=[0, fold_size], original_shape=True)
            train_set = EchoNetConcatenate(data_path=args.data_path, batch=args.batch_dir, split=['train', 'val', 'test'], phase=args.phase,
                                    target=args.target, input_channels=args.input_channels, size=args.size, augmentation=False,
                                    range_cv=[fold_size, i*fold_size], original_shape=True) + EchoNetConcatenate(data_path=args.data_path, batch=args.batch_dir, split=['train', 'val', 'test'], phase=args.phase,
                                    target=args.target, input_channels=args.input_channels, size=args.size, augmentation=False,
                                    range_cv=[(i+1)*fold_size, len_real_dataset], original_shape=True)
            # train_set = torch.utils.data.Subset(real_dataset, range(fold_size, i*fold_size)) + torch.utils.data.Subset(real_dataset, range((i+1)*fold_size, len_real_dataset))
            # test_set = torch.utils.data.Subset(real_dataset, range(0, fold_size))
            # print(range(fold_size, i*fold_size), range((i+1)*fold_size, len_real_dataset))
            # print(range(0, fold_size))
        
        ## add generated data on the training set
        if args.percentace > 0.0: train_set = torch.utils.data.ConcatDataset([train_set, train_gen_set])
        else : train_set = train_set
        

        training_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, prefetch_factor=args.prefetch_factor)
        validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True, prefetch_factor=args.prefetch_factor)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True, prefetch_factor=args.prefetch_factor)
        print(f'Train: {len(train_set)}, Validation: {len(validation_set)}, Test: {len(test_set)}')

        ## create folder for the i+1 cross validation folder
        save_dir = os.path.join(save_dir_main, f'fold_{i+1}')
        os.makedirs(os.path.join(save_dir))
        
        ## TRAIN
        print(f'Fold {i+1}) Start training...')
        loss_fn = cfg['loss']
        model = cfg['model'].to(device)
        print('start reinitialization')
        reset_model_weights(model)
        print('end reinitialization')
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        losses = fit(training_loader, validation_loader, model, loss_fn, optimizer, epochs=args.epochs, device=device, 
                    save_dir=save_dir)

        ## save the args dictionary in a file
        with open(os.path.join(save_dir,'losses.json'), 'w') as f:
            json.dump(losses, f, default=convert_to_serializable, indent=4)

        args_dict = vars(args)
        with open(os.path.join(save_dir,'args.json'), 'w') as f:
            json.dump(args_dict, f, indent=4)

       ## TEST
        print(f'Fold {i+1}) Start testing...')
        
        testing_cross_validation(train_dir=save_dir, test_loader=test_loader, device=device)
