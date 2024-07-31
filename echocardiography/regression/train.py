"""
Main file to train the PLAX regression model. The code provides flags for hyperparameters tuning and directory path where load the data 
as well as save the output. Here a list of important flags:

- DATA flags:
    --data_path: Directory of the dataset, can be a local path or a remote one
    --save_dir: Directory to save the model
    --batch_dir: Batch number of video folder, e.g. Batch1, Batch2, Batch3, Batch4
    --phase: select the phase of the heart, diastole or systole

- HYPERPARAMETERS flags:
    --epochs: Number of epochs to train the model
    --batch_size: Batch size for the dataloader
    --lr: Learning rate for the optimizer
    --weight_decay: L2 regularization for the optimizer, default=0 that means no regularization
    --threshold_wloss: Threshold for the weighted loss, if 0. all the weights are 1 and the lass fall back to regular ones
    --input_channels: Number of input channels, default=1 for grayscale images, 3 for the RGB
    --size: Size of image, default is (256, 256), aspect ratio (240, 320)

- MODEL flags:
    --target: select the target to predict, e.g. keypoints, heatmaps, segmentation
    --model: model architecture to use, e.g. resnet50, unet, plaxmodel
"""
import os
import argparse

import torch
from torchvision import transforms
import time
import tqdm
import random

import json
import numpy as np
import matplotlib.pyplot as plt

from echocardiography.regression.dataset import EchoNetDataset, convert_to_serializable
from echocardiography.regression.models import ResNet50Regression, PlaxModel, UNet, UNet_up
from echocardiography.regression.losses import RMSELoss, WeightedRMSELoss, WeightedMSELoss
from echocardiography.regression.cfg import train_config

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
    for i, (inputs, labels) in enumerate(training_loader):
        ## load data
        time_load_end = time.time()
        time_load_tot = time_load_end - time_load_start
        print(f'time loading data {i}: {time_load_tot:.5f}')

        # if i == 0: print(f'time loading data {i}: {time.time() - time_start}')
        time_move_to_device = time.time()
        inputs, labels = inputs.to(device), labels.to(device)       # Every data instance is an input + label pair
        time_move_to_device_end = time.time()
        time_move = time_move_to_device_end - time_move_to_device
        print(f'time move to device {i}: {time_move:.5f}')

        time_loss = time.time()
        optimizer.zero_grad()                           # Zero your gradients for every batch!
        outputs = model(inputs)                         # Make predictions for this batch
        if len(outputs) == 2: outputs = outputs[-1]
        loss = loss_fn(outputs.float(), labels.float()) # Compute the loss and its gradientsù
        print(f'loss {i}: {loss.device}')
        loss.backward()
        
        optimizer.step() # Adjust learning weights
        time_loss_end = time.time()
        time_loss_tot = time_loss_end - time_loss
        print(f'time loss {i}: {time_loss_tot:.5f}')

        # Gather data and report
        time_report = time.time()
        running_loss += loss.item()
        print(f'running loss {i}: {type(running_loss)}') 
        # print(f'running loss {i}: {running_loss.device}') 

        # if i == len(training_loader) - 1:
        #      #torch.tensor((i + 1)).to(device) 
        #     # print(f'last batch loss: {last_loss}')
        #     # tb_x = epoch_index * len(training_loader) + i + 1     # add time step to the tensorboard
        #     # tb_writer.add_scalar('Loss/train', last_loss, tb_x)   # 
        #     running_loss = 0.
        time_report_end = time.time()
        time_report_tot = time_report_end - time_report
        print(f'time report {i}: {time_report_tot:.5f}')
        print(f'TOT {i}: {time_move + time_loss_tot + time_report_tot + time_load_tot:.5f}\n')
        time_load_start = time.time()
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
                vinputs, vlabels = vdata
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
    train_set = EchoNetDataset(batch=args.batch_dir, split='train', phase=args.phase, label_directory=None, data_path=args.data_path,
                              target=args.target, input_channels=args.input_channels, size=args.size, augmentation=False)

    validation_set = EchoNetDataset(batch=args.batch_dir, split='val', phase=args.phase, label_directory=None, data_path=args.data_path,
                              target=args.target, input_channels=args.input_channels, size=args.size, augmentation=False)
    
    training_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, prefetch_factor=args.prefetch_factor)
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True, prefetch_factor=args.prefetch_factor)
    
    ## TRAIN
    print('start training...')
    loss_fn = cfg['loss']
    model = cfg['model'].to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print(loss_fn)
    print(model) 

    #check if the save directory exist, if not create it
    save_dir = os.path.join(args.save_dir, args.batch_dir, args.phase)
    if not os.path.exists(save_dir):
        save_dir = os.path.join(save_dir, 'trial_1')
        os.makedirs(os.path.join(save_dir))
    else:
        current_trial = len(os.listdir(save_dir))
        save_dir = os.path.join(save_dir, f'trial_{current_trial + 1}')
        os.makedirs(os.path.join(save_dir))


    losses = fit(training_loader, validation_loader, model, loss_fn, optimizer, epochs=args.epochs, device=device, 
                save_dir=save_dir)

    ## save the args dictionary in a file
    with open(os.path.join(save_dir,'losses.json'), 'w') as f:
        json.dump(losses, f, default=convert_to_serializable, indent=4)

    args_dict = vars(args)
    with open(os.path.join(save_dir,'args.json'), 'w') as f:
        json.dump(args_dict, f, indent=4)

    ## plot the loss
    fig, ax = plt.subplots(figsize=(10, 6), num='Losses')
    ax.plot(np.array(losses['train']), label='train')
    ax.plot(np.array(losses['valid']), label='valid')
    ax.set_xlabel('Epochs', fontsize=15)
    ax.set_ylabel('Loss', fontsize=15)
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.legend(fontsize=15)
    plt.savefig(os.path.join(save_dir, 'losses.png'))
