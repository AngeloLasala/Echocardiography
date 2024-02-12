"""
Main file to train the PLAX regression model
"""
import os
import argparse

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchsummary import summary 
from datetime import datetime
import time
import tqdm

import json
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tqdm
from dataset import EchoNetDataset, convert_to_serializable
from models import ResNet50Regression, PlaxModel, UNet
from losses import RMSELoss, WeightedRMSELoss, WeightedMSELoss
from scipy.stats import multivariate_normal
from scipy import ndimage
import cv2
import math

## Auxiliar class for data augmentation and loss (UPDATIND: move to other module.py)
class AdjustGamma(object):
    """
    Gamma correction for the image
    """
    def __init__(self, gamma):
        self.gamma = gamma
    def __call__(self, img):
        return transforms.functional.adjust_gamma(img, self.gamma)


## FIT functions
def dataset_iteration(dataloader):
    """
    Iterate over the dataset

    Parameters
    ----------
    dataloader : torch.utils.data.DataLoader
        DataLoader object that contains the dataset
    """
    for batch_idx, (data, target) in enumerate(dataloader):
        # Your training code goes here
        # 'data' contains the input images
        # 'target' contains the corresponding labels or any other relevant information
        
        # Example: Print the shape of the batch
        print(f'Batch {batch_idx + 1}/{len(training_loader)} - Data Shape: {data.shape}, Target Shape: {target.shape}')
        image = data.numpy().transpose((0, 2, 3, 1))[0]
        label = target.numpy()[0]
        print(np.min(image), np.max(image))
        
        plt.figure(figsize=(14,14), num='Example - batch ' + str(batch_idx + 1))
        plt.imshow(image, cmap='gray')
        plt.scatter(label[0] * image.shape[0], label[1] * image.shape[1], color='green', marker='o', s=100, alpha=0.5) 
        plt.scatter(label[2] * image.shape[0], label[3] * image.shape[1], color='green', marker='o', s=100, alpha=0.5)

        plt.scatter(label[4] * image.shape[0], label[5] * image.shape[1], color='red', marker='o', s=100, alpha=0.5) 
        plt.scatter(label[6] * image.shape[0], label[7] * image.shape[1], color='red', marker='o', s=100, alpha=0.5)

        plt.scatter(label[8] * image.shape[0], label[9] * image.shape[1], color='blue', marker='o', s=100, alpha=0.5) 
        plt.scatter(label[10] * image.shape[0], label[11] * image.shape[1], color='blue', marker='o', s=100, alpha=0.5)
        plt.axis('off')
        plt.show()

def train_config(target, device):
    """
    return the model and the loss function based on the target

    Parameters
    ----------
    target : str
        target to predict, e.g. keypoints, heatmaps, segmentation

    Returns
    -------
    """
    cfg = {}
    if target == 'keypoints': 
        cfg['model'] = ResNet50Regression(num_labels=12)
        cfg['loss'] = torch.nn.MSELoss()

    elif target == 'heatmaps': 
        cfg['model'] = PlaxModel(num_classes=6)
        # cfg['model'] = UNet(num_classes=6)
        cfg['loss'] = WeightedRMSELoss(device=device)
        
    elif target == 'segmentation':
        cfg['model'] = PlaxModel(num_classes=6)
        # cfg['model'] = UNet(in_channels=3, num_classes=6)
        cfg['loss'] = torch.nn.BCELoss()
       
    else:
        raise ValueError(f'target {target} is not valid. Available targets are keypoints, heatmaps, segmentation')

    return cfg

def train_one_epoch(training_loader, model, loss, optimizer, device, tb_writer = None):
    """
    Funtion that performe the training of the model for one epoch
    """
    running_loss = 0.
    loss = 0.           ## this have to be update with the last_loss
    for i, (inputs, labels) in enumerate(training_loader):
        inputs, labels = inputs.to(device), labels.to(device)       # Every data instance is an input + label pair
        
        optimizer.zero_grad()                           # Zero your gradients for every batch!
        outputs = model(inputs)                         # Make predictions for this batch

        loss = loss_fn(outputs.float(), labels.float()) # Compute the loss and its gradients
        loss.backward()
        
        optimizer.step() # Adjust learning weights

        # Gather data and report
        running_loss += loss.item()  
        if i == len(training_loader) - 1:
            last_loss = running_loss / (i + 1) 
            # print(f'last batch loss: {last_loss}')
            # tb_x = epoch_index * len(training_loader) + i + 1     # add time step to the tensorboard
            # tb_writer.add_scalar('Loss/train', last_loss, tb_x)   # 
            running_loss = 0.
        time_end = time.time()

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
        avg_loss = train_one_epoch(training_loader, model, loss_fn, optimizer, device=device)

        running_vloss = 0.0 
        model.eval() # Set the model to evaluation mode, disabling dropout and using population statistics for batch normalization.
        
        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for i, vdata in enumerate(validation_loader):
                vinputs, vlabels = vdata
                vinputs = vinputs.to(device)
                vlabels = vlabels.to(device)

                voutputs = model(vinputs).to(device)
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
        print(f'Epoch {epoch}/{EPOCHS} | Train Loss: {avg_loss:.6f} | Validation Loss: {avg_vloss:.6f} | Time: {time.time() - start:.2f}s')
    return losses
        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Read the dataset')
    parser.add_argument('--data_dir', type=str, default="/media/angelo/OS/Users/lasal/Desktop/Phd notes/Echocardiografy/EchoNet-LVH", help='Directory of the dataset')
    parser.add_argument('--batch_dir', type=str, default='Batch2', help='Batch number of video folder, e.g. Batch1, Batch2, Batch3, Batch4')
    parser.add_argument('--phase', type=str, default='diastole', help='select the phase of the heart, diastole or systole')
    parser.add_argument('--target', type=str, default='keypoints', help='select the target to predict, e.g. keypoints, heatmaps, segmentation')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs to train the model')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for the dataloader')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for the optimizer')
    parser.add_argument('--save_dir', type=str, default='TRAINED_MODEL', help='Directory to save the model')
    args = parser.parse_args()
    
    ## device and reproducibility    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    np.random.seed(42)
    cfg = train_config(args.target, device=device)
    print(f'Using device: {device}')
    
    ## initialize the prepocessing and data augmentation
    transform_train = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            AdjustGamma(gamma=0.5)
        ], p=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    ## initialize the prepocessing and data augmentation
    transform_val = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    transform_target = transforms.Compose([transforms.Resize((256, 256))])

    print('start creating the dataset...')
    train_set = EchoNetDataset(batch=args.batch_dir, split='train', phase=args.phase, label_directory=None,
                              target=args.target, transform=transform_train, transform_target=transform_target)
    validation_set = EchoNetDataset(batch=args.batch_dir, split='val', phase=args.phase, label_directory=None, 
                              target=args.target, transform=transform_val, transform_target=transform_target)

    print('start creating the dataloader...')
    training_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    ## TRAIN
    print('start training...')
    loss_fn = cfg['loss']
    print(loss_fn) 
    model = cfg['model'].to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

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
