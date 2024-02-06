"""
Main file to train the PLAX regression model
"""
import os
import argparse

import torch
from torch.utils.data import Dataset
from torchvision import transforms

import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tqdm
from dataset import EchoNetLVH

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Read the dataset')
    parser.add_argument('--data_dir', type=str, default="/media/angelo/OS/Users/lasal/Desktop/Phd notes/Echocardiografy/EchoNet-LVH", help='Directory of the dataset')
    args = parser.parse_args()
    
    ## initialize the reshape and normalization for image in a transfrom object
    transform = transforms.Compose([transforms.Resize((256,256)),
                                    transforms.ToTensor()])

    print('start loading the dataset...')
    training_set = EchoNetLVH(data_dir=args.data_dir, batch='Batch1', split='train', transform=transform, only_diastole=True)
    print('dataset loaded')
    
    # Create data loaders for our datasets; shuffle for training, not for validation
    print('start creating the dataloader...')
    training_loader = torch.utils.data.DataLoader(training_set, batch_size=4, shuffle=True)
    print('dataloader created')

    #print the first element of the dataloader
    image, label = next(iter(training_loader))
    #convert the tensor into numpy
    image = image.numpy().transpose((0, 2, 3, 1))
    label = label.numpy()
    print(image.shape, label.shape)

    for batch_idx, (data, target) in enumerate(training_loader):
        # Your training code goes here
        # 'data' contains the input images
        # 'target' contains the corresponding labels or any other relevant information
        
        # Example: Print the shape of the batch
        print(f'Batch {batch_idx + 1}/{len(training_loader)} - Data Shape: {data.shape}, Target Shape: {target.shape}')
        
    
