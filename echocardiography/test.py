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

import json
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tqdm
from dataset import EchoNetDataset, convert_to_serializable
from models import ResNet50Regression
from scipy.stats import multivariate_normal
from scipy import ndimage
import cv2
import math
import numpy as np

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

    ## load test dataset
    transform_val = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    transform_target = transforms.Compose([transforms.Resize((256, 256))])

    test_set = EchoNetDataset(batch=args.batch, split='test', phase=args.phase, label_directory=None, 
                              target=trained_args['target'], transform=transform_val, transform_target=transform_target)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    ## load the model
    best_model = get_best_model(train_dir)
    model_dir = os.path.join('TRAINED_MODEL', args.batch, args.phase, args.trial)
    model = ResNet50Regression(num_labels=12)
    model.load_state_dict(torch.load(os.path.join(train_dir, f'model_{best_model}')))
    model.to(device)
    
    ## test the model
    model.eval()
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
                print(label)
                print(output)

                plt.figure(figsize=(14,14), num='Example')
                plt.imshow(image, cmap='gray', alpha=0.2)

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
                plt.show()
               
            
                

        
    
