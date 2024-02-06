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
from models import ResNet50Regression

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
        plt.scatter(label[0] * image.shape[1], label[1] * image.shape[0], color='green', marker='o', s=100, alpha=0.5) 
        plt.scatter(label[2] * image.shape[1], label[3] * image.shape[0], color='green', marker='o', s=100, alpha=0.5)

        plt.scatter(label[4] * image.shape[1], label[5] * image.shape[0], color='red', marker='o', s=100, alpha=0.5) 
        plt.scatter(label[6] * image.shape[1], label[7] * image.shape[0], color='red', marker='o', s=100, alpha=0.5)

        plt.scatter(label[8] * image.shape[1], label[9] * image.shape[0], color='blue', marker='o', s=100, alpha=0.5) 
        plt.scatter(label[10] * image.shape[1], label[11] * image.shape[0], color='blue', marker='o', s=100, alpha=0.5)
        plt.axis('off')
        plt.show()

def train_one_epoch(training_loader, model, loss, optimizer, device, tb_writer = None):
    """
    Funtion that performe the training of the model for one epoch
    """
    running_loss = 0.
    loss = 0.           ## this have to be update with the last_loss
    for i, data in enumerate(training_loader):
        inputs, labels = data       # Every data instance is an input + label pair
        inputs = inputs.to(device)  # Move inputs and labels to GPU
        labels = labels.to(device)

        optimizer.zero_grad()      # Zero your gradients for every batch!
        outputs = model(inputs)    # Make predictions for this batch

        # Compute the loss and its gradients
        loss = loss_fn(outputs.float(), labels.float())
        loss.backward()
        
        optimizer.step() # Adjust learning weights

        # Gather data and report
        running_loss += loss.item()  
        if i == len(training_loader) - 1:  
            last_loss = running_loss / (i + 1) 
            print(f'last batch loss: {last_loss}')
            # tb_x = epoch_index * len(training_loader) + i + 1     # add time step to the tensorboard
            # tb_writer.add_scalar('Loss/train', last_loss, tb_x)   # 
            running_loss = 0.

    return last_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Read the dataset')
    parser.add_argument('--data_dir', type=str, default="/media/angelo/OS/Users/lasal/Desktop/Phd notes/Echocardiografy/EchoNet-LVH", help='Directory of the dataset')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    ## initialize the reshape and normalization for image in a transfrom object
    transform = transforms.Compose([transforms.Resize((256,256)),
                                    transforms.ToTensor()])

    print('start loading the dataset...')
    training_set = EchoNetLVH(data_dir=args.data_dir, batch='Batch1', split='train', transform=transform, only_diastole=True)
    validation_set = EchoNetLVH(data_dir=args.data_dir, batch='Batch1', split='val', transform=transform, only_diastole=True)
    print('dataset loaded')
    
    # Create data loaders for our datasets; shuffle for training, not for validation
    print('start creating the dataloader...')
    training_loader = torch.utils.data.DataLoader(training_set, batch_size=4, shuffle=True, num_workers=16)
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=4, shuffle=False, num_workers=16)
    print('dataloader created')

    ##sanity check of the dataloader
    # dataset_iteration(training_loader)

    loss_fn = torch.nn.MSELoss()
    model = ResNet50Regression(num_labels=12)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.to(device)

     
    epoch_number = 0
    EPOCHS = 5
    best_vloss = 1_000_000.     # initialize the current best validation loss with a large value

    for epoch in range(EPOCHS):
        print(f'EPOCH {epoch + 1}')

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss = train_one_epoch(training_loader, model, loss_fn, optimizer, device=device)

        running_vloss = 0.0 
        
        model.eval() # Set the model to evaluation mode, disabling dropout and using population statistics for batch normalization.
        # print(model.eval)
        print('start validation')
        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for i, vdata in enumerate(validation_loader):
                vinputs, vlabels = vdata
                vinputs = vinputs.to(device)
                vlabels = vlabels.to(device)

                voutputs = model(vinputs).to(device)
                vloss = loss_fn(voutputs, vlabels)
                running_vloss += vloss

        avg_vloss = running_vloss / (i + 1)
        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

        # Log the running loss averaged per batch
        # for both training and validation
        # writer.add_scalars('Training vs. Validation Loss',
        #                 { 'Training' : avg_loss, 'Validation' : avg_vloss },
        #                 epoch_number + 1)
        # writer.flush()

        # # Track best performance, and save the model's state
        # if avg_vloss < best_vloss:
        #     best_vloss = avg_vloss
        #     model_path = 'model_{}_{}'.format(timestamp, epoch_number)
        #     torch.save(model.state_dict(), model_path)

        # epoch_number += 1
    


    
    
