"""
Main file to train the PLAX regression model
"""
import os
import argparse

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from datetime import datetime
import time
import tqdm

import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tqdm
from dataset import EchoNetLVH, select_patients
from models import ResNet50Regression

class EchoNetDataset(Dataset):
    def __init__(self, batch, split, phase, label_directory, transform=None):
        """
        Args:
            data_dir (string): Directory with all the video.
            batch (string): Batch number of video folder, e.g. 'Batch1', 'Batch2', 'Batch3', 'Batch4'.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.transform = transform
        self.split = split
        self.batch = batch
        self.phase = phase

        label = pd.read_csv(label_directory, index_col=0)
        self.label = label
        self.data_dir = os.path.join('DATA', self.batch, self.split, self.phase)

    
    def __len__(self):
        """
        Return the total number of patiel in selected batch
        """ 
        return len(os.listdir(self.data_dir))

    def __getitem__(self, idx):
        """
        Get the image and the label of the patient
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        patient = os.listdir(self.data_dir)[idx].split('.')[0]
        a = time.time()
        patient_label = self.get_keypoint(patient)
        # print(f'time = {time.time()-a:.5f}')


        # read the image wiht PIL
        image = Image.open(os.path.join(self.data_dir, patient+'.png')) 
        
        # read the label  
        keypoints_label = []
        for heart_part in ['LVPWd', 'LVIDd', 'IVSd']:
            if patient_label[heart_part] is not None:
                x1_heart_part = patient_label[heart_part]['x1'] / image.size[0]
                y1_heart_part = patient_label[heart_part]['y1'] / image.size[1]
                x2_heart_part = patient_label[heart_part]['x2'] / image.size[0]
                y2_heart_part = patient_label[heart_part]['y2'] / image.size[1]
                keypoints_label.append([x1_heart_part, y1_heart_part, x2_heart_part, y2_heart_part])

        keypoints_label = (np.array(keypoints_label)).flatten()

        if self.transform:
            image = self.transform(image)

        return image, keypoints_label


    def get_patiens(self):
        """
        get the list of patient in the entire dataset
        """
        return np.unique(self.label['HashedFileName'].values)

    def get_keypoint(self, patient_hash):
        """
        Get the keypoint from the label dataset file

        Parameters
        ----------
        patient_hash : str
            Hashed file name of the patient

        Returns
        -------
        label_dict : dict
            Dictionary containing the keypoint information
        """
        label = self.label
        label_dict = {'LVIDd': None, 'IVSd': None, 'LVPWd': None, 
                    'LVIDs': None, 'IVSs': None, 'LVPWs': None}

        for value in label[label['HashedFileName'] == patient_hash]['Calc'].values:
            x1 = label.loc[(label['HashedFileName'] == patient_hash) & (label['Calc'] == value), 'X1'].array[0]
            x2 = label.loc[(label['HashedFileName'] == patient_hash) & (label['Calc'] == value), 'X2'].array[0]
            y1 = label.loc[(label['HashedFileName'] == patient_hash) & (label['Calc'] == value), 'Y1'].array[0]
            y2 = label.loc[(label['HashedFileName'] == patient_hash) & (label['Calc'] == value), 'Y2'].array[0]
            
            calc_value = label.loc[(label['HashedFileName'] == patient_hash) & (label['Calc'] == value), 'CalcValue'].array[0]
            label_dict[value] = {'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2, 'calc_value': calc_value}

        return label_dict

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
    for i, (inputs, labels) in enumerate(training_loader):
        # print(f'batch {i+1}')
        time_start = time.time()
        inputs, labels = inputs.to(device), labels.to(device)       # Every data instance is an input + label pair
        
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
            # print(f'last batch loss: {last_loss}')
            # tb_x = epoch_index * len(training_loader) + i + 1     # add time step to the tensorboard
            # tb_writer.add_scalar('Loss/train', last_loss, tb_x)   # 
            running_loss = 0.
        time_end = time.time()
        # print(f'time for batch = {time_end - time_start:.5f}')

    return last_loss

def fit(training_loader, validation_loader,
        model, loss_fn, optimizer, 
        epochs=5, device='cpu'):
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

    for epoch in tqdm.tqdm(range(EPOCHS)):
        time_start = time.time()
        epoch += 1
        # print(f'EPOCH {epoch}')

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss = train_one_epoch(training_loader, model, loss_fn, optimizer, device=device)
        time_end = time.time()
        # print(f'Training took {time_end - time_start} seconds')

        running_vloss = 0.0 
        model.eval() # Set the model to evaluation mode, disabling dropout and using population statistics for batch normalization.
        
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
        # print(f'LOSS train {avg_loss:.5f} - valid {avg_vloss:.5f}')

        # Log the running loss averaged per batch
        # for both training and validation
        # writer.add_scalars('Training vs. Validation Loss',
        #                 { 'Training' : avg_loss, 'Validation' : avg_vloss },
        #                 epoch_number + 1)
        # writer.flush()

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            # print('best model found')
            best_vloss = avg_vloss
            model_path = f'model_{epoch}'
            torch.save(model.state_dict(), model_path)
        # print('============================================')
        

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Read the dataset')
    parser.add_argument('--data_dir', type=str, default="/media/angelo/OS/Users/lasal/Desktop/Phd notes/Echocardiografy/EchoNet-LVH", help='Directory of the dataset')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

    print(f'Using device: {device}')
    

    ## initialize the reshape and normalization for image in a transfrom object
    transform = transforms.Compose([transforms.Resize((256,256)),
                                    transforms.ToTensor(), 
                                    transforms.Normalize((0.5 ), (0.5 ))])

    train_set = EchoNetDataset(batch='Batch2', split='train', phase='diastole', label_directory='MeasurementsList.csv', transform=transform)
    validation_set = EchoNetDataset(batch='Batch2', split='val', phase='diastole', label_directory='MeasurementsList.csv', transform=transform)

    print('start creating the dataloader...')
    training_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True, num_workers=8, pin_memory=True)
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=32, shuffle=False, num_workers=8, pin_memory=True)
    print('dataloader created')
    # dataset_iteration(training_loader) #sanity check of the dataloader

    loss_fn = torch.nn.MSELoss()
    model = ResNet50Regression(num_labels=12).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    fit(training_loader, validation_loader, model, loss_fn, optimizer, epochs=50, device=device)

    
    


    
    
