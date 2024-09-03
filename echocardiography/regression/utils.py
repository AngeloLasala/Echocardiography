"""
This file contains the utility functions for the regression model
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2


def get_corrdinate_from_heatmap_ellipses(heatmap):
    """
    Get the coordinate from the heatmaps. retrive the ellipse with a thr of 0.5
    and get the center value

    Parameters
    ----------
    heatmap : torch.Tensor
        Tensor containing the heatmap

    Returns
    -------
    list
        List containing the coordinates
    """
    label_list = []
    for ch in range(heatmap.shape[0]):
        ## in 0-1 range
        heatmap[ch] = (heatmap[ch] - np.min(heatmap[ch])) / (np.max(heatmap[ch]) - np.min(heatmap[ch]))
        ## create a mask of 1 and 0 with threshold 0.5
        mask = (heatmap[ch] > 0.5).astype(np.uint8) * 255 
        ## fit ellispse
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)        
        for cnt in contours:
            if len(cnt) >= 5:    
                ellipse = cv2.fitEllipse(cnt)
                label_list.append(ellipse[0][0])
                label_list.append(ellipse[0][1])    
    return label_list

def get_corrdinate_from_heatmap(heatmap):
    """
    Get the coordinate from the heatmap

    Parameters
    ----------
    heatmap : torch.Tensor
        Tensor containing the heatmap

    Returns
    -------
    list
        List containing the coordinates
    """
    label_list = []
    for ch in range(heatmap.shape[0]):
        max_value = np.max(heatmap[ch])
        coor = np.where(heatmap[ch] == max_value)
        label_list.append(coor[1][0])
        label_list.append(coor[0][0])
    return label_list

def get_corrdinate_from_heatmap_torch(heatmap):
    """
    Get the coordinate from the heatmap

    Parameters
    ----------
    heatmap : torch.Tensor
        Tensor containing the heatmap

    Returns
    -------
    list
        List containing the coordinates
    """
def get_coordinate_from_heatmap_torch(heatmap):
    """
    Get the coordinate from the heatmap

    Parameters
    ----------
    heatmap : torch.Tensor
        Tensor containing the heatmap (B,C,W,H)

    Returns
    -------
    list
        List containing the coordinates
    """
    label_list = []
    for ch in range(heatmap.shape[1]):
        max_value, max_index = torch.max(heatmap[0,ch,:,:].view(-1), dim=0)
        coor = torch.nonzero(heatmap[0,ch,:,:] == max_value)
        label_list.append(coor[0][1].item())
        label_list.append(coor[0][0].item())
    return label_list


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


def echocardiografic_parameters(label):
    """
    given the array of 12 labels compute the RWT and LVmass

    Parameters
    ----------
    label: np.array
        array of 12 labels [ in corrofinate order: x1, y1, x2, y2, x1, y1, x2, y2, x1, y1, x2, y2] * img shape

    Returns
    -------
    RWT: float
        Relative Wall Thickness

    LVmass: float
        Left Ventricular Mass
    """
    
    ## compute the RWT
    LVPWd = np.sqrt((label[2] - label[0])**2 + (label[3] - label[1])**2)
    LVIDd = np.sqrt((label[6] - label[4])**2 + (label[7] - label[5])**2)
    IVSd = np.sqrt((label[10] - label[8])**2 + (label[11] - label[9])**2)
    
    rwt = 2 * LVPWd / LVIDd
    rst = 2 * IVSd / LVIDd
    return rwt, rst