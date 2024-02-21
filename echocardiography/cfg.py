"""
Configuration of models and loss functions for training
"""
import os
import argparse

import numpy as np
from dataset import EchoNetDataset, convert_to_serializable
from models import ResNet50Regression, PlaxModel, UNet, UNet_up
from losses import RMSELoss, WeightedRMSELoss, WeightedMSELoss

def train_config(target, threshold_wloss, model, device):
    """
    return the model and the loss function based on the target

    Parameters
    ----------
    target : str
        target to predict, e.g. keypoints, heatmaps, segmentation

    theta_wloss : float
        threshold for the weighted loss, if 0. all the weights are 1 and the lass fall back to regular ones

    Returns
    -------
    """
    cfg = {}
    if target == 'keypoints': 
        cfg['model'] = ResNet50Regression(num_labels=12)
        cfg['loss'] = torch.nn.MSELoss()

    elif target == 'heatmaps': 
        cfg['model'] = PlaxModel(num_classes=6)
        if model == 'unet': cfg['model'] = UNet(num_classes=6)
        if model == 'unet_up': cfg['model'] = UNet_up(num_classes=6)
        if model == 'plax': cfg['model'] = PlaxModel(num_classes=6)
        cfg['loss'] = WeightedRMSELoss(threshold=threshold_wloss, device=device)
        
    elif target == 'segmentation':
        if model == 'unet': cfg['model'] = UNet(num_classes=6)
        if model == 'plax': cfg['model'] = PlaxModel(num_classes=6)
        cfg['loss'] = torch.nn.MSELoss()
       
    else:
        raise ValueError(f'target {target} is not valid. Available targets are keypoints, heatmaps, segmentation')

    return cfg