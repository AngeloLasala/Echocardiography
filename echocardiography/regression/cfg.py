"""
Configuration of models and loss functions for training
"""
import os
import argparse

import numpy as np
from echocardiography.regression.dataset import EchoNetDataset, convert_to_serializable
from echocardiography.regression.models import ResNet50Regression, ResNet101Regression, ResNet152Regression, PlaxModel, UNet, UNet_up, UNet_up_hm, SwinTransformerTiny, SwinTransformerSmall, SwinTransformerBase, ResNet152Regression, Unet_ResSkip
from echocardiography.regression.losses import RMSELoss, WeightedRMSELoss, WeightedMSELoss, WeighteRMSELoss_l2MAE
import torch

def train_config(target, threshold_wloss, model, input_channels, device):
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
        if model == 'resnet50' : cfg['model'] = ResNet50Regression(input_channels=input_channels, num_labels=12)
        if model == 'resnet101': cfg['model'] = ResNet101Regression(input_channels=input_channels, num_labels=12)
        if model == 'resnet152': cfg['model'] = ResNet152Regression(input_channels=input_channels, num_labels=12)
        if model == 'swin_tiny' : cfg['model'] = SwinTransformerTiny(input_channels=input_channels, num_labels=12)
        if model == 'swin_small': cfg['model'] = SwinTransformerSmall(input_channels=input_channels, num_labels=12)
        if model == 'swin_base': cfg['model'] = SwinTransformerBase(input_channels=input_channels, num_labels=12)
        cfg['loss'] = torch.nn.MSELoss()
        cfg['input_channels'] = input_channels

    elif target == 'heatmaps': 
        if model == 'unet_base': cfg['model'] = PlaxModel(num_classes=6)
        if model == 'unet': cfg['model'] = UNet(input_channels=input_channels, num_classes=6)
        if model == 'unet_up': cfg['model'] = UNet_up(input_channels=input_channels, num_classes=6)
        if model == 'unet_res_skip': 
            model_config = {
                            'down_channels': [ 64, 128, 256, 256],
                            'mid_channels': [ 256, 256],
                            'down_sample': [ True, True, True],
                            'attn_down' : [False,False,False],
                            'norm_channels' : 32,
                            'num_heads' : 8,
                            'conv_out_channels' : 128,
                            'num_down_layers': 1,
                            'num_mid_layers': 1,
                            'num_up_layers': 1,
                        }
            cfg['model'] = Unet_ResSkip(im_channels=input_channels, num_classes=6, model_config=model_config)

        if model == 'unet_res_skip_att': 
            model_config = {
                            'down_channels': [ 64, 128, 256, 256],
                            'mid_channels': [ 256, 256],
                            'down_sample': [ True, True, True],
                            'attn_down' : [True,True,True],
                            'norm_channels' : 32,
                            'num_heads' : 8,
                            'conv_out_channels' : 128,
                            'num_down_layers': 1,
                            'num_mid_layers': 1,
                            'num_up_layers': 1,
                        }
            cfg['model'] = Unet_ResSkip(im_channels=input_channels, num_classes=6, model_config=model_config)

        if model == 'unet_res_skip_base': 
            model_config = {
                            'down_channels': [ 64, 128, 256, 256],
                            'mid_channels': [ 256, 256],
                            'down_sample': [ True, True, True],
                            'attn_down' : [False,False,False],
                            'norm_channels' : 32,
                            'num_heads' : 8,
                            'conv_out_channels' : 128,
                            'num_down_layers': 2,
                            'num_mid_layers': 2,
                            'num_up_layers': 2,
                        }
            cfg['model'] = Unet_ResSkip(im_channels=input_channels, num_classes=6, model_config=model_config)

        if model == 'unet_res_skip_base_att': 
            model_config = {
                            'down_channels': [ 64, 128, 256, 256],
                            'mid_channels': [ 256, 256],
                            'down_sample': [ True, True, True],
                            'attn_down' : [True,True,True],
                            'norm_channels' : 32,
                            'num_heads' : 8,
                            'conv_out_channels' : 128,
                            'num_down_layers': 2,
                            'num_mid_layers': 2,
                            'num_up_layers': 2,
                        }
            cfg['model'] = Unet_ResSkip(im_channels=input_channels, num_classes=6, model_config=model_config)

        if model == 'plax': cfg['model'] = PlaxModel(num_classes=6)
        
        cfg['loss'] = WeighteRMSELoss_l2MAE(threshold=threshold_wloss, alpha=1.5, device = device)
        cfg['input_channels'] = input_channels
        
    elif target == 'segmentation':
        if model == 'unet': cfg['model'] = UNet(input_channels=input_channels, num_classes=6)
        if model == 'unet_up': cfg['model'] = UNet_up(input_channels=input_channels, num_classes=6)
        if model == 'plax': cfg['model'] = PlaxModel(num_classes=6)
        cfg['loss'] = WeighteRMSELoss_l2MAE(threshold=threshold_wloss, alpha=1.5, device = device)
        cfg['input_channels'] = input_channels

    elif target == 'heatmaps_sigma':
        cfg['model'] =  UNet_up_hm(num_classes=6)
        cfg['loss'] = WeighteRMSELoss_l2MAE(threshold=threshold_wloss, alpha=1.5, device = device)
       
    else:
        raise ValueError(f'target {target} is not valid. Available targets are keypoints, heatmaps, segmentation')

    return cfg