"""
Create the class of EchoNet-LVH dataset
"""
import os
import argparse

import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tqdm
import json
from scipy.stats import multivariate_normal
from scipy import ndimage
import cv2
import math
import time

from echocardiography.regression.data_reader import read_video, read_video

class EchoNetDataset(Dataset):
    """
    EchoNet-LVH class to read the data. This class is used to training validation and test the model

    Current version: 19/07/2024
    ---------------------------
    The data come from the 'DATA' folder inside of this repository. The tree is the follow
    DATA
    ├── Batch1
    │   ├── test
    │   │   ├── diastole
    │   │   │   ├── image
    │   │   │   └── label
    │   ├── train
    │   │   ├── diastole
    │   │   │   ├── image
    │   │   │   └── label
    │   └── val
    │   │   ├── diastole
    │   │   │   ├── image
    │   │   │   └── label
    ├── Batch2
    ├── Batch3
    └── Batch4

    where 'DATA' is fix in the code while batch split and phase are given as input
    """
    def __init__(self, batch, split, phase, target, input_channels, size, 
                data_path=None, label_directory=None, transform=None, augmentation=False, 
                original_shape=False):
        """
        Args:
            batch (string): Batch number of video folder, e.g. 'Batch1', 'Batch2', 'Batch3', 'Batch4'.
            split (string): train, validation or test
            phase (string): diastole or systole
            target (string): keypoint, heatmap, segmentation
            input_channels (int): Number of input channels, must be 3 for RGB or 1 for grayscale
            size (tuple): Size of the image 
            label_directory (string): Directory of the label dataset, default None that means read the file from the json file
            transform (callable, optional): Optional transform to be applied on a sample.    
            transform_target (callable, optional): Optional transform to be applied on a sample.
            augmentation (bool): Apply data augmentation to the image and the label
        """
        self.split = split
        self.batch = batch
        self.phase = phase
        self.target = target
        self.augmentation = augmentation
        self.size = size
        self.input_channels = input_channels
        self.data_path = data_path
        self.original_shape = original_shape

        if self.data_path is not None: ## take the data in local storage, here i have collect the data in the same repository
            self.data_dir = os.path.join(self.data_path, self.batch, self.split, self.phase)
        else:                 ## take the data from a given path
            self.data_dir = os.path.join('DATA', self.batch, self.split, self.phase)

        self.patient_files = [patient_hash.split('.')[0] for patient_hash in os.listdir(os.path.join(self.data_dir, 'image'))]

    
        if label_directory is not None:
            label = pd.read_csv(os.path.join(label_directory, 'MeasurementsList.csv'), index_col=0)
            self.label = label
            self.keypoints_dict = {patient_hash: self.get_keypoint(patient_hash) for patient_hash in tqdm.tqdm(self.patient_files)}
        else:
            #load a directory from a json file
            with open(os.path.join(self.data_dir, 'label', 'label.json'), 'r') as f:
                # print(f.read())
                self.keypoints_dict = json.load(f)

    def __len__(self):
        """
        Return the total number of patiel in selected batch
        """ 
        return len(self.patient_files)

    def __getitem__(self, idx):
        """
        Get the image and the label of the patient
        """
        # print('START GET ITEM')
        # start_get_item = time.time()
        if torch.is_tensor(idx):
            idx = idx.tolist()

      
        ## trasform the label based on target: keypoints, heatmaps, segmentations
        if self.target == 'keypoints': 
            # image_label_start = time.time()
            image, label, calc_value, original_shape = self.get_image_label(idx)
            # image_label_stop = time.time()
            # print(f'Time to get image and label: {image_label_stop - image_label_start:.5f}')

            if self.augmentation:
                image, label = self.data_augmentation_kp(image, label)
            else:
                resize = transforms.Resize(size=self.size)
                image = resize(image)
                if self.input_channels == 1: image = image.convert('L')
                label = torch.tensor(label)
                image = transforms.functional.to_tensor(image)
                # image = transforms.functional.normalize(image, (0.5), (0.5))  
                # im_tensor = torchvision.transforms.ToTensor()(im)

                image = (2 * image) - 1  

        elif self.target == 'heatmaps_sigma':
            # image_label_start = time.time()
            image, label, calc_value, original_shape = self.get_image_label(idx)
            # image_label_stop = time.time()
            # print(f'Time to get image and label: {image_label_stop - image_label_start:.5f}')
            if self.augmentation:
                image, label = self.data_augmentation_kp(image, label)
            else:
                resize = transforms.Resize(size=self.size)
                image = resize(image)
                if self.input_channels == 1: image = image.convert('L')
                label = torch.tensor(label)
                image = transforms.functional.to_tensor(image)
                # image = transforms.functional.normalize(image, (0.5), (0.5))  
                # im_tensor = torchvision.transforms.ToTensor()(im)

                image = (2 * image) - 1  

        elif self.target == 'heatmaps': 
            # image_label_start = time.time()
            image, label, calc_value, original_shape, heatmap = self.get_image_label(idx)
            # image_label_stop = time.time()
            # print(f'Time to get image and label: {image_label_stop - image_label_start:.5f}')

            # label_start = time.time()   
            label = heatmap #self.get_heatmap(idx)
            # label_stop = time.time()
            # print(f'Time to get heatmap: {label_stop - label_start:.5f}')
            ## apply data augmentation only on training set, else simply resize the image and the label
            if self.augmentation:
                aug_start = time.time()
                image, label = self.data_augmentation(image, label)
                aug_stop = time.time()
                # print(f'Time to apply data augmentation: {aug_stop - aug_start:.5f}')
            else:
                image, label = self.trasform(image, label)
                
        elif self.target == 'segmentation':
            # image_label_start = time.time()
            image, label, calc_value, original_shape, heatmap = self.get_image_label(idx)
            # image_label_stop = time.time()
            # print(f'Time to get image and label: {image_label_stop - image_label_start:.5f}')

            label = self.get_heatmap(idx)
            label = (label > 0.5).astype(np.float32)
            ## apply data augmentation only on training set, else simply resize the image and the label
            if self.augmentation:
                image, label = self.data_augmentation(image, label)
            else:
                image, label = self.trasform(image, label)
           
        else:
            raise ValueError(f'target {self.target} is not valid. Available targets are keypoints, heatmaps, segmentation, heatmpas_sigma')
        # stop_get_item = time.time()
        # print(f'Time to get item: {stop_get_item - start_get_item:.5f}')

        ## for the evaluation in real dimention, the calc_value and original shape are needed
        if self.original_shape: return image, label, calc_value, original_shape
        else: return image, label


    def trasform(self, image, label):
        """
        Simple trasformaztion of the label and image. Resize and normalize the image and resize the label
        """
        # convert each channel in PIL image
        label = [Image.fromarray(label[:,:,ch]) for ch in range(label.shape[2])]

        ## Resize
        resize = transforms.Resize(size=self.size)
        image = resize(image)
        label = [resize(ch) for ch in label]

        ## convert to tensor and normalize
        label = np.array([np.array(ch) for ch in label])
        label = torch.tensor(label)

        if self.input_channels == 1: image = image.convert('L')
        image = transforms.functional.to_tensor(image)
        # image = transforms.functional.normalize(image, (0.5), (0.5))
        # im_tensor = torchvision.transforms.ToTensor()(im)
        image = (2 * image) - 1    
        return image, label

    def data_augmentation(self, image, label):
        """
        Set of trasformation to apply to image and label (heatmaps).
        This function contain all the data augmentation trasformations and the normalization.
        Note: torchvision.transforms often rely on PIL as the underlying library, 
            so each channel of heatmap need to transform  separately (channels are in the first dimension)
        """
        # convert each channel in PIL image
        label = [Image.fromarray(label[:,:,ch]) for ch in range(label.shape[2])]

        ## Resize
        resize = transforms.Resize(size=self.size)
        image = resize(image)
        label = [resize(ch) for ch in label]
        
        ## random rotation to image and label
        if torch.rand(1) > 0.5:
            angle = np.random.randint(-15, 15)
            image = transforms.functional.rotate(image, angle)
            label = [transforms.functional.rotate(ch, angle) for ch in label]

        ## random translation to image and label in each direction
        if torch.rand(1) > 0.5:
            translate = transforms.RandomAffine.get_params(degrees=(0.,0.), 
                                                        translate=(0.10, 0.10),
                                                        scale_ranges=(1.0,1.0),
                                                        shears=(0.,0.), 
                                                        img_size=self.size)
            image = transforms.functional.affine(image, *translate)
            label = [transforms.functional.affine(ch, *translate) for ch in label]

        ## random horizontal flip
        if torch.rand(1) > 0.5:
            image = transforms.functional.hflip(image)
            label = [transforms.functional.hflip(ch) for ch in label]
            
        ## random brightness and contrast
        if torch.rand(1) > 0.5:
            image = transforms.ColorJitter(brightness=0.5, contrast=0.5)(image)
        
        ## random gamma correction
        if torch.rand(1) > 0.5:
            gamma = np.random.uniform(0.5, 1.5)
            image = transforms.functional.adjust_gamma(image, gamma)

        ## Convert to tensor and normalize
        label = np.array([np.array(ch) for ch in label])
        label = torch.tensor(label)

        if self.input_channels == 1: image = image.convert('L')
        image = transforms.functional.to_tensor(image)
        # image = transforms.functional.normalize(image, (0.5), (0.5))
        # im_tensor = torchvision.transforms.ToTensor()(im)
        image = (2 * image) - 1  
        return image, label

    def data_augmentation_kp(self, image, label):
        ## Resize
        resize = transforms.Resize(size=self.size)
        image = resize(image)

        ## random orizontal flip
        if torch.rand(1) > 0.5:
            image = transforms.functional.hflip(image)
            label[0::2] =  1 - label[0::2]

        # random rotation
        if torch.rand(1) > 0.5:
            angle = np.random.randint(-15, 15)
            image = transforms.functional.rotate(image, angle)
            angle = angle * np.pi/180
            center = (0.5, 0.5)
            rot_matrix = np.zeros((2, 3))
            rot_matrix[0, 0] = np.cos(angle)
            rot_matrix[0, 1] = np.sin(angle)
            rot_matrix[1, 0] = -np.sin(angle)
            rot_matrix[1, 1] = np.cos(angle)
            rot_matrix[0, 2] = (1 - np.cos(angle)) * center[0] - np.sin(angle) * center[1]
            rot_matrix[1, 2] = np.sin(angle) * center[0] + (1 - np.cos(angle)) * center[1]

            # Apply rotation to each label coordinate
            for i in range(len(label) // 2):
                # Extract x and y coordinates
                x_coord = label[i * 2]
                y_coord = label[i * 2 + 1]
                # Translate to origin
                x_coord -= center[0]
                y_coord -= center[1]
                # Apply rotation
                label[i * 2] = x_coord * rot_matrix[0, 0] + y_coord * rot_matrix[0, 1] + center[0]
                label[i * 2 + 1] = x_coord * rot_matrix[1, 0] + y_coord * rot_matrix[1, 1] + center[1]

        ## random brightness and contrast
        if torch.rand(1) > 0.5:
            image = transforms.ColorJitter(brightness=0.5, contrast=0.5)(image)
        
        ## random gamma correction
        if torch.rand(1) > 0.5:
            gamma = np.random.uniform(0.5, 1.5)
            image = transforms.functional.adjust_gamma(image, gamma)

        ## Convert to tensor and normalize
        label = torch.tensor(label)

        if self.input_channels == 1: image = image.convert('L')
        image = transforms.functional.to_tensor(image)
        # image = transforms.functional.normalize(image, (0.5), (0.5))
        # im_tensor = torchvision.transforms.ToTensor()(im)
        image = (2 * image) - 1  
        return image, label


    def get_heatmap(self, idx):
        """
        given a index of the patient return the 6D heatmap of the keypoints
        """
        if self.target == 'heatmaps': image, labels, calc_value, original_shape, _ = self.get_image_label(idx)
        else: image, labels, calc_value, original_shape = self.get_image_label(idx)

        ## mulptiple the labels by the image size
        converter = np.tile([image.size[0], image.size[1]], 6)
        labels = labels * converter

        x, y = np.meshgrid(np.arange(0, image.size[0]), np.arange(0, image.size[1]))
        pos = np.dstack((x, y)) 

        std_dev = int(image.size[0] * 0.05) 
        covariance = np.array([[std_dev * 20, 0.], [0., std_dev]])
        
        # Initialize an empty 6-channel heatmap vector
        heatmaps_label= np.zeros((image.size[1], image.size[0], 6), dtype=np.float32)
        for hp, heart_part in enumerate([labels[0:4], labels[4:8], labels[8:12]]): ## LVIDd, IVSd, LVPWd
            ## compute the angle of the heart part
            x_diff = heart_part[0:2][0] - heart_part[2:4][0]
            y_diff = heart_part[2:4][1] - heart_part[0:2][1]
            angle = math.degrees(math.atan2(y_diff, x_diff))

            for i in range(2): ## each heart part has got two keypoints with the same angle
                mean = (int(heart_part[i*2]), int(heart_part[(i*2)+1]))
                gaussian = multivariate_normal(mean=mean, cov=covariance)
                base_heatmap = gaussian.pdf(pos)

                rotation_matrix = cv2.getRotationMatrix2D(mean, angle + 90, 1.0)
                base_heatmap = cv2.warpAffine(base_heatmap, rotation_matrix, (base_heatmap.shape[1], base_heatmap.shape[0]))
                base_heatmap = base_heatmap / np.max(base_heatmap)
                channel_index = hp * 2 + i
                heatmaps_label[:, :, channel_index] = base_heatmap

        return heatmaps_label

    def get_image_label(self, idx):
        """
        from index return the image and the label 
        the labels are the normalized coordinates of the keypoints
        """
        patient = self.patient_files[idx]
        patient_label = self.keypoints_dict[patient]

        # read the image wiht PIL
        image = Image.open(os.path.join(self.data_dir, 'image', patient+'.png'))
        
        # read the label  
        keypoints_label, calc_value_list = [], []
        for heart_part in ['LVPWd', 'LVIDd', 'IVSd']:
            if patient_label[heart_part] is not None:
                x1_heart_part = patient_label[heart_part]['x1'] / patient_label[heart_part]['width']
                y1_heart_part = patient_label[heart_part]['y1'] / patient_label[heart_part]['height']
                x2_heart_part = patient_label[heart_part]['x2'] / patient_label[heart_part]['width']
                y2_heart_part = patient_label[heart_part]['y2'] / patient_label[heart_part]['height']
                heart_part_value = patient_label[heart_part]['calc_value']
                keypoints_label.append([x1_heart_part, y1_heart_part, x2_heart_part, y2_heart_part])
                calc_value_list.append(heart_part_value)

        keypoints_label = (np.array(keypoints_label)).flatten()
        calc_value_list = np.array(calc_value_list).flatten()
        original_shape = np.array([patient_label[heart_part]['height'], patient_label[heart_part]['width']])

        if self.target == 'heatmaps' or self.target == 'segmentation':
            # read the npy file of the heatmpa
            heatmap = np.load(os.path.join(self.data_dir, 'heatmap', patient+'.npy'))
            heatmap = heatmap.astype(np.float32)
            return image, keypoints_label, calc_value_list, original_shape, heatmap
        else:
            return image, keypoints_label, calc_value_list, original_shape

        
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

class EchoNetGeneretedDataset(Dataset):
    """
    Class datset for the generated images of EchoNet-LVH dataset
    """
    def __init__(self, par_dir, trial, experiment, guide_w, epoch, phase, target, input_channels, size, 
                data_path=None, label_directory=None, transform=None, augmentation=False, 
                original_shape=False,
                percentace=1.0):
        """
        Args:
            par_dir (string): Directory with all the generated images.
            trial (string): Trial number of the experiment
            experiment (string): Experiment number of the trial
            guide_w (string): Guide weight of the experiment
            epoch (string): Epoch number of the experiment
            phase (string): diastole or systole
            target (string): keypoint, heatmap, segmentation
            input_channels (int): Number of input channels, must be 3 for RGB or 1 for grayscale
            size (tuple): Size of the image
            transform (callable, optional): Optional transform to be applied on a sample.    
            transform_target (callable, optional): Optional transform to be applied on a sample.
            augmentation (bool): Apply data augmentation to the image and the label
            percentace (float): Percentage of the dataset to use, default 1.0
        """
        self.par_dir = par_dir
        self.trial = trial
        self.experiment = experiment
        self.guide_w = guide_w
        self.epoch = epoch
        self.phase = phase
        self.target = target
        self.augmentation = augmentation
        self.size = size
        self.input_channels = input_channels
        self.data_path = data_path
        self.original_shape = original_shape
        self.percentace = percentace

        self.data_dir = os.path.join(par_dir, trial, experiment, 'test', f'w_{guide_w}', f'samples_ep_{epoch}')

        ## label dict
        self.label_dict = self.get_label_dict()

        ## patient files
        self.patient_files = list(self.label_dict.keys())
        self.patient_files_subsample = self.get_random_subsample()


    def __len__(self):
        """
        Return the total number of patiel in selected batch
        """ 
        return len(self.patient_files_subsample)

    def __getitem__(self, idx):
        """
        get the image and the label of the patient
        """

        if torch.is_tensor(idx):
            idx = idx.tolist()

        ## trasform the label based on target: keypoints, heatmaps, segmentations
        if self.target == 'keypoints': 
            # image_label_start = time.time()
            image, label, calc_value, original_shape = self.get_image_label(idx)
           
            if self.augmentation:
                image, label = self.data_augmentation_kp(image, label)
            else:
                resize = transforms.Resize(size=self.size)
                image = resize(image)
                if self.input_channels == 1: image = image.convert('L')
                label = torch.tensor(label)
                image = transforms.functional.to_tensor(image)
                # image = transforms.functional.normalize(image, (0.5), (0.5))  
                # im_tensor = torchvision.transforms.ToTensor()(im)

                image = (2 * image) - 1  

        elif self.target == 'heatmaps': 
            # image_label_start = time.time()
            image, label, calc_value, original_shape, heatmap = self.get_image_label(idx)
            
            # label_start = time.time()   
            label = heatmap #self.get_heatmap(idx)
            # label_stop = time.time()
            # print(f'Time to get heatmap: {label_stop - label_start:.5f}')
            ## apply data augmentation only on training set, else simply resize the image and the label
            if self.augmentation:
                image, label = self.data_augmentation(image, label)
                # print(f'Time to apply data augmentation: {aug_stop - aug_start:.5f}')
            else:
                image, label = self.trasform(image, label)

        if self.original_shape: return image, label, calc_value, original_shape
        else: return image, label
        
    
    def get_image_label(self, idx):
        """
        from index return the image and the label 
        the labels are the normalized coordinates of the keypoints
        """
        patient = self.patient_files_subsample[idx]
        patient_label = self.label_dict[patient]
        #read the image wiht PIL
        image = Image.open(os.path.join(self.data_dir, patient+'.png'))
        
        # read the label  
        keypoints_label, calc_value_list = [], []
        for heart_part in ['LVPWd', 'LVIDd', 'IVSd']:
            if patient_label[heart_part] is not None:
                x1_heart_part = patient_label[heart_part]['x1'] / patient_label[heart_part]['width']
                y1_heart_part = patient_label[heart_part]['y1'] / patient_label[heart_part]['height']
                x2_heart_part = patient_label[heart_part]['x2'] / patient_label[heart_part]['width']
                y2_heart_part = patient_label[heart_part]['y2'] / patient_label[heart_part]['height']
                heart_part_value = patient_label[heart_part]['calc_value']
                keypoints_label.append([x1_heart_part, y1_heart_part, x2_heart_part, y2_heart_part])
                calc_value_list.append(heart_part_value)

        keypoints_label = (np.array(keypoints_label)).flatten()
        calc_value_list = np.array(calc_value_list).flatten()
        original_shape = np.array([patient_label[heart_part]['height'], patient_label[heart_part]['width']])


        if self.target == 'heatmaps' or self.target == 'segmentation':
            heat_idx = patient.split('_')[1]
            shape_guided_idx = patient.split('_')[2]
            heatmap = np.load(os.path.join(self.data_dir, f'heatmap_{heat_idx}.npy'))[int(shape_guided_idx)]
            heatmap = heatmap.astype(np.float32)

            return image, keypoints_label, calc_value_list, original_shape, heatmap
        else:
            return image, keypoints_label, calc_value_list, original_shape
    
    def data_augmentation(self, image, label):
        """
        Set of trasformation to apply to image and label (heatmaps).
        This function contain all the data augmentation trasformations and the normalization.
        Note: torchvision.transforms often rely on PIL as the underlying library, 
            so each channel of heatmap need to transform  separately (channels are in the first dimension)
        """
        # convert each channel in PIL image
        label = [Image.fromarray(label[ch,:,:]) for ch in range(label.shape[0])]

        ## Resize
        resize = transforms.Resize(size=self.size)
        image = resize(image)
        label = [resize(ch) for ch in label]
        
        ## random rotation to image and label
        if torch.rand(1) > 0.5:
            angle = np.random.randint(-15, 15)
            image = transforms.functional.rotate(image, angle)
            label = [transforms.functional.rotate(ch, angle) for ch in label]

        ## random translation to image and label in each direction
        if torch.rand(1) > 0.5:
            translate = transforms.RandomAffine.get_params(degrees=(0.,0.), 
                                                        translate=(0.10, 0.10),
                                                        scale_ranges=(1.0,1.0),
                                                        shears=(0.,0.), 
                                                        img_size=self.size)
            image = transforms.functional.affine(image, *translate)
            label = [transforms.functional.affine(ch, *translate) for ch in label]

        ## random horizontal flip
        if torch.rand(1) > 0.5:
            image = transforms.functional.hflip(image)
            label = [transforms.functional.hflip(ch) for ch in label]
            
        ## random brightness and contrast
        if torch.rand(1) > 0.5:
            image = transforms.ColorJitter(brightness=0.5, contrast=0.5)(image)
        
        ## random gamma correction
        if torch.rand(1) > 0.5:
            gamma = np.random.uniform(0.5, 1.5)
            image = transforms.functional.adjust_gamma(image, gamma)

        ## Convert to tensor and normalize
        label = np.array([np.array(ch) for ch in label])
        label = torch.tensor(label)

        if self.input_channels == 1: image = image.convert('L')
        image = transforms.functional.to_tensor(image)
        # image = transforms.functional.normalize(image, (0.5), (0.5))
        # im_tensor = torchvision.transforms.ToTensor()(im)
        image = (2 * image) - 1  
        return image, label
    
    def data_augmentation_kp(self, image, label):
        ## Resize
        resize = transforms.Resize(size=self.size)
        image = resize(image)

        ## random orizontal flip
        if torch.rand(1) > 0.5:
            image = transforms.functional.hflip(image)
            label[0::2] =  1 - label[0::2]

        # random rotation
        if torch.rand(1) > 0.5:
            angle = np.random.randint(-15, 15)
            image = transforms.functional.rotate(image, angle)
            angle = angle * np.pi/180
            center = (0.5, 0.5)
            rot_matrix = np.zeros((2, 3))
            rot_matrix[0, 0] = np.cos(angle)
            rot_matrix[0, 1] = np.sin(angle)
            rot_matrix[1, 0] = -np.sin(angle)
            rot_matrix[1, 1] = np.cos(angle)
            rot_matrix[0, 2] = (1 - np.cos(angle)) * center[0] - np.sin(angle) * center[1]
            rot_matrix[1, 2] = np.sin(angle) * center[0] + (1 - np.cos(angle)) * center[1]

            # Apply rotation to each label coordinate
            for i in range(len(label) // 2):
                # Extract x and y coordinates
                x_coord = label[i * 2]
                y_coord = label[i * 2 + 1]
                # Translate to origin
                x_coord -= center[0]
                y_coord -= center[1]
                # Apply rotation
                label[i * 2] = x_coord * rot_matrix[0, 0] + y_coord * rot_matrix[0, 1] + center[0]
                label[i * 2 + 1] = x_coord * rot_matrix[1, 0] + y_coord * rot_matrix[1, 1] + center[1]

        ## random brightness and contrast
        if torch.rand(1) > 0.5:
            image = transforms.ColorJitter(brightness=0.5, contrast=0.5)(image)
        
        ## random gamma correction
        if torch.rand(1) > 0.5:
            gamma = np.random.uniform(0.5, 1.5)
            image = transforms.functional.adjust_gamma(image, gamma)

        ## Convert to tensor and normalize
        label = torch.tensor(label)

        if self.input_channels == 1: image = image.convert('L')
        image = transforms.functional.to_tensor(image)
        # image = transforms.functional.normalize(image, (0.5), (0.5))
        # im_tensor = torchvision.transforms.ToTensor()(im)
        image = (2 * image) - 1  
        return image, label
    
    def trasform(self, image, label):
        """
        Simple trasformaztion of the label and image. Resize and normalize the image and resize the label
        """
        # convert each channel in PIL image
        label = [Image.fromarray(label[ch,:,:]) for ch in range(label.shape[0])]

        ## Resize
        resize = transforms.Resize(size=self.size)
        image = resize(image)
        label = [resize(ch) for ch in label]

        ## convert to tensor and normalize
        label = np.array([np.array(ch) for ch in label])
        label = torch.tensor(label)

        if self.input_channels == 1: image = image.convert('L')
        image = transforms.functional.to_tensor(image)
        # image = transforms.functional.normalize(image, (0.5), (0.5))
        # im_tensor = torchvision.transforms.ToTensor()(im)
        image = (2 * image) - 1    
        return image, label

    def get_label_dict(self):
        """
        read the json file of the label with the keypoints
        """
        path_label = os.path.join(self.par_dir, self.trial, self.experiment, 'test', f'w_{self.guide_w}', f'samples_ep_{self.epoch}')
        with open(os.path.join(path_label, 'label.json'), 'r') as f:
            label_dict = json.load(f)
        return label_dict

    def get_random_subsample(self):
        """
        Get a random subsample of the dataset
        """
        ## randmon subsample of the patient files
        np.random.seed(42)
        len_subsample = int(len(self.patient_files) * self.percentace) 
        random_subsample = np.random.choice(self.patient_files, len_subsample, replace=False)
        return random_subsample
        

class EchoNetLVH(Dataset):
    """
    Update of 19/07/2024: I use this class only to create the folder for given input following the EchoNet-LVH dataset structure
    """
    def __init__(self, data_dir, batch, split, patients, phase, transform=None):
        """
        Args:
            data_dir (string): Directory with all the video.
            batch (string): Batch number of video folder, e.g. 'Batch1', 'Batch2', 'Batch3', 'Batch4'.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data_dir = data_dir
        self.batch = batch
        self.transform = transform
        self.label = self.get_keypoint_dataset()

        ## train, validatioor test dataset
        self.split = split

        ## list of patients in the selected batch, this comes from the ausiliar functions
        self.patients = patients

        ## diastole or systole frame
        self.phase = phase

        self.keypoints_dict = {patient_hash: self.get_keypoint(patient_hash) for patient_hash in tqdm.tqdm(self.patients)}

    
    def __len__(self):
        """
        Return the total number of patiel in selected batch
        """
        return len(self.patients)

    def __getitem__(self, idx):
        """
        Get the image and the label of the patient
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        patient = self.patients[idx]
        patient_label = self.keypoints_dict[patient]
        
        ## read the video
        video_dir = os.path.join(self.data_dir, self.batch, patient+'.avi')
        video = read_video(video_dir)
        

        img_diastole = video[patient_label[self.phase]]
        image = Image.fromarray(img_diastole)
        keypoints_label = []
        for heart_part in ['LVPWd', 'LVIDd', 'IVSd']:
            if patient_label[heart_part] is not None:
                x1_heart_part = patient_label[heart_part]['x1'] / img_diastole.shape[1]
                y1_heart_part = patient_label[heart_part]['y1'] / img_diastole.shape[0]
                x2_heart_part = patient_label[heart_part]['x2'] / img_diastole.shape[1]
                y2_heart_part = patient_label[heart_part]['y2'] / img_diastole.shape[0]
                keypoints_label.append([x1_heart_part, y1_heart_part, x2_heart_part, y2_heart_part])

        keypoints_label = (np.array(keypoints_label)).flatten()

        if self.transform:
            image = self.transform(image)

        return image, keypoints_label

    def get_keypoint_dataset(self):
        """
        get the dataset of keypoint for all the patients
        """
        label_dir = os.path.join(self.data_dir, 'MeasurementsList.csv')
        label = pd.read_csv(label_dir, index_col=0)
        return label

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
                    'LVIDs': None, 'IVSs': None, 'LVPWs': None,
                    'diastole': None, 'systole': None, 'split': None}

        for value in label[label['HashedFileName'] == patient_hash]['Calc'].values:
            x1 = label.loc[(label['HashedFileName'] == patient_hash) & (label['Calc'] == value), 'X1'].array[0]
            x2 = label.loc[(label['HashedFileName'] == patient_hash) & (label['Calc'] == value), 'X2'].array[0]
            y1 = label.loc[(label['HashedFileName'] == patient_hash) & (label['Calc'] == value), 'Y1'].array[0]
            y2 = label.loc[(label['HashedFileName'] == patient_hash) & (label['Calc'] == value), 'Y2'].array[0]
            if value.endswith('s'):
                systole = label.loc[(label['HashedFileName'] == patient_hash) & (label['Calc'] == value), 'Frame'].array[0]
                label_dict['systole'] = systole-1
            elif value.endswith('d'):
                diastole = label.loc[(label['HashedFileName'] == patient_hash) & (label['Calc'] == value), 'Frame'].array[0]
                label_dict['diastole'] = diastole-1

            calc_value = label.loc[(label['HashedFileName'] == patient_hash) & (label['Calc'] == value), 'CalcValue'].array[0]
            label_dict[value] = {'x1': x1.item(), 'x2': x2.item(), 'y1': y1.item(), 'y2': y2.item(), 'calc_value': calc_value.item()}
        


        split_set = label.loc[(label['HashedFileName'] == patient_hash) & (label['Calc'] == value), 'split'].array[0]
        label_dict['split'] = split_set

        return label_dict

    def get_diastole_patient(self):
        """
        Becouse the dataset label are not equal across the patients, 
        this function return the patient with all diastole label
        """
        diastole_patient = []
        for patient in tqdm.tqdm(self.patients_batch[:]):
            patient_label = self.get_keypoint(patient)
            if patient_label[self.phase] is not None :
                if patient_label['LVIDd'] is not None and patient_label['IVSd'] is not None and patient_label['LVPWd'] is not None:
                    if patient_label['split'] == self.split:
                        diastole_patient.append(patient)
        return diastole_patient
    
    def show_img_with_keypoints(self, idx):
        """
        plot the images and the available keypoints for selected patient
        """
        patient = self.patients[idx]
        patient_label = self.get_keypoint(patient)

        video_dir = os.path.join(self.data_dir, self.batch, patient+'.avi')
        video = read_video(video_dir)
        keypoint_dict = self.get_keypoint(patient)

        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20,10), tight_layout=True, num=patient)
        if keypoint_dict['diastole'] is not None:
            ax[0].set_title(f'Diastole - frame {keypoint_dict["diastole"]}', fontsize=30)
            ax[0].imshow(video[patient_label['diastole']], cmap='gray')


        if keypoint_dict['systole'] is not None:
            ax[1].set_title(f'Systole - frame {keypoint_dict["systole"]}', fontsize=30)
            ax[1].imshow(video[patient_label['systole']],cmap='gray')

        color_dict = {'LVIDd': 'red', 'IVSd': 'blue', 'LVPWd': 'green',
                      'LVIDs': 'red', 'IVSs': 'blue', 'LVPWs': 'green',}
        for key, value in keypoint_dict.items():
            if key.endswith('d') and value is not None:
                ax[0].plot(value['x1'], value['y1'], color=color_dict[key], marker='o', markersize=15, alpha=0.7)
                ax[0].plot(value['x2'], value['y2'], color=color_dict[key], marker='o', markersize=15, alpha=0.7)
            elif key.endswith('s') and value is not None:
                ax[1].plot(value['x1'], value['y1'], color=color_dict[key], marker='o', markersize=15, alpha=0.7)
                ax[1].plot(value['x2'], value['y2'], color=color_dict[key], marker='o', markersize=15, alpha=0.7)
                 
        #desable the axis
        ax[0].axis('off')
        ax[1].axis('off')
        plt.show()

    def save_img_and_label(self):
        """
        Create the directory to save the images for traing validation and test
        """
        
        patients = self.patients

        # create the directory to save the images 
        dir_save = os.path.join('DATA', self.batch, self.split, self.phase)
        dir_save_img = os.path.join(dir_save, 'image')
        dir_save_label = os.path.join(dir_save, 'label')

        if not os.path.exists(dir_save): os.makedirs(dir_save)   
        if not os.path.exists(dir_save_img): os.makedirs(dir_save_img)
        if not os.path.exists(dir_save_label): os.makedirs(dir_save_label)

        with open(os.path.join(dir_save_label,'label.json'), 'w') as f:
            json.dump(self.keypoints_dict, f, default=convert_to_serializable, indent=4)
         
        for patient in tqdm.tqdm(patients):
            patient_label = self.get_keypoint(patient)

            video_dir = os.path.join(self.data_dir, self.batch, patient+'.avi')
            video = read_video(video_dir)

            img_frame = video[patient_label[self.phase]]
            image = Image.fromarray(img_frame)
            # print(patient, image.size)
            image.save(os.path.join(dir_save_img,  f'{patient}.png'))

        ## save the label in a json file
        
        


## ausiliar functions to get the subset of the dataset 
def select_patients(data_dir, batch, phase):
    """
    Because the dataset label are not equal across the patients, 
    this function return the list of patient with all diastole label

    """

    #ma the phase with a single letter
    phase_letter = {'diastole': 'd', 'systole': 's'}

    ## Read the entire batch patients' name
    batch_dir = os.path.join(data_dir, batch)
    patients_batch = [i.split('.')[0] for i in os.listdir(batch_dir) if i.endswith('.avi')]

    ## Read the label dataset
    label_dir = os.path.join(data_dir, 'MeasurementsList.csv')
    label = pd.read_csv(label_dir, index_col=0)
    
    phase_patient = {'train': [], 'val': [], 'test': []}
    for patient in tqdm.tqdm(patients_batch):
        patient_label = select_keypoint(patient, label)
        if patient_label[phase] is not None :  ## check if the patient has the diastole label
            if patient_label['LVID' + phase_letter[phase]] is not None and patient_label['IVS'+ phase_letter[phase]] is not None and patient_label['LVPW'+ phase_letter[phase]] is not None: ## check if the patient has all the label
                phase_patient[patient_label['split']].append(patient)
    
    return phase_patient

def select_keypoint(patient_hash, label):
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
    label_dict = {'LVIDd': None, 'IVSd': None, 'LVPWd': None, 
                'LVIDs': None, 'IVSs': None, 'LVPWs': None,
                'diastole': None, 'systole': None, 'split': None}

    for value in label[label['HashedFileName'] == patient_hash]['Calc'].values:
        x1 = label.loc[(label['HashedFileName'] == patient_hash) & (label['Calc'] == value), 'X1'].array[0]
        x2 = label.loc[(label['HashedFileName'] == patient_hash) & (label['Calc'] == value), 'X2'].array[0]
        y1 = label.loc[(label['HashedFileName'] == patient_hash) & (label['Calc'] == value), 'Y1'].array[0]
        y2 = label.loc[(label['HashedFileName'] == patient_hash) & (label['Calc'] == value), 'Y2'].array[0]
        if value.endswith('s'):
            systole = label.loc[(label['HashedFileName'] == patient_hash) & (label['Calc'] == value), 'Frame'].array[0]
            label_dict['systole'] = systole-1
        elif value.endswith('d'):
            diastole = label.loc[(label['HashedFileName'] == patient_hash) & (label['Calc'] == value), 'Frame'].array[0]
            label_dict['diastole'] = diastole-1

        calc_value = label.loc[(label['HashedFileName'] == patient_hash) & (label['Calc'] == value), 'CalcValue'].array[0]
        label_dict[value] = {'x1': x1.astype(int), 'x2': x2.astype(int), 'y1': y1.astype(int), 'y2': y2.astype(int), 'calc_value': calc_value}
    


    split_set = label.loc[(label['HashedFileName'] == patient_hash), 'split'].array[0]
    label_dict['split'] = split_set

    return label_dict

def convert_to_serializable(obj):
    if isinstance(obj, (np.ndarray, np.integer)):
        return int(obj)
    return obj

        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Read the dataset')
    parser.add_argument('--data_dir', type=str, default="/media/angelo/OS/Users/lasal/Desktop/Phd notes/Echocardiografy/EchoNet-LVH", help='Directory of the dataset')
    parser.add_argument('--batch', type=str, default='Batch2', help='Batch number of video folder, e.g. Batch1, Batch2, Batch3, Batch4')
    parser.add_argument('--phase', type=str, default='diastole', help='select the phase of the heart, diastole or systole')
    args = parser.parse_args()

    par_dir = "/media/angelo/OS/Users/lasal/OneDrive - Scuola Superiore Sant'Anna/PhD_notes/Echocardiografy/trained_model/diffusion/eco"
    trial = "trial_2"
    experiment = "cond_ldm_1"
    guide_w = "0.6"
    epoch = "100"
    percentace = 0.2

    dataset = EchoNetGeneretedDataset(par_dir=par_dir, trial=trial, experiment=experiment, guide_w=guide_w, epoch=epoch, phase='diastole',
                                      target='heatmaps', input_channels=1, size=(320, 240), augmentation=True,
                                      percentace=percentace)
    
    # transform = transforms.Compose([transforms.Resize((256,256)),
    #                                 transforms.ToTensor(),
    #                                 transforms.Normalize((0.5 ), (0.5 ))])
    
    # patients = select_patients(args.data_dir, args.batch, args.phase)
    # for key in patients.keys():
    #     print(f'Number of patient in {key} = {len(patients[key])}')

    #     echonet_dataset = EchoNetLVH(data_dir=args.data_dir, batch=args.batch, split=key, transform=transform, patients=patients[key], phase=args.phase)
    #     # echonet_dataset.save_img_and_label()
        

    # image, label = echonet_dataset[6]

    # ## convert the tor into numpy
    # image = image.numpy().transpose((1, 2, 0))
    # print(np.min(image), np.max(image))

    # plt.figure()
    # plt.hist(image.ravel())
    # plt.show()
    

    # plt.figure(figsize=(14,14), num='Example')
    # plt.imshow(image, cmap='gray')
    # plt.scatter(label[0] * image.shape[1], label[1] * image.shape[0], color='green', marker='o', s=100, alpha=0.5) 
    # plt.scatter(label[2] * image.shape[1], label[3] * image.shape[0], color='green', marker='o', s=100, alpha=0.5)

    # # plt.scatter(label[4] * image.shape[1], label[5] * image.shape[0], color='red', marker='o', s=100, alpha=0.5) 
    # # plt.scatter(label[6] * image.shape[1], label[7] * image.shape[0], color='red', marker='o', s=100, alpha=0.5)

    # # plt.scatter(label[8] * image.shape[1], label[9] * image.shape[0], color='blue', marker='o', s=100, alpha=0.5) 
    # # plt.scatter(label[10] * image.shape[1], label[11] * image.shape[0], color='blue', marker='o', s=100, alpha=0.5)

    # #0X10F154DF2CD47783
    # echonet_dataset.show_img_with_keypoints(6)
    
    # plt.show()