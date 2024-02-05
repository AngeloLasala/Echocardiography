"""
Create the class of EchoNet-LVH dataset
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

from data_reader import read_video_grayscale

class EchoNetLVH(Dataset):
    def __init__(self, data_dir, batch, split, only_diastole=False, transform=None):
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

        ## list of patients in the selected batch
        batch_dir = os.path.join(self.data_dir,self.batch)
        patients_batch = [i.split('.')[0] for i in os.listdir(batch_dir) if i.endswith('.avi')]
        self.patients_batch = patients_batch

        ## due to the label are not equal across the patients, 
        # we can select only the patient with all diastole label
        self.only_diastole = only_diastole

    def __len__(self):
        """
        Return the total number of patiel in selected batch
        """
        batch_dir = os.path.join(self.data_dir,self.batch)
        patients_batch = [i.split('.')[0] for i in os.listdir(batch_dir) if i.endswith('.avi')]
        return len(patients_batch)

    def __getitem__(self, idx):
        """
        Get the image and the label of the patient
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.only_diastole:
            patients = self.get_diastole_patient()
        else:
            self.patients_batch 

        patient = self.patients_batch[idx]
        print(f'{self.split} - {len(patients)}')
        patient_label = self.get_keypoint(patient)
        
        ## read the video
        video_dir = os.path.join(self.data_dir, self.batch, patient+'.avi')
        video = read_video_grayscale(video_dir)

        img_diastole = video[patient_label['diastole']]
        keypoints_label = []
        print(patient_label.keys())
        for heart_part in ['LVPWd', 'LVIDd', 'IVSd']:
            if patient_label[heart_part] is not None:
                x1_heart_part = patient_label[heart_part]['x1']
                y1_heart_part = patient_label[heart_part]['y1']
                x2_heart_part = patient_label[heart_part]['x2']
                y2_heart_part = patient_label[heart_part]['y2']
                keypoints_label.append([x1_heart_part, y1_heart_part, x2_heart_part, y2_heart_part])

        keypoints_label = (np.array(keypoints_label)).flatten()

        if self.transform:
            image = self.transform(image)

        return img_diastole, keypoints_label

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
            label_dict[value] = {'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2, 'calc_value': calc_value}
        


        split_set = label.loc[(label['HashedFileName'] == patient_hash) & (label['Calc'] == value), 'split'].array[0]
        label_dict['split'] = split_set

        return label_dict

    def get_diastole_patient(self):
        """
        Becouse the dataset label are not equal across the patients, 
        this function return the patient with all diastole label
        """
        diastole_patient = []
        for patient in tqdm.tqdm(self.patients_batch[:100]):
            patient_label = self.get_keypoint(patient)
            if patient_label['diastole'] is not None :
                if patient_label['LVIDd'] is not None and patient_label['IVSd'] is not None and patient_label['LVPWd'] is not None:
                    if patient_label['split'] == self.split:
                        diastole_patient.append(patient)
        return diastole_patient
        
    
    def show_img_with_keypoints(self, idx):
        """
        plot the images and the available keypoints for selected patient
        """
        patient = self.patients_batch[idx]
        patient_label = self.get_keypoint(patient)

        video_dir = os.path.join(self.data_dir, self.batch, patient+'.avi')
        video = read_video_grayscale(video_dir)
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Read the dataset')
    parser.add_argument('--data_dir', type=str, default="/media/angelo/OS/Users/lasal/Desktop/Phd notes/Echocardiografy/EchoNet-LVH", help='Directory of the dataset')
    args = parser.parse_args()
    
    echonet_dataset = EchoNetLVH(data_dir=args.data_dir, batch='Batch1', split='train', transform=None, only_diastole=True)
    
    image, label = echonet_dataset[0]

    plt.figure(figsize=(14,14), num='Example')
    plt.imshow(image, cmap='gray')
    plt.scatter(label[0], label[1], color='green', marker='o', s=100, alpha=0.5) 
    plt.scatter(label[2], label[3], color='green', marker='o', s=100, alpha=0.5)

    plt.scatter(label[4], label[5], color='red', marker='o', s=100, alpha=0.5) 
    plt.scatter(label[6], label[7], color='red', marker='o', s=100, alpha=0.5)

    plt.scatter(label[8], label[9], color='blue', marker='o', s=100, alpha=0.5) 
    plt.scatter(label[10], label[11], color='blue', marker='o', s=100, alpha=0.5)

    echonet_dataset.show_img_with_keypoints(0)
    print(image, label)
    
    plt.show()