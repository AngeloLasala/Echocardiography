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

from data_reader import read_video, read_video

class EchoNetDataset(Dataset):
    def __init__(self, batch, split, phase, target, label_directory=None, transform=None, transform_target=None):
        """
        Args:
            batch (string): Batch number of video folder, e.g. 'Batch1', 'Batch2', 'Batch3', 'Batch4'.
            split (string): train, validation or test
            phase (string): diastole or systole
            target (string): keypoint, heatmap, segmentation
            label_directory (string): Directory of the label dataset, default None that means read the file from the json file
            transform (callable, optional): Optional transform to be applied
        """
        self.split = split
        self.batch = batch
        self.phase = phase
        self.target = target
        self.transform = transform
        self.transform_target = transform_target


        self.data_dir = os.path.join('DATA', self.batch, self.split, self.phase)
        self.patient_files = [patient_hash.split('.')[0] for patient_hash in os.listdir(os.path.join(self.data_dir, 'image'))]

    
        if label_directory is not None:
            label = pd.read_csv(os.path.join(label_directory, 'MeasurementsList.csv'), index_col=0)
            self.label = label
            self.keypoints_dict = {patient_hash: self.get_keypoint(patient_hash) for patient_hash in tqdm.tqdm(self.patient_files)}
        else:
            #load a directory from a json file
            with open(os.path.join(self.data_dir, 'label', 'label.json'), 'r') as f:
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
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image, label = self.get_image_label(idx)

        if self.target == 'keypoints': 
            label = label

        elif self.target == 'heatmap': 
            label = self.get_heatmap(idx)
            if self.transform_target:
                #convert each channel in a pil image
                label = [Image.fromarray(label[:,:,i]) for i in range(label.shape[2])]
                label = [self.transform_target(i) for i in label]
                label = np.array([np.array(i) for i in label])

        elif self.target == 'segmentation':
            label = self.get_heatmap(idx)
            label = (label > 0.5).astype(np.float32)
            if self.transform_target:
                #convert each channel in a pil image
                label = [Image.fromarray(label[:,:,i]) for i in range(label.shape[2])]
                label = [self.transform_target(i) for i in label]
                label = np.array([np.array(i) for i in label])
        
        if self.transform:
            image = self.transform(image)

        return image, label

    def get_heatmap(self, idx):
        """
        given a index of the patient return the 6D heatmap of the keypoints
        """
        image, labels = self.get_image_label(idx)

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
                # print(base_heatmap.shape, np.min(base_heatmap), np.max(base_heatmap))
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
        keypoints_label = []
        for heart_part in ['LVPWd', 'LVIDd', 'IVSd']:
            if patient_label[heart_part] is not None:
                x1_heart_part = patient_label[heart_part]['x1'] / image.size[0]
                y1_heart_part = patient_label[heart_part]['y1'] / image.size[1]
                x2_heart_part = patient_label[heart_part]['x2'] / image.size[0]
                y2_heart_part = patient_label[heart_part]['y2'] / image.size[1]
                keypoints_label.append([x1_heart_part, y1_heart_part, x2_heart_part, y2_heart_part])

        keypoints_label = (np.array(keypoints_label)).flatten()
        return image, keypoints_label

        
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

class EchoNetLVH(Dataset):
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
         

        for patient in tqdm.tqdm(patients):
            patient_label = self.get_keypoint(patient)

            video_dir = os.path.join(self.data_dir, self.batch, patient+'.avi')
            video = read_video(video_dir)

            img_frame = video[patient_label[self.phase]]
            image = Image.fromarray(img_frame)
            # print(patient, image.size)
            image.save(os.path.join(dir_save_img,  f'{patient}.png'))

        ## save the label in a json file
        with open(os.path.join(dir_save_label,'label.json'), 'w') as f:
            json.dump(self.keypoints_dict, f, default=convert_to_serializable, indent=4)
        


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

    transform = transforms.Compose([transforms.Resize((256,256)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5 ), (0.5 ))])
    
    patients = select_patients(args.data_dir, args.batch, args.phase)
    for key in patients.keys():
        print(f'Number of patient in {key} = {len(patients[key])}')

        echonet_dataset = EchoNetLVH(data_dir=args.data_dir, batch=args.batch, split=key, transform=transform, patients=patients[key], phase=args.phase)
        echonet_dataset.save_img_and_label()
        

    image, label = echonet_dataset[6]

    ## convert the tor into numpy
    image = image.numpy().transpose((1, 2, 0))
    print(np.min(image), np.max(image))

    plt.figure()
    plt.hist(image.ravel())
    plt.show()
    

    plt.figure(figsize=(14,14), num='Example')
    plt.imshow(image, cmap='gray')
    plt.scatter(label[0] * image.shape[1], label[1] * image.shape[0], color='green', marker='o', s=100, alpha=0.5) 
    plt.scatter(label[2] * image.shape[1], label[3] * image.shape[0], color='green', marker='o', s=100, alpha=0.5)

    # plt.scatter(label[4] * image.shape[1], label[5] * image.shape[0], color='red', marker='o', s=100, alpha=0.5) 
    # plt.scatter(label[6] * image.shape[1], label[7] * image.shape[0], color='red', marker='o', s=100, alpha=0.5)

    # plt.scatter(label[8] * image.shape[1], label[9] * image.shape[0], color='blue', marker='o', s=100, alpha=0.5) 
    # plt.scatter(label[10] * image.shape[1], label[11] * image.shape[0], color='blue', marker='o', s=100, alpha=0.5)

    #0X10F154DF2CD47783
    echonet_dataset.show_img_with_keypoints(6)
    
    plt.show()