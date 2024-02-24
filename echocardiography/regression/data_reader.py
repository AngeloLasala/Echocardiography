"""
File that read the video and the correspinding label from the dataset directory
"""
import os
import argparse
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import tqdm

def read_video(file_path):
    """
    Read a video file and return a NumPy array

    Parameters
    ----------
    file_path : str
        Path to the video file

    Returns
    -------
    video_array : ndarray
        NumPy array of shape (num_frames, height, width, channels) containing the video frames
    """
    # Open the video file
    cap = cv2.VideoCapture(file_path)

    # Check if the video file is opened successfully
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return None

    # Read the first frame to get video properties
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read the first frame.")
        return None

    # Get video properties
    height, width, channels = frame.shape
    fps = cap.get(cv2.CAP_PROP_FPS)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initialize an empty NumPy array to store video frames
    video_array = np.empty((num_frames, height, width, channels), dtype=np.uint8)

    # Read and store all frames in the array
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        video_array[frame_count] = frame
        frame_count += 1

    # Release the video capture object
    cap.release()

    return video_array

def read_video_grayscale(file_path):
    """
    Read a video file and return a NumPy array in grayscale (one channel)

    Parameters
    ----------
    file_path : str
        Path to the video file

    Returns
    -------
    video_array : ndarray
        NumPy array of shape (num_frames, height, width) containing the grayscale video frames
    """
    # Open the video file
    cap = cv2.VideoCapture(file_path)

    # Check if the video file is opened successfully
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return None

    # Read the first frame to get video properties
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read the first frame.")
        return None

    # Get video properties
    height, width = frame.shape[:2]
    fps = cap.get(cv2.CAP_PROP_FPS)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initialize an empty NumPy array to store grayscale video frames
    video_array = np.empty((num_frames, height, width), dtype=np.uint8)

    # Read and store all frames in the array
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert the frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        video_array[frame_count] = gray_frame
        frame_count += 1

    # Release the video capture object
    cap.release()

    return video_array

def get_keypoint(label, patient_hash):
    """
    Get the keypoint from the label dataset file

    Parameters
    ----------
    label : pandas.DataFrame
        DataFrame containing the label data
    patient_hash : str
        Hashed file name of the patient

    Returns
    -------
    label_dict : dict
        Dictionary containing the keypoint information
    """
    label_dict = {'LVIDd': None, 'LVIDd': None, 'LVPWd': None, 
                  'LVIDs': None, 'LVIDs': None, 'LVPWs': None,
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


def main(data_dir):
    """
    Main function to read the dataset
    """
    # Read the dataset
    dataset_dir = "/media/angelo/OS/Users/lasal/Desktop/Phd notes/Echocardiografy/EchoNet-LVH"
    print(os.listdir(dataset_dir))

    ## ECconet-LVH is composed by 5 folders: 4 'Batch' with video and 'MeasurementsList.csv' with the label
    # Read the label
    label_dir = os.path.join(dataset_dir, 'MeasurementsList.csv')
    label = pd.read_csv(label_dir, index_col=0)

    patients = np.unique(label['HashedFileName'].values)
    print(label.head())
    
    batch_dir = os.path.join(dataset_dir,'Batch1')
    batch_dict = {'diastole': None, 'systole':None, 'keypoint': None}
    for patient in tqdm.tqdm(os.listdir(batch_dir)):
        patient_hash = patient.split('.')[0] 
        # print(patient)
        
        video_dir = os.path.join(batch_dir, patient)
        video = read_video(video_dir)

        # print(label[label['HashedFileName'] == patient_hash])
        keypoint_dict = get_keypoint(label, patient_hash)
        if keypoint_dict['diastole'] is not None and keypoint_dict['systole'] is not None:
            frame_1 = video[keypoint_dict['diastole']]
            frame_2 = video[keypoint_dict['systole']]
        if keypoint_dict['diastole'] is not None and keypoint_dict['systole'] is None:
            frame_1 = video[keypoint_dict['diastole']]
            frame_2 = None
        if keypoint_dict['diastole'] is None and keypoint_dict['systole'] is not None:
            frame_1 = None
            frame_2 = video[keypoint_dict['systole']]
        batch_dict[patient_hash] = {'keypoint': keypoint_dict}
          
        # plt.figure(figsize=(14,14), num=patient_hash + ' diastole')
        # plt.imshow(batch_dict[patient_hash]['diastole'])
        # ##add the marcker with coordinate x1_d and y1_d
        # plt.plot(keypoint_dict['LVIDd']['x1'], keypoint_dict['LVIDd']['y1'], color='red', marker='o', markersize=10, alpha=0.5)
        # plt.plot(keypoint_dict['LVIDd']['x2'], keypoint_dict['LVIDd']['y2'], color='red', marker='o', markersize=10, alpha=0.5)
        # plt.plot(keypoint_dict['LVPWd']['x1'], keypoint_dict['LVPWd']['y1'], color='blue', marker='o', markersize=5)
        # plt.plot(keypoint_dict['LVPWd']['x2'], keypoint_dict['LVPWd']['y2'], color='blue', marker='o', markersize=5)
        # plt.plot(keypoint_dict['IVSd']['x1'], keypoint_dict['IVSd']['y1'], color='green', marker='o', markersize=5)
        # plt.plot(keypoint_dict['IVSd']['x2'], keypoint_dict['IVSd']['y2'], color='green', marker='o', markersize=5)
        
        # plt.figure(figsize=(14,14), num=patient_hash + ' systole')
        # plt.imshow(batch_dict[patient_hash]['systole'])
        # ##add the marcker with coordinate x1_s and y1_s
        # plt.plot(keypoint_dict['LVIDs']['x1'], keypoint_dict['LVIDs']['y1'], 'ro')
        # plt.plot(keypoint_dict['LVIDs']['x2'], keypoint_dict['LVIDs']['y2'], 'ro')
        # plt.show()

    return batch_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Read the dataset')
    parser.add_argument('--data_dir', type=str, default='"/media/angelo/OS/Users/lasal/Desktop/Phd notes/Echocardiografy/EchoNet-LVH"', help='Directory of the dataset')
    args = parser.parse_args()

    batch_dict = main(args.data_dir)

