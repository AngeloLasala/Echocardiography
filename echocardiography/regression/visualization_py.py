"""
Visualize the hypertrophy dataset as starting point of sampling task with latent diffusion model
"""

import os
import argparse
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import tqdm
import json
from scipy.stats import shapiro

def get_metrics(patients_list, resize):
    """
    Compute relevant metric for the hypertrophy dataset

    Parameters:
    -----------
    patients_list: list
        list of patients
    resize: tuple
        tuple of the new resolution, preserve aspect ratio (240, 320) -> H/W = 0.75 (median value of the dataset)
    """
    rwt_list = []                    ## RWT = 2 * PWT / LVIDd
    lv_mass_list = []                ## LVM = 0.8 * (1.04 * ((LVIDd + IVSd + PWT) ** 3 - LVIDd ** 3)) + 0.6
    relative_distance_list = []      ## Relative distance = 2 * PWT / LVIDd (computed on the distances not to measure)
    rwt_error_list = []              ## RWT error 
    rst_error_list = []              ## RST error 
    hypertrofy_list = []     
    cm_px_list, cm_px_256_list, cm_px_ar_list, aspect_ratio_list = [], [], [], []
    hypertrophy_dict = {}
    split_list = []
    for patient in tqdm.tqdm(patients_list):
        patient_label = label[label['HashedFileName'] == patient]        
        ivsd = patient_label[patient_label['Calc'] == 'IVSd']['CalcValue'].values
        pwd = patient_label[patient_label['Calc'] == 'LVPWd']['CalcValue'].values
        lvidd = patient_label[patient_label['Calc'] == 'LVIDd']['CalcValue'].values
        resolution = patient_label[patient_label['Calc'] == 'Resolution']['CalcValue'].values
        # retorn the patient split
        split = np.array(patient_label['split'].values[0])
        split_list.append(split)
        
        if len(ivsd) == 0 or len(pwd) == 0 or len(lvidd) == 0:
            pass
        
        else:
            resolution = (float(patient_label['Height'].values[0]), float(patient_label['Width'].values[0]))
            aspect_ratio = resolution[0] / resolution[1]  ## H/W
            ivsd = ivsd[0]
            pwd = pwd[0]
            lvidd = lvidd[0]

            ## PW
            pw_x1 = patient_label[patient_label['Calc'] == 'LVPWd']['X1'].values[0]
            pw_y1 = patient_label[patient_label['Calc'] == 'LVPWd']['Y1'].values[0]
            pw_x2 = patient_label[patient_label['Calc'] == 'LVPWd']['X2'].values[0]
            pw_y2 = patient_label[patient_label['Calc'] == 'LVPWd']['Y2'].values[0]

            pw_distance = np.sqrt((pw_x2 - pw_x1) ** 2 + (pw_y2 - pw_y1) ** 2)
            pw_distance_256 = 256 * np.sqrt((1/resolution[1]**2)*(pw_x2 - pw_x1) ** 2 + (1 / resolution[0]**2)*(pw_y2 - pw_y1) ** 2) 
            pw_distance_ar = np.sqrt(((resize[1]/resolution[1])*(pw_x2 - pw_x1)) ** 2 + ((resize[0]/resolution[0])*(pw_y2 - pw_y1)) ** 2) 

            ## LVID
            lvid_x1 = patient_label[patient_label['Calc'] == 'LVIDd']['X1'].values[0]
            lvid_y1 = patient_label[patient_label['Calc'] == 'LVIDd']['Y1'].values[0]
            lvid_x2 = patient_label[patient_label['Calc'] == 'LVIDd']['X2'].values[0]
            lvid_y2 = patient_label[patient_label['Calc'] == 'LVIDd']['Y2'].values[0]

            lvid_distance = np.sqrt((lvid_x2 - lvid_x1) ** 2 + (lvid_y2 - lvid_y1) ** 2)
            lvid_distance_256 = 256 * np.sqrt((1/resolution[1]**2)*(lvid_x2 - lvid_x1) ** 2 + (1 / resolution[0]**2)*(lvid_y2 - lvid_y1) ** 2)
            lvid_distance_ar = np.sqrt(((resize[1]/resolution[1])*(lvid_x2 - lvid_x1)) ** 2 + ((resize[0]/resolution[0])*(lvid_y2 - lvid_y1)) ** 2)

            ## IVSD
            ivsd_x1 = patient_label[patient_label['Calc'] == 'IVSd']['X1'].values[0]
            ivsd_y1 = patient_label[patient_label['Calc'] == 'IVSd']['Y1'].values[0]
            ivsd_x2 = patient_label[patient_label['Calc'] == 'IVSd']['X2'].values[0]
            ivsd_y2 = patient_label[patient_label['Calc'] == 'IVSd']['Y2'].values[0]

            ivsd_distance = np.sqrt((ivsd_x2 - ivsd_x1) ** 2 + (ivsd_y2 - ivsd_y1) ** 2)
            ivsd_distance_256 = 256 * np.sqrt((1/resolution[1]**2)*(ivsd_x2 - ivsd_x1) ** 2 + (1 / resolution[0]**2)*(ivsd_y2 - ivsd_y1) ** 2)
            ivsd_distance_ar = np.sqrt(((resize[1]/resolution[1])*(ivsd_x2 - ivsd_x1)) ** 2 + ((resize[0]/resolution[0])*(ivsd_y2 - ivsd_y1)) ** 2)

            # print(f'PW: {pw_distance:.4f}')
            # print(f'LVID: {lvid_distance:.4f}')
            # print(f'IVSD: {ivsd_distance:.4f}')
            # print(f'cm/pixel (original): {pwd/pw_distance:.4f} - {lvidd/lvid_distance:.4f} - {ivsd/ivsd_distance:.4f}')
            # print(f'cm/pixel (256): {pwd/pw_distance_256:.4f} - {lvidd/lvid_distance_256:.4f} - {ivsd/ivsd_distance_256:.4f}')
            # print(f'cm/pixel (aspect ratio): {pwd/pw_distance_ar:.4f} - {lvidd/lvid_distance_ar:.4f} - {ivsd/ivsd_distance_ar:.4f}')
            cm_px_list.append([lvidd/lvid_distance])
            cm_px_256_list.append([lvidd/lvid_distance_256])
            cm_px_ar_list.append([lvidd/lvid_distance_ar])

            ## Compute the RWT
            rwt = 2 * pwd / lvidd                                   ## RWT compute on dimention (cm)
            relative_distance = 2 * pw_distance / lvid_distance     ## RWT compute on distance (pixel - original resolution)
            rwt_256 = 2 * pw_distance_256 / lvid_distance_256       ## RWT compute on distance (pixel - 256 resolution)
            rwt_ar = 2 * pw_distance_ar / lvid_distance_ar          ## RWT compute on distance (pixel - aspect ratio)

            ## Compute the RST
            rst = 2 * ivsd / lvidd                                  ## RST compute on dimention (cm)
            rst_distance = 2 * ivsd_distance / lvid_distance        ## RST compute on distance (pixel - original resolution)
            rst_256 = 2 * ivsd_distance_256 / lvid_distance_256     ## RST compute on distance (pixel - 256 resolution)
            rst_ar = 2 * ivsd_distance_ar / lvid_distance_ar        ## RST compute on distance (pixel - aspect ratio)

            lv_mass = 0.8 * (1.04 * ((lvidd + ivsd + pwd) ** 3 - lvidd ** 3)) + 0.6
            lv_mass_d = 0.8 * (1.04 * ((lvidd/lvidd + ivsd/lvidd + pwd/lvidd) ** 3 - (lvidd/lvidd) ** 3)) + 0.6
            # print(f'lv_mass_d: {lv_mass_d:.4f}')

            # print(f'rwt: {rwt}, lv_mass: {lv_mass}')
            # print('RWT')
            # print(f'rws_distance: {np.abs(rwt - relative_distance)}')
            # print(f'rwt_256: {np.abs(rwt - rwt_256)}')
            # print(f'rwt_ar: {np.abs(rwt - rwt_ar)}')
            # print()
            # print('RST')
            # print(f'rst_distance: {np.abs(rst - rst_distance)}')
            # print(f'rst_256: {np.abs(rst - rst_256)}')
            # print(f'rst_ar: {np.abs(rst - rst_ar)}')
            rwt_list.append(rwt)
            lv_mass_list.append(lv_mass)
            relative_distance_list.append(relative_distance)
            aspect_ratio_list.append(aspect_ratio)
            hypertrofy_list.append([lv_mass, rwt, lv_mass_d, rst_ar])
            rwt_error_list.append([rwt - relative_distance, rwt - rwt_256, rwt - rwt_ar])
            rst_error_list.append([rst - rst_distance, rst - rst_256, rst - rst_ar])
            hypertrophy_dict[patient] = [lv_mass, rwt]

        # print('==============================================') 
    
    return rwt_list, hypertrofy_list, cm_px_list, cm_px_256_list, cm_px_ar_list, aspect_ratio_list, rwt_error_list, rst_error_list, hypertrophy_dict, split_list

if __name__ == '__main__':   
    parser = argparse.ArgumentParser(description='Visualize the hypertrophy dataset')
    parser.add_argument('--compute_metrics', action='store_true', help="compute the metrics accross all the data, otherwise load the precomputed one, default=False")
    args = parser.parse_args()

dataset_dir = "/home/angelo/Documenti/Echocardiography/echocardiography"
print(os.listdir(dataset_dir))

## ECconet-LVH is composed by 5 folders: 4 'Batch' with video and 'MeasurementsList.csv' with the label
# Read the label
label_dir = os.path.join(dataset_dir, 'MeasurementsList.csv')
label = pd.read_csv(label_dir, index_col=0)
print(label.head())

patients = label['HashedFileName'].unique()

if args.compute_metrics:

    rwt_list, hypertrofy_list, cm_px_list, cm_px_256_list, cm_px_ar_list, aspect_ratio_list, rwt_error_list, rst_error_list, hypertrofy_dict, split_list = get_metrics(patients, (240, 320))  

    hypertrofy_list = np.array(hypertrofy_list)
    cm_px_list = np.array(cm_px_list)
    cm_px_256_list = np.array(cm_px_256_list)
    cm_px_ar_list = np.array(cm_px_ar_list)
    aspect_ratio_list = np.array(aspect_ratio_list)
    rwt_error_list = np.array(rwt_error_list)
    rst_error_list = np.array(rst_error_list)
    split_list = np.array(split_list)

    save_folder = 'dataset_metrics'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    else:
        pass
    np.save(os.path.join(save_folder,'hypertrofy_list.npy'), hypertrofy_list)
    np.save(os.path.join(save_folder,'cm_px_list.npy'), cm_px_list)
    np.save(os.path.join(save_folder,'cm_px_256_list.npy'), cm_px_256_list)
    np.save(os.path.join(save_folder,'cm_px_ar_list.npy'), cm_px_ar_list)
    np.save(os.path.join(save_folder,'aspect_ratio_list.npy'), aspect_ratio_list)
    np.save(os.path.join(save_folder,'rwt_error_list.npy'), rwt_error_list)
    np.save(os.path.join(save_folder,'rst_error_list.npy'), rst_error_list)
    np.save(os.path.join(save_folder,'split_list.npy'), split_list)

    ## save hypertrophy_dict as json
    with open(os.path.join(save_folder, 'hypertrophy_dict.json'), 'w') as f:
        json.dump(hypertrofy_dict, f)

else:
    save_folder = 'dataset_metrics'
    hypertrofy_list = np.load(os.path.join(save_folder,'hypertrofy_list.npy'))
    cm_px_list = np.load(os.path.join(save_folder,'cm_px_list.npy'))
    cm_px_256_list = np.load(os.path.join(save_folder,'cm_px_256_list.npy'))
    cm_px_ar_list = np.load(os.path.join(save_folder,'cm_px_ar_list.npy'))
    aspect_ratio_list = np.load(os.path.join(save_folder,'aspect_ratio_list.npy'))
    rwt_error_list = np.load(os.path.join(save_folder,'rwt_error_list.npy'))
    rst_error_list = np.load(os.path.join(save_folder,'rst_error_list.npy'))
    hypertrofy_dict = json.load(open(os.path.join(save_folder, 'hypertrophy_dict.json')))
    split_list = np.load(os.path.join(save_folder,'split_list.npy'))

print('INFO')

batch_list = ['Batch1', 'Batch2', 'Batch3', 'Batch4']
data_dir = "/media/angelo/OS/Users/lasal/Desktop/DATA_h"
print(f'Number of patients: {len(patients)} - {len(hypertrofy_dict.keys())}')
for batch in batch_list:
    print(f'{batch}')
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(data_dir, batch, split, 'diastole', 'label', 'label.json' )
        with open(split_dir) as json_file:
            split_label = json.load(json_file)
        
        patients = split_label.keys()
        NG, EH, CR, CH = 0, 0, 0, 0
        for patient in patients:
            if patient in hypertrofy_dict.keys():
                lvm, rwt = hypertrofy_dict[patient][0], hypertrofy_dict[patient][1]
                if lvm < 200 and rwt < 0.42:
                    NG += 1
                elif lvm > 200 and rwt < 0.42:
                    EH += 1
                elif lvm < 200 and rwt > 0.42:
                    CR += 1
                else:
                    CH += 1
            else:
                split_label[patient]['hypertrophy'] = [0, 0]
        print(f'{split}: {len(patients)}) NG: {NG} - EH: {EH} - CR: {CR} - CH: {CH}')
    print('==============================================')

## return the histogram of the split list 
for split in ['train', 'val', 'test']:
    print(f'{split}')
    print(np.unique(split_list[split_list == split], return_counts=True))
    print('==============================================')

## 2D scatter plots #####################################################################################################

## only colored scatter plot
print('INFO')
print(rwt_error_list.shape)
print(rst_error_list.shape)
print('Unique vale of cm/px in aspect ratio 240 320')
print(np.unique(cm_px_ar_list).shape)


fig, ax = plt.subplots(figsize=(10, 10), num='RWT vs LVM', tight_layout=True)
color = np.where((hypertrofy_list[:, 0] < 200) & (hypertrofy_list[:, 1] < 0.42), 'green',
                 np.where((hypertrofy_list[:, 0] >= 200) & (hypertrofy_list[:, 1] < 0.42), 'olive',
                          np.where((hypertrofy_list[:, 0] < 200) & (hypertrofy_list[:, 1] >= 0.42), 'orange', 'red')))
ax.scatter(hypertrofy_list[:, 0], hypertrofy_list[:, 1], c=color, marker='o', alpha=0.2)
ax.fill_between([0, 200], 0, 0.42, color='green', alpha=0.3, label='Normal Geometry')
ax.fill_between([200, 1000], 0, 0.42, color='olive', alpha=0.3, label='Eccentric Hypertrophy')
ax.fill_between([0, 200], 0.42, 2, color='orange', alpha=0.3, label='Concentric Remodeling')
ax.fill_between([200, 1000], 0.42, 2, color='red', alpha=0.3, label='Concentric Hypertrophy')

ax.grid(linestyle='--', linewidth=0.5)
ax.set_xlabel('Left Ventricular Mass (LVM)', fontsize=22)
ax.set_ylabel('Relative Wall Thickness (RWT)', fontsize=22)
ax.tick_params(axis='x', labelsize=18)
ax.tick_params(axis='y', labelsize=18)
ax.legend(fontsize=18)

fig, ax = plt.subplots(figsize=(10, 10), num='RWT_ar vs LVM_2', tight_layout=True)
color = np.where((hypertrofy_list[:, 0] < 200) & (hypertrofy_list[:, 1] < 0.42), 'green',
                 np.where((hypertrofy_list[:, 0] >= 200) & (hypertrofy_list[:, 1] < 0.42), 'olive',
                          np.where((hypertrofy_list[:, 0] < 200) & (hypertrofy_list[:, 1] >= 0.42), 'orange', 'red')))
ax.scatter(hypertrofy_list[:, 2], hypertrofy_list[:, 1], c=color, marker='o', alpha=0.2)
# ax.fill_between([0, 200], 0, 0.42, color='green', alpha=0.3, label='Normal Geometry')
# ax.fill_between([200, 1000], 0, 0.42, color='olive', alpha=0.3, label='Eccentric Hypertrophy')
# ax.fill_between([0, 200], 0.42, 2, color='orange', alpha=0.3, label='Concentric Remodeling')
# ax.fill_between([200, 1000], 0.42, 2, color='red', alpha=0.3, label='Concentric Hypertrophy')

ax.grid(linestyle='--', linewidth=0.5)
ax.set_ylabel('Relative Wall Thickness (RWT)', fontsize=22)
ax.set_xlabel('Left Ventricle Mass d  (LVM_d)', fontsize=22)
ax.tick_params(axis='x', labelsize=18)
ax.tick_params(axis='y', labelsize=18)
ax.legend(fontsize=18)

## colored zone with b/w scatter
fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(hypertrofy_list[:, 0], hypertrofy_list[:, 1], c='gray', marker='o', alpha=0.1)
ax.fill_between([0, 200], 0, 0.42, color='green', alpha=0.4)
ax.fill_between([200, 1000], 0, 0.42, color='olive', alpha=0.4)
ax.fill_between([0, 200], 0.42, 2, color='orange', alpha=0.4)
ax.fill_between([200, 1000], 0.42, 2, color='red', alpha=0.4)

ax.grid(linestyle='--', linewidth=0.5)
ax.set_xlabel('Left Ventricular Mass (LVM)', fontsize=22)
ax.set_ylabel('Relative Wall Thickness (RWT)', fontsize=22)
ax.tick_params(axis='x', labelsize=18)
ax.tick_params(axis='y', labelsize=18)

## plot the histogram of the cm/px
fig, ax = plt.subplots(figsize=(8, 8), num='cm/px', tight_layout=True)
stat, p = shapiro(cm_px_list)
print(f'Shapiro-Wilk Test (origin): p-value: {p}')
stat, p = shapiro(cm_px_256_list)
print(f'Shapiro-Wilk Test (256): p-value: {p}')
ax.hist(cm_px_list, bins=25, color='gray', alpha=0.7, label= f'Original res- mean: {np.mean(cm_px_list):.4f} - std: {np.std(cm_px_list):.4f}')
ax.hist(cm_px_256_list, bins=25, color='blue', alpha=0.7, label= f'256 res- mean: {np.mean(cm_px_256_list):.4f} - std: {np.std(cm_px_256_list):.4f}')
ax.hist(cm_px_ar_list, bins=25, color='red', alpha=0.7, label= f'Aspect ratio res- mean: {np.mean(cm_px_ar_list):.4f} - std: {np.std(cm_px_ar_list):.4f}')
ax.set_xlabel('cm/px', fontsize=22)
ax.set_ylabel('Number of patient', fontsize=22)
ax.tick_params(axis='x', labelsize=18)
ax.tick_params(axis='y', labelsize=18)
ax.legend(fontsize=18)

## plot the aspect ratio
fig, ax = plt.subplots(figsize=(10, 10))
ax.hist(aspect_ratio_list, color='red', alpha=0.7, label= f'mean: {np.mean(aspect_ratio_list):.4f} - std: {np.std(aspect_ratio_list):.4f}')
ax.set_xlabel('Aspect ratio', fontsize=22)
ax.set_ylabel('Number of patient', fontsize=22)
ax.tick_params(axis='x', labelsize=18)
ax.tick_params(axis='y', labelsize=18)
ax.legend(fontsize=18)

## plot the error of the RWT
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(20, 7), num='RWT error', tight_layout=True)
ax[0].set_title('Original resolution', fontsize=22)
ax[0].hist(rwt_error_list[:, 0], bins=30, color='gray', alpha=0.7, label= f'mean: {np.mean(rwt_error_list[:, 0]):.4f} - std: {np.std(rwt_error_list[:, 0]):.4f}')
ax[0].set_xlabel('Error', fontsize=22)
ax[0].set_ylabel('Number of patient', fontsize=22)
ax[0].tick_params(axis='x', labelsize=18)
ax[0].tick_params(axis='y', labelsize=18)
ax[0].legend(fontsize=18)

ax[1].set_title('256 resolution', fontsize=22)
ax[1].hist(rwt_error_list[:, 1], bins=30, color='blue', alpha=0.7, label= f'mean: {np.mean(rwt_error_list[:, 1]):.4f} - std: {np.std(rwt_error_list[:, 1]):.4f}')
ax[1].set_xlabel('Error', fontsize=22)
ax[1].set_ylabel('Number of patient', fontsize=22)
ax[1].tick_params(axis='x', labelsize=18)
ax[1].tick_params(axis='y', labelsize=18)
ax[1].legend(fontsize=18)

ax[2].set_title('Aspect ratio resolution', fontsize=22)
ax[2].hist(rwt_error_list[:, 2], bins=30, color='red', alpha=0.7, label= f'mean: {np.mean(rwt_error_list[:, 2]):.4f} - std: {np.std(rwt_error_list[:, 2]):.4f}')
ax[2].set_xlabel('Error', fontsize=22)
ax[2].set_ylabel('Number of patient', fontsize=22)
ax[2].tick_params(axis='x', labelsize=18)
ax[2].tick_params(axis='y', labelsize=18)
ax[2].legend(fontsize=18)

## plot the error of the RST
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(20, 7), num='RST error', tight_layout=True)
ax[0].set_title('Original resolution', fontsize=22)
ax[0].hist(rst_error_list[:, 0], bins=30, color='gray', alpha=0.7, label= f'mean:{np.mean(rst_error_list[:, 0]):.4f}-std:{np.std(rst_error_list[:, 0]):.4f}')
ax[0].set_xlabel('Error', fontsize=22)
ax[0].set_ylabel('Number of patient', fontsize=22)
ax[0].tick_params(axis='x', labelsize=18)
ax[0].tick_params(axis='y', labelsize=18)
ax[0].legend(fontsize=18)

ax[1].set_title('256 resolution', fontsize=22)
ax[1].hist(rst_error_list[:, 1], bins=30, color='blue', alpha=0.7, label= f'mean:{np.mean(rst_error_list[:, 1]):.4f}-std: {np.std(rst_error_list[:, 1]):.4f}')
ax[1].set_xlabel('Error', fontsize=22)
ax[1].set_ylabel('Number of patient', fontsize=22)
ax[1].tick_params(axis='x', labelsize=18)
ax[1].tick_params(axis='y', labelsize=18)
ax[1].legend(fontsize=18)

ax[2].set_title('Aspect ratio resolution', fontsize=22)
ax[2].hist(rst_error_list[:, 2], bins=30, color='red', alpha=0.7, label= f'mean:{np.mean(rst_error_list[:, 2]):.4f}-std:{np.std(rst_error_list[:, 2]):.4f}')
ax[2].set_xlabel('Error', fontsize=22)
ax[2].set_ylabel('Number of patient', fontsize=22)
ax[2].tick_params(axis='x', labelsize=18)
ax[2].tick_params(axis='y', labelsize=18)
ax[2].legend(fontsize=18)


##  3D histogram plot ####################################################################
fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(projection='3d')
x = np.array(hypertrofy_list)[:, 0] # Left Ventricular Mass
y = np.array(hypertrofy_list)[:, 1] # Relative Wall Thickness

hist, xedges, yedges = np.histogram2d(x, y, bins=30)

# Construct arrays for the anchor positions of the 16 bars.
xpos, ypos = np.meshgrid(xedges[:-1] + 0, yedges[:-1] + 0., indexing="ij")
xpos = xpos.ravel()
ypos = ypos.ravel()
zpos = 0

# Construct arrays with the dimensions for the 16 bars.
dy = 0.07 * np.ones_like(zpos)
dx = 25 * np.ones_like(zpos)
dz = hist.ravel()

# Add plane for y = 0.42

ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average', alpha=0.5, color=np.where((xpos < 200) & (ypos < 0.42), 'green',
                                                                         np.where((xpos >= 200) & (ypos < 0.42), 'olive', 
                                                                         np.where((xpos < 200) & (ypos >= 0.42), 'orange', 'red'))))
ax.set_xlabel('\n Left Ventricular Mass', fontsize=18)
ax.set_ylabel('\n Relative Wall Thickness', fontsize=18)
ax.set_zlabel('\n Number of patient', fontsize=18)
ax.tick_params(axis='x', labelsize=15)
ax.tick_params(axis='y', labelsize=15)
ax.tick_params(axis='z', labelsize=15)
####################################################################################################

plt.show()
    