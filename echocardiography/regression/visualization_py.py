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

dataset_dir = "/media/angelo/OS/Users/lasal/OneDrive - Scuola Superiore Sant'Anna/Desktop/Phd notes/Echocardiografy/EchoNet-LVH"
print(os.listdir(dataset_dir))

## ECconet-LVH is composed by 5 folders: 4 'Batch' with video and 'MeasurementsList.csv' with the label
# Read the label
label_dir = os.path.join(dataset_dir, 'MeasurementsList.csv')
label = pd.read_csv(label_dir, index_col=0)

patients = label['HashedFileName'].unique()
rwt_list = []
lv_mass_list = []
relative_distance_list = []
hypertrofy_list = []
cm_px_list, cm_px_256_list, aspect_ratio_list = [], [], []
Save_list = True
if Save_list:
    for patient in tqdm.tqdm(patients):
        patient_label = label[label['HashedFileName'] == patient]        
        ivsd = patient_label[patient_label['Calc'] == 'IVSd']['CalcValue'].values
        pwd = patient_label[patient_label['Calc'] == 'LVPWd']['CalcValue'].values
        lvidd = patient_label[patient_label['Calc'] == 'LVIDd']['CalcValue'].values
        resolution = patient_label[patient_label['Calc'] == 'Resolution']['CalcValue'].values
        
        if len(ivsd) == 0 or len(pwd) == 0 or len(lvidd) == 0:
            pass
            # print(len(ivsd), len(pwd), len(lvidd))
            # print(f'Patient: {patient} has no measurements')
        
        else:
            resolution = (float(patient_label['Width'].values[0]), float(patient_label['Height'].values[0]))
            aspect_ratio = resolution[1] / resolution[0]
            # print(resolution[0], resolution[1])
            ivsd = ivsd[0]
            pwd = pwd[0]
            lvidd = lvidd[0]

            pw_x1 = patient_label[patient_label['Calc'] == 'LVPWd']['X1'].values[0]
            pw_y1 = patient_label[patient_label['Calc'] == 'LVPWd']['Y1'].values[0]
            pw_x2 = patient_label[patient_label['Calc'] == 'LVPWd']['X2'].values[0]
            pw_y2 = patient_label[patient_label['Calc'] == 'LVPWd']['Y2'].values[0]

            pw_x1_256 = (pw_x1 / resolution[0]) * 256
            pw_y1_256 = (pw_y1 / resolution[1]) * 256
            pw_x2_256 = (pw_x2 / resolution[0]) * 256
            pw_y2_256 = (pw_y2 / resolution[1]) * 256

            pw_distance = np.sqrt((pw_x2 - pw_x1) ** 2 + (pw_y2 - pw_y1) ** 2)
            # pw_distance_256 = np.sqrt((pw_x2_256 - pw_x1_256) ** 2 + (pw_y2_256 - pw_y1_256) ** 2)
            pw_distance_256 = 256 * np.sqrt((1/resolution[0]**2)*(pw_x2 - pw_x1) ** 2 + (1 / resolution[1]**2)*(pw_y2 - pw_y1) ** 2) 
            pw_distance_ar = np.sqrt(((256/resolution[0])*(pw_x2 - pw_x1)) ** 2 + ((192/resolution[1])*(pw_y2 - pw_y1)) ** 2) 

            lvid_x1 = patient_label[patient_label['Calc'] == 'LVIDd']['X1'].values[0]
            lvid_y1 = patient_label[patient_label['Calc'] == 'LVIDd']['Y1'].values[0]
            lvid_x2 = patient_label[patient_label['Calc'] == 'LVIDd']['X2'].values[0]
            lvid_y2 = patient_label[patient_label['Calc'] == 'LVIDd']['Y2'].values[0]

            lvid_x1_256 = (lvid_x1 / resolution[0]) * 256
            lvid_y1_256 = (lvid_y1 / resolution[1]) * 256
            lvid_x2_256 = (lvid_x2 / resolution[0]) * 256
            lvid_y2_256 = (lvid_y2 / resolution[1]) * 256

            lvid_distance = np.sqrt((lvid_x2 - lvid_x1) ** 2 + (lvid_y2 - lvid_y1) ** 2)
            # lvid_distance_256 = np.sqrt((lvid_x2_256 - lvid_x1_256) ** 2 + (lvid_y2_256 - lvid_y1_256) ** 2)
            lvid_distance_256 = 256 * np.sqrt((1/resolution[0]**2)*(lvid_x2 - lvid_x1) ** 2 + (1 / resolution[1]**2)*(lvid_y2 - lvid_y1) ** 2)
            lvid_distance_ar = np.sqrt(((256/resolution[0])*(lvid_x2 - lvid_x1)) ** 2 + ((192/resolution[1])*(lvid_y2 - lvid_y1)) ** 2)


            ivsd_x1 = patient_label[patient_label['Calc'] == 'IVSd']['X1'].values[0]
            ivsd_y1 = patient_label[patient_label['Calc'] == 'IVSd']['Y1'].values[0]
            ivsd_x2 = patient_label[patient_label['Calc'] == 'IVSd']['X2'].values[0]
            ivsd_y2 = patient_label[patient_label['Calc'] == 'IVSd']['Y2'].values[0]

            ivsd_x1_256 = (ivsd_x1 / resolution[0]) * 256
            ivsd_y1_256 = (ivsd_y1 / resolution[1]) * 256
            ivsd_x2_256 = (ivsd_x2 / resolution[0]) * 256
            ivsd_y2_256 = (ivsd_y2 / resolution[1]) * 256


            ivsd_distance = np.sqrt((ivsd_x2 - ivsd_x1) ** 2 + (ivsd_y2 - ivsd_y1) ** 2)
            # ivsd_distance_256 = np.sqrt((ivsd_x2_256 - ivsd_x1_256) ** 2 + (ivsd_y2_256 - ivsd_y1_256) ** 2)
            ivsd_distance_256 = 256 * np.sqrt((1/resolution[0]**2)*(ivsd_x2 - ivsd_x1) ** 2 + (1 / resolution[1]**2)*(ivsd_y2 - ivsd_y1) ** 2)
            ivsd_distance_ar = np.sqrt(((256/resolution[0])*(ivsd_x2 - ivsd_x1)) ** 2 + ((192/resolution[1])*(ivsd_y2 - ivsd_y1)) ** 2)

            # print(f'PW: {pw_distance:.4f}')
            # print(f'LVID: {lvid_distance:.4f}')
            # print(f'IVSD: {ivsd_distance:.4f}')
            print(f'cm/pixel (original): {pwd/pw_distance:.4f} - {lvidd/lvid_distance:.4f} - {ivsd/ivsd_distance:.4f}')
            print(f'cm/pixel (256): {pwd/pw_distance_256:.4f} - {lvidd/lvid_distance_256:.4f} - {ivsd/ivsd_distance_256:.4f}')
            print(f'cm/pixel (aspect ratio): {pwd/pw_distance_ar:.4f} - {lvidd/lvid_distance_ar:.4f} - {ivsd/ivsd_distance_ar:.4f}')
            cm_px_list.append([lvidd/lvid_distance])
            cm_px_256_list.append([lvidd/lvid_distance_256])

            relative_distance = 2 * pw_distance / lvid_distance
            
            rwt = 2 * pwd / lvidd
            rwt_256 = 2 * pw_distance_256 / lvid_distance_256
            rwt_ar = 2 * pw_distance_ar / lvid_distance_ar
            lv_mass = 0.8 * (1.04 * ((lvidd + ivsd + pwd) ** 3 - lvidd ** 3)) + 0.6
            print()
            print(f'rwt: {rwt}, lv_mass: {lv_mass}')
            print(f'relative distance: {relative_distance}')
            print(f'rwt_256: {rwt_256}')
            print(f'rwt_ar: {rwt_ar}')
            print(f'pw: {pw_distance}, lvid: {lvid_distance}')
            rwt_list.append(rwt)
            lv_mass_list.append(lv_mass)
            relative_distance_list.append(relative_distance)
            aspect_ratio_list.append(aspect_ratio)
            hypertrofy_list.append([lv_mass, rwt])
        print('==============================================') 


    hypertrofy_list = np.array(hypertrofy_list)
    cm_px_list = np.array(cm_px_list)
    cm_px_256_list = np.array(cm_px_256_list)
    # save this list
    np.save('hypertrofy_list.npy', hypertrofy_list)
    np.save('cm_px_list.npy', cm_px_list)
    np.save('cm_px_256_list.npy', cm_px_256_list)
    np.save('aspect_ratio_list.npy', aspect_ratio_list)

else:
    hypertrofy_list = np.load('hypertrofy_list.npy')    
    cm_px_list = np.load('cm_px_list.npy')
    cm_px_256_list = np.load('cm_px_256_list.npy')
    aspect_ratio_list = np.load('aspect_ratio_list.npy')
    print(np.mean(aspect_ratio_list), np.median(aspect_ratio_list), np.std(aspect_ratio_list))

## 2D scatter plots #####################################################################################################

## only colored scatter plot
fig, ax = plt.subplots(figsize=(10, 10),  tight_layout=True)
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
fig, ax = plt.subplots(figsize=(8, 8))
from scipy.stats import shapiro
stat, p = shapiro(cm_px_list)
print(f'Shapiro-Wilk Test (origin): p-value: {p}')
stat, p = shapiro(cm_px_256_list)
print(f'Shapiro-Wilk Test (256): p-value: {p}')
ax.hist(cm_px_list, bins=25, color='gray', alpha=0.7, label= f'Original res- mean: {np.mean(cm_px_list):.4f} - std: {np.std(cm_px_list):.4f}')
ax.hist(cm_px_256_list, bins=25, color='blue', alpha=0.7, label= f'256 res- mean: {np.mean(cm_px_256_list):.4f} - std: {np.std(cm_px_256_list):.4f}')
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
    