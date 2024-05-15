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
Save_list = False
if Save_list:
    for patient in tqdm.tqdm(patients):
        patient_label = label[label['HashedFileName'] == patient]
        ivsd = patient_label[patient_label['Calc'] == 'IVSd']['CalcValue'].values
        pwd = patient_label[patient_label['Calc'] == 'LVPWd']['CalcValue'].values
        lvidd = patient_label[patient_label['Calc'] == 'LVIDd']['CalcValue'].values
        
        if len(ivsd) == 0 or len(pwd) == 0 or len(lvidd) == 0:
            pass
            # print(len(ivsd), len(pwd), len(lvidd))
            # print(f'Patient: {patient} has no measurements')
        
        else:
            ivsd = ivsd[0]
            pwd = pwd[0]
            lvidd = lvidd[0]

            pw_x1 = patient_label[patient_label['Calc'] == 'LVPWd']['X1'].values[0]
            pw_y1 = patient_label[patient_label['Calc'] == 'LVPWd']['Y1'].values[0]
            pw_x2 = patient_label[patient_label['Calc'] == 'LVPWd']['X2'].values[0]
            pw_y2 = patient_label[patient_label['Calc'] == 'LVPWd']['Y2'].values[0]

            pw_distance = np.sqrt((pw_x2 - pw_x1) ** 2 + (pw_y2 - pw_y1) ** 2)

            lvid_x1 = patient_label[patient_label['Calc'] == 'LVIDd']['X1'].values[0]
            lvid_y1 = patient_label[patient_label['Calc'] == 'LVIDd']['Y1'].values[0]
            lvid_x2 = patient_label[patient_label['Calc'] == 'LVIDd']['X2'].values[0]
            lvid_y2 = patient_label[patient_label['Calc'] == 'LVIDd']['Y2'].values[0]

            lvid_distance = np.sqrt((lvid_x2 - lvid_x1) ** 2 + (lvid_y2 - lvid_y1) ** 2)
            relative_distance = 2 * pw_distance / lvid_distance
            
            rwt = 2 * pwd / lvidd
            lv_mass = 0.8 * (1.04 * ((lvidd + ivsd + pwd) ** 3 - lvidd ** 3)) + 0.6
            # print(f'rwt: {rwt}, lv_mass: {lv_mass}')
            # print(f'relative distance: {relative_distance}')
            # print(f'pw: {pw_distance}, lvid: {lvid_distance}')
            rwt_list.append(rwt)
            lv_mass_list.append(lv_mass)
            relative_distance_list.append(relative_distance)
            hypertrofy_list.append([lv_mass, rwt])
        # print('==============================================') 


    hypertrofy_list = np.array(hypertrofy_list)
    # save this list
    np.save('hypertrofy_list.npy', hypertrofy_list)

else:
    hypertrofy_list = np.load('hypertrofy_list.npy')    

## 2D scatter plots #####################################################################################################

## only colored scatter plot
fig, ax = plt.subplots(figsize=(10, 10))
color = np.where((hypertrofy_list[:, 0] < 200) & (hypertrofy_list[:, 1] < 0.42), 'green',
                 np.where((hypertrofy_list[:, 0] >= 200) & (hypertrofy_list[:, 1] < 0.42), 'olive',
                          np.where((hypertrofy_list[:, 0] < 200) & (hypertrofy_list[:, 1] >= 0.42), 'orange', 'red')))
ax.scatter(hypertrofy_list[:, 0], hypertrofy_list[:, 1], c=color, marker='o', alpha=0.2)
ax.fill_between([0, 200], 0, 0.42, color='green', alpha=0.3)
ax.fill_between([200, 1000], 0, 0.42, color='olive', alpha=0.3)
ax.fill_between([0, 200], 0.42, 2, color='orange', alpha=0.3)
ax.fill_between([200, 1000], 0.42, 2, color='red', alpha=0.3)

ax.grid(linestyle='--', linewidth=0.5)
ax.set_xlabel('Left Ventricular Mass', fontsize=18)
ax.set_ylabel('Relative Wall Thickness', fontsize=18)
ax.tick_params(axis='x', labelsize=15)
ax.tick_params(axis='y', labelsize=15)

## colored zone with b/w scatter
fig, ax = plt.subplots(figsize=(10, 10))
ax.scatter(hypertrofy_list[:, 0], hypertrofy_list[:, 1], c='gray', marker='o', alpha=0.1)
ax.fill_between([0, 200], 0, 0.42, color='green', alpha=0.4)
ax.fill_between([200, 1000], 0, 0.42, color='olive', alpha=0.4)
ax.fill_between([0, 200], 0.42, 2, color='orange', alpha=0.4)
ax.fill_between([200, 1000], 0.42, 2, color='red', alpha=0.4)

ax.grid(linestyle='--', linewidth=0.5)
ax.set_xlabel('Left Ventricular Mass', fontsize=18)
ax.set_ylabel('Relative Wall Thickness', fontsize=18)
ax.tick_params(axis='x', labelsize=15)
ax.tick_params(axis='y', labelsize=15)



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
    