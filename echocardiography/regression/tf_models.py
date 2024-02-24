"""
convert and performe the same toy eample for in tensorflow
"""
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import os
from PIL import Image
import pandas as pd
import json
from scipy.stats import multivariate_normal
from scipy import ndimage
import cv2
import math
import matplotlib.pyplot as plt

def get_heatmap(image, labels):
    """
    given a index of the patient return the 6D heatmap of the keypoints
    """
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

def get_image_label(sample_path):
        """
        from index return the image and the label 
        the labels are the normalized coordinates of the keypoints
        """
        image_path = sample_path
        # patient = tf.strings.split(tf.strings.split(sample_path, '/')[-1], '.')[0]
        # label_path = tf.strings.join([
        #                             tf.strings.join([tf.strings.split(image_path, 'image')[0], 'label']),
        #                             tf.strings.split(tf.strings.split(image_path, '/')[-1], '.')[0],
        #                         ], '/')


        # with open(os.path.join(label_path, 'label.json'), 'r') as f:
        #     label = json.load(f)

        # read the image wiht PIL
        image = Image.open(sample_path) 

        # read the label dict
        patient_name = patient.split('/')[-1].split('.')[0]
        patient_label =label[patient_name]

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

        label = get_heatmap(image, keypoints_label)

        image, label = np.array(image), np.array(label)
        # image, label  = tf.cast(image, tf.float32), tf.cast(label, tf.float32)

        return image, label

def load_sample_train(sample_path):
    """
    Load and preproces train_file
    Parameters
    ----------
    image_file : string
        image's path
    Returns
    -------
    image : tensorflow tensor
        preprocessed CAM 
    real_image : tensorflow tensor
        preprocessed US image
    """
    image, label = get_image_label(sample_path)
    image, label = random_jitter(image, label)
    image, label= normalize(image, label)

    return image, label

def normalize(image, label):
    """
    Normalize the image and the label
    """
    # image = tf.cast(image, tf.float32)
    # label = tf.cast(label, tf.float32)

    image = (image / 127.5) - 1.
    # label = (label / 127.5) - 1

    return image, label

@tf.function()
def random_jitter(image, label, rot=False):
	"""
	Complete image preprocessing for Segmentation

	Parameters
	----------
	image : tensorflow tensor
		input imgage, i.e. CAM 
	label : tensorflow tensor
		real image, i.e. US image
	
	"""
	# resize
	image, label = tf.image.resize(image, [256, 256]), tf.image.resize(label, [256, 256])

    ## add data augmentation
										
	return image, label
    
def UNet(num_classes):
    # Encoder
    inputs = tf.keras.Input(shape=(None, None, 3))  # Assumes 3 channels for input images

    conv1 = layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = layers.BatchNormalization(epsilon=1e-05, momentum=0.1)(conv1)
    conv1 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv1)
    conv1 = layers.BatchNormalization()(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = layers.Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = layers.BatchNormalization()(conv2)
    conv2 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv2)
    conv2 = layers.BatchNormalization()(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = layers.Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = layers.BatchNormalization()(conv3)
    conv3 = layers.Conv2D(256, 3, activation='relu', padding='same')(conv3)
    conv3 = layers.BatchNormalization()(conv3)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = layers.Conv2D(512, 3, activation='relu', padding='same')(pool3)
    conv4 = layers.BatchNormalization()(conv4)
    conv4 = layers.Conv2D(512, 3, activation='relu', padding='same')(conv4)
    conv4 = layers.BatchNormalization()(conv4)
    pool4 = layers.MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = layers.Conv2D(1024, 3, activation='relu', padding='same')(pool4)
    conv5 = layers.BatchNormalization()(conv5)
    conv5 = layers.Conv2D(1024, 3, activation='relu', padding='same')(conv5)
    conv5 = layers.BatchNormalization()(conv5)

    # Decoder
    up6 = layers.UpSampling2D(size=(2, 2))(conv5)
    up6 = layers.Conv2D(512, 2, activation='relu', padding='same')(up6)
    merge6 = layers.concatenate([conv4, up6], axis=3)
    conv6 = layers.Conv2D(512, 3, activation='relu', padding='same')(merge6)
    conv6 = layers.BatchNormalization()(conv6)
    conv6 = layers.Conv2D(512, 3, activation='relu', padding='same')(conv6)
    conv6 = layers.BatchNormalization()(conv6)

    up7 = layers.UpSampling2D(size=(2, 2))(conv6)
    up7 = layers.Conv2D(256, 2, activation='relu', padding='same')(up7)
    merge7 = layers.concatenate([conv3, up7], axis=3)
    conv7 = layers.Conv2D(256, 3, activation='relu', padding='same')(merge7)
    conv7 = layers.BatchNormalization()(conv7)
    conv7 = layers.Conv2D(256, 3, activation='relu', padding='same')(conv7)
    conv7 = layers.BatchNormalization()(conv7)

    up8 = layers.UpSampling2D(size=(2, 2))(conv7)
    up8 = layers.Conv2D(128, 2, activation='relu', padding='same')(up8)
    merge8 = layers.concatenate([conv2, up8], axis=3)
    conv8 = layers.Conv2D(128, 3, activation='relu', padding='same')(merge8)
    conv8 = layers.BatchNormalization()(conv8)
    conv8 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv8)
    conv8 = layers.BatchNormalization()(conv8)

    up9 = layers.UpSampling2D(size=(2, 2))(conv8)
    up9 = layers.Conv2D(64, 2, activation='relu', padding='same')(up9)
    merge9 = layers.concatenate([conv1, up9], axis=3)
    conv9 = layers.Conv2D(64, 3, activation='relu', padding='same')(merge9)
    conv9 = layers.BatchNormalization()(conv9)
    conv9 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv9)
    conv9 = layers.BatchNormalization()(conv9)

    # Output layer
    outputs = layers.Conv2D(num_classes, 1, activation='sigmoid')(conv9)

    model = models.Model(inputs=inputs, outputs=outputs)

    return model

if __name__ == "__main__":

    model = UNet(num_classes=6)
    print(model.summary())
    ## toy image of 224x224
    input_img = np.random.rand(224, 224, 3)
    input_img = np.expand_dims(input_img, axis=0)
    print(input_img.shape)
    prediction = model.predict(input_img)
    print(prediction.shape)

    

    plt.figure()
    plt.imshow(input_img[0])

    plt.figure()
    plt.imshow(prediction[0, :, :, 0])
    plt.show()


    # ##Dataset
    # train_dataset = tf.data.Dataset.list_files(train_list, shuffle=True)
    # train_dataset = train_dataset.map(load_sample_train, num_parallel_calls=tf.data.AUTOTUNE)

    # model = UNet(num_classes=6)
    # print(model.summary())