import glob
import os

import torchvision
from PIL import Image
from tqdm import tqdm
import json
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import numpy as np
import math
from scipy.stats import multivariate_normal
import cv2


from echocardiography.regression.test import get_best_model, show_prediction
from echocardiography.regression.utils import echocardiografic_parameters
from echocardiography.regression.cfg import train_config

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class MnistDataset(Dataset):
    r"""
    Nothing special here. Just a simple dataset class for mnist images.
    Created a dataset class rather using torchvision to allow
    replacement with any other image dataset
    """
    def __init__(self, split, size, im_path, im_ext='png'):
        r"""
        Init method for initializing the dataset properties
        :param split: train/test to locate the image files
        :param im_path: root folder of images
        :param im_ext: image extension. assumes all
        images would be this type.
        """
        self.split = split
        self.size = size
        self.im_ext = im_ext
        self.im_path = im_path
        self.images, self.labels = self.load_images(im_path)

    def load_images(self, im_path):
        """
        Gets all images from the path specified
        and stacks them all up
        :param im_path:
        :return:
        """
        im_path = os.path.join(self.get_data_directory())
        assert os.path.exists(im_path), f"images path {im_path} does not exist"
        ims = []
        labels = []
        for d_name in tqdm(os.listdir(im_path)):
            for fname in glob.glob(os.path.join(im_path, d_name, '*.{}'.format(self.im_ext))):
                ims.append(fname)
                labels.append(int(d_name))
        print('Found {} images for split {}'.format(len(ims), self.split))
        return ims, labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        im = Image.open(self.images[index])
        im = im.resize(self.size)
        im_tensor = torchvision.transforms.ToTensor()(im)

        # Convert input to -1 to 1 range.
        im_tensor = (2 * im_tensor) - 1
        return im_tensor

    def get_data_directory(self):
        """
        Return the data directory from the current directory
        """
        current_dir = os.getcwd()
        while current_dir.split('/')[-1] != 'diffusion':
            current_dir = os.path.dirname(current_dir)
        data_dir = os.path.join(current_dir, self.im_path)
        return data_dir

class EcoDataset():
    def __init__(self, split, size, im_path, dataset_batch, phase,  
                dataset_batch_regression=None, trial=None, condition_config=None, im_ext='png'):
        self.split = split
        self.im_ext = im_ext
        self.size = size
        self.im_path = im_path  ## '\data'

        # self.images, self.labels = self.load_images(im_path)
        self.dataset_batch = dataset_batch ## 'Batch_n' number of batch in Echonet-LVH, task generation
        self.phase = phase                 ## cardiac phase: 'diastole' or 'systole'
        
        ## regression part
        self.trial = trial     ## trial number for the trained model for regression
        self.dataset_batch_regression = dataset_batch_regression ## 'Batch_n' number of batch in Echonet-LVH, task regression

        ## data directory
        self.get_data_directory()
        # self.data_dir = os.path.join(self.get_data_directory(),  self.split) # this is for the 'data' in 'diffusion/data'
        self.data_dir, self.data_dir_label = self.get_data_directory() # this is for the 'data' in 'echocardiography/regression/DATA/Batch_n/split/phase/image'
        with open(os.path.join(self.data_dir_label, 'label.json'), 'r') as f:
            self.data_label = json.load(f)
        
        self.condition_types = [] if condition_config is None else condition_config['condition_types']
        ## spatial condition 
        # self.train_dir = ## add the file path to be compatible with the function get_model_regression 


        self.patient_files = [patient_hash.split('.')[0] for patient_hash in os.listdir(os.path.join(self.data_dir))]

    def __len__(self):
        return len(self.patient_files)

    def __getitem__(self, index):
        im, keypoints_label, calc_value_list = self.get_image_label(index)
        # print(im.size, keypoints_label, calc_value_list)
    
        cond_inputs = {}    ## add to this dict the condition inputs
        if len(self.condition_types) > 0:  # check if there is at least one condition in ['class', 'text', 'image']
            
            ################ IMAGE CONDITION ###################################
            if 'image' in self.condition_types:
                ## NEW VERSION. get the heatmaps from the label
                # this is equal to how i load the data for the regression without the data augumentation
                heatmaps_label = self.get_heatmap(index)
                im_tensor, heatmaps_label = self.trasform(im, heatmaps_label)
                calc_value_list = torch.tensor(calc_value_list)
                cond_inputs['image'] = heatmaps_label

                ## OLD VERSION. compute the heatmap for the regrassion network
                ## load model and compute the heatmap
                # model = self.get_model_regression()
                # model.eval()
                # with torch.no_grad():
                #     image = torchvision.transforms.ToTensor()(im)
                #     image = transforms.functional.normalize(image, (0.5), (0.5))    
                #     image = image.unsqueeze(0)
                #     image = image.to(device)
                #     output = model(image).to(device)

                # # Convert input to -1 to 1 range.
                # im = im.convert('L')
                # im_tensor = torchvision.transforms.ToTensor()(im)
                # im_tensor = (2 * im_tensor) - 1
                # cond_inputs['image'] = output[0]
            #####################################################################

            ###### CLASS CONDITION ##############################################
            if 'class' in self.condition_types:
                # process the image
                im = im.convert('L')
                im_tensor = torchvision.transforms.ToTensor()(im)
                im_tensor = (2 * im_tensor) - 1

                class_label = self.get_class_hypertrophy(keypoints_label, calc_value_list)
                cond_inputs['class'] = class_label
            #####################################################################
            
            return im_tensor, cond_inputs   

        else: # no condition
            im = im.resize(self.size)
            im = im.convert('L')
            im_tensor = torchvision.transforms.ToTensor()(im)
            im_tensor = (2 * im_tensor) - 1
            return im_tensor

    def get_image_label(self, index):
        patient_hash = self.patient_files[index]
        patient_path = os.path.join(self.data_dir, f'{patient_hash}.' + self.im_ext)

        im = Image.open(patient_path)
        lab = self.data_label[patient_hash]

        ## da questi ricava i valori normalizzati alla self.size mentre il calc no!!
        keypoints_label, calc_value_list = [], []
        for heart_part in ['LVPWd', 'LVIDd', 'IVSd']:
            x1_heart_part = lab[heart_part]['x1'] / im.size[0]
            y1_heart_part = lab[heart_part]['y1'] / im.size[1]
            x2_heart_part = lab[heart_part]['x2'] / im.size[0]
            y2_heart_part = lab[heart_part]['y2'] / im.size[1]
            heart_part_value = lab[heart_part]['calc_value']
            keypoints_label.append([x1_heart_part, y1_heart_part, x2_heart_part, y2_heart_part])
            calc_value_list.append(heart_part_value)

        keypoints_label = np.array(keypoints_label).flatten()
        calc_value_list = np.array(calc_value_list).flatten()

        return im, keypoints_label, calc_value_list

    def get_class_hypertrophy(self, keypoints_label, calc_value_list):
        """
        get the class label of each patient
        """
        rwt, rst = echocardiografic_parameters(keypoints_label)
        ivsd = calc_value_list[2]
        pwd = calc_value_list[0]
        lvidd = calc_value_list[1]

        rwt = 2 * pwd / lvidd
        lv_mass = 0.8 * (1.04 * ((lvidd + ivsd + pwd) ** 3 - lvidd ** 3)) + 0.6
        
        if rwt > 0.42 and lv_mass > 200:  ## Concentric hypertrophy
            class_label = 0
        if rwt > 0.42 and lv_mass < 200:  ## Concentric remodeling
            class_label = 1
        if rwt < 0.42 and lv_mass > 200:  ## Eccentric hypertrophy
            class_label = 2 
        if rwt < 0.42 and lv_mass < 200:  ## Normal geometry
            class_label = 3

        # convert the class label to one hot encoding with torch
        class_label = torch.nn.functional.one_hot(torch.tensor(class_label), num_classes=4)
        return class_label


    def get_heatmap(self, idx):
        """
        given a index of the patient return the 6D heatmap of the keypoints
        """
        image, labels, calc_value = self.get_image_label(idx)

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
        image = image.convert('L')
        label = [Image.fromarray(label[:,:,ch]) for ch in range(label.shape[2])]

        ## Resize
        resize = transforms.Resize(size=self.size)
        image = resize(image)
        label = [resize(ch) for ch in label]

        ## convert to tensor and normalize
        label = np.array([np.array(ch) for ch in label])
        label = torch.tensor(label)

        image = transforms.functional.to_tensor(image)
        image = (2 * image) - 1    
        return image, label

    def get_data_directory(self):
        """
        Return the data directory from the current directory
        """
        current_dir = os.getcwd()

        ## this part is for for the image in 'diffusion/data'
        # while current_dir.split('/')[-1] != 'diffusion':
        #     current_dir = os.path.dirname(current_dir)
        # data_dir = os.path.join(current_dir, self.im_path) ## this is right for the image in 'diffusion/data' when i copy and paste from the '/regression'
        
        ## this part is for the image in 'echocardiography/regression/DATA/Batch_n/split/phase/image'
        while current_dir.split('/')[-1] != 'echocardiography':
            current_dir = os.path.dirname(current_dir)
        data_dir_regre = os.path.join(current_dir, 'regression')
        data_dir_regre_img = os.path.join(data_dir_regre, self.im_path, self.dataset_batch, self.split, self.phase, 'image') ## change this to the path of the trained model
        data_dir_regre_lab = os.path.join(data_dir_regre, self.im_path, self.dataset_batch, self.split, self.phase, 'label')
        return data_dir_regre_img, data_dir_regre_lab

    def get_model_regression(self):
        """
        For each image in the dataset, get the heatmap of regression points
        """
        # Possible error
        if self.trial is None:
            raise ValueError('The trial value is None, please provide a trial value to get the model, i.e. trial_1')

        if self.dataset_batch_regression is None:
            raise ValueError('The dataset_batch_regression value is None, please provide a dataset_batch_regression value to get the model, i.e. Batch_1')

        if self.dataset_batch_regression == self.dataset_batch:
            raise ValueError('The dataset_batch_regression value is equal to the dataset_batch value, please provide a different value for dataset_batch_regression')

        ## get the current directory
        current_dir = os.getcwd()
        while current_dir.split('/')[-1] != 'echocardiography':
            current_dir = os.path.dirname(current_dir)
        data_dir = os.path.join(current_dir, 'regression')

        train_dir = os.path.join(data_dir, 'TRAINED_MODEL', self.dataset_batch_regression, self.phase, self.trial) ## change this to the path of the trained model
        with open(os.path.join(train_dir, 'args.json')) as json_file:
            trained_args = json.load(json_file)
        cfg = train_config(trained_args['target'],
                        threshold_wloss=trained_args['threshold_wloss'],
                        model=trained_args['model'],
                        device=device)

        best_model = get_best_model(train_dir)
        model = cfg['model'].to(device)
        model.load_state_dict(torch.load(os.path.join(train_dir, f'model_{best_model}')))
        model.to(device)
        return model

class CelebDataset(Dataset):
    """
    Celeb dataset will by default centre crop and resize the images.
    This can be replaced by any other dataset. As long as all the images
    are under one directory.
    """

    def __init__(self, split, im_path, size=256, im_channels=3, im_ext='jpg',
                 use_latents=False, latent_path=None, condition_config=None):
        self.split = split
        self.size = size
        self.im_channels = im_channels
        self.im_ext = im_ext
        self.im_path = im_path
        self.latent_maps = None
        self.use_latents = False

        self.condition_types = [] if condition_config is None else condition_config['condition_types']

        self.idx_to_cls_map = {}
        self.cls_to_idx_map ={}

        if 'image' in self.condition_types:
            self.mask_channels = condition_config['image_condition_config']['image_condition_input_channels']
            self.mask_h = condition_config['image_condition_config']['image_condition_h']
            self.mask_w = condition_config['image_condition_config']['image_condition_w']

        self.images, self.texts, self.masks = self.load_images(im_path)

        # Whether to load images or to load latents
        if use_latents and latent_path is not None:
            latent_maps = load_latents(latent_path)
            if len(latent_maps) == len(self.images):
                self.use_latents = True
                self.latent_maps = latent_maps
                print('Found {} latents'.format(len(self.latent_maps)))
            else:
                print('Latents not found')

    def load_images(self, im_path):
        r"""
        Gets all images from the path specified
        and stacks them all up
        """
        im_path = os.path.join(self.get_data_directory())
        assert os.path.exists(im_path), "images path {} does not exist".format(im_path)
        ims = []
        fnames = glob.glob(os.path.join(im_path, 'CelebA-HQ-img/*.{}'.format('png')))
        fnames += glob.glob(os.path.join(im_path, 'CelebA-HQ-img/*.{}'.format('jpg')))
        fnames += glob.glob(os.path.join(im_path, 'CelebA-HQ-img/*.{}'.format('jpeg')))
        texts = []
        masks = []

        if 'image' in self.condition_types:
            label_list = ['skin', 'nose', 'eye_g', 'l_eye', 'r_eye', 'l_brow', 'r_brow', 'l_ear', 'r_ear', 'mouth',
                          'u_lip', 'l_lip', 'hair', 'hat', 'ear_r', 'neck_l', 'neck', 'cloth']
            self.idx_to_cls_map = {idx: label_list[idx] for idx in range(len(label_list))}
            self.cls_to_idx_map = {label_list[idx]: idx for idx in range(len(label_list))}

        for fname in tqdm(fnames):
            ims.append(fname)

            if 'text' in self.condition_types:
                im_name = os.path.split(fname)[1].split('.')[0]
                captions_im = []
                with open(os.path.join(im_path, 'celeba-caption/{}.txt'.format(im_name))) as f:
                    for line in f.readlines():
                        captions_im.append(line.strip())
                texts.append(captions_im)

            if 'image' in self.condition_types:
                im_name = int(os.path.split(fname)[1].split('.')[0])
                masks.append(os.path.join(im_path, 'CelebAMask-HQ-mask', '{}.png'.format(im_name)))
        if 'text' in self.condition_types:
            assert len(texts) == len(ims), "Condition Type Text but could not find captions for all images"
        if 'image' in self.condition_types:
            assert len(masks) == len(ims), "Condition Type Image but could not find masks for all images"
        print('Found {} images'.format(len(ims)))
        print('Found {} masks'.format(len(masks)))
        print('Found {} captions'.format(len(texts)))
        return ims, texts, masks

    def get_mask(self, index):
        r"""
        Method to get the mask of WxH
        for given index and convert it into
        Classes x W x H mask image
        :param index:
        :return:
        """
        mask_im = Image.open(self.masks[index])
        mask_im = np.array(mask_im)
        im_base = np.zeros((self.mask_h, self.mask_w, self.mask_channels))
        for orig_idx in range(len(self.idx_to_cls_map)):
            im_base[mask_im == (orig_idx+1), orig_idx] = 1
        mask = torch.from_numpy(im_base).permute(2, 0, 1).float()
        return mask

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        ######## Set Conditioning Info ########
        cond_inputs = {}
        if 'text' in self.condition_types:
            cond_inputs['text'] = random.sample(self.texts[index], k=1)[0]
        if 'image' in self.condition_types:
            mask = self.get_mask(index)
            cond_inputs['image'] = mask
        #######################################

        if self.use_latents:
            latent = self.latent_maps[self.images[index]]
            if len(self.condition_types) == 0:
                return latent
            else:
                return latent, cond_inputs
        else:
            im = Image.open(self.images[index])
            im_tensor = torchvision.transforms.Compose([
                torchvision.transforms.Resize(self.size),
                torchvision.transforms.CenterCrop(self.size),
                torchvision.transforms.ToTensor(),
            ])(im)
            im.close()

            # Convert input to -1 to 1 range.
            im_tensor = (2 * im_tensor) - 1
            if len(self.condition_types) == 0:
                return im_tensor
            else:
                return im_tensor, cond_inputs

    def get_data_directory(self):
        """
        Return the data directory from the current directory
        """
        current_dir = os.getcwd()
        while current_dir.split('/')[-1] != 'diffusion':
            current_dir = os.path.dirname(current_dir)
        data_dir = os.path.join(current_dir, self.im_path)
        return data_dir


if __name__ == '__main__':
    import yaml
    
    conf = '/home/angelo/Documents/Echocardiography/echocardiography/diffusion/conf/eco_image_cond.yaml'
    with open(conf, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    data = EcoDataset(split='train', size=(256,256), im_path='DATA', dataset_batch='Batch3', phase='diastole', 
                      dataset_batch_regression='Batch2', trial='trial_2', condition_config=config['ldm_params']['condition_config']) #, condition_config=False)
    # data_loader = DataLoader(data, batch_size=1, shuffle=True, num_workers=4, timeout=10)
    # print(data.condition_types)
    # print(data[1][0].shape, data[1][1]['image'].shape)
    # print(data.patient_files[1])
    # print(data.data_dir, data.data_dir_label)
    print()
    print(data[13])
    