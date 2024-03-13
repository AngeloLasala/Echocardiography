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


from echocardiography.regression.test import get_best_model, show_prediction
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
    def __init__(self, split, size, im_path, spatial_condiction=False, im_ext='png'):
        self.split = split
        self.im_ext = im_ext
        self.size = size
        self.im_path = im_path  ## '\data'
        # self.images, self.labels = self.load_images(im_path)
       
        ## data directory
        self.get_data_directory()
        self.data_dir = os.path.join(self.get_data_directory(),  self.split)

        ## spatial condition 
        # self.train_dir = ## add the file path to be compatible with the function get_model_regression 
        self.spatial_condiction = spatial_condiction


        self.patient_files = [patient_hash.split('.')[0] for patient_hash in os.listdir(os.path.join(self.data_dir))]

    def __len__(self):
        return len(self.patient_files)

    def __getitem__(self, index):
        patient_hash = self.patient_files[index]
        patient_path = os.path.join(self.data_dir, f'{patient_hash}.' + self.im_ext)

        im = Image.open(patient_path)
        im = im.resize(self.size)
        im_tensor = torchvision.transforms.ToTensor()(im)

        ## here the part for the heatmaps
        if self.spatial_condiction:
            model = self.get_model_regression()
            model.eval()
            with torch.no_grad():
                image = im_tensor
                image = transforms.functional.normalize(image, (0.5), (0.5))    
                image = image.unsqueeze(0)
                image = image.to(device)
                output = model(image).to(device)

                # image = image.cpu().numpy().transpose((0, 2, 3, 1))
                # output = output.cpu().numpy()
                
                # for i in range(image.shape[0]):
                #     image = image[i]
                #     output = output[i]
                #     show_prediction(image, output, output, target='heatmaps')
                #     plt.show()

            # Convert input to -1 to 1 range.
            im = im.convert('L')
            im_tensor = (2 * im_tensor) - 1
            return im_tensor, output[0]

        else:
            # Convert input to -1 to 1 range.
            im = im.convert('L')
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

    def get_model_regression(self):
        """
        For each image in the dataset, get the heatmap of regression points
        """
        ## get the current directory
        current_dir = os.getcwd()
        while current_dir.split('/')[-1] != 'echocardiography':
            current_dir = os.path.dirname(current_dir)
        data_dir = os.path.join(current_dir, 'regression')

        train_dir = os.path.join(data_dir, 'TRAINED_MODEL', 'Batch2', 'diastole', 'trial_21') ## change this to the path of the trained model
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
    data = EcoDataset(split='train_eco_train', size=(256,256), im_path='data', spatial_condiction=False)
    # data_loader = DataLoader(data, batch_size=1, shuffle=True, num_workers=4, timeout=10)
    
    print(data[3][0].shape, data[3][1].shape)

