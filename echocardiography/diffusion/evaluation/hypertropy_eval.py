"""
Evaluate if the generete image are in lime with the given condition
Note that it make sense only for conditioning model
"""
import os
import argparse
import yaml
import numpy as np

import torch
from torchvision import transforms

from echocardiography.diffusion.models.unet_cond_base import get_config_value
from echocardiography.diffusion.dataset.dataset import MnistDataset, EcoDataset, CelebDataset
from torch.utils.data import DataLoader
from echocardiography.regression.utils import echocardiografic_parameters, get_corrdinate_from_heatmap

import matplotlib.pyplot as plt
from PIL import Image


class GenerateDataset(torch.utils.data.Dataset):
    """
    Dataset of generated image loaded from the path
    """
    def __init__(self, trial, experiment, epoch, size, input_channels):
        self.trial = trial
        self.experiment = experiment
        self.epoch = epoch
        self.size = size
        self.input_channels = input_channels

        self.data_dir_label = self.get_eco_path()
        self.files = [os.path.join(self.data_dir_label, f'x0_{i}.png') for i in range(len(os.listdir(self.data_dir_label)))]

    def __len__(self):
        return len(os.listdir(self.data_dir_label))

    def __getitem__(self, idx):
        image_path = self.files[idx]

        # read the image wiht PIL
        image = Image.open(image_path)
        resize = transforms.Resize(size=self.size)
        image = resize(image)
        if self.input_channels == 1: image = image.convert('L')
        image = transforms.functional.to_tensor(image)
        image = (2 * image) - 1 

        return image

    def get_eco_path(self):
        """
        retrive the path 'eco' from current directory
        """
        current_dir = os.getcwd()

        while current_dir.split('/')[-1] != 'echocardiography':
            current_dir = os.path.dirname(current_dir)
        data_dir_diff= os.path.join(current_dir, 'diffusion')
        data_dir_diff_sample= os.path.join(data_dir_diff, 'trained_model', 'eco', self.trial, self.experiment, f'samples_ep_{self.epoch}')
        return data_dir_diff_sample

def get_hypertrophy_class(one_hot_label):
    """
    Get the hypertrophy class from the one hot label
    """
    class_idx = torch.argmax(one_hot_label, dim=1)

    return class_idx

def get_hypertrophy_class_generated(model, generated_images):
    """
    Get the hypertrophy class from the generated images
    """
    model.eval()
    with torch.no_grad():
        generated_images = generated_images.to(device)
        generated_prediction = model(generated_images)
        generated_prediction = generated_prediction.cpu().numpy()

        ## get coordinate from the heatmap
        for jj in range(generated_prediction.shape[0]):
            label = get_corrdinate_from_heatmap(generated_prediction[jj])
            distances = []
            for i in range(3):
                x1, y1 = label[(i*4)], label[(i*4)+1]
                x2, y2 = label[(i*4)+2], label[(i*4)+3]
                distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                distances.append(distance)
            print(distances)
        print()

    return generated_prediction

def main(conf, epoch, show_plot=False):
    """
    Compute the alignment of the generated image with the given condition
    """
    ## read the configuration file
    with open(conf, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    
    diffusion_config = config['diffusion_params']
    dataset_config = config['dataset_params']
    diffusion_model_config = config['ldm_params']
    autoencoder_model_config = config['autoencoder_params']
    train_config = config['train_params']

    ## because i need only the class of hypertrophy, not the true conditioning type
    condition_config = get_config_value(autoencoder_model_config, key='condition_config', default_value=None)
    if condition_config is not None:
        assert 'condition_types' in condition_config, \
            "condition type missing in conditioning config"
        condition_types = condition_config['condition_types']
    print(condition_types)

    
    ## Initialize the dataset, note that cthe class dataset retrive also the regresion model
    ## note that the true label is in this datset for evaluating the hypertrophy
    ## up to date the only evaluation is related to the normalized measure: rwt and rvs
    im_dataset_cls = {
        'mnist': MnistDataset,
        'celebhq': CelebDataset,
        'eco': EcoDataset,
    }.get(dataset_config['name'])    

    ## Load the Validation dataset
    print('dataset', dataset_config['dataset_batch'])
    data_list = []
    for dataset_batch in dataset_config['dataset_batch']:
        data_batch = im_dataset_cls(split=dataset_config['split_val'], size=(dataset_config['im_size_h'], dataset_config['im_size_w']),
                            parent_dir=dataset_config['parent_dir'], im_path=dataset_config['im_path'], dataset_batch=dataset_batch , phase=dataset_config['phase'],
                            condition_config=condition_config)
        data_list.append(data_batch)
    
    data_img = torch.utils.data.ConcatDataset(data_list)
    print('len of the dataset', len(data_img))
    data_loader = DataLoader(data_img, batch_size=train_config['ldm_batch_size']//2, shuffle=False, num_workers=8)

    ## load the regression model
    # regression_model = data_img.get_model_regression().to(device)
    # regression_model.eval()
    # print(data_img.size)


    # ## load the generated image from path
    # data_gen = GenerateDataset(trial=args.trial, experiment=args.experiment, epoch=epoch,
    #                            size=(dataset_config['im_size'], dataset_config['im_size']), input_channels=dataset_config['im_channels'])
    # data_loader_gen = DataLoader(data_gen, batch_size=train_config['ldm_batch_size']//2, shuffle=False, num_workers=8)

    # for i in data_gen:
    #     print(i.shape)
        
    # for data, gen_data in zip(data_loader, data_loader_gen):
    #     cond_input = None
    #     if condition_config is not None:
    #         im, cond_input = data  # im is the image (batch_size=8), cond_input is the conditional input ['image for the mask']
    #         for key in cond_input.keys(): ## for all the type of condition, we move the  tensor on the device
    #             cond_input[key] = cond_input[key].to(device)
    #     else:
    #         im = data


    #     print(get_hypertrophy_class(cond_input['class']))
    #     real_labels = get_hypertrophy_class(cond_input['class'])

    #     ## predict the label for the generated image
    #     get_hypertrophy_class_generated(regression_model, gen_data)
    #     print()


    #     # plot the real and generate image
    #     if show_plot:
    #         for ii in range(train_config['ldm_batch_size']//2):
    #             fig, ax = plt.subplots(1, 2, figsize=(12, 7), tight_layout=True)
    #             #set the title
    #             ax[0].set_title(f'Real image - class {real_labels[ii]}', fontsize=20)
    #             ax[0].imshow(im[ii].squeeze().cpu().detach().numpy(), cmap='gray')
    #             ax[1].imshow(gen_data[ii].squeeze().cpu().detach().numpy(), cmap='gray')
    #             for aaa in ax:
    #                 aaa.axis('off')
    #         plt.show()


        
        
    


   
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compute hypertrophy loss score.")
    parser.add_argument('--par_dir', type=str, default='/home/angelo/Documents/Echocardiography/echocardiography/diffusion/trained_model/eco',
                         help="""parent directory of the trained model
                        local: /home/angelo/Documents/Echocardiography/echocardiography/diffusion/trained_model/eco
                        cluster: /leonardo_work/IscrC_Med-LMGM/Angelo/trained_model/diffusion/eco""")
    parser.add_argument('--trial', type=str, default='trial_1', help='trial name for saving the model, it is the trial folde that contain the VAE model')
    parser.add_argument('--experiment', type=str, default='cond_ldm', help="""name of expermient, it is refed to the type of condition and in general to the 
                                                                              hyperparameters (file .yaml) that is used for the training, it can be cond_ldm, cond_ldm_2, """)
    parser.add_argument('--guide_w', type=float, default=0.0, help='guide_w for the conditional model, w=-1 [unconditional], w=0 [vanilla conditioning], w>0 [guided conditional]')
    
    # parser.add_argument('--epoch', type=int, default=99, help='epoch to sample, this is the epoch of cond ldm model')
    args = parser.parse_args()
    device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")

    current_directory = os.path.dirname(__file__)
    par_dir = os.path.dirname(current_directory)
    par_dir = os.path.join(par_dir, 'trained_model', 'eco')

    experiment_dir = os.path.join(args.par_dir, args.trial, args.experiment)
    config = os.path.join(experiment_dir, 'config.yaml')
    
    epoch = 100
    main(config, epoch=epoch)