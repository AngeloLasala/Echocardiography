"""
Compute the fid score to evaluate the quality of the generated images.
Modify from (https://github.com/mseitzer/pytorch-fid/tree/master?tab=readme-ov-file)
"""
import os
import argparse
import yaml
import torch
import numpy as np
import pathlib
from PIL import Image
from torchvision import transforms
import torchvision.transforms as TF
from PIL import Image
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d
import matplotlib.pyplot as plt
print('loading inception')
from pytorch_fid.inception import InceptionV3
print('FINISH')
class ImagePathDataset(torch.utils.data.Dataset):
    def __init__(self, files, transforms=None):
        self.files = files
        self.transforms = transforms

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path = self.files[i]
        img = Image.open(path).convert("RGB")
        img = img.resize((299, 299))
        if self.transforms is not None:
            img = self.transforms(img)
        return img


def get_activations(
    files, model, batch_size=50, dims=2048, device="cpu", num_workers=1
):
    """Calculates the activations of the pool_3 layer for all images.

    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : Batch size of images for the model to process at once.
                     Make sure that the number of samples is a multiple of
                     the batch size, otherwise some samples are ignored. This
                     behavior is retained to match the original FID score
                     implementation.
    -- dims        : Dimensionality of features returned by Inception
    -- device      : Device to run calculations
    -- num_workers : Number of parallel dataloader workers

    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """
    model.eval()

    if batch_size > len(files):
        print(
            (
                "Warning: batch size is bigger than the data size. "
                "Setting batch size to data size"
            )
        )
        batch_size = len(files)

    dataset = ImagePathDataset(files, transforms=TF.ToTensor())
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )

    pred_arr = np.empty((len(files), dims))

    start_idx = 0

    for batch in dataloader:
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch)[0]

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        pred = pred.squeeze(3).squeeze(2).cpu().numpy()

        pred_arr[start_idx : start_idx + pred.shape[0]] = pred

        start_idx = start_idx + pred.shape[0]

    return pred_arr


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert (
        mu1.shape == mu2.shape
    ), "Training and test mean vectors have different lengths"
    assert (
        sigma1.shape == sigma2.shape
    ), "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = (
            "fid calculation produces singular product; "
            "adding %s to diagonal of cov estimates"
        ) % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


def calculate_activation_statistics(
    files, model, batch_size=50, dims=2048, device="cpu", num_workers=1
):
    """Calculation of the statistics used by the FID.
    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : The images numpy array is split into batches with
                     batch size batch_size. A reasonable batch size
                     depends on the hardware.
    -- dims        : Dimensionality of features returned by Inception
    -- device      : Device to run calculations
    -- num_workers : Number of parallel dataloader workers

    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the inception model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the inception model.
    """
    act = get_activations(files, model, batch_size, dims, device, num_workers)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def compute_statistics_of_path(path, model, batch_size, dims, device, num_workers=1):
    """
    Compute the statistics of the images in the list of paths
    """
    files = []
    for p in path:
        p_path = pathlib.Path(p)
        file_list = sorted(
            [file for ext in {"bmp", "jpg", "jpeg", "pgm", "png", "ppm", "tif", "tiff", "webp"} for file in p_path.glob("*.{}".format(ext))]
        )
        files += file_list
    m, s = calculate_activation_statistics(
        files, model, batch_size, dims, device, num_workers
    )

    return m, s


def calculate_fid_given_paths(paths, batch_size, device, dims, num_workers=1):
    """
    Calculates the FID of two paths. I want ot generalize this ginev tuo set of paths
    Params:
    paths : list of two list [[real/path/1, real/path/2, ...],[fake/path/1, fake/path/2, ...]]
            paths[0] list of paths of real images
            paths[1] list of paths of fake images
    batch_size  : batch size for the dataloader
    device      : device to run calculations
    dims        : dimensionality of features returned by Inception
    num_workers : number of parallel dataloader workers
    
    """
    ## check the leangth of the paths, must be two
    if len(paths) != 2:
        raise ValueError("Expected two list of paths, got %d" % len(paths))

    for reals in paths[0]:
        if not os.path.exists(reals):
            raise RuntimeError("Invalid path: %s" % reals)

    for fakes in paths[1]:
        if not os.path.exists(fakes):
            raise RuntimeError("Invalid path: %s" % fakes)

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]

    model = InceptionV3([block_idx]).to(device)

    m1, s1 = compute_statistics_of_path(
        paths[0], model, batch_size, dims, device, num_workers
    )
    m2, s2 = compute_statistics_of_path(
        paths[1], model, batch_size, dims, device, num_workers
    )
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)

    return fid_value


def fid_epoch(conf, experiment_dir, epoch, batch_size=50, device=None, dims=2048, num_workers=4):
    """
    FID score of cond LDM model for selected epoch.
    
    Parameters
    ----------
    conf: str
        path to the configuration file

    experiment_dir: str
        path to the experiment directory

    epoch: int
        epoch of the model

    batch_size: int
        batch size for the dataloader
    
    device: str
        device to run calculations

    dims: int
        dimensionality of features returned by Inception

    num_workers: int
        number of parallel dataloader workers

    Returns
    -------
    fid_value: float
        FID score

    """
    with open(conf, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    dataset_config = config['dataset_params']
    
    # Real train images
    # current_dir = os.getcwd()
    # while current_dir.split('/')[-1] != 'echocardiography':
    #     current_dir = os.path.dirname(current_dir)
    # data_dir_regre = os.path.join(current_dir, 'regression')
    # data_real = os.path.join(data_dir_regre, 'DATA', dataset_config['dataset_batch'], dataset_config['split'], dataset_config['phase'], 'image')
    # data_real = []
    for batch in dataset_config['dataset_batch']:
        data_real = os.path.join(dataset_config['parent_dir'], dataset_config['im_path'], batch,
                             dataset_config['split'], dataset_config['phase'], 'image')
        # data_real += data_batch

    # Fake images validation
    data_fake = os.path.join(experiment_dir, f'samples_ep_{epoch}')

    fid_value = calculate_fid_given_paths([data_real, data_fake], batch_size, device, dims, num_workers)

    return fid_value

def fid_experiment(conf, experiment_dir, batch_size=50, device=None, dims=2048, num_workers=4):
    """
    FID score of cond LDM model for all epochs.
    
    Parameters
    ----------
    config: str
        path to the configuration file

    experiment_dir: str
        path to the experiment directory

    Returns
    -------
    fid_values: list
        FID scores for all epochs

    """
    folder_list = [folder for folder in os.listdir(experiment_dir) if folder.startswith("samples_ep")]
    epoch_list = [int(folder.split('_')[-1]) for folder in folder_list]
    epoch_list.sort()

    with open(conf, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    dataset_config = config['dataset_params']
    
    data_real = []
    for batch in dataset_config['dataset_batch']:
        data_batch = os.path.join(dataset_config['parent_dir'], dataset_config['im_path'], batch,
                             dataset_config['split'], dataset_config['phase'], 'image')
        data_real.append(data_batch)

    # Fake images validation
    fid_values = {}
    for epoch in epoch_list:
        data_fake = [os.path.join(experiment_dir, f'samples_ep_{epoch}')]
        fid_value = calculate_fid_given_paths([data_real, data_fake], batch_size, device, dims, num_workers)
        fid_values[epoch] = fid_value

    return fid_values



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute FID score.")
    parser.add_argument('--par_dir', type=str, default='/home/angelo/Documents/Echocardiography/echocardiography/diffusion/trained_model/eco',
                         help="""parent directory of the trained model
                        local: /home/angelo/Documents/Echocardiography/echocardiography/diffusion/trained_model/eco
                        cluster: /leonardo_work/IscrC_Med-LMGM/Angelo/trained_model/diffusion/eco""")
    parser.add_argument('--trial', type=str, default='trial_1', help='trial name for saving the model, it is the trial folde that contain the VAE model')
    parser.add_argument('--experiment', type=str, default='cond_ldm', help="""name of expermient, it is refed to the type of condition and in general to the 
                                                                              hyperparameters (file .yaml) that is used for the training, it can be cond_ldm, cond_ldm_2, """)
    parser.add_argument('--guide_w', type=float, default=0.0, help='guide_w for the conditional model, w=-1 [unconditional], w=0 [vanilla conditioning], w>0 [guided conditional]')
    parser.add_argument('--show_plot', action='store_true', help="show and save the FID plot, default=False")


    # parser.add_argument('--epoch', type=int, default=99, help='epoch to sample, this is the epoch of cond ldm model')
    args = parser.parse_args()
    device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")

    # current_directory = os.path.dirname(__file__)
    # par_dir = os.path.dirname(current_directory)
    # par_dir = os.path.join(par_dir, 'trained_model', 'eco')

    experiment_dir = os.path.join(args.par_dir, args.trial, args.experiment)
    config = os.path.join(experiment_dir, 'config.yaml')
    experiment_dir_w = os.path.join(experiment_dir, f'w_{args.guide_w}')
    
    fid = fid_experiment(config, experiment_dir_w, device=device)
    for key, value in fid.items():
        print(f'Epoch: {key}, FID: {value}')    
    
    ## save the FID score
    with open(os.path.join(experiment_dir, f'w_{args.guide_w}', 'FID_score.txt'), 'w') as f:
        for key, value in fid.items():
            f.write(f'Epoch: {key}, FID: {value}\n')
        
    if args.show_plot:
        fig,ax = plt.subplots(nrows=1, ncols=1, figsize=(8,5), num=f'FID score', tight_layout=True)
        ax.plot(list(fid.keys()), list(fid.values()), marker='o', color='b')
        ax.set_xlabel('Epoch', fontsize=20)
        ax.set_ylabel('FID', fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.grid('dotted')
        plt.savefig(os.path.join(experiment_dir, 'FID_score.png'))
        plt.show()
 

   
    