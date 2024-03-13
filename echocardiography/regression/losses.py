import torch
import torch.nn as nn
import os
import argparse
from torchvision import transforms
import matplotlib.pyplot as plt

from echocardiography.regression.dataset import EchoNetDataset, convert_to_serializable
from echocardiography.regression.utils import get_corrdinate_from_heatmap, get_corrdinate_from_heatmap_torch
## define weighted MSE loss
class WeightedMSELoss(nn.Module):
    def __init__(self, threshold, device):
        super(WeightedMSELoss, self).__init__()
        self.device = device
        self.threshold = threshold

    def forward(self, output, label):
        weight = torch.ones(label.shape).to(self.device)
        ## binary the targert with 0.5
        mask = (label >= self.threshold).float().to(self.device)
        ## give me the total numer of 0 and 1
        num_0 = torch.sum(mask == 0).float().to(self.device)
        num_1 = torch.sum(mask == 1).float().to(self.device)
        ## give me the frequency of 0 and 1
        freq_0 = num_0 / (num_0 + num_1).to(self.device)
        freq_1 = num_1 / (num_0 + num_1).to(self.device)

        # create a weight tensor substituting 1 with 1/num_1 and 0 with 1/num_0
        weight = torch.where(mask == 1, 1/freq_1, weight).to(self.device)
        weight = torch.where(mask == 0, 1/freq_0, weight).to(self.device)
        # print(f'Weight: {weight.min()} - {weight.max()}')
        return torch.mean(weight * (label - output) ** 2)

class RMSELoss(torch.nn.Module):
    """
    Root Mean Square Error Loss
    """
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = torch.nn.MSELoss()
        self.eps = eps
        
    def forward(self,label,target):
        loss = torch.sqrt(self.mse(label,target) + self.eps)
        return loss

class WeightedRMSELoss(torch.nn.Module):
    """
    Weighted Root Mean Square Error Loss
    """
    def __init__(self, threshold, device, eps=1e-6):
        super().__init__()
        self.mse = torch.nn.MSELoss()
        self.eps = eps
        self.device = device
        self.threshold = threshold
        
    def forward(self,output,label):
        weight = torch.ones(label.shape).to(self.device)

        ## binary the targert with selected threshold
        mask = (label >= self.threshold).float().to(self.device)
 
        ## give me the total numer of 0 and 1
        num_0 = torch.sum(mask == 0).float().to(self.device)
        num_1 = torch.sum(mask == 1).float().to(self.device)
        
        ## give me the frequency of 0 and 1
        freq_0 = num_0 / (num_0 + num_1).to(self.device)
        freq_1 = num_1 / (num_0 + num_1).to(self.device)
        
        # create a weight tensor substituting 1 with 1/num_1 and 0 with 1/num_0
        weight = torch.where(mask == 1, 1/freq_1, weight).to(self.device)
        weight = torch.where(mask == 0, 1/freq_0, weight).to(self.device)

        loss = torch.sqrt((torch.mean(weight * (label - output) ** 2)) + self.eps)
        return loss

class WeighteRMSELoss_l2MAE(torch.nn.Module):
    """
    Weighted Root Mean Square Error Loss with L2 regularization function for the mass center
    """
    def __init__(self, threshold, device, mu=1., alpha=1., eps=1e-6):
        super().__init__()
        self.eps = eps
        self.device = device
        self.threshold = threshold
        self.w_rmse = WeightedRMSELoss(self.threshold, self.device, self.eps)
        self.mu = mu
        self.alpha = alpha
        
    def forward(self,output,label):
        ## weighted RMSE
        w_rmse = self.w_rmse(output, label)

        # ## L2 regularization
        mass_center_output = self.get_cordinate(output).float()
        mass_center_label = self.get_cordinate(label).float()
        l2_reg = torch.mean((mass_center_output - mass_center_label) ** 2)

        loss = self.mu * w_rmse + self.alpha * l2_reg
        return loss

    def get_cordinate(self, heatmap):
        """
        Get the coordinate from the heatmap

        Parameters
        ----------
        heatmap : torch.Tensor
            Tensor containing the heatmap (B x C * W * H)

        Returns
        -------
        list
            List containing the coordinates
        """
        batch_list, label_list = [], []
        for batch in range(heatmap.shape[0]):
            for ch in range(heatmap.shape[1]):
                # Get the channel slice
                ch_map = heatmap[0, ch, :, :]
                max_index = torch.argmax(ch_map)
                max_coordinates = divmod(max_index.item(), ch_map.size(1))
                
                label_list.append(max_coordinates[1]/ch_map.size(1))
                label_list.append(max_coordinates[0]/ch_map.size(0))   
            batch_list.append(label_list)
        return torch.tensor(batch_list)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Read the dataset')
    parser.add_argument('--data_dir', type=str, default="/media/angelo/OS/Users/lasal/Desktop/Phd notes/Echocardiografy/EchoNet-LVH", help='Directory of the dataset')
    parser.add_argument('--batch_dir', type=str, default='Batch2', help='Batch number of video folder, e.g. Batch1, Batch2, Batch3, Batch4')
    parser.add_argument('--phase', type=str, default='diastole', help='select the phase of the heart, diastole or systole')
    parser.add_argument('--target', type=str, default='keypoints', help='select the target to predict, e.g. keypoints, heatmaps, segmentation')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs to train the model')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for the dataloader')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for the optimizer')
    parser.add_argument('--save_dir', type=str, default='TRAINED_MODEL', help='Directory to save the model')
    args = parser.parse_args()
    
    ## device and reproducibility    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device('cpu')
    print(f'Using device: {device}')
    

    print('start creating the dataset...')
    validation_set = EchoNetDataset(batch=args.batch_dir, split='val', phase=args.phase, label_directory=None, 
                              target=args.target, augmentation=True)

    print('start creating the dataloader...')
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    for ii in range(1):
        image, label = validation_set[0]
        print(f'image shape: {image.size} - label shape: {type(label)}')
        print(f'min max image: {image.min()} - {image.max()}')

        plt.figure(figsize=(24, 10), tight_layout=True)
        plt.subplot(1, 4, 1)
        plt.imshow(image[0], cmap='gray')
        plt.title('Image')
        plt.axis('off')
        plt.subplot(1, 4, 2)
        plt.imshow(image[0], cmap='gray')
        plt.imshow(label[0,:,:], cmap='jet', alpha=0.5)
        plt.imshow(label[-1,:,:], cmap='jet', alpha=0.5)
        plt.title('Image')
        plt.axis('off')
        plt.subplot(1, 4, 3)
        plt.imshow(label[0,:,:], cmap='jet')
        plt.title('Label')
        plt.axis('off')
        plt.subplot(1, 4, 4)
        plt.imshow(label[-1,:,:], cmap='jet')
        plt.title('Heatmap')
        plt.axis('off')
        # plt.show()

    
    
    
    for i, (image, label) in enumerate(validation_loader):
        ## cretate a target tensfor random with the same shape of the label
        print(image.shape, label.shape)

        plt.figure(figsize=(24, 10))
        plt.subplot(1, 4, 1)
        plt.imshow(image[0,0,:,:], cmap='gray')
        plt.title('Image')
        plt.axis('off')
        plt.subplot(1, 4, 2)
        plt.imshow(image[0,0,:,:], cmap='gray')
        plt.imshow(label[0,0,:,:], cmap='jet', alpha=0.5)
        plt.imshow(label[0,-1,:,:], cmap='jet', alpha=0.5)
        plt.title('Image')
        plt.axis('off')
        plt.subplot(1, 4, 3)
        plt.imshow(label[0,0,:,:], cmap='jet')
        plt.title('Label')
        plt.axis('off')
        plt.subplot(1, 4, 4)
        plt.imshow(label[0,-1,:,:], cmap='jet')
        plt.title('Heatmap')
        plt.axis('off')

        output = torch.rand(label.shape).to(device)
        label = label.to(device)

        loss = torch.nn.MSELoss()
        loss = nn.MSELoss()
        
        w_mse = WeightedMSELoss(threshold=0.1, device = device)(output, label)
        w_rmse = WeightedRMSELoss(threshold=0.1, device = device)(output, label)
        wl_rmse = WeighteRMSELoss_l2MAE(threshold=0.0, device = device)(output, label)
        rmse = RMSELoss()(output, label)

        print(f'W_MSE: {w_mse}')
        print(f'W_RMSE: {w_rmse}')
        print(f'RMSE: {rmse}')
        print(f'WL_RMSE: {wl_rmse}')
        print('===============================================')
        plt.show()
