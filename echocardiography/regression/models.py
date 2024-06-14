"""
Structure of the models used in automatic detection of keypoits in PLAX echocardiography.
"""
import torch
import torch.nn as nn
import torchvision
from torchvision import models
import torchvision.transforms.functional as functional
import torch.nn.functional as F
from pathlib import Path
from torchsummary import summary 
from torch.distributions.multivariate_normal import MultivariateNormal
import timm
from scipy import ndimage
import cv2
import numpy as np
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ResNet50Regression(nn.Module):
    def __init__(self, input_channels, num_labels):
        super(ResNet50Regression, self).__init__()

        self.input_channels = input_channels
        self.num_labels = num_labels

        # Load the pre-trained ResNet50 model
        resnet50 = models.resnet50(weights='DEFAULT')
         
        # Add a new regression layer
        self.resnet50 = resnet50  
        self.resnet50.conv1 = nn.Conv2d(self.input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)  
        self.resnet50.fc = nn.Sequential(nn.Linear(resnet50.fc.in_features, self.num_labels),
                                                    nn.Sigmoid())

    def forward(self, x):
        x = self.resnet50(x)
        x = x.view(x.size(0), -1)
        return x

class SwinTransformerTiny(nn.Module):
    """
    Swin trasformed v2 modela with a modified first convolutional layer to accept the desired number of input channels.
    """
    def __init__(self, input_channels, num_labels):
        super(SwinTransformerTiny, self).__init__()
        self.input_channels = input_channels
        self.num_classes = num_labels

        # Load the pre-trained Swin Transformer Tiny model
        self.model = timm.create_model('swinv2_tiny_window8_256', pretrained=True, num_classes=self.num_classes)

        # Modify the first convolutional layer to accept the desired number of input channels
        self.model.patch_embed.proj = nn.Conv2d(self.input_channels, self.model.patch_embed.proj.out_channels,
                                                kernel_size=(4, 4), stride=(4, 4))

    def forward(self, x):

        # Obtain the features from the backbone, normalized but not with the pooling
        features = self.model.forward_features(x)
        
        # Obtain the final classification output
        output = self.model.head(features)

        return features, output


class PlaxModel(torch.nn.Module):

    """Model used for prediction of PLAX measurement points.
    Output channels correspond to heatmaps for the endpoints of
    measurements of interest.
    """

    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.model = torchvision.models.segmentation.deeplabv3_resnet50(num_classes=self.num_classes)
        # self.model = torchvision.models.segmentation.fcn_resnet50(num_classes=self.num_classes)

    def forward(self, x):
        return torch.sigmoid(self.model(x)['out'])

class UNet_up(nn.Module):
    def __init__(self, input_channels, num_classes):
        super().__init__()
        self.input_channels = input_channels

        # Encoder
        self.e11 = nn.Conv2d(self.input_channels, 64, kernel_size=3, padding=1) # output: 570x570x64
        self.bn11 = nn.BatchNorm2d(64)
        self.e12 = nn.Conv2d(64, 64, kernel_size=3, padding=1) # output: 568x568x64
        self.bn12 = nn.BatchNorm2d(64)

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 284x284x64

        # input: 284x284x64
        self.e21 = nn.Conv2d(64, 128, kernel_size=3, padding=1) # output: 282x282x128
        self.bn21 = nn.BatchNorm2d(128)
        self.e22 = nn.Conv2d(128, 128, kernel_size=3, padding=1) # output: 280x280x128
        self.bn22 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 140x140x128

        # input: 140x140x128
        self.e31 = nn.Conv2d(128, 256, kernel_size=3, padding=1) # output: 138x138x256
        self.bn31 = nn.BatchNorm2d(256)
        self.e32 = nn.Conv2d(256, 256, kernel_size=3, padding=1) # output: 136x136x256
        self.bn32 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 68x68x256

        # input: 68x68x256
        self.e41 = nn.Conv2d(256, 512, kernel_size=3, padding=1) # output: 66x66x512
        self.bn41 = nn.BatchNorm2d(512)
        self.e42 = nn.Conv2d(512, 512, kernel_size=3, padding=1) # output: 64x64x512
        self.bn42 = nn.BatchNorm2d(512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 32x32x512

        # input: 32x32x512
        self.e51 = nn.Conv2d(512, 1024, kernel_size=3, padding=1) # output: 30x30x1024
        self.bn51 = nn.BatchNorm2d(1024)
        self.e52 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1) # output: 28x28x1024
        self.bn52 = nn.BatchNorm2d(1024)


        # Decoder
        self.upconv1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1_1 = nn.Conv2d(1024,512, kernel_size=1, stride=1)

        self.d11 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.bn_d11 = nn.BatchNorm2d(512)
        self.d12 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn_d12 = nn.BatchNorm2d(512)

        self.upconv2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1_2 = nn.Conv2d(512, 256, kernel_size=1, stride=1)

        self.d21 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.bn_d21 = nn.BatchNorm2d(256)
        self.d22 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn_d22 = nn.BatchNorm2d(256)

        self.upconv3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1_3 = nn.Conv2d(256, 128, kernel_size=1, stride=1)

        self.d31 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.bn_d31 = nn.BatchNorm2d(128)
        self.d32 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn_d32 = nn.BatchNorm2d(128)

        self.upconv4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1_4 = nn.Conv2d(128, 64, kernel_size=1, stride=1)

        self.d41 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn_d41 = nn.BatchNorm2d(64)
        self.d42 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn_d42 = nn.BatchNorm2d(64)

        # Output layer
        self.outconv = nn.Conv2d(64, num_classes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Encoder
        xe11 = F.relu(self.bn11(self.e11(x)))
        xe12 = F.relu(self.bn12(self.e12(xe11)))
        xp1 = self.pool1(xe12)

        xe21 = F.relu(self.bn21(self.e21(xp1)))
        xe22 = F.relu(self.bn22(self.e22(xe21)))
        xp2 = self.pool2(xe22)

        xe31 = F.relu(self.bn31(self.e31(xp2)))
        xe32 = F.relu(self.bn32(self.e32(xe31)))
        xp3 = self.pool3(xe32)

        xe41 = F.relu(self.bn41(self.e41(xp3)))
        xe42 = F.relu(self.bn42(self.e42(xe41)))
        xp4 = self.pool4(xe42)

        xe51 = F.relu(self.bn51(self.e51(xp4)))
        xe52 = F.relu(self.bn52(self.e52(xe51)))
        
        # Decoder
        xu1 = self.conv1_1(self.upconv1(xe52))
        xu11 = torch.cat([xu1, xe42], dim=1)
        xd11 = F.relu(self.bn_d11(self.d11(xu11)))
        xd12 = F.relu(self.bn_d12(self.d12(xd11)))

        xu2 = self.conv1_2(self.upconv2(xd12))
        xu22 = torch.cat([xu2, xe32], dim=1)
        xd21 = F.relu(self.bn_d21(self.d21(xu22)))
        xd22 = F.relu(self.bn_d22(self.d22(xd21)))

        xu3 = self.conv1_3(self.upconv3(xd22))
        xu33 = torch.cat([xu3, xe22], dim=1)
        xd31 = F.relu(self.bn_d31(self.d31(xu33)))
        xd32 = F.relu(self.bn_d32(self.d32(xd31)))

        xu4 = self.conv1_4(self.upconv4(xd32))
        xu44 = torch.cat([xu4, xe12], dim=1)
        xd41 = F.relu(self.bn_d41(self.d41(xu44)))
        xd42 = F.relu(self.bn_d42(self.d42(xd41)))

        # Output layer
        out = self.outconv(xd42)
        out = self.sigmoid(out)

        return out

class UNet(nn.Module):
    def __init__(self, input_channels, num_classes):
        super().__init__()
        self.input_channels = input_channels

        # Encoder
        self.e11 = nn.Conv2d(self.input_channels, 64, kernel_size=3, padding=1) # output: 570x570x64
        self.bn12 = nn.BatchNorm2d(64)
        self.e12 = nn.Conv2d(64, 64, kernel_size=3, padding=1) # output: 568x568x64
        self.bn12 = nn.BatchNorm2d(64)

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 284x284x64

        # input: 284x284x64
        self.e21 = nn.Conv2d(64, 128, kernel_size=3, padding=1) # output: 282x282x128
        self.bn21 = nn.BatchNorm2d(128)
        self.e22 = nn.Conv2d(128, 128, kernel_size=3, padding=1) # output: 280x280x128
        self.bn22 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 140x140x128

        # input: 140x140x128
        self.e31 = nn.Conv2d(128, 256, kernel_size=3, padding=1) # output: 138x138x256
        self.bn31 = nn.BatchNorm2d(256)
        self.e32 = nn.Conv2d(256, 256, kernel_size=3, padding=1) # output: 136x136x256
        self.bn32 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 68x68x256

        # input: 68x68x256
        self.e41 = nn.Conv2d(256, 512, kernel_size=3, padding=1) # output: 66x66x512
        self.bn41 = nn.BatchNorm2d(512)
        self.e42 = nn.Conv2d(512, 512, kernel_size=3, padding=1) # output: 64x64x512
        self.bn42 = nn.BatchNorm2d(512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 32x32x512

        # input: 32x32x512
        self.e51 = nn.Conv2d(512, 1024, kernel_size=3, padding=1) # output: 30x30x1024
        self.bn51 = nn.BatchNorm2d(1024)
        self.e52 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1) # output: 28x28x1024
        self.bn52 = nn.BatchNorm2d(1024)


        # Decoder
        self.upconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.d11 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.bn_d11 = nn.BatchNorm2d(512)
        self.d12 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn_d12 = nn.BatchNorm2d(512)

        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.d21 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.bn_d21 = nn.BatchNorm2d(256)
        self.d22 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        # self.bn_d22 = nn.BatchNorm2d(256)

        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.d31 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.bn_d31 = nn.BatchNorm2d(128)
        self.d32 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn_d32 = nn.BatchNorm2d(128)

        self.upconv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.d41 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn_d41 = nn.BatchNorm2d(64)
        self.d42 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn_d42 = nn.BatchNorm2d(64)

        # Output layer
        self.outconv = nn.Conv2d(64, num_classes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Encoder
        xe11 = F.relu(self.e11(x))
        xe12 = F.relu(self.e12(xe11))
        xp1 = self.pool1(xe12)

        xe21 = F.relu(self.e21(xp1))
        xe22 = F.relu(self.e22(xe21))
        xp2 = self.pool2(xe22)

        xe31 = F.relu(self.e31(xp2))
        xe32 = F.relu(self.e32(xe31))
        xp3 = self.pool3(xe32)

        xe41 = F.relu(self.e41(xp3))
        xe42 = F.relu(self.e42(xe41))
        xp4 = self.pool4(xe42)

        xe51 = F.relu(self.e51(xp4))
        xe52 = F.relu(self.e52(xe51))
        
        # Decoder
        xu1 = self.upconv1(xe52)
        xu11 = torch.cat([xu1, xe42], dim=1)
        xd11 = F.relu(self.d11(xu11))
        xd12 = F.relu(self.d12(xd11))

        xu2 = self.upconv2(xd12)
        xu22 = torch.cat([xu2, xe32], dim=1)
        xd21 = F.relu(self.d21(xu22))
        xd22 = F.relu(self.d22(xd21))

        xu3 = self.upconv3(xd22)
        xu33 = torch.cat([xu3, xe22], dim=1)
        xd31 = F.relu(self.d31(xu33))
        xd32 = F.relu(self.d32(xd31))

        xu4 = self.upconv4(xd32)
        xu44 = torch.cat([xu4, xe12], dim=1)
        xd41 = F.relu(self.d41(xu44))
        xd42 = F.relu(self.d42(xd41))

        # Output layer
        out = self.outconv(xd42)
        out = self.sigmoid(out)

        return out

class HeatmapLayer(nn.Module):
    def __init__(self, num_classes, max_sigma=0.49999, min_sigma=0.00001):
        super(HeatmapLayer, self).__init__()
        # Glorot (Xavier) uniform initialization for the weight parameter
        self.num_classes = num_classes

        ## give me a vector of size batch_size of random values between 0 and max_sigma
        self.log_weight = nn.Parameter(data = torch.Tensor(1,1,1,1) , requires_grad=True)
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.log_weight)
      
    def forward(self, x, labels):
        ## x shape = (batch_size, 3, 256, 256)
        ## labels shape = (batch_size, 12)
        image, labels = x, labels
        device = labels.device  # Get the device of the labels tensor
        
        ## mulptiple the labels by the image size
        converter = torch.tensor([image.shape[2], image.shape[3]], device=device).repeat(self.num_classes)
        labels = labels * converter

        ## create a meshgrid of the image size
        x, y = torch.meshgrid(torch.arange(0, image.shape[2], device=device), torch.arange(0, image.shape[3], device=device))
        pos = torch.stack((x, y), dim=2)
     
        #create a torch tensrof of the same shaepe if self.weight and all the value 256
        images_widths = torch.ones((image.shape[0], self.num_classes), device=device) * image.shape[2]
        # sigmas = images_widths * self.weight.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        sigmas = images_widths * self.log_weight.exp()

        # create a tensor of covariace matrix with shape (batch_size, num_classes, 2, 2)
        covariances = torch.zeros((image.shape[0], self.num_classes, 2, 2), device=device)
        covariances[:, :, 0, 0] = sigmas 
        covariances[:, :, 1, 1] = sigmas * 20
    
        
        # create the tensor of angles
        x_diff = labels[:, [0,4,8]] - labels[:, [2,6,10]] 
        y_diff = labels[:, [3,7,11]] - labels[:, [1,5,9]]

        angle_deg = torch.atan2(y_diff, x_diff) * 180 / torch.pi
        angle = torch.zeros((image.shape[0], self.num_classes), device=device)
        angle[:,0], angle[:,1]= angle_deg[:,0], angle_deg[:,0]
        angle[:,2], angle[:,3]= angle_deg[:,1], angle_deg[:,1]
        angle[:,4], angle[:,5]= angle_deg[:,2], angle_deg[:,2]


        # cretate the tensor of means (the center of heatmaps)
        mean = torch.zeros((image.shape[0], self.num_classes,2), device=device)
        mean[:, :, 0] = labels[:, [0,2,4,6,8,10]] # x_coordinate of labels
        mean[:, :, 1] = labels[:, [1,3,5,7,9,11]] # y_coordinate of labels

        ## Create the heatmaps
        # Initialize an empty 6-channel heatmap vector
        heatmaps_labels= torch.zeros((image.shape[0], self.num_classes, image.shape[2], image.shape[3]), device=device)
        for batch in range(image.shape[0]): # for each image in the batch
            for hp in range(self.num_classes):
                # create a multivariate normal distribution
                gaussian = MultivariateNormal(loc=torch.tensor(mean[batch,hp]), covariance_matrix=covariances[batch,hp])
                log_prob = gaussian.log_prob(pos)
                base_heatmap = torch.exp(log_prob)

                #normalize the heatmap
                base_heatmap = base_heatmap / torch.max(base_heatmap)
                heatmaps_labels[batch, hp, :, :] = base_heatmap
        
        
        # create the rotation matrix
        # rot_matrix = torch.zeros((image.shape[0], self.num_classes, 2, 2), device=device)
        # rot_matrix[:, :, 0, 0] = torch.cos(angle * torch.pi / 180)
        # rot_matrix[:, :, 0, 1] = -torch.sin(angle * torch.pi / 180)
        # rot_matrix[:, :, 1, 0] = torch.sin(angle * torch.pi / 180)
        # rot_matrix[:, :, 1, 1] = torch.cos(angle * torch.pi / 180)
        # print('rot_matrix', rot_matrix.shape)
        # print('heatmaps_labels', heatmaps_labels.shape)

        # # create the affine grid
        # grid = F.affine_grid(rot_matrix, heatmaps_labels.size())
        # print('grid', grid.shape)

        
        return heatmaps_labels

class UNet_up_hm(nn.Module):
    """
    UNet model with heatmap layer
    """
    def __init__(self, num_classes):
        super().__init__()
        # Encoder
        self.e11 = nn.Conv2d(3, 64, kernel_size=3, padding=1) # output: 570x570x64
        self.bn11 = nn.BatchNorm2d(64)
        self.e12 = nn.Conv2d(64, 64, kernel_size=3, padding=1) # output: 568x568x64
        self.bn12 = nn.BatchNorm2d(64)

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 284x284x64

        # input: 284x284x64
        self.e21 = nn.Conv2d(64, 128, kernel_size=3, padding=1) # output: 282x282x128
        self.bn21 = nn.BatchNorm2d(128)
        self.e22 = nn.Conv2d(128, 128, kernel_size=3, padding=1) # output: 280x280x128
        self.bn22 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 140x140x128

        # input: 140x140x128
        self.e31 = nn.Conv2d(128, 256, kernel_size=3, padding=1) # output: 138x138x256
        self.bn31 = nn.BatchNorm2d(256)
        self.e32 = nn.Conv2d(256, 256, kernel_size=3, padding=1) # output: 136x136x256
        self.bn32 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 68x68x256

        # input: 68x68x256
        self.e41 = nn.Conv2d(256, 512, kernel_size=3, padding=1) # output: 66x66x512
        self.bn41 = nn.BatchNorm2d(512)
        self.e42 = nn.Conv2d(512, 512, kernel_size=3, padding=1) # output: 64x64x512
        self.bn42 = nn.BatchNorm2d(512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 32x32x512

        # input: 32x32x512
        self.e51 = nn.Conv2d(512, 1024, kernel_size=3, padding=1) # output: 30x30x1024
        self.bn51 = nn.BatchNorm2d(1024)
        self.e52 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1) # output: 28x28x1024
        self.bn52 = nn.BatchNorm2d(1024)


        # Decoder
        self.upconv1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1_1 = nn.Conv2d(1024,512, kernel_size=1, stride=1)

        self.d11 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.bn_d11 = nn.BatchNorm2d(512)
        self.d12 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn_d12 = nn.BatchNorm2d(512)

        self.upconv2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1_2 = nn.Conv2d(512, 256, kernel_size=1, stride=1)

        self.d21 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.bn_d21 = nn.BatchNorm2d(256)
        self.d22 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn_d22 = nn.BatchNorm2d(256)

        self.upconv3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1_3 = nn.Conv2d(256, 128, kernel_size=1, stride=1)

        self.d31 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.bn_d31 = nn.BatchNorm2d(128)
        self.d32 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn_d32 = nn.BatchNorm2d(128)

        self.upconv4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1_4 = nn.Conv2d(128, 64, kernel_size=1, stride=1)

        self.d41 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn_d41 = nn.BatchNorm2d(64)
        self.d42 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn_d42 = nn.BatchNorm2d(64)

        # Output layer
        self.outconv = nn.Conv2d(64, num_classes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

        ## Heatmap layer
        self.heatmap_layer = HeatmapLayer(num_classes=num_classes)

    def forward(self, x, labels):
        # Encoder
        xe11 = F.relu(self.bn11(self.e11(x)))
        xe12 = F.relu(self.bn12(self.e12(xe11)))
        xp1 = self.pool1(xe12)

        xe21 = F.relu(self.bn21(self.e21(xp1)))
        xe22 = F.relu(self.bn22(self.e22(xe21)))
        xp2 = self.pool2(xe22)

        xe31 = F.relu(self.bn31(self.e31(xp2)))
        xe32 = F.relu(self.bn32(self.e32(xe31)))
        xp3 = self.pool3(xe32)

        xe41 = F.relu(self.bn41(self.e41(xp3)))
        xe42 = F.relu(self.bn42(self.e42(xe41)))
        xp4 = self.pool4(xe42)

        xe51 = F.relu(self.bn51(self.e51(xp4)))
        xe52 = F.relu(self.bn52(self.e52(xe51)))
        
        # Decoder
        xu1 = self.conv1_1(self.upconv1(xe52))
        xu11 = torch.cat([xu1, xe42], dim=1)
        xd11 = F.relu(self.bn_d11(self.d11(xu11)))
        xd12 = F.relu(self.bn_d12(self.d12(xd11)))

        xu2 = self.conv1_2(self.upconv2(xd12))
        xu22 = torch.cat([xu2, xe32], dim=1)
        xd21 = F.relu(self.bn_d21(self.d21(xu22)))
        xd22 = F.relu(self.bn_d22(self.d22(xd21)))

        xu3 = self.conv1_3(self.upconv3(xd22))
        xu33 = torch.cat([xu3, xe22], dim=1)
        xd31 = F.relu(self.bn_d31(self.d31(xu33)))
        xd32 = F.relu(self.bn_d32(self.d32(xd31)))

        xu4 = self.conv1_4(self.upconv4(xd32))
        xu44 = torch.cat([xu4, xe12], dim=1)
        xd41 = F.relu(self.bn_d41(self.d41(xu44)))
        xd42 = F.relu(self.bn_d42(self.d42(xd41)))

        # Output layer
        out = self.outconv(xd42)
        out = self.sigmoid(out)

        ## Heatmap layer
        heatmaps = self.heatmap_layer(x, labels)

        return out, heatmaps
        
        

if __name__ == '__main__':
    ## SimpleRegression model
    model = ResNet50Regression(input_channels=1, num_labels=12)
    x = torch.randn(1, 1, 256, 256)
    print(model)
    print()


    # ## PLAX model
    # model = PlaxModel(num_classes=6)
    # x = torch.randn(2, 3, 256, 256)
    # model.train()
    # print(summary(model))
    # print(model(x).shape, model(x).min(), model(x).max())
    # pred = model(x).detach().numpy()
    # # print max and min values
    # print(pred.max(), pred.min())
    # print(pred.shape)

    import matplotlib.pyplot as plt
    # plt.figure()
    # plt.imshow(pred[0, 0, :, :])
    # # plt.show()

    # ## UNet model
    # model = UNet(num_classes=6)
    # model.train()
    # x = torch.randn(2, 3, 256, 256)
    # print(summary(model))
    # print(model(x).shape, model(x).min(), model(x).max())
    # pred = model(x).detach().numpy()

    # plt.figure()
    # plt.imshow(pred[0, 0, :, :])
    # # plt.show()

    ## SimpleLayer model
    model = HeatmapLayer(num_classes=6)
    x = torch.randn(5, 3, 256, 256)
    labels = torch.rand(5, 12)
    heatmaps = model(x, labels).detach().numpy()
    print(heatmaps.shape)


    print('model unet up')
    model = UNet_up(num_classes=6)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {total_params}")
    x = torch.randn(2, 3, 256, 256)
    pred = model(x).detach().numpy()
    plt.figure(num='Un_up')
    plt.imshow(pred[0, 0, :, :])

    print('model unet up hm')
    model = UNet_up_hm(num_classes=6)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {total_params}")
    x = torch.randn(5, 3, 256, 256)
    labels = torch.rand(5, 12)
    pred, heatmaps = model(x, labels)
    heatmaps = heatmaps.detach().numpy()

    for i in range(6):
        plt.figure(num=f'Un_up_hm {i}')
        plt.imshow(heatmaps[0, i, :, :])
    # plt.show()

    for i in range(6):
        plt.figure(num=f'Un_up_hm edfas {i}')
        plt.imshow(heatmaps[1, i, :, :])
    plt.show()
  
    
    # Iterate through the layers inside the Sequential blockKC
    # for layer_idx, layer in enumerate(sequential_block.children()):
    #     print(f"Layer {layer_idx + 1}: {layer}")


    ############################################################################################################
    # a = time.time()
    # image, label = train_set[1456]
    # print('time to get the image and label:', time.time() - a)
    # print(image.size, label.shape, type(image), type(label))
    # image = image.numpy().transpose((1, 2, 0))
    
    # a = time.time()
    # heatmap = train_set.get_heatmap(0)
    # print('time to get the heatmap:', time.time() - a)
    # print(heatmap.shape)

    # plt.figure(figsize=(14,14), num='Example_dataset_heatmap')
    # plt.imshow(image, cmap='gray')
    # for i in range(label.shape[0]):
    #     plt.imshow(label[i,:,:], cmap='jet', alpha=0.2)
    # plt.show()

    # for i in training_loader:
    #     print(i[0].shape, i[1].shape)
    #     label = i[1].numpy()        
    #     print(label.shape)
    #     plt.figure()
    #     plt.imshow(i[0].numpy().transpose((0, 2, 3, 1))[0], cmap='gray')
    #     plt.imshow(label[0, 0, :, :], alpha=0.3)
    #     plt.imshow(label[0, -1, :, :], alpha=0.3)
    #     plt.show()