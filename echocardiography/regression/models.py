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

class ResNet101Regression(nn.Module):
    def __init__(self, input_channels, num_labels):
        super(ResNet101Regression, self).__init__()

        self.input_channels = input_channels
        self.num_labels = num_labels

        # Load the pre-trained ResNet50 model
        resnet101 = models.resnet101(weights='DEFAULT')
         
        # Add a new regression layer
        self.resnet101 = resnet101  
        self.resnet101.conv1 = nn.Conv2d(self.input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)  
        self.resnet101.fc = nn.Sequential(nn.Linear(resnet101.fc.in_features, self.num_labels),
                                                    nn.Sigmoid())

    def forward(self, x):
        x = self.resnet101(x)
        x = x.view(x.size(0), -1)
        return x

class ResNet152Regression(nn.Module):
    def __init__(self, input_channels, num_labels):
        super(ResNet152Regression, self).__init__()

        self.input_channels = input_channels
        self.num_labels = num_labels

        # Load the pre-trained ResNet152 model
        resnet152 = models.resnet152(pretrained=True)
         
        # Add a new regression layer
        self.resnet152 = resnet152  
        self.resnet152.conv1 = nn.Conv2d(self.input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)  
        self.resnet152.fc = nn.Sequential(nn.Linear(resnet152.fc.in_features, self.num_labels),
                                                    nn.Sigmoid())

    def forward(self, x):
        x = self.resnet152(x)
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

class SwinTransformerSmall(nn.Module):
    """
    Swin trasformed v2 modela with a modified first convolutional layer to accept the desired number of input channels.
    """
    def __init__(self, input_channels, num_labels):
        super(SwinTransformerSmall, self).__init__()
        self.input_channels = input_channels
        self.num_classes = num_labels

        # Load the pre-trained Swin Transformer Tiny model
        self.model = timm.create_model('swinv2_small_window8_256', pretrained=True, num_classes=self.num_classes)

        # Modify the first convolutional layer to accept the desired number of input channels
        self.model.patch_embed.proj = nn.Conv2d(self.input_channels, self.model.patch_embed.proj.out_channels,
                                                kernel_size=(4, 4), stride=(4, 4))

    def forward(self, x):

        # Obtain the features from the backbone, normalized but not with the pooling
        features = self.model.forward_features(x)
        
        # Obtain the final classification output
        output = self.model.head(features)

        return features, output

class SwinTransformerBase(nn.Module):
    """
    Swin trasformed v2 Base model with a modified first convolutional layer to accept the desired number of input channels.
    """
    def __init__(self, input_channels, num_labels):
        super(SwinTransformerBase, self).__init__()
        self.input_channels = input_channels
        self.num_classes = num_labels

        # Load the pre-trained Swin Transformer Tiny model
        self.model = timm.create_model('swinv2_base_window8_256', pretrained=True, num_classes=self.num_classes)

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
        
class DownBlock(nn.Module):
    """
    Down conv block with attention.
    Sequence of following block
    1. Resnet block 
    2. Attention block
    3. Downsample

    Parameters
    ----------
    in_channels: Number of input channels
    out_channels: Number of output channels
    """
    
    def __init__(self, in_channels, out_channels, down_sample, num_heads, num_layers, attn, norm_channels):
        super().__init__()
        self.num_layers = num_layers
        self.down_sample = down_sample
        self.attn = attn
        
        self.resnet_conv_first = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(norm_channels, in_channels if i == 0 else out_channels),
                    nn.SiLU(),
                    nn.Conv2d(in_channels if i == 0 else out_channels, out_channels,
                              kernel_size=3, stride=1, padding=1),
                )
                for i in range(num_layers)
            ]
        )
        
        self.resnet_conv_second = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(norm_channels, out_channels),
                    nn.SiLU(),
                    nn.Conv2d(out_channels, out_channels,
                              kernel_size=3, stride=1, padding=1),
                )
                for _ in range(num_layers)
            ]
        )
        
        if self.attn:
            self.attention_norms = nn.ModuleList(
                [nn.GroupNorm(norm_channels, out_channels)
                 for _ in range(num_layers)]
            )
            
            self.attentions = nn.ModuleList(
                [nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
                 for _ in range(num_layers)]
            )
        

        self.residual_input_conv = nn.ModuleList(
            [
                nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=1)
                for i in range(num_layers)
            ]
        )
        self.down_sample_conv = nn.Conv2d(out_channels, out_channels,
                                          4, 2, 1) if self.down_sample else nn.Identity()
    
    def forward(self, x, t_emb=None, context=None):
        out = x
        for i in range(self.num_layers):
            # Resnet block of Unet
            resnet_input = out
            out = self.resnet_conv_first[i](out)
            out = self.resnet_conv_second[i](out)
            out = out + self.residual_input_conv[i](resnet_input)
            
            if self.attn:
                # Attention block of Unet
                batch_size, channels, h, w = out.shape
                # print(f'     Attention block {i}) input shape: {out.shape}')
                in_attn = out.reshape(batch_size, channels, h * w)
                in_attn = self.attention_norms[i](in_attn)
                in_attn = in_attn.transpose(1, 2)
                # print(f'     Attention block {i}) in_attn: {in_attn.shape} reshape and normalize')
                out_attn, _ = self.attentions[i](in_attn, in_attn, in_attn)
                # print(f'     Attention block {i}) out_attn: {out_attn.shape} ')
                out_attn = out_attn.transpose(1, 2).reshape(batch_size, channels, h, w)
                out = out + out_attn
                    
        # Downsample
        out = self.down_sample_conv(out)
        return out

class MidBlock(nn.Module):
    """
    Mid conv block with attention.
    Sequence of following blocks
    1. Resnet block with time embedding
    2. Attention block
    3. Resnet block with time embedding
    """
    
    def __init__(self, in_channels, out_channels, num_heads, num_layers, norm_channels):
        super().__init__()
        self.num_layers = num_layers

        self.resnet_conv_first = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(norm_channels, in_channels if i == 0 else out_channels),
                    nn.SiLU(),
                    nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=3, stride=1,
                              padding=1),
                )
                for i in range(num_layers + 1)
            ]
        )
        
        
        self.resnet_conv_second = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(norm_channels, out_channels),
                    nn.SiLU(),
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                )
                for _ in range(num_layers + 1)
            ]
        )
        
        self.attention_norms = nn.ModuleList(
            [nn.GroupNorm(norm_channels, out_channels)
             for _ in range(num_layers)]
        )
        
        self.attentions = nn.ModuleList(
            [nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
             for _ in range(num_layers)]
        )
        
        self.residual_input_conv = nn.ModuleList(
            [
                nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=1)
                for i in range(num_layers + 1)
            ]
        )

    def forward(self, x, t_emb=None, context=None):
        out = x
        
        # First resnet block
        resnet_input = out
        out = self.resnet_conv_first[0](out)
        out = self.resnet_conv_second[0](out)
        out = out + self.residual_input_conv[0](resnet_input)
        
        for i in range(self.num_layers):
            # Attention Block
            batch_size, channels, h, w = out.shape
            in_attn = out.reshape(batch_size, channels, h * w)
            in_attn = self.attention_norms[i](in_attn)
            in_attn = in_attn.transpose(1, 2)
            out_attn, _ = self.attentions[i](in_attn, in_attn, in_attn)
            out_attn = out_attn.transpose(1, 2).reshape(batch_size, channels, h, w)
            out = out + out_attn
            
            # Resnet Block
            resnet_input = out
            out = self.resnet_conv_first[i + 1](out)
            out = self.resnet_conv_second[i + 1](out)
            out = out + self.residual_input_conv[i + 1](resnet_input)
        
        return out

class UpBlock(nn.Module):
    """
    Up conv block with attention.
    Sequence of following blocks
    1. Upsample
    1. Concatenate Down block output
    2. Resnet block 
    3. Attention Block
    """
    
    def __init__(self, in_channels, out_channels, up_sample, num_heads, num_layers, attn, norm_channels):
        super().__init__()
        self.num_layers = num_layers
        self.up_sample = up_sample
        self.attn = attn

        self.resnet_conv_first = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(norm_channels, in_channels if i == 0 else out_channels),
                    nn.SiLU(),
                    nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=3, stride=1,
                              padding=1),
                )
                for i in range(num_layers)
            ]
        )
                
        self.resnet_conv_second = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(norm_channels, out_channels),
                    nn.SiLU(),
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                )
                for _ in range(num_layers)
            ]
        )

        if self.attn:
            self.attention_norms = nn.ModuleList(
                [
                    nn.GroupNorm(norm_channels, out_channels)
                    for _ in range(num_layers)
                ]
            )
            
            self.attentions = nn.ModuleList(
                [
                    nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
                    for _ in range(num_layers)
                ]
            )
            
        self.residual_input_conv = nn.ModuleList(
            [
                nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=1)
                for i in range(num_layers)
            ]
        )
        self.up_sample_conv = nn.ConvTranspose2d(in_channels, in_channels,
                                                 4, 2, 1) \
            if self.up_sample else nn.Identity()
    
    def forward(self, x, out_down=None):
        # Upsample
        x = self.up_sample_conv(x)
        
        # Concat with Downblock output
        if out_down is not None:
            print(x.shape, out_down.shape  )
            x = torch.cat([x, out_down], dim=1)

        out = x
        for i in range(self.num_layers):
            # Resnet Block
            resnet_input = out
            out = self.resnet_conv_first[i](out)
            out = self.resnet_conv_second[i](out)
            out = out + self.residual_input_conv[i](resnet_input)
            
            # Self Attention
            if self.attn:
                batch_size, channels, h, w = out.shape
                in_attn = out.reshape(batch_size, channels, h * w)
                in_attn = self.attention_norms[i](in_attn)
                in_attn = in_attn.transpose(1, 2)
                out_attn, _ = self.attentions[i](in_attn, in_attn, in_attn)
                out_attn = out_attn.transpose(1, 2).reshape(batch_size, channels, h, w)
                out = out + out_attn
        return out


class UpBlockUnet(nn.Module):
    """
    Up conv block with attention.
    Sequence of following blocks
    1. Upsample
    1. Concatenate Down block output
    2. Resnet block with time embedding
    3. Attention Block
    """
    
    def __init__(self, in_channels, out_channels, up_sample, num_heads, num_layers, attn, norm_channels):
        super().__init__()
        self.num_layers = num_layers
        self.up_sample = up_sample
        self.attn = attn

        self.resnet_conv_first = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(norm_channels, in_channels if i == 0 else out_channels),
                    nn.SiLU(),
                    nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=3, stride=1,
                              padding=1),
                )
                for i in range(num_layers)
            ]
        )
        
            
        self.resnet_conv_second = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(norm_channels, out_channels),
                    nn.SiLU(),
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                )
                for _ in range(num_layers)
            ]
        )
        
        if self.attn:
            self.attention_norms = nn.ModuleList(
                [
                    nn.GroupNorm(norm_channels, out_channels)
                    for _ in range(num_layers)
                ]
            )
            
            self.attentions = nn.ModuleList(
                [
                    nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
                    for _ in range(num_layers)
                ]
            )
            
       
        self.residual_input_conv = nn.ModuleList(
            [
                nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=1)
                for i in range(num_layers)
            ]
        )
        self.up_sample_conv = nn.ConvTranspose2d(in_channels // 2, in_channels // 2,
                                                 4, 2, 1) \
            if self.up_sample else nn.Identity()
    
    def forward(self, x, out_down=None, t_emb=None, context=None):
        
        x = self.up_sample_conv(x)
        if out_down is not None:
            x = torch.cat([x, out_down], dim=1)
        
        out = x
        for i in range(self.num_layers):
            # Resnet
            resnet_input = out
            out = self.resnet_conv_first[i](out)
            out = self.resnet_conv_second[i](out)
            out = out + self.residual_input_conv[i](resnet_input)
            # Self Attention
            if self.attn:
                batch_size, channels, h, w = out.shape
                in_attn = out.reshape(batch_size, channels, h * w)
                in_attn = self.attention_norms[i](in_attn)
                in_attn = in_attn.transpose(1, 2)
                out_attn, _ = self.attentions[i](in_attn, in_attn, in_attn)
                out_attn = out_attn.transpose(1, 2).reshape(batch_size, channels, h, w)
                out = out + out_attn

        
        return out

class UNet_ResNoSkip(nn.Module):
    def __init__(self, im_channels, num_classes, model_config):
        super().__init__()
        self.down_channels = model_config['down_channels']
        self.down_sample = model_config['down_sample']
        self.num_down_layers = model_config['num_down_layers']
        self.num_up_layers = model_config['num_up_layers']
        
        # To disable attention in Downblock of Encoder and Upblock of Decoder
        self.attns = model_config['attn_down']
        
        # Latent Dimension
        self.norm_channels = model_config['norm_channels']
        self.num_heads = model_config['num_heads']
        
        # Assertion to validate the channel information
        assert len(self.down_sample) == len(self.down_channels) - 1
        assert len(self.attns) == len(self.down_channels) - 1
        

        # Wherever we use downsampling in encoder correspondingly use
        # upsampling in decoder
        
        #print(f"VAE MODEL")
        self.up_sample = list(reversed(self.down_sample))
        
        ##################### Encoder ######################
        self.encoder_conv_in = nn.Conv2d(im_channels, self.down_channels[0], kernel_size=3, padding=(1, 1))
        
        #print(f'input channel {im_channels}, first out channel Conv2d {self.down_channels[0]}')
        
        #print('--------------------------------')
        
        # Downblock + Midblock
        
        #print('ENCODER')
        self.encoder_layers = nn.ModuleList([])
        for i in range(len(self.down_channels) - 1):
            
            #print(f'layer {i}) input channel {self.down_channels[i]}, out channel Conv2d {self.down_channels[i + 1]}')
            self.encoder_layers.append(DownBlock(self.down_channels[i], self.down_channels[i + 1], 
                                                 down_sample=self.down_sample[i],
                                                 num_heads=self.num_heads,
                                                 num_layers=self.num_down_layers,
                                                 attn=self.attns[i],
                                                 norm_channels=self.norm_channels))
        
        
        ##################### Decoder ######################   
        #print('Decoder Layers')
        self.decoder_layers = nn.ModuleList([])
        for i in reversed(range(1, len(self.down_channels))):
            
            #print(f'layer {i}) input channel {self.down_channels[i]}, out channel Conv2d {self.down_channels[i - 1]}')
            self.decoder_layers.append(UpBlock(self.down_channels[i] , self.down_channels[i - 1],
                                               up_sample=self.down_sample[i - 1],
                                               num_heads=self.num_heads,
                                               num_layers=self.num_up_layers,
                                               attn=self.attns[i - 1],
                                               norm_channels=self.norm_channels))
        
        self.decoder_norm_out = nn.GroupNorm(self.norm_channels, self.down_channels[0])
        
        #print(f'GroupNorm {self.norm_channels}, last out channel Conv2d {self.down_channels[0]}')
        self.decoder_conv_out = nn.Conv2d(self.down_channels[0], num_classes, kernel_size=3, padding=1)
        
        #print(f'output channel Conv2d {im_channels}')
    
    def encode(self, x):
        # print('Input')
        out = self.encoder_conv_in(x)
        # print(f'input channel {x.shape}, first out channel Conv2d {out.shape}')
        
        #print('first Conv2d',out.shape)
        # print('Encoder')
        for idx, down in enumerate(self.encoder_layers):
            # print(f'Encoder layer {idx} - input channel {out.shape}')
            out = down(out)
            # print(f'Encoder layer {idx} - out channel {out.shape}')
            # print()
        return out
    
    def decode(self, z):
        out = z
                   
        for idx, up in enumerate(self.decoder_layers):
            # print(f'Decoder layer {idx} - input channel {out.shape}')
            out = up(out)
            # print(f'Decoder layer {idx} - out channel {out.shape}')
            
            

        out = self.decoder_norm_out(out)
        
        #print('GroupNorm',out.shape)
        out = nn.SiLU()(out)
        out = self.decoder_conv_out(out)
        out = torch.sigmoid(out)    
        # print('Decoder Output',out.shape)
        return out

    def forward(self, x):
        
        #print('FORWARD PASS')
        
        #print('input',x.shape)
        encoder_output = self.encode(x)
        out = self.decode(encoder_output)
                
        #print('output',out.shape)
        return encoder_output, out

class Unet_ResSkip(nn.Module):
    """
    Unet model comprising
    Down blocks, Midblocks and Uplocks
    """
    
    def __init__(self, im_channels, num_classes, model_config):
        super().__init__()
        self.down_channels = model_config['down_channels']
        self.mid_channels = model_config['mid_channels']
        self.down_sample = model_config['down_sample']
        self.num_down_layers = model_config['num_down_layers']
        self.num_mid_layers = model_config['num_mid_layers']
        self.num_up_layers = model_config['num_up_layers']
        self.attns = model_config['attn_down']
        self.norm_channels = model_config['norm_channels']
        self.num_heads = model_config['num_heads']
        self.conv_out_channels = model_config['conv_out_channels']
        
        assert self.mid_channels[0] == self.down_channels[-1]
        assert self.mid_channels[-1] == self.down_channels[-2]
        assert len(self.down_sample) == len(self.down_channels) - 1
        assert len(self.attns) == len(self.down_channels) - 1
        
        
        self.up_sample = list(reversed(self.down_sample))
        self.conv_in = nn.Conv2d(im_channels, self.down_channels[0], kernel_size=3, padding=1)
        
        self.downs = nn.ModuleList([])
        for i in range(len(self.down_channels) - 1):
            self.downs.append(DownBlock(self.down_channels[i], self.down_channels[i + 1],
                                        down_sample=self.down_sample[i],
                                        num_heads=self.num_heads,
                                        num_layers=self.num_down_layers,
                                        attn=self.attns[i], norm_channels=self.norm_channels))
        
        self.mids = nn.ModuleList([])
        for i in range(len(self.mid_channels) - 1):
            self.mids.append(MidBlock(self.mid_channels[i], self.mid_channels[i + 1], 
                                      num_heads=self.num_heads,
                                      num_layers=self.num_mid_layers,
                                      norm_channels=self.norm_channels))
        
        self.ups = nn.ModuleList([])
        for i in reversed(range(len(self.down_channels) - 1)):
            self.ups.append(UpBlockUnet(self.down_channels[i] * 2, self.down_channels[i - 1] if i != 0 else self.conv_out_channels,
                                        up_sample=self.down_sample[i],
                                        num_heads=self.num_heads,
                                        attn=self.attns[i - 1],
                                        num_layers=self.num_up_layers,
                                        norm_channels=self.norm_channels))
        
        self.norm_out = nn.GroupNorm(self.norm_channels, self.conv_out_channels)
        self.conv_out = nn.Conv2d(self.conv_out_channels, num_classes, kernel_size=3, padding=1)
    
    def forward(self, x):
        # Shapes assuming downblocks are [C1, C2, C3, C4]
        # Shapes assuming midblocks are [C4, C4, C3]
        # Shapes assuming downsamples are [True, True, False]
        # B x C x H x W
        # print('input',x.shape)
        out = self.conv_in(x)
        # B x C1 x H x W
        # print('conv_in',out.shape)
        
        down_outs = []
        for idx, down in enumerate(self.downs):
            down_outs.append(out)
            out = down(out)
            # print(f'down {idx})',out.shape)
        # down_outs  [B x C1 x H x W, B x C2 x H/2 x W/2, B x C3 x H/4 x W/4]
        # out B x C4 x H/4 x W/4
        
        for idx, mid in enumerate(self.mids):
            out = mid(out)
            # print(f'mid {idx})',out.shape)
        # out B x C3 x H/4 x W/4
        
        for idx, up in enumerate(self.ups):
            down_out = down_outs.pop()
            # print(f'up {idx}) input',out.shape, down_out.shape)
            out = up(out, down_out)
            # print(f'up {idx})',out.shape)
            # out [B x C2 x H/4 x W/4, B x C1 x H/2 x W/2, B x 16 x H x W]
            
        out = self.norm_out(out)
        out = nn.SiLU()(out)
        out = self.conv_out(out)
        out = torch.sigmoid(out)
        # print('output',out.shape)
        # out B x C x H x W
        return out
if __name__ == '__main__':
    ## SimpleRegression model
    model = SwinTransformerSmall(input_channels=1, num_labels=12)
    x = torch.randn(1, 1, 256, 256)
    embedd, out = model(x)
    print(embedd.shape, out.shape)
    print()

    ## UNet_Res model
    model_config = {
        'down_channels': [32, 64, 128, 256],
        'down_sample': [True, True, True],
        'attn_down': [False, False, False],
        'norm_channels': 32,
        'num_heads': 16,
        'num_down_layers': 1,
        'num_mid_layers': 1,
        'num_up_layers': 1
    }

    unet_res = UNet_ResNoSkip(im_channels=1, num_classes=6, model_config=model_config)
    x = torch.randn(1, 1, 256, 256)
    out = unet_res(x)
    # print(unet_res)
    print(out[0].shape, out[1].shape)

    model_config = {
        'down_channels': [ 32, 64, 256, 256],
        'mid_channels': [ 256, 256],
        'down_sample': [ True, True, True ],
        'attn_down' : [False,False,False],
        'norm_channels' : 16,
        'num_heads' : 8,
        'conv_out_channels' : 128,
        'num_down_layers': 2,
        'num_mid_layers': 2,
        'num_up_layers': 2,
    }
    unet_res_skip = Unet_ResSkip(im_channels=1, num_classes=6, model_config=model_config)
    x = torch.randn(1, 1, 256, 256)
    out = unet_res_skip(x)
    print(out.shape)
    ## print the summary


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