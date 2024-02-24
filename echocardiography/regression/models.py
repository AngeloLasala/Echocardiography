"""
Structure of the models used in automatic detection of keypoits in PLAX echocardiography.
"""
import torch
import torch.nn as nn
import torchvision
from torchvision import models
import torch.nn.functional as F
from pathlib import Path
from torchsummary import summary 

class ResNet50Regression(nn.Module):
    def __init__(self, 
                input_channels=3, 
                num_labels=12):
        super(ResNet50Regression, self).__init__()
        # Load the pre-trained ResNet50 model
        resnet50 = models.resnet50(weights='DEFAULT')
         
        # Add a new regression layer
        self.resnet50 = resnet50    
        self.resnet50.fc = nn.Sequential(nn.Linear(resnet50.fc.in_features, num_labels),
                                                    nn.Sigmoid())

    def forward(self, x):
        x = self.resnet50(x)
        x = x.view(x.size(0), -1)
        return x


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
    def __init__(self, num_classes):
        super().__init__()
        
        # Encoder
        self.e11 = nn.Conv2d(3, 64, kernel_size=3, padding=1) # output: 570x570x64
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

if __name__ == '__main__':
    # ## SimpleRegression model
    # model = ResNet50Regression()
    # x = torch.randn(1, 3, 256, 256)
    # print(model(x))


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

    # import matplotlib.pyplot as plt
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


    model = UNet_up(num_classes=6)
    print(summary(model))
    x = torch.randn(2, 3, 256, 256)
    pred = model(x).detach().numpy()
    plt.figure(num='Un_up')
    plt.imshow(pred[0, 0, :, :])
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