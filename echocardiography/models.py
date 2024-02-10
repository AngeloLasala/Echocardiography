"""
Structure of the models used in automatic detection of keypoits in PLAX echocardiography.
"""
import torch
import torchvision
from torchvision import models
from pathlib import Path
from torchsummary import summary 

class ResNet50Regression(torch.nn.Module):
    def __init__(self, 
                input_channels=3, 
                num_labels=12):
        super(ResNet50Regression, self).__init__()
        # Load the pre-trained ResNet50 model
        resnet50 = models.resnet50(weights='DEFAULT')
         
        # Add a new regression layer
        self.resnet50 = resnet50    
        self.resnet50.fc = torch.nn.Sequential(torch.nn.Linear(resnet50.fc.in_features, num_labels),
                                                    torch.nn.Sigmoid())

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

    def forward(self, x):
        return torch.sigmoid(self.model(x)['out'])

if __name__ == '__main__':
    ## SimpleRegression model
    model = ResNet50Regression()
    x = torch.randn(1, 3, 256, 256)
    print(model(x))


    ## PLAX model
    model = PlaxModel(num_classes=6)
    x = torch.randn(2, 3, 256, 256)
    model.train()
    print(model(x).shape)
    pred = model(x).detach().numpy()
    # print max and min values
    print(pred.max(), pred.min())
    print(pred.shape)

    import matplotlib.pyplot as plt
    plt.figure()
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