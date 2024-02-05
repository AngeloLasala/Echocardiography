"""
Structure of the models used in automatic detection of keypoits in PLAX echocardiography.
"""
import torch
import torchvision
from torchvision import models
from pathlib import Path
from torchsummary import summary 

class ResNet50Regression(torch.nn.Module):
    def __init__(self, input_channels=3, num_labels=6):
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

    def __init__(self, 
            measurements=['LVPW', 'LVID', 'IVS'], 
        ) -> None:
        super().__init__()
        self.model = torchvision.models.segmentation.deeplabv3_resnet50(num_classes=len(measurements) + 1)

    def forward(self, x):
        return torch.sigmoid(self.model(x)['out'])


### SimpleRegression model
model = ResNet50Regression()
x = torch.randn(1, 3, 256, 256)
print(model(x))


### PLAX model
# model = PlaxModel()
# # print(summary(model, (3, 224, 224)))
# sequential_block = model.model.backbone  # Replace with the specific Sequential block you want to inspect
# print(sequential_block)

# Iterate through the layers inside the Sequential block
# for layer_idx, layer in enumerate(sequential_block.children()):
#     print(f"Layer {layer_idx + 1}: {layer}")