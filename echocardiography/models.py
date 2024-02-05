"""
Structure of the models used in automatic detection of keypoits in PLAX echocardiography.
"""
import torch
import torchvision
from pathlib import Path
from torchsummary import summary 


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

model = PlaxModel()
# print(summary(model, (3, 224, 224)))
sequential_block = model.model.backbone  # Replace with the specific Sequential block you want to inspect
print(sequential_block)

# Iterate through the layers inside the Sequential block
# for layer_idx, layer in enumerate(sequential_block.children()):
#     print(f"Layer {layer_idx + 1}: {layer}")