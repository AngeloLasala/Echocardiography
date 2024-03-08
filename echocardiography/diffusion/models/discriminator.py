"""
PatchGAN Discriminator model for Adversarial loss function fro the VAE model
"""
import torch
import torch.nn as nn


class Discriminator(nn.Module):
    """
    PatchGAN Discriminator.
    """
    
    def __init__(self, im_channels=3,
                 conv_channels=[64, 128, 256],
                 kernels=[4,4,4,4],
                 strides=[2,2,2,1],
                 paddings=[1,1,1,1]):
        super().__init__()
        self.im_channels = im_channels
        activation = nn.LeakyReLU(0.2)
        layers_dim = [self.im_channels] + conv_channels + [1]
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(layers_dim[i], layers_dim[i + 1],
                          kernel_size=kernels[i],
                          stride=strides[i],
                          padding=paddings[i],
                          bias=False if i !=0 else True),
                nn.BatchNorm2d(layers_dim[i + 1]) if i != len(layers_dim) - 2 and i != 0 else nn.Identity(),
                activation if i != len(layers_dim) - 2 else nn.Identity()
            )
            for i in range(len(layers_dim) - 1)
        ])
    
    def forward(self, x):
        out = x
        # print('input', out.shape)
        for layer in self.layers:
            out = layer(out)
            # print(out.shape)

        return out


if __name__ == '__main__':
    x = torch.randn((2, 3, 256, 256))
    prob = Discriminator(im_channels=3)(x)
    print(prob.shape)