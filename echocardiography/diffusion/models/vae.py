"""
Variational Autoencorder (VAE) model design for the Latent Diffusion Model (LDM)
"""

import torch
import torch.nn as nn
from echocardiography.diffusion.models.blocks import DownBlock, MidBlock, UpBlock
from torchsummary import summary 


class VAE(nn.Module):
    def __init__(self, im_channels, model_config):
        super().__init__()
        self.down_channels = model_config['down_channels']
        self.mid_channels = model_config['mid_channels']
        self.down_sample = model_config['down_sample']
        self.num_down_layers = model_config['num_down_layers']
        self.num_mid_layers = model_config['num_mid_layers']
        self.num_up_layers = model_config['num_up_layers']
        
        # To disable attention in Downblock of Encoder and Upblock of Decoder
        self.attns = model_config['attn_down']
        
        # Latent Dimension
        self.z_channels = model_config['z_channels']
        self.norm_channels = model_config['norm_channels']
        self.num_heads = model_config['num_heads']
        
        # Assertion to validate the channel information
        assert self.mid_channels[0] == self.down_channels[-1]
        assert self.mid_channels[-1] == self.down_channels[-1]
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
                                                 t_emb_dim=None, down_sample=self.down_sample[i],
                                                 num_heads=self.num_heads,
                                                 num_layers=self.num_down_layers,
                                                 attn=self.attns[i],
                                                 norm_channels=self.norm_channels))

        
        #print('Mid Encoder Layers')
        self.encoder_mids = nn.ModuleList([])
        for i in range(len(self.mid_channels) - 1):
            
            #print(f'layer {i}) input channel {self.mid_channels[i]}, out channel Conv2d {self.mid_channels[i + 1]}')
            self.encoder_mids.append(MidBlock(self.mid_channels[i], self.mid_channels[i + 1],
                                              t_emb_dim=None,
                                              num_heads=self.num_heads,
                                              num_layers=self.num_mid_layers,
                                              norm_channels=self.norm_channels))
        
        #print('--------------------------------')
        
        #print('MEAN AND STD')
        self.encoder_norm_out = nn.GroupNorm(self.norm_channels, self.down_channels[-1])
        
        #print(f'GroupNorm {self.norm_channels}, last out channel Conv2d {self.down_channels[-1]}')
        self.encoder_conv_out = nn.Conv2d(self.down_channels[-1], 2*self.z_channels, kernel_size=3, padding=1)
        
        #print(f'Encoder Output: input:{self.down_channels[-1]}, output:{2*self.z_channels}')
        # Latent Dimension is 2*Latent because we are predicting mean & variance
        self.pre_quant_conv = nn.Conv2d(2*self.z_channels, 2*self.z_channels, kernel_size=1)
        
        #print(f'Pre Quant Conv: input:{2*self.z_channels}, output:{2*self.z_channels}')
        
        #print('--------------------------------')   
        ####################################################
        
        
        ##################### Decoder ######################
        
        #print('DECODER')
        self.post_quant_conv = nn.Conv2d(self.z_channels, self.z_channels, kernel_size=1)
        
        #print(f'Post Quant Conv: intput:{self.z_channels}, output:{self.z_channels}')
        self.decoder_conv_in = nn.Conv2d(self.z_channels, self.mid_channels[-1], kernel_size=3, padding=(1, 1))
        
        #print(f'Decoder conv in, input channel {self.z_channels}, first out channel Conv2d {self.mid_channels[-1]}')
        
        # Midblock + Upblock
        
        #print('Mid Decoder Layers')
        self.decoder_mids = nn.ModuleList([])
        for i in reversed(range(1, len(self.mid_channels))):
            
            #print(f'layer {i}) input channel {self.mid_channels[i]}, out channel Conv2d {self.mid_channels[i - 1]}')
            self.decoder_mids.append(MidBlock(self.mid_channels[i], self.mid_channels[i - 1],
                                              t_emb_dim=None,
                                              num_heads=self.num_heads,
                                              num_layers=self.num_mid_layers,
                                              norm_channels=self.norm_channels))

        
        #print('Decoder Layers')
        self.decoder_layers = nn.ModuleList([])
        for i in reversed(range(1, len(self.down_channels))):
            
            #print(f'layer {i}) input channel {self.down_channels[i]}, out channel Conv2d {self.down_channels[i - 1]}')
            self.decoder_layers.append(UpBlock(self.down_channels[i], self.down_channels[i - 1],
                                               t_emb_dim=None, up_sample=self.down_sample[i - 1],
                                               num_heads=self.num_heads,
                                               num_layers=self.num_up_layers,
                                               attn=self.attns[i - 1],
                                               norm_channels=self.norm_channels))
        
        self.decoder_norm_out = nn.GroupNorm(self.norm_channels, self.down_channels[0])
        
        #print(f'GroupNorm {self.norm_channels}, last out channel Conv2d {self.down_channels[0]}')
        self.decoder_conv_out = nn.Conv2d(self.down_channels[0], im_channels, kernel_size=3, padding=1)
        
        #print(f'output channel Conv2d {im_channels}')
    
    def encode(self, x):
        out = self.encoder_conv_in(x)
        
        #print('first Conv2d',out.shape)
        for idx, down in enumerate(self.encoder_layers):
            out = down(out)
            
            #print(f'Encoder layer {idx})',out.shape)
        for mid in self.encoder_mids:
            out = mid(out)
            
            #print(f'Encoder mid layer',out.shape)
        out = self.encoder_norm_out(out)
        
        #print('GroupNorm',out.shape)
        out = nn.SiLU()(out)
        out = self.encoder_conv_out(out)
        
        #print('Encoder Output',out.shape)
        out = self.pre_quant_conv(out)
        
        #print('Pre Quant Conv',out.shape)
        mean, logvar = torch.chunk(out, 2, dim=1)
        std = torch.exp(0.5 * logvar)
        sample = mean + std * torch.randn(mean.shape).to(device=x.device)
        return sample, out
    
    def decode(self, z):
        out = z
        
        #print('Decoder Input',out.shape)
        out = self.post_quant_conv(out)
        
        #print('Post Quant Conv',out.shape)
        out = self.decoder_conv_in(out)
        
        #print('Decoder Conv In',out.shape)
        for idx, mid in enumerate(self.decoder_mids):
            out = mid(out)
            
            #print(f'Decoder mid layer {idx})',out.shape)
        for idx, up in enumerate(self.decoder_layers):
            out = up(out)
            
            #print(f'Decoder layer {idx})',out.shape)

        out = self.decoder_norm_out(out)
        
        #print('GroupNorm',out.shape)
        out = nn.SiLU()(out)
        out = self.decoder_conv_out(out)
        
        #print('Decoder Output',out.shape)
        return out

    def forward(self, x):
        
        #print('FORWARD PASS')
        
        #print('input',x.shape)
        z, encoder_output = self.encode(x)
        out = self.decode(z)
        
        #print('output',out.shape)
        return out, encoder_output

if __name__ == '__main__':
    model_config = {
        'z_channels': 2,
        'down_channels': [32, 64, 128, 256],
        'mid_channels': [256, 256],
        'down_sample': [True, True, True],
        'attn_down': [False, False, False],
        'norm_channels': 32,
        'num_heads': 16,
        'num_down_layers': 1,
        'num_mid_layers': 1,
        'num_up_layers': 1
    }
    model = VAE(1, model_config)
    x = torch.randn(1, 1, 256, 256)
    out = model(x)
    # 
    # #print(summary(model))
