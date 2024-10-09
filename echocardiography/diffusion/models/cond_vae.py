"""
Conditional Variational Autoencorder (VAE) model design for the Latent Diffusion Model (LDM)
"""

import torch
import torch.nn as nn
from echocardiography.diffusion.models.blocks import DownBlock, MidBlock, UpBlock
from torchsummary import summary 


class condVAE(nn.Module):
    def __init__(self, im_channels, model_config, condition_config):
        super().__init__()
        self.down_channels = model_config['down_channels']
        self.mid_channels = model_config['mid_channels']
        self.down_sample = model_config['down_sample']
        self.num_down_layers = model_config['num_down_layers']
        self.num_mid_layers = model_config['num_mid_layers']
        self.num_up_layers = model_config['num_up_layers']
        self.condition_config = condition_config

        # self.class_conditioning = True if condition_config['condition_types'] is not None else False
        # if 'num_classes' in condition_config['class_condition_config']:
        #     self.num_classes = condition_config['class_condition_config']['num_classes']
        # if 'image_condition_input_channels' in condition_config['class_condition_config']:
        #     self.num_classes = condition_config['class_condition_config']['image_condition_input_channels']
        # else:
        #     ValueError('Number of classes not provided in class_condition_config')
        
        ## conditioning types
        self.class_conditioning = False
        self.image_conditioning = False
        assert 'condition_types' in self.condition_config, 'Condition Type not provided in model config'
        condition_types = self.condition_config['condition_types']
        if 'class' in condition_types:
            self.class_conditioning = True
            self.num_classes = self.condition_config['class_condition_config']['num_classes']
        if 'image' in condition_types:
            self.image_conditioning = True
            self.im_cond_input_ch = self.condition_config['image_condition_config']['image_condition_input_channels']
            self.im_cond_output_ch = self.condition_config['image_condition_config']['image_condition_output_channels']
            

        print(f'Class conditioning: {self.class_conditioning}')
        print(f'Image conditioning: {self.image_conditioning}')
        
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
        ## here i have to change the input channel of the first conv becouse i have to stack the class conditioninh
        # im_channels = im_channels + num_classes
        if self.class_conditioning == True:
            print('sto dentro encoder class conditioning')
            self.encoder_conv_in = nn.Conv2d(im_channels + self.num_classes, self.down_channels[0], kernel_size=3, padding=(1, 1))
        elif self.image_conditioning == True:
            print('sto dentro encoder image conditioning')
            # Map the mask image to a N channel image and
            # concat that with input across channel dimension
            # similar to the Unet cond acrctitecture for the DDPM
            self.cond_conv_in = nn.Conv2d(in_channels=self.im_cond_input_ch,
                                          out_channels=self.im_cond_output_ch,
                                          kernel_size=1,
                                          bias=False)
            self.encoder_conv_in = nn.Conv2d(im_channels + self.im_cond_output_ch, self.down_channels[0], kernel_size=3, padding=(1, 1))
        else:
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
        ## Also the decoder should take the class conditioning, so i have to stacjìk the condition to the input of the decoder
        if self.class_conditioning == True: 
            self.post_quant_conv = nn.Conv2d(self.z_channels + self.num_classes, self.z_channels, kernel_size=1)
        elif self.image_conditioning == True:
            self.post_quant_conv = nn.Conv2d(self.z_channels + self.im_cond_output_ch, self.z_channels, kernel_size=1)
        else:
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
    
    def encode(self, x, y):
        """
        The encoder takes twuo input, the image x and the class y (one_hot).
        the input of the encoder is the stack of the x and the reshape of the y to the same shape of x
        """
        #print('Encoder Input',x.shape) 
        if self.class_conditioning == True:
            y = y.view(-1, self.num_classes, 1, 1)
            y = y.repeat(1, 1, x.shape[2], x.shape[3])
            x = torch.cat([x, y], dim=1)

        if self.image_conditioning == True:
            # resize y to the same shape of xonly for the -2 and -1 dimentions
            y = nn.functional.interpolate(y, size=x.shape[-2:], mode='bilinear', align_corners=False)
            y = self.cond_conv_in(y)
            x = torch.cat([x, y], dim=1)
    
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
    
    def decode(self, z, y):
        """
        The decoder takes two input, the latent z and the class y (one_hot).
        the input of the decoder is the stack of the z and the reshape of the y to the same shape of z
        """
        if self.class_conditioning == True:
            y = y.view(-1, self.num_classes, 1, 1)
            y = y.repeat(1, 1, z.shape[2], z.shape[3])
            z = torch.cat([z, y], dim=1)

        if self.image_conditioning == True:
            # resize y to the same shape of z only for the -2 and -1 dimentions
            y = nn.functional.interpolate(y, size=z.shape[-2:], mode='bilinear', align_corners=False)
            y = self.cond_conv_in(y)
            z = torch.cat([z, y], dim=1)
        
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
        
        # print('Decoder Output',out.shape)
        return out

    def forward(self, x, y):
        
        # print('FORWARD PASS')
        
        # print('input',x.shape)
        z, encoder_output = self.encode(x, y)
        out = self.decode(z, y)
        # print('z',z.shape)
        # print('encoder_output',encoder_output.shape)    
        # print('decoder output',out.shape)
        return out, encoder_output

if __name__ == '__main__':
    model_config = {
        'z_channels': 3,
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

    condition_class = {
        'condition_types': ['class'],
        'class_condition_config': {
            'num_classes': 4
        }
    }

    condition_image = {
        'condition_types': ['image'],
        'image_condition_config': {
            'image_condition_input_channels': 6,
            'image_condition_output_channels': 3
        }
    }   

    model = condVAE(1, model_config, condition_image)
    x = torch.randn(2, 1, 240, 320)
    class_x = torch.tensor([[0,1,0,0], [0,0,1,0]])
    image_x = torch.randn(2, 6, 240, 320)
    out = model(x, image_x)
    print(out[0].shape)
    
    # #print(summary(model))
