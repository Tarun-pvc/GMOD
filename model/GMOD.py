import torch.nn as nn
from model.network import *
from utils.data_utils import normalize_tensor


class GMOD(nn.Module):
    def __init__(self, num_bands, l_resolution=32, h_resolution = 64, use_nonlocal = False, use_ECA = False, use_attention = False):
        super().__init__()
        self.use_ECA = use_ECA
        self.use_nonlocal = use_nonlocal
        self.upscale_factor = int(h_resolution//l_resolution)
        self.sfeb = SFEB(num_bands, l_resolution, h_resolution, use_nonlocal=use_nonlocal, use_eca=use_ECA, use_attention= use_attention)

        self.bn_conv_sr3 = nn.BatchNorm2d(num_bands)
        self.bn_conv_final1 = nn.BatchNorm2d(num_bands)
        self.bn_conv_final3 = nn.BatchNorm2d(num_bands)
        self.bn_upconv = nn.BatchNorm2d(num_bands)
        self.bn_bicubic_conv = nn.BatchNorm2d(num_bands)

        self.conv_final1 = DilatedConvBlock(num_bands, num_bands, kernel_size=3, dilation_rate=1) # Reduced the dilation rate from 4 to 1

        self.conv_final3 = nn.Conv2d(num_bands, num_bands, kernel_size=3, padding=1, stride=1)
        
        self.upconv = nn.Conv2d(3, num_bands, kernel_size=1, stride = 1)
        self.bicubic_conv = nn.Conv2d(num_bands, num_bands, kernel_size = 3, padding = 1, stride = 1)

        if use_nonlocal:
            self.nonlocalblock = NonLocalConvBlock(num_bands)

        if use_ECA:
            self.sa1 = eca_layer(num_bands)
            self.sa2 = eca_layer(num_bands)
            self.sa3 = eca_layer(num_bands)

    

    def forward(self, x, sr3):
        sfeb_out = self.sfeb(x)
        if self.use_ECA:
            sfeb_out = self.sa1(sfeb_out)

        conv_sr3 = self.upconv(sr3)
        conv_sr3 = self.bn_upconv(conv_sr3) # batch normalization 

        combined = sfeb_out + conv_sr3

        if self.use_ECA:
            combined = self.sa2(combined)


        smoothed = F.leaky_relu(self.bn_conv_final1(self.conv_final1(combined)), negative_slope=0.01)
        smoothed = F.leaky_relu(self.bn_conv_final3(self.conv_final3(smoothed)), negative_slope=0.01)

        bicubic_upsampled = F.interpolate(x, scale_factor=self.upscale_factor, mode='bicubic', align_corners=False) 
        bicubic_upsampled = self.bicubic_conv(bicubic_upsampled)
        bicubic_upsampled = self.bn_bicubic_conv(bicubic_upsampled) # Might have to make a change here later 

        if self.use_nonlocal:
            bicubic_upsampled = self.nonlocalblock(bicubic_upsampled)
        if self.use_ECA:
            bicubic_upsampled = self.sa3(bicubic_upsampled)

        output = smoothed + bicubic_upsampled
        output = normalize_tensor(output)
        # print("output max: ", torch.max(output))
        
        return output
        # return transform2tensor(output)

