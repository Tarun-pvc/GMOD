import torch.nn as nn
from network import *

l_resolution = 32
h_resolution = 64
scale = int(h_resolution//l_resolution)

class GMOD(nn.Module):
    def __init__(self, num_bands, upscale_factor, use_nonlocal = False, use_ECA = False):
        super().__init__()
        self.use_ECA = use_ECA
        self.use_nonlocal = use_nonlocal
        self.nonlocalblock = NonLocalConvBlock(num_bands)
        self.sfeb = SFEB(num_bands, upscale_factor, use_nonlocal=use_nonlocal)
        self.conv_sr3 = nn.Conv2d(in_channels=3, out_channels=num_bands, kernel_size=3, padding=1, stride=1)
        # self.conv_final1 = nn.Conv2d(num_bands, num_bands, kernel_size=3, padding=1, stride=1)
        self.conv_final1 = DilatedConvBlock(num_bands, num_bands, kernel_size=3, dilation_rate=4)
        # self.conv_final2 = nn.Conv2d(num_bands, num_bands, kernel_size=3, padding=1, stride=1)
        # self.conv_final2 = DilatedConvBlock(num_bands, num_bands, kernel_size=3, dilation_rate=2)
        self.conv_final3 = nn.Conv2d(num_bands, num_bands, kernel_size=3, padding=1, stride=1)
        # self.sr3_weights_path = sr3_weights_path

        self.sa1 = eca_layer(num_bands)
        # self.sa2 = SelfAttention(num_bands, l_resolution)
        self.sa2 = eca_layer(num_bands)
        self.sa3 = eca_layer(num_bands)

        self.upconv = nn.Conv2d(3, num_bands, kernel_size=1, stride = 1)
        self.bicubic_conv = nn.Conv2d(num_bands, num_bands, kernel_size = 3, padding = 1, stride = 1)
    

    def forward(self, x, sr3):
        sfeb_out = self.sfeb(x)
        if self.use_ECA:
            sfeb_out = self.sa1(sfeb_out)

        # sr3_out = SR3(perform_pca(x), opt, weights_path=self.sr3_weights_path)
        # sr3_out = generator(torch.tensor(sfeb_out.cpu(), dtype = (torch.float)))
        # sr3_out = sr3_out.cuda()
        # sr3_out = sr3_out[1].to(device)
        # sr3_out = self.sa2(sr3_out)
        # combined = sfeb_out + 0.01*self.conv_sr3(sr3_out.to("cuda"))
        conv_sr3 = self.upconv(sr3)
        # conv_sr3 = F.relu(conv_sr3)
        combined = sfeb_out + conv_sr3
        # combined = F.relu(combined)

        if self.use_ECA:
            combined = self.sa2(combined)


        smoothed = F.leaky_relu(self.conv_final1(combined), negative_slope=0.01)
        # smoothed = F.leaky_relu(self.conv_final2(smoothed), negative_slope=0.01)
        smoothed = F.leaky_relu(self.conv_final3(smoothed), negative_slope=0.01)

        bicubic_upsampled = F.interpolate(x, scale_factor=scale, mode='bicubic', align_corners=True)

        bicubic_upsampled = self.bicubic_conv(bicubic_upsampled)+bicubic_upsampled
        if self.use_nonlocal:
            bicubic_upsampled = self.nonlocalblock(bicubic_upsampled)
        if self.use_ECA:
            bicubic_upsampled = self.sa3(bicubic_upsampled)

        output = smoothed + bicubic_upsampled
        return output
