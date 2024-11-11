from layers import *
import torch

l_resolution = 32
h_resolution = 64

class MSAB(nn.Module):
    def __init__(self, in_channels, use_nonlocal=False, use_spefe = False):
        super().__init__()
        self.use_nonlocal = use_nonlocal
        self.conv2d1 = DilatedConvBlock(in_channels, in_channels, kernel_size=3, dilation_rate=1)
        self.conv2d2 = DilatedConvBlock(in_channels, in_channels, kernel_size=3, dilation_rate=2)
        
        self.use_spefe = use_spefe

        self.conv3d1 = nn.Conv3d(in_channels, in_channels, kernel_size=(3,1,1), padding=(1,0,0))
        self.conv3d2 = nn.Conv3d(in_channels, in_channels, kernel_size=(3,1,1), padding=(1,0,0))
        self.conv3d3 = nn.Conv3d(in_channels, in_channels, kernel_size=(3,1,1), padding=(1,0,0))
        self.conv3d4 = nn.Conv3d(in_channels, in_channels, kernel_size=(5,1,1), padding=(2,0,0))
        self.conv3d5 = nn.Conv3d(in_channels, in_channels, kernel_size=(5,1,1), padding=(2,0,0))
        self.conv3d6 = nn.Conv3d(in_channels, in_channels, kernel_size=(5,1,1), padding=(2,0,0))
        self.spefe = spefe(in_channels)

        # self.nonlocalconv = NonLocalConvBlock(in_channels)

    def forward(self, F_in):
        h = F_in.clone()
        h = self.conv2d1(h)
        # h = F.relu(h)
        h = self.conv2d2(h)
        # h = F.relu(h)



        conv3image = F.leaky_relu(self.conv3d1(h.unsqueeze(-1)), negative_slope=0.01)
        conv3image = F.leaky_relu(self.conv3d2(conv3image), negative_slope=0.01)
        conv3image = F.leaky_relu(self.conv3d3(conv3image), negative_slope=0.01)

        conv5image = F.leaky_relu(self.conv3d4(h.unsqueeze(-1)), negative_slope=0.01)
        conv5image = F.leaky_relu(self.conv3d5(conv5image), negative_slope=0.01)
        conv5image = F.leaky_relu(self.conv3d6(conv5image), negative_slope=0.01)

        h = self.spefe(h)

        F_mid = conv3image.squeeze(-1) + conv5image.squeeze(-1)

        # if self.use_nonlocal:
        #     F_mid = self.nonlocalconv(F_mid)

        F_mid = F.softmax(F_mid, dim=1)
        F_out = F_mid * h
        return F_out



class SubNetwork(nn.Module):
    def __init__(self, in_channels, upscale_factor=1, use_nonlocal = False):
        super().__init__()
        self.msab1 = MSAB(in_channels)
        # self.attention1 = SelfAttention(in_channels, l_resolution)
        self.attention1 = eca_layer(in_channels)
        self.msab2 = MSAB(in_channels)
        # self.attention1 = eca_layer(in_channels
                                    # )
        self.msab3 = MSAB(in_channels, use_nonlocal)
        self.msab4 = MSAB(in_channels, use_nonlocal)
        self.msab5 = MSAB(in_channels, use_nonlocal)
        self.msab6 = MSAB(in_channels, use_nonlocal)
        self.msab7 = MSAB(in_channels, use_nonlocal)
        self.msab8 = MSAB(in_channels, use_nonlocal)
        self.conv = nn.Conv2d(in_channels, in_channels * upscale_factor**2, kernel_size=3, stride=1, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

        self.relu = nn.ReLU()

    def forward(self, x):
        h = x
        m1 = self.msab1(h)
        # m1 = F.relu(m1)
        m2 = self.msab2(m1)
        m2 += m1
        m2 = self.relu(m2)
        # m2 = self.attention1(m2)
        m3 = self.msab3(m2)
        m3 += m2
        m3 = self.relu(m3)
        # m3 = self.attention1(m3)
        m4 = self.msab4(m3)
        m4 += m3
        m4 = self.relu(m4)

        m5 = self.msab5(m4)
        m5 += m4
        m5 = self.relu(m5)

        m6 = self.msab6(m5)
        m6 += m5
        m6 = self.relu(m6)

        m7 = self.msab7(m6)
        m7 = self.relu(m7)
        m8 = self.msab8(m7)

        m8 += x
        m8 = self.relu(m8)
        m8 = self.conv(m8)
        m8 = self.pixel_shuffle(m8)
        # m8 = F.relu(m8)
        return m8

class Trunk(nn.Module):
    def __init__(self, in_channels, out_channels, upscale_factor = 2):
        super().__init__()
        self.inconv = Piece3DConv(in_channels, in_channels)
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=1)

        # self.sa1 = SelfAttention(in_channels, l_resolution)
        self.sa1 = eca_layer(in_channels)
        self.sa2 = SelfAttention(in_channels, l_resolution)
        self.sa3 = SelfAttention(in_channels, l_resolution)
        # self.sa4 = SelfAttention(in_channels, l_resolution)
        
        self.msab1 = MSAB(in_channels)
        self.msab2 = MSAB(in_channels)
        self.msab3 = MSAB(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels * upscale_factor**2, kernel_size=3, stride=1, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    # Changes made: 
    def forward(self, x):
        h = self.inconv(x)
        h = self.conv1(h)
        # h = F.relu(h)
        m1 = self.msab1(h)
        # m1 = F.relu(m1)
        # m1 = self.sa1(m1)
        m2 = self.msab2(m1)
        # m2 = F.relu(m2)
        m2 = self.sa2(m2)
        m3 = self.msab3(m2)
        # m3 = F.relu(m3)
        m3 = self.sa3(m3)
        
        m3 += x
        m3 += m2
        m3 += m1
        m3 = self.conv2(m3)
        h = m3
        h = self.conv(h)
        h = self.pixel_shuffle(h)
        # h = F.relu(h)
        # h = self.sa4(h)
        return h

class SFEB(nn.Module):
    def __init__(self, num_bands, upscale_factor, use_nonlocal = False):
        super().__init__()
        self.use_nonlocal = use_nonlocal
        self.number_of_groups = 6
        self.bands_per_group = num_bands // self.number_of_groups
        self.group_nets = nn.ModuleList([
            SubNetwork(self.bands_per_group, use_nonlocal=use_nonlocal) for _ in range(self.number_of_groups)
        ])
        self.trunk = Trunk(self.number_of_groups * self.bands_per_group, num_bands)
        self.nonlocalblock = NonLocalConvBlock(num_bands)

    def forward(self, x):

        group_outputs = [self.group_nets[i](x[:, i*self.bands_per_group:(i+1)*self.bands_per_group, :, :])
                         for i in range(self.number_of_groups)]
        concatenated_output = torch.cat(group_outputs, dim=1)

        final_output = self.trunk(concatenated_output)
        return final_output
