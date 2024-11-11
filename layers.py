import torch
import torch.nn as nn
import torch.nn.functional as F


# Efficient Channel Attention Layer
class eca_layer(nn.Module):
    def __init__(self, channel, k_size=3):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1) # !! What's with the 1 here? 
        self.conv = nn.Conv1d(in_channels=1, out_channels=1, kernel_size = k_size, padding = (k_size-1)//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1,-2)).transpose(-1,-2).unsqueeze(-1) # !! 1Two branches of ECA
        y = self.sigmoid(y)

        return x * y.expand_as(x)


class NonLocalConvBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.g = nn.Conv3d(in_channels = in_channels, out_channels = in_channels//2, kernel_size=1)
        self.phi = nn.Conv3d(in_channels = in_channels, out_channels= in_channels//2, kernel_size= 1)
        self.theta = nn.Conv3d(in_channels = in_channels, out_channels = in_channels //2, kernel_size = 1)
        self.conv3d = nn.Conv3d(in_channels = in_channels // 2, out_channels = in_channels, kernel_size=1)
        
        self.max_pool = nn.MaxPool2d(kernel_size =2, stride = 2) 
    
    def forward(self, x):
        B, C, H, W = x.shape

        intermediate_channels = C // 2 


        ## !! should already work. Why unsqueeze again?
        theta  = self.theta(x.unsqueeze(-1))
        phi = self.phi(x.unsqueeze(-1))
        g = self.g(x.unsqueeze(-1))

        theta = theta.reshape(-1, intermediate_channels)
        phi = phi.reshape(-1, intermediate_channels)
        g = g.reshape(-1, intermediate_channels)

        theta_phi_matmul = F.softmax(torch.matmul(theta,phi), dim = 1)

        theta_phi_g_matmul = F.softmax(torch.matmul(theta_phi_matmul, g), dim = 1)

        final = self.conv3d((theta_phi_g_matmul.unsqueeze(-1))).squeeze(-1) + x

        return final

class SelfAttention(nn.Module):
    def __init__(self, channels, size):
        super().__init__()

        self.channels= channels
        self.size = size 
        
        self.num_heads = 4
        self.head_dim = channels // self.num_heads
        self.embed_dim = self.head_dim * self.num_heads         

        self.mha = nn.MultiheadAttention(self.embed_dim, self.num_heads, batch_first = True)
        self.ln = nn.LayerNorm([self.embed_dim])
        
        # feedforward net
        # !! Maybe use spectral normalization as well  
        self.ff_self = nn.Sequential(
            nn.LayerNorm([self.embed_dim]),
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, self.embed_dim)
        )

        self.project_in = nn.Linear(channels, self.embed_dim) if channels != self.embed_dim else nn.Identity()
        self.project_out = nn.Linear(self.embed_dim, channels) if channels != self.embed_dim else nn.Identity()

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.view(b, c, h*w).transpose(1,2) # !! why transpose? wouldn't (b, h*w, c) have worked the same way?  

        x = self.project_in( x )
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        attention_value = self.project_out(attention_value)

        return attention_value.transpose(1,2).view(b,c,h,w)

class Piece3DConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        
        # 1x3x3 convolution
        self.conv_1x3x3 = nn.Conv3d(in_channels, out_channels, 
                                    kernel_size=(1, kernel_size, kernel_size),
                                    stride=stride, 
                                    padding=(0, padding, padding))
        
        # 3x1x1 convolution
        self.conv_3x1x1 = nn.Conv3d(in_channels, out_channels, 
                                    kernel_size=(kernel_size, 1, 1),
                                    stride=stride, 
                                    padding=(padding, 0, 0))

    def forward(self, x):
        # Perform 1x3x3 convolution
        x = x.unsqueeze(-1)
        out_1x3x3 = self.conv_1x3x3(x)
        
        # Perform 3x1x1 convolution
        out_3x1x1 = self.conv_3x1x1(x)
        
        # Elementwise addition of the two convolution results
        out = out_1x3x3 + out_3x1x1
        out = out.squeeze(-1)
        return out

class spefe(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.piece1 = Piece3DConv(in_channels, in_channels)
        # self.leaky_relu = F.leaky_relu()
        self.piece2 = Piece3DConv(in_channels, in_channels)
    
    def forward(self, x):
        residual = x
        h = x.clone()
        h = self.piece1(h)
        h = F.leaky_relu(h, negative_slope=0.01)
        h = self.piece2(h)
        h = F.leaky_relu(h, negative_slope=0.01)
        h += residual

        return h

class DilatedConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation_rate):
        super(DilatedConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, 
                              padding=dilation_rate, dilation=dilation_rate)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))
