
import torch
import torch.nn as nn

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding, 
            groups=in_channels,  
            bias=False
        )
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, 
            kernel_size=1, 
            stride=1, 
            padding=0, 
            bias=True
        )
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class DepthwiseSeparableDeconvPixelShuffle(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=None):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        
        expand_ratio = stride * stride
        intermediate_channels = in_channels * expand_ratio
        
        self.pointwise_expand = nn.Conv2d(
            in_channels, intermediate_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )
        
        self.pixel_shuffle = nn.PixelShuffle(stride)
        
        # Depthwise convolution
        self.depthwise = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            groups=in_channels,
            bias=False
        )
        
        self.pointwise_final = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True
        )
        
    
    def forward(self, x):
        x = self.pointwise_expand(x)   
        x = self.pixel_shuffle(x)       
        x = self.depthwise(x)           
        x = self.pointwise_final(x)     
        return x