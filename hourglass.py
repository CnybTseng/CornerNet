# -*- coding: utf-8 -*-
# file: hourglass.py
# brief: Hourglass module.
# author: Zeng Zhiwei
# date: 2019/12/11

import torch
import torch.nn.functional as F

class Residual(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(Residual, self).__init__()
        padding = (kernel_size - 1) // 2
        
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bnrm1 = torch.nn.BatchNorm2d(out_channels)
        self.relu1 = torch.nn.ReLU(inplace=True)
        
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=padding, bias=False)
        self.bnrm2 = torch.nn.BatchNorm2d(out_channels)
        
        if stride != 1 or in_channels != out_channels:
            self.skip = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False)
        else:
            self.skip = torch.nn.Sequential()
        
        self.relu = torch.nn.ReLU(inplace=True)
    
    def forward(self, x):
        y = self.conv1(x)
        y = self.bnrm1(y)
        y = self.relu1(y)
        y = self.conv2(y)
        y = self.bnrm2(y)
        z = self.skip(x)
        return self.relu(y + z)

class Upsample(torch.nn.Module):
    def __init__(self, scale_factor):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
    
    def forward(self, x):
        return F.interpolate(input=x, scale_factor=self.scale_factor, mode='nearest')

class Hourglass(torch.nn.Module):
    def __init__(self, degree, num_features, repeats):
        super(Hourglass, self).__init__()
        ios_channels = num_features[0]  # input channels, output channels, skip block IO channels
        mid_channels = num_features[1]  # middle block IO channels
        ios_repeats = repeats[0]        # repeats of input, output and skip residual blocks
        mid_repeats = repeats[1]        # repeats of middle residual blocks
        
        # input residual blocks
        self.input  = [Residual(ios_channels, mid_channels, stride=2)]
        self.input += [Residual(mid_channels, mid_channels) for _ in range(ios_repeats-1)]
        self.input  = torch.nn.Sequential(*self.input)
        
        # middle hourglass module or residual blocks
        if degree > 1:
            self.middle = Hourglass(degree-1, num_features[1:], repeats[1:])
        else:
            self.middle = [Residual(mid_channels, mid_channels) for _ in range(mid_repeats)]
            self.middle = torch.nn.Sequential(*self.middle)
        
        # output residual blocks
        self.output  = [Residual(mid_channels, ios_channels) for _ in range(ios_repeats-1)]
        self.output += [Residual(ios_channels, ios_channels)]
        self.output += [Upsample(scale_factor=2)]
        self.output  = torch.nn.Sequential(*self.output)
        
        # skip residual blocks
        self.skip = [Residual(ios_channels, ios_channels) for _ in range(ios_repeats)]
        self.skip = torch.nn.Sequential(*self.skip)
        
    def forward(self, x):
        y = self.input(x)
        y = self.middle(y)
        y = self.output(y)
        z = self.skip(x)
        return y + z

if __name__ == '__main__':
    x = torch.rand(4,3,128,128)
    residual_block1 = Residual(3, 64)
    residual_block2 = Residual(3, 64, stride=2)
    y = residual_block1(x)
    print(f'residual_block1 input size {x.size()}, output size {y.size()}')
    y = residual_block2(x)
    print(f'residual_block2 input size {x.size()}, output size {y.size()}')
    
    x = torch.rand(4,256,128,128)
    hourglass = Hourglass(degree=5, num_features=[256,256,384,384,384,512], repeats=[2,2,2,2,2,4])
    y = hourglass(x)
    print(f'hourglass input size {x.size()}, output size {y.size()}')