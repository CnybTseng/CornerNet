# -*- coding: utf-8 -*-
# file: cornernet.py
# brief: CornerNet.
# author: Zeng Zhiwei
# date: 2019/12/11

import torch
import hourglass

class Convolution(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, with_bn=True):
        super(Convolution, self).__init__()
        padding = (kernel_size - 1) // 2
        
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=not with_bn)
        self.bnrm = torch.nn.BatchNorm2d(out_channels) if with_bn else torch.nn.Sequential()
        self.relu = torch.nn.ReLU(inplace=True)
        
    def forward(self, x):
        y = self.conv(x)
        y = self.bnrm(y)
        return self.relu(y)

class CornerNet(torch.nn.Module):
    def __init__(self):
        super(CornerNet, self).__init__()
        self.degree = 5
        self.num_features = [256,256,384,384,384,512]
        self.repeats = [2,2,2,2,2,4]
        
        self.input  = [Convolution(in_channels=3, out_channels=128, kernel_size=7, stride=2, with_bn=False)]
        self.input += [hourglass.Residual(in_channels=128, out_channels=256, stride=2)]
        self.input  = torch.nn.Sequential(*self.input)
        
        self.hourglass = [hourglass.Hourglass(self.degree, self.num_features, self.repeats) for _ in range(2)]
        
        
    
    def forward(self, x):
        output = self.input(x)
        return output

if __name__ == '__main__':
    x = torch.rand(4, 3, 128, 128)
    net = CornerNet()
    print(net)
    y = net(x)
    print(f'net input size {x.size()}, output size{y.size()}')