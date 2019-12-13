# -*- coding: utf-8 -*-
# file: cornerpool.py
# brief: Corner pooling.
# author: Zeng Zhiwei
# date: 2019/12/12

import torch

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

class CornerPool(torch.nn.Module):
    def __init__(self, in_channels, pool1, pool2):
        super(CornerPool, self).__init__()
        
        self.pconv1 = Convolution(in_channels, out_channels=128, kernel_size=3, stride=1)
        self.pconv2 = Convolution(in_channels, out_channels=128, kernel_size=3, stride=1)
        
        self.merge1  = [torch.nn.Conv2d(in_channels=128, out_channels=in_channels, kernel_size=3, stride=1, padding=1, bias=False)]
        self.merge1 += [torch.nn.BatchNorm2d(num_features=in_channels)]
        self.merge1 = torch.nn.Sequential(*self.merge1)
        
        self.skip  = [torch.nn.Conv2d(in_channels, out_channels=in_channels, kernel_size=1, stride=1, padding=0, bias=False)]
        self.skip += [torch.nn.BatchNorm2d(num_features=in_channels)]
        self.skip = torch.nn.Sequential(*self.skip)
        
        self.merge2  = [torch.nn.ReLU(inplace=True)]
        self.merge2 += [Convolution(in_channels, out_channels=in_channels, kernel_size=3, stride=1)]
        self.merge2 = torch.nn.Sequential(*self.merge2)
        
        self.pool1 = pool1()
        self.pool2 = pool2()
    
    def forward(self, x):
        y = self.pconv1(x)
        y = self.pool1(y)
        z = self.pconv2(x)
        z = self.pool2(z)
        m = self.merge1(y + z)
        s = self.skip(x)
        return self.merge2(m + s)

if __name__ == '__main__':
    x = torch.rand(4, 256, 128, 128)
    cpool = CornerPool(in_channels=256)
    y = cpool(x)
    print(f'corner pool input size {x.size()}, output size {y.size()}')