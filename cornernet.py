# -*- coding: utf-8 -*-
# file: cornernet.py
# brief: CornerNet.
# author: Zeng Zhiwei
# date: 2019/12/11

import torch
import hourglass
import cornerpool
from pool import TopPool, LeftPool, BottomPool, RightPool

class Intermediate(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Intermediate, self).__init__()
        self.resi = hourglass.Residual(in_channels, out_channels)        
        self.input = self.__convbn(in_channels, out_channels)        
        self.output = self.__convbn(in_channels, out_channels)       
        self.relu = torch.nn.ReLU(inplace=True)
    
    def forward(self, x, y):
        a = self.input(x)
        b = self.output(y)
        c = self.relu(a + b)
        return self.resi(c)
    
    def __convbn(self, in_channels, out_channels, kernel_size=1, stride=1):
        padding = (kernel_size - 1) // 2
        modules  = [torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)]
        modules += [torch.nn.BatchNorm2d(num_features=256)]
        return torch.nn.Sequential(*modules)

class CornerNet(torch.nn.Module):
    def __init__(self, classes=80):
        super(CornerNet, self).__init__()
        self.degree = 5
        self.num_features = [256,256,384,384,384,512]
        self.repeats = [2,2,2,2,2,4]
        
        # backbone
        
        self.input  = [cornerpool.Convolution(in_channels=3, out_channels=128, kernel_size=7, stride=2)]
        self.input += [hourglass.Residual(in_channels=128, out_channels=256, stride=2)]
        self.input  = torch.nn.Sequential(*self.input)

        self.hourglass = [hourglass.Hourglass(self.degree, self.num_features, self.repeats) for _ in range(2)]
        self.hourglass = torch.nn.Sequential(*self.hourglass)
        
        self.outputs = [cornerpool.Convolution(in_channels=256, out_channels=256, kernel_size=3, stride=1) for _ in range(2)]
        self.outputs = torch.nn.Sequential(*self.outputs)
        
        self.intermediates = [Intermediate(in_channels=256, out_channels=256)]
        self.intermediates = torch.nn.Sequential(*self.intermediates)
        
        # prediction module
        
        self.tl_pools = [cornerpool.CornerPool(in_channels=256, pool1=TopPool, pool2=LeftPool) for _ in range(2)]
        self.tl_pools = torch.nn.Sequential(*self.tl_pools)
        self.br_pools = [cornerpool.CornerPool(in_channels=256, pool1=BottomPool, pool2=RightPool) for _ in range(2)]
        self.br_pools = torch.nn.Sequential(*self.br_pools)
        
        self.tl_heats = [self.__predict(in_channels=256, mid_channels=256, out_channels=classes) for _ in range(2)]
        self.tl_heats = torch.nn.Sequential(*self.tl_heats)        
        self.br_heats = [self.__predict(in_channels=256, mid_channels=256, out_channels=classes) for _ in range(2)]
        self.br_heats = torch.nn.Sequential(*self.br_heats)
        
        self.tl_embds = [self.__predict(in_channels=256, mid_channels=256, out_channels=1) for _ in range(2)]
        self.tl_embds = torch.nn.Sequential(*self.tl_embds)
        self.br_embds = [self.__predict(in_channels=256, mid_channels=256, out_channels=1) for _ in range(2)]
        self.br_embds = torch.nn.Sequential(*self.br_embds)
        
        self.tl_offss = [self.__predict(in_channels=256, mid_channels=256, out_channels=2) for _ in range(2)]
        self.tl_offss = torch.nn.Sequential(*self.tl_offss)
        self.br_offss = [self.__predict(in_channels=256, mid_channels=256, out_channels=2) for _ in range(2)]
        self.br_offss = torch.nn.Sequential(*self.br_offss)
    
    def forward(self, x):
        y = self.__backbone(x)

        tl_pool = self.tl_pools[-1](y[-1])
        tl_heat = self.tl_heats[-1](tl_pool)
        tl_embd = self.tl_embds[-1](tl_pool)
        tl_offs = self.tl_offss[-1](tl_pool)
        
        br_pool = self.br_pools[-1](y[-1])
        br_heat = self.br_heats[-1](br_pool)
        br_embd = self.br_embds[-1](br_pool)
        br_offs = self.br_offss[-1](br_pool)
        
        return (tl_heat, br_heat, tl_embd, br_embd, tl_offs, br_offs)
    
    def __backbone(self, x):
        x = self.input(x)
        outputs = []
        for i, (hg, out) in enumerate(zip(self.hourglass, self.outputs)):
            y = hg(x)
            y = out(y)
            outputs.append(y)
            if i == len(self.hourglass) - 1:
                break
            x = self.intermediates[i](x, y)           
        return outputs
    
    def __predict(self, in_channels, mid_channels, out_channels):
        modules =  [cornerpool.Convolution(in_channels, out_channels=mid_channels, kernel_size=3, stride=1, with_bn=False)]
        modules += [torch.nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=True)]
        return torch.nn.Sequential(*modules)

class CornerDecoder(torch.nn.Module):
    def __init__(self, K=100):
        super(CornerDecoder, self).__init__()
        self.K = 100
        self.ae_thresh = 0.5
        self.num_dets = 1000

    def forward(self, x):
        tl_heat, br_heat, tl_embd, br_embd, tl_offs, br_offs = x
        
        tl_heat = torch.sigmoid(tl_heat)    # 1 * classes * netw/4 * neth/4
        br_heat = torch.sigmoid(br_heat)        
        tl_heat = self.__nms(tl_heat, kernel_size=3)
        br_heat = self.__nms(br_heat, kernel_size=3)
        
        tl_scores, tl_cs, tl_ys, tl_xs = self.__topk(tl_heat, self.K)
        br_scores, br_cs, br_ys, br_xs = self.__topk(br_heat, self.K)
        
        tl_tags = tl_embd[0][0, tl_ys, tl_xs]  
        br_tags = br_embd[0][0, br_ys, br_xs]
        
        tl_biases = tl_offs[0][:, tl_ys, tl_xs]
        br_biases = br_offs[0][:, br_ys, br_xs]
        
        tl_ys = tl_ys.int().float()
        tl_xs = tl_xs.int().float()
        br_ys = br_ys.int().float()
        br_xs = br_xs.int().float()
        
        tl_ys += tl_biases[1]
        tl_xs += tl_biases[0]
        br_ys += br_biases[1]
        br_xs += br_biases[0]
        
        tl_ys = tl_ys.view(-1, 1).expand(self.K, self.K).contiguous().view(1, -1)
        tl_xs = tl_xs.view(-1, 1).expand(self.K, self.K).contiguous().view(1, -1)
        br_ys = br_ys.expand(self.K, self.K).contiguous().view(1, -1)
        br_xs = br_xs.expand(self.K, self.K).contiguous().view(1, -1)        
        
        # reject object based on embedding distance
        tl_tags = tl_tags.view(-1, 1).expand(self.K, self.K).contiguous().view(1, -1)
        br_tags = br_tags.expand(self.K, self.K).contiguous().view(1, -1)
        dists = torch.abs(tl_tags - br_tags)
        dist_ids = dists > self.ae_thresh
        
        tl_scores = tl_scores.view(-1, 1).expand(self.K, self.K).contiguous().view(1, -1)
        br_scores = br_scores.expand(self.K, self.K).contiguous().view(1, -1)
        scores = (tl_scores + br_scores) / 2
        
        # reject object based on class index
        tl_cs = tl_cs.view(-1, 1).expand(self.K, self.K).contiguous().view(1, -1)
        br_cs = br_cs.expand(self.K, self.K).contiguous().view(1, -1)
        class_ids = tl_cs != br_cs
        
        # reject object based on size
        width_ids = tl_xs > br_xs
        height_ids = tl_ys > br_ys
        
        scores[dist_ids]   = -1
        scores[class_ids]  = -1
        scores[width_ids]  = -1
        scores[height_ids] = -1
        
        scores, indices = torch.topk(scores, self.num_dets)
        bboxes = torch.cat((tl_xs[0,indices], tl_ys[0,indices], br_xs[0,indices], br_ys[0,indices]), dim=0)
        
        print(f'bboxes size {bboxes.size()}')
        print(f'{tl_xs.size()} {tl_ys.size()} {br_xs.size()} {br_ys.size()}')
        print(f'{tl_tags.size()} {br_tags.size()}')
        print(f'{dist_ids.size()} {class_ids.size()} {width_ids.size()} {height_ids.size()}')
        
        return None

    def __nms(self, x, kernel_size=3):
        padding = (kernel_size - 1) // 2
        y = torch.nn.functional.max_pool2d(x, kernel_size, stride=1, padding=padding)
        mask = (x == y).float()
        return mask * x
    
    def __topk(self, x, K=100):
        n, c, h, w = x.size()
        y = x.view(n, -1)
        scores, indices = torch.topk(y, K)
        cs = indices / (w * h)
        indices = indices % (w * h)
        ys = indices / w
        xs = indices % w
        return scores, cs, ys, xs

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.load('model/dog.pth')
    net = CornerNet().to(device)
    net.load_state_dict(torch.load('model/cornernet.pth'))
    net.eval()
    print(net)
    ys = net(x)
    print(f'net input size {x.size()}')
    for y in ys:
        print(f'net output size {y.size()}')
    
    refr = torch.load('model/net_output.pth')
    for r, t in zip(refr, ys):
        print(f'reference output is equal to test output? {torch.equal(r, t)}')
    
    decoder = CornerDecoder()
    zs = decoder(ys)