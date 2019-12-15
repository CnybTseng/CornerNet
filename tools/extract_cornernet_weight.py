# -*- coding: utf-8 -*-
# file: extract_cornernet_weight.py
# brief: Extract CornerNet weights from public model.
# author: Zeng Zhiwei
# date: 2019/12/12

import sys
import torch
import argparse
sys.path.append('.')
import cornernet

def get_data(it, size):
    try:
        param = next(it)
        assert size == param.data.size()
        return param
    except StopIteration:
        print('StopIteration error!')
        sys.exit(-1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-model', '-s', dest='sm', type=str, help='public CornerNet model file')
    parser.add_argument('--destination-model', '-d', dest='dm', type=str, help='destination model file')
    parser.add_argument('--classes', type=int, default=80, help='number of classes [80]')
    args = parser.parse_args()
    
    model = cornernet.CornerNet(args.classes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    with open(args.sm, 'rb') as file:
        params = torch.load(file, map_location=device)
        file.close()
    
    params = [v for v in params.values()]
    it = iter(params)
    
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            weight = get_data(it, module.weight.data.size())
            module.weight.data = weight.clone()
            if module.bias is not None:
                bias = get_data(it, module.bias.data.size())
                module.bias.data = bias.clone()
        elif isinstance(module, torch.nn.BatchNorm2d):
            weight = get_data(it, module.weight.data.size())
            module.weight.data = weight.clone()
            bias = get_data(it, module.bias.data.size())
            module.bias.data = bias.data.clone()
            running_mean = get_data(it, module.running_mean.data.size())
            module.running_mean.data = running_mean.data.clone()
            running_var = get_data(it, module.running_var.data.size())
            module.running_var.data = running_var.data.clone()
    
    torch.save(model.state_dict(), f'{args.dm}')