# -*- coding: utf-8 -*-
# file: test.py
# brief: Test detecting object based on CornerNet.
# author: Zeng Zhiwei
# date: 2019/12/14

import cv2
import sys
import torch
import argparse
import cornernet
import transform
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', '-i', type=str, help='test image')
    args = parser.parse_args()
    
    bgr = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if not bgr.any():
        print(f'read image {args.image} fail!')
        sys.exit()
    
    h, w = bgr.shape[:2]
    maker = transform.TestImageMaker()
    image, border, offset, ratio = maker(bgr)
    image = torch.from_numpy(image).cuda()
    print(f'{border} {offset} {ratio}')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # x = torch.load('model/dog.pth')
    x = image
    net = cornernet.CornerNet().to(device)
    net.load_state_dict(torch.load('model/cornernet.pth'))
    net.eval()
    ys = net(x)
    
    print(f'net input size {x.size()}')
    for y in ys:
        print(f'net output size {y.size()}')
    
    refr = torch.load('model/net_output.pth')
    for r, t in zip(refr, ys):
        print(f'reference output is equal to test output? {torch.equal(r, t)}')
    
    decoder = cornernet.CornerDecoder()
    z = decoder(ys)
    
    refr = torch.load('model/dets.pth')
    print(f'{refr.size()} {z.size()}')
    print(f'reference detections is equal to test detections? {torch.equal(refr[0], z)}')
    
    z = z.detach().cpu().numpy()
    xs, ys = z[:,0:4:2], z[:,1:4:2]
    xs /= ratio[1]
    ys /= ratio[0]
    
    xs -= border[2]
    ys -= border[0]
    
    xs = np.clip(xs, 0, w-1)
    ys = np.clip(ys, 0, h-1)
    
    indices = z[:,4]>0.48
    z = z[indices,:]
    print(z.shape)
    
    color = (0,255,255)
    for det in z:
        pt1 = (det[0], det[1])
        pt2 = (det[2], det[3])
        bgr = cv2.rectangle(bgr, pt1, pt2, color)
    cv2.imwrite('result.jpg', bgr)