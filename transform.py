# -*- coding: utf-8 -*-
# file: transform.py
# brief: Image transform module for CornerNet.
# author: Zeng Zhiwei
# date: 2019/12/14

import torch
import numpy as np

class ToTensor(object):
    def __call__(self, image, target=None):
        image = torch.from_numpy(image)
        image = torch.unsqueeze(image, 0).permute(0, 3, 1, 2).contiguous()
        if target:
            return image, target
        else:
            return image

class TestImageMaker(object):
    def __init__(self):
        self.input_size = np.array([511,511])
        self.output_size = np.array([128,128])
        self.mean = np.array([0.40789654, 0.44719302, 0.47026115], dtype=np.float32)
        self.std  = np.array([0.28863828, 0.27408164, 0.27809835], dtype=np.float32)
    
    def __call__(self, image):
        h, w = image.shape[:2]
        ih, iw = h | 127, w | 127
        
        yscale, xscale = (self.input_size + 1) / self.output_size
        oh, ow = (ih + 1) / yscale, (iw + 1) / xscale
        ratio = (oh / ih, ow / iw)
        
        image = image.astype(np.float32)
        output_image = np.zeros((ih, iw, 3), dtype=image.dtype)
        
        cy, cx = h // 2, w // 2
        minx, maxx = max(0, cx-iw//2), min(w, cx+iw//2)
        miny, maxy = max(0, cy-ih//2), min(h, cy+ih//2)
        left, right = cx-minx, maxx-cx
        top, bottom = cy-miny, maxy-cy
        
        ciy, cix = ih // 2, iw // 2
        y_slice = slice(ciy - top, ciy + bottom)
        x_slice = slice(cix - left, cix + right)
        output_image[y_slice, x_slice, :] = image[miny:maxy, minx:maxx, :]

        output_image = output_image / 255
        output_image = output_image - self.mean
        output_image = output_image / self.std
        output_image = np.expand_dims(output_image.transpose((2, 0, 1)), axis=0)
        
        border = np.array([ciy - top, ciy + bottom, cix - left, cix + right], dtype=np.float32)
        offset = np.array([cy - ih // 2, cx - iw // 2])
        
        return output_image, border, offset, ratio

if __name__ == '__main__':
    import cv2
    import sys
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', '-i', type=str, help='test image')
    args = parser.parse_args()
    
    image = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if not image.any():
        print(f'read image {args.image} fail!')
        sys.exit()
    
    maker = TestImageMaker()
    image, border, offset, ratio = maker(image)
    print(border)
    print(offset)
    print(ratio)
    
    image = torch.from_numpy(image).cuda()
    refer = torch.load('model/dog.pth')
    print(f'{image.size()} {refer.size()}')
    print(f'the reference image is equal to test image? {torch.equal(image, refer)}')