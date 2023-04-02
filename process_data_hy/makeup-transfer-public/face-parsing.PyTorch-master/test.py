#!/usr/bin/python
# -*- encoding: utf-8 -*-

from logger import setup_logger
from model import BiSeNet

import torch

import os
import os.path as osp
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import cv2
import argparse

import pdb

def vis_parsing_maps(im, parsing_anno, stride, save_im=True, save_path='vis_results/parsing_map_on_im.jpg',save_path2=''):
    # Colors for all 20 parts
    part_colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0],
                   [255, 0, 85], [255, 0, 170],
                   [0, 255, 0], [85, 255, 0], [170, 255, 0],
                   [0, 255, 85], [0, 255, 170],
                   [0, 0, 255], [85, 0, 255], [170, 0, 255],
                   [0, 85, 255], [0, 170, 255],
                   [255, 255, 0], [255, 255, 85], [255, 255, 170],
                   [255, 0, 255], [255, 85, 255], 
                   [255, 170, 255], [0, 255, 255], [85, 255, 255], [170, 255, 255]]

    im = np.array(im)
    vis_im = im.copy().astype(np.uint8)
    # pdb.set_trace()
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
    vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255

    num_of_class = np.max(vis_parsing_anno)

    for pi in range(1, num_of_class + 1):
        index = np.where(vis_parsing_anno == pi)
        vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]

    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
    # print(vis_parsing_anno_color.shape, vis_im.shape)
    vis_im = cv2.addWeighted(cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR), 0.4, vis_parsing_anno_color, 0.6, 0)

    record = np.zeros((1,20))

    # file = '/home/hanxinyang/face-parsing.PyTorch-master/res/record.txt'
    # with open(file, 'w') as f:
    #     for i in range(len(vis_parsing_anno)):
    #         for j in range(len(vis_parsing_anno[i])):
    #             # pdb.set_trace()
    #             record[vis_parsing_anno[i][j]]
    #             f.write('%u ' %vis_parsing_anno[i][j])
    #         f.write('\n')

    # print(record)

    # Save result or not
    if save_im:
        cv2.imwrite(save_path2, vis_parsing_anno)
        cv2.imwrite(save_path, vis_im, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    # return vis_im

def evaluate(ckpt, respth='./res/test_res', dspth='./data', cp='model_final_diss.pth'):

    if not os.path.exists(respth):
       os.makedirs(respth)
    respth1=respth.replace('test_res','test_res1')
    if not os.path.exists(respth1):
        os.makedirs(respth1)
    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    net.cuda()
    save_pth = osp.join(ckpt, cp)
    net.load_state_dict(torch.load(save_pth))
    net.eval()

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    with torch.no_grad():
        for image_path in os.listdir(dspth):
            img = Image.open(osp.join(dspth, image_path))
            image = img.resize((512, 512), Image.BILINEAR)
            img = to_tensor(image)
            img = torch.unsqueeze(img, 0)
            img = img.cuda()
            out = net(img)[0]
            parsing = out.squeeze(0).cpu().numpy().argmax(0)
            # print(parsing)
            print(np.unique(parsing))
            
            # return parsing
            # pdb.set_trace()

            vis_parsing_maps(image, parsing, stride=1, save_im=True, save_path=osp.join(respth, image_path),save_path2=osp.join(respth1, image_path))
    # return parsing
# H's parsing
# def face_parsing(ckpt="/data/hanxinyang/MuNeRF_latest/process_data_hy/makeup-transfer-publicface-parsing.PyTorch-master/res/cp", respth="/data/hanxinyang/MuNeRF_latest/process_data_hy/makeup-transfer-publicface-parsing.PyTorch-master/res/test_res", dspth="/data/hanxinyang/MuNeRF_latest/process_data_hy/makeup-transfer-publicdataH/crop/before", cp='model_final_diss.pth'):
#     n_classes = 19
#     net = BiSeNet(n_classes=n_classes)
#     net.cuda()
#     save_pth = osp.join(ckpt, cp)
#     net.load_state_dict(torch.load(save_pth))
#     net.eval()
#     to_tensor = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
#     ])
#     with torch.no_grad():
#         for image_path in os.listdir(dspth):
#             img = Image.open(osp.join(dspth, image_path))
#             image = img.resize((512, 512), Image.BILINEAR)
#             img = to_tensor(image)
#             img = torch.unsqueeze(img, 0)
#             img = img.cuda()
#             out = net(img)[0]
#             parsing = out.squeeze(0).cpu().numpy().argmax(0)
#             # print(parsing)
#             print(np.unique(parsing))

#             # pdb.set_trace()

#             vis_parsing_maps(image, parsing, stride=1, save_im=True, save_path=osp.join(respth, image_path),save_path2=osp.join(respth1, image_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", default="/data/hanxinyang/MuNeRF_latest/process_data_hy/makeup-transfer-publicface-parsing.PyTorch-master/res/cp", type=str)
    parser.add_argument("--respth", default="/data/hanxinyang/MuNeRF_latest/process_data_hy/makeup-transfer-publicface-parsing.PyTorch-master/res/test_res", type=str)
    parser.add_argument("--dspth", default="/data/hanxinyang/MuNeRF_latest/process_data_hy/makeup-transfer-publicdataH/crop/before", type=str)
    args = parser.parse_args()
    ckpt = args.ckpt_path
    respth =  args.respth
    dspth = args.dspth
    evaluate(ckpt, respth, dspth=dspth, cp='79999_iter.pth')