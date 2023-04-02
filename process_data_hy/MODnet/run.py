#!/usr/bin/python
# -*- coding: UTF-8 -*-
from modnet import MODNet
from PIL import Image
import torch
from torchvision import transforms
import numpy as np
import cv2
import os
import json
import argparse
import pdb
trans = transforms.Compose([#transforms.ToTensor(),
			 transforms.Resize(512),
                         transforms.ToTensor(),
			 transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]) 


def getmask(modnet, imgs_t):
    # input: [B,3,H,W], [-1,1]
    #####
    frame = imgs_t[0].clone()
    frame = frame.cpu().detach().transpose(0, 2).transpose(0, 1).numpy()
    # pdb.set_trace()

    var = ((frame + 1) / 2)
    var[var < 0] = 0
    var[var > 1] = 1
    var = var * 255
    frame_np = var.astype(np.uint8)

    h, w = frame_np.shape[:2]
    '''if w >= h:
        rh = 512 ### don't change
        rw = int(w / h * 512)
    else:
        rw = 512
        rh = int(h / w * 512)
    rh = rh - rh % 32
    rw = rw - rw % 32
    frame_np = cv2.resize(frame_np, (rw, rh), cv2.INTER_AREA)'''
    frame_PIL = Image.fromarray(frame_np)
    frame_tensor = trans(frame_PIL)
    frame_tensor = frame_tensor[None, :, :, :].cuda()

    with torch.no_grad():
        _, _, matte_tensor = modnet(frame_tensor, True)
    matte_tensor = matte_tensor.repeat(1, 3, 1, 1)[0]
    var = matte_tensor
    var = var.cpu().detach().transpose(0, 2).transpose(0, 1).numpy()
    # var = ((var + 1) / 2)
    var[var < 0] = 0
    var[var > 1] = 1
    var = var * 255
    var = var.astype(np.uint8)
    var = cv2.resize(var,(w, h))
    return var

def getbox(mask):
    colorval = mask
    ai,aj,bi,bj = 0,0,0,0
    h = 512 #640
    w = 512 #368
    for i in range(h):
        for j in range(w):
            if colorval[i][j][0]>60: 
              ai = i
              break
        else:
            continue
        break
        
    for i in reversed(range(h)):
        for j in range(w):
            if colorval[i][j][0]>60: 
              bi = i
              break
        else:
            continue
        break
    
    for j in range(w):
        for i in range(h):
            if colorval[i][j][0]>60: 
                aj = j
                break
        else:
            continue
        break
        
    for i in reversed(range(h)):
        for j in reversed(range(w)):
            if colorval[i][j][0]>60: 
                bj = j
                break
        else:
            continue
        break
    return [(ai-5)/512, (bi-5)/512, (aj+5)/512, (bj-5)/512]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--modnet", type=str, default='/data/hanxinyang/MuNeRF_latest/process_data_hy/MODnet/modnet_webcam_portrait_matting.ckpt', help="Path to modnet model."
    )
    parser.add_argument(
        "--inputdata",
        type=str,
        default="/data/hanxinyang/MuNeRF_latest/process_data_hy/input/",
        help="Path to load input frames",
    )
    parser.add_argument(
        "--jsonfiles",
        type=str,
        default="/data/hanxinyang/MuNeRF_latest/process_data_hy/MODnet/debug/box.json",
        help="Path to save json files",
    )
    args = parser.parse_args()

    modnet = torch.nn.DataParallel(MODNet(backbone_pretrained=False))
    modnet.load_state_dict(torch.load(args.modnet))
    modnet = modnet.module
    modnet.cuda()
    modnet.eval()
    
    datadir = args.inputdata
    boxdict = {}
    jsonname = args.jsonfiles
    for filename in sorted(os.listdir(datadir)):
        filepath = datadir+filename
        if not filename.endswith('.png'):
            continue
        print('&&&&&&&', filename, '&&&&&&&')
        image = Image.open(filepath)
        # print(image.shape)
        # pdb.set_trace()

        img = trans(image)
        img = img.unsqueeze(0) # 填充一维
        mask = getmask(modnet,img)
        boxdict[filename] = getbox(mask)
    with open(jsonname,"w") as f:
            json.dump(boxdict,f,indent=1)
  
