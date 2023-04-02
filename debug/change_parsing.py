import numpy as np
import os
import cv2

def change(i):
    if i == 1:
        return 6
    elif i == 2:
        return 7
    elif i == 5:
        return 11
    elif i == 6:
        return 1
    elif i == 7:
        return 2
    elif i == 11:
        return 5
    else:
        return i

vfunc = np.vectorize(change)

def handle(img, save_path):
    img = vfunc(img)
    cv2.imwrite(save_path, img)

targets = ['/data/hanxinyang/MuNeRF_latest/dataset/girl10/00005/mask/warp_makeup_00005','/data/hanxinyang/MuNeRF_latest/dataset/girl10/00006/mask/warp_makeup_00006','/data/hanxinyang/MuNeRF_latest/dataset/boy4/00005/mask/warp_makeup_00005','/data/hanxinyang/MuNeRF_latest/dataset/boy4/00006/mask/warp_makeup_00006']

for target in targets:
    current_path = target
    print("Handling %s"%current_path)
    things = os.listdir(current_path)
    for thing in things:
        img_path = os.path.join(current_path, thing)
        img = cv2.imread(img_path)
        handle(img, img_path)