targets = ['/data/hanxinyang/MuNeRF_latest/dataset/boy4/ori_imgs', '/data/hanxinyang/MuNeRF_latest/dataset/boy4/test', '/data/hanxinyang/MuNeRF_latest/dataset/boy4/train', '/data/hanxinyang/MuNeRF_latest/dataset/boy4/val', '/data/hanxinyang/MuNeRF_latest/dataset/boy4/bg']

import os
import cv2

min_x = 426
min_y = 0
max_x = 512 
max_y = 480

img_path = "/data/hanxinyang/MuNeRF_latest/dataset/boy4/ori_imgs/1533.png"
save_path = "/data/hanxinyang/MuNeRF_latest/debug/boy4.png"
# Trial
# img = cv2.imread(img_path)
# cv2.rectangle(img, (min_x, min_y), (max_x, max_y), (255, 0, 0), thickness=-1)
# cv2.imwrite(save_path, img)

# Formally Mask 
for target in targets:
    things = os.listdir(target)
    for thing in things:
        if not thing.endswith('png'):
            continue
        else:
            img_path = os.path.join(target, thing)
            img = cv2.imread(img_path)
            cv2.rectangle(img, (min_x, min_y), (max_x, max_y), (9,14,10), thickness=-1)
            cv2.imwrite(img_path, img)
