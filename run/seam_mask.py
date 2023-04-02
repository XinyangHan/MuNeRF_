import cv2
import os
import numpy as np
# path = "/home/yuanyujie/cvpr23/dataset/girl10/ori_imgs/ellipse_mask/0000.png"
# img = cv2.imread(path)
# print(np.unique(img))
import pdb
root = "/home/yuanyujie/cvpr23/dataset/girl10/ori_imgs/ellipse_mask"
save_root = "/home/yuanyujie/cvpr23/dataset/girl10/00005/warp_makeup_00005"
raw_root = "/home/yuanyujie/cvpr23/dataset/girl10/ori_imgs"
makeup_root = "/home/yuanyujie/cvpr23/dataset/girl10/00005/warp_makeup_00005_normal"

for thing in os.listdir(root):
    path = os.path.join(root, thing)
    mask = cv2.imread(path)
    raw_path = os.path.join(raw_root, thing)
    makeup_path = os.path.join(makeup_root, thing)

    raw = cv2.imread(raw_path)
    makeup = cv2.imread(makeup_path)
    # pdb.set_trace()
    nonzero_x, nonzero_y = mask[:,:,0].nonzero()
    x1, x2 = nonzero_x.min(), nonzero_x.max()
    y1, y2 = nonzero_y.min(), nonzero_y.max()
    x = y1
    y = x1
    w = (y2-y1)
    h = (x2-x1)

    poisson_face = cv2.seamlessClone(makeup, raw, mask.astype(np.uint8) * 255, (x+w//2,y+h//2), cv2.NORMAL_CLONE)
    cv2.imwrite(os.path.join(save_root, thing), poisson_face)

    