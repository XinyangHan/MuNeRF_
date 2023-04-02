import os
import numpy as np
import cv2

path6 = "/home/yuanyujie/cvpr23/dataset/girl1/00094/mask/warp_makeup_00094/0001.png"
path10 = "/home/yuanyujie/cvpr23/dataset/girl1/00099/mask/warp_makeup_00099/0001.png"

img6 = cv2.imread(path6)
img10 = cv2.imread(path10)

def img_to_index(img, save_path):
    for index in range(19):
        save_img = np.zeros((img.shape[0], img.shape[1]))
        for j in range(img.shape[0]):
            for k in range(img.shape[1]):
                if img[j][k][0] == index :
                    save_img[j][k] = 255
        cv2.imwrite(os.path.join(save_path, "%d.png"%index), save_img)

save_path6 = "/home/yuanyujie/cvpr23/debug/H/parsing_6"
save_path9 = "/home/yuanyujie/cvpr23/debug/H/parsing_10"

os.makedirs(save_path6, exist_ok = True)
os.makedirs(save_path9, exist_ok = True)

img_to_index(img6, save_path6)
img_to_index(img10, save_path9)
