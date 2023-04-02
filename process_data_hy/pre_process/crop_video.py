import os
import cv2

root = "/data/hanxinyang/MuNeRF_latest/dataset/girl15/ori_imgs_uncropped"
save_root = "/data/hanxinyang/MuNeRF_latest/dataset/girl15/ori_imgs"

os.makedirs(save_root, exist_ok=True)

for thing in os.listdir(root):
    path = os.path.join(root, thing)
    save_path = os.path.join(save_root, thing)
    
    img = cv2.imread(path)
    x0 = 280
    y0 = 20
    d = 320
    cropped = img[y0:y0+ d, x0:x0+d]
    cv2.imwrite(save_path, cropped)