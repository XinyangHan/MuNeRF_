import os
import cv2



path =  "/data/hanxinyang/MuNeRF_latest/dataset/girl15/ori_imgs_uncropped/0000.png"
save_path =  "/data/hanxinyang/MuNeRF_latest/debug/H/crop.png"

img = cv2.imread(path)
x0 = 280
y0 = 20
d = 320
print(img.shape)
cropped = img[y0:y0+ d, x0:x0+d]
cv2.imwrite(save_path, cropped)