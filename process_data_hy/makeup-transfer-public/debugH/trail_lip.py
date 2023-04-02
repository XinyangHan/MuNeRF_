import os
import cv2
import numpy as np

path1 = "/data/hanxinyang/MuNeRF_latest/process_data_hy/makeup-transfer-public/debugH/after_transformed.jpg"
path2 = "/data/hanxinyang/MuNeRF_latest/process_data_hy/makeup-transfer-public/debugH/before.jpg"

img1 = cv2.imread(path1)
img2 = cv2.imread(path2)

img = img1/4 + img2/2

save_path = "/data/hanxinyang/MuNeRF_latest/process_data_hy/makeup-transfer-public/debugH/result.jpg"
cv2.imwrite(save_path, img)
