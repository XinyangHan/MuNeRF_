import cv2

path = "/home/yuanyujie/cvpr23/dataset/boy4/ori_imgs/ellipse_mask/0006.png"
img = cv2.imread(path)*255
cv2.imwrite("/home/yuanyujie/cvpr23/debug/H/ellipse/2.png", img)