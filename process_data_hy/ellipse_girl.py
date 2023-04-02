#!/usr/bin/python
# -*- coding: UTF-8 -*-
import cv2
import numpy as np
import pdb
import os
import math
import argparse

try_num = "1081"


img_path = "/data/hanxinyang/MuNeRF_latest/dataset/girl10/00005/warp_makeup_00005_ori/%s.png"%try_num
landmark_path = "/data/hanxinyang/MuNeRF_latest/dataset/girl10/ori_imgs/landmark/%s.npy"%try_num
output_path = "/data/hanxinyang/MuNeRF_latest/debug/H/ellipse"

img = cv2.imread(img_path)


text = np.load(landmark_path, allow_pickle=True)
text = text[()]['faces'][0]['landmark']
# pdb.set_trace()
L_keys = text.keys()
# print(text.keys())

important_keys = ["left_eye_center", "right_eye_center", "left_eye_left_corner", "left_eye_right_corner", "right_eye_left_corner", "right_eye_right_corner"]
left_center = text[important_keys[0]]
right_center = text[important_keys[1]]
left_eye_left_corner = text["left_eye_left_corner"]
left_eye_right_corner = text["left_eye_right_corner"]
right_eye_left_corner = text["right_eye_left_corner"]
right_eye_right_corner = text["right_eye_right_corner"]
delta_y_left = left_eye_left_corner['y'] - left_eye_right_corner['y']
delta_y_right = right_eye_left_corner['y'] - right_eye_right_corner['y']
delta_x_left = left_eye_right_corner['x'] - left_eye_left_corner['x']
delta_x_right = right_eye_right_corner['x'] - right_eye_left_corner['x']
tan_left = delta_y_left / delta_x_left
tan_right = delta_y_right / delta_x_right

value_left = math.atan(tan_left)
value_right = math.atan(tan_right)
angle_left = -math.degrees(value_left)
angle_right = -math.degrees(value_right)
left_long = abs(int(0.74*delta_x_left))
left_short = int(left_long*0.53)
right_long = abs(int(0.45*delta_x_right))
right_short = int(right_long*0.6)


# print(right_center['x'])
# print(L_keys)
# landmarks = text['landmark']
# pdb.set_trace()
cv2.ellipse(img, (int(left_center['x'] - 5) , int(left_center['y']-4)), (left_long, left_short), angle_left, 0, 360, (255, 255, 255),-1) #画椭圆
cv2.ellipse(img, (int(right_center['x']), int(right_center['y']-10)), (right_long, right_short), angle_right, 0, 360, (255, 255, 255),-1) #画椭圆
    
save_path = os.path.join(output_path, "girl.png")
cv2.imwrite(save_path, img)

d = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:1, 10:0, 11:0, 12:0, 13:1, 14:0, 15:0, 16:0, 17:0, 18:0}

# mask ellipse

parser = argparse.ArgumentParser()

parser.add_argument(
    "--mask_root",
    type=str, 
    default="/data/hanxinyang/MuNeRF_latest/dataset/girl10/00005/mask/warp_makeup_00005", 
    help="whether to save depth images.",
)

parser.add_argument(
    "--ldmk_root",
    type=str, 
    default="/data/hanxinyang/MuNeRF_latest/dataset/girl10/ori_imgs/landmark", 
    help="/data/hanxinyang/MuNeRF_latest/dataset/girl10/ori_imgs/landmark",
)

parser.add_argument(
    "--ori_img_root",
    type=str, 
    default="/data/hanxinyang/MuNeRF_latest/dataset/girl10/ori_imgs", 
    help="/data/hanxinyang/MuNeRF_latest/dataset/girl10/ori_imgs",
)
args = parser.parse_args()

mask_root = args.mask_root
ldmk_root = args.ldmk_root

os.makedirs(os.path.join(args.ori_img_root, "ellipse_mask"), exist_ok=True)
masks = os.listdir(mask_root)
for thing in masks:
    mask_path = os.path.join(mask_root, thing)
    landmark_path = os.path.join(ldmk_root, thing[:-4]+'.npy')
    ori_mask = cv2.imread(mask_path)
    # final_mask = np.empty(ori_mask.shape[0], ori_mask.shape[1], ori_mask.shape[2])
    text = np.load(landmark_path, allow_pickle=True)
    text = text[()]['faces'][0]['landmark']
    # pdb.set_trace()
    L_keys = text.keys()
    # print(text.keys())

    important_keys = ["left_eye_center", "right_eye_center", "left_eye_left_corner", "left_eye_right_corner", "right_eye_left_corner", "right_eye_right_corner"]
    left_center = text[important_keys[0]]
    right_center = text[important_keys[1]]
    left_eye_left_corner = text["left_eye_left_corner"]
    left_eye_right_corner = text["left_eye_right_corner"]
    right_eye_left_corner = text["right_eye_left_corner"]
    right_eye_right_corner = text["right_eye_right_corner"]
    delta_y_left = left_eye_left_corner['y'] - left_eye_right_corner['y']
    delta_y_right = right_eye_left_corner['y'] - right_eye_right_corner['y']
    delta_x_left = left_eye_right_corner['x'] - left_eye_left_corner['x']
    delta_x_right = right_eye_right_corner['x'] - right_eye_left_corner['x']
    tan_left = delta_y_left / delta_x_left
    tan_right = delta_y_right / delta_x_right

    value_left = math.atan(tan_left)
    value_right = math.atan(tan_right)
    angle_left = -math.degrees(value_left)
    angle_right = -math.degrees(value_right)
    left_long = abs(int(0.74*delta_x_left))
    left_short = int(left_long*0.53)
    right_long = abs(int(0.45*delta_x_right))
    right_short = int(right_long*0.6)
    
    cv2.ellipse(ori_mask, (int(left_center['x'] - 5) , int(left_center['y']-4)), (left_long, left_short), angle_left, 0, 360, (9, 9, 9),-1) #画椭圆
    cv2.ellipse(ori_mask, (int(right_center['x']), int(right_center['y']-10)), (right_long, right_short), angle_right, 0, 360, (9, 9, 9),-1) #画椭圆
    final_mask = np.vectorize(d.get)(ori_mask)
    
    target_root = os.path.join(args.ori_img_root, "ellipse_mask")
    os.makedirs(target_root, exist_ok=True)
    save_path = os.path.join(target_root, thing)
    # print(save_path)
    cv2.imwrite(save_path, final_mask)



