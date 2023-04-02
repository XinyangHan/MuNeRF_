from PIL import Image
import torch
from torchvision import transforms
import numpy as np
import cv2
import os
import json
import argparse
import pdb

parser = argparse.ArgumentParser()
parser.add_argument(
    "--name", type=str, default='girl10', help="Path to modnet model."
)

args = parser.parse_args()

path = os.path.join("/data/hanxinyang/MuNeRF_latest/dataset", args.name, "raw_imgs")
save_path = os.path.join("/data/hanxinyang/MuNeRF_latest/dataset", args.name,  "ori_imgs")
if not os.path.exists(save_path):
    os.mkdir(save_path)

for img_name in os.listdir(path):
    img_path = os.path.join(path, img_name)
    img = cv2.imread(img_path)
    dim = (512,512)
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    img_save_path = os.path.join(save_path, img_name)
    cv2.imwrite(img_save_path, resized)
    # pdb.set_trace()
