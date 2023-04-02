target = "/data/hanxinyang/MuNeRF_latest/dataset/boy4/bg/bc.png"

import cv2
import os
from PIL import Image as Image

im = Image.open(target)
im.save(target[:-3] + 'jpg')
