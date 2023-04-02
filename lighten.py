import matplotlib.pyplot as plt
from numpy.lib.npyio import save
from skimage import exposure, img_as_float
from PIL import Image
import os
import numpy as np

path1 = "/home/yuanyujie/makeupnerf/dataset/girl9/train/"
path2 = "/home/yuanyujie/makeupnerf/dataset/girl9/train_/"
filelist1 = sorted(os.listdir(path1))

for i in range(len(filelist1)):
    file_name = os.path.join(path1,filelist1[i])
    save_path = os.path.join(path2,filelist1[i])
    img = Image.open(file_name)
    img = img_as_float(img)
    img_out= exposure.adjust_gamma(img, 0.8)
    img_out = img_out*255.
    img_out = Image.fromarray(np.uint8(img_out))
    img_out.save(save_path)
