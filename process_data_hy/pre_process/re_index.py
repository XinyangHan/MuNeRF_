path = "/data/hanxinyang/MuNeRF_latest/dataset/girl10/ori_imgs_jpg"
save_path = "/data/hanxinyang/MuNeRF_latest/dataset/girl10/ori_imgs"

import os
from PIL import Image as Image
for img_name in os.listdir(path):
    index = int(img_name[:-4])
    index = "%04d"%index
    im = Image.open(os.path.join(path, img_name))

    img_name = index + ".png"
    im.save(os.path.join(save_path, img_name))

    

