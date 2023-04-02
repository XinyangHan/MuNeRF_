path = "/data/hanxinyang/MuNeRF_latest/dataset/girl10/ori_imgs_jpg"
save_path = "/data/hanxinyang/MuNeRF_latest/dataset/girl10/ori_imgs"

import os
from PIL import Image as Image
# im = Image.open('test.jpg')
# im.save('test.tiff') # or 'test.tif

img_names = os.listdir(path)
for img_name in img_names:
    im = Image.open(os.path.join(path, img_name))
    im.save(os.path.join(save_path, img_name[:-4] + ".png"))

