import os
from PIL import Image
x1, y1, w, h = 30, 0, 196, 256
frame_dir = '/data/hanxinyang/MuNeRF_latest/crop/'
save_dir = '/data/hanxinyang/MuNeRF_latest/cropped/'
if not os.path.isdir(save_dir):
    os.mkdir(save_dir)
for file in os.listdir(frame_dir):
    img_path = os.path.join(frame_dir, file)
    save_path = os.path.join(save_dir, file)
    img = Image.open(img_path)
    img_crop = img.crop((x1,y1,x1+w,y1+h))
    img_crop.save(save_path)
