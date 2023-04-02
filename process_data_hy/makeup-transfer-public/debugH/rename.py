import os
from PIL import Image
import pdb

root = "/data/hanxinyang/MuNeRF_latest/process_data_hy/makeup-transfer-public/dataH/boy4/after"
save = "/data/hanxinyang/MuNeRF_latest/process_data_hy/makeup-transfer-public/dataH/boy4/_"
print(os.listdir(root))
for i, thing in enumerate(os.listdir(root)):
    save_path = os.path.join(save, "%s.png"%i)
    img_path = os.path.join(root, thing)
    # print(img_path)
    

    img = Image.open(img_path)
    img.save(save_path)
