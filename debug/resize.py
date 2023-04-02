import os
import cv2

targets = ["/data/hanxinyang/MuNeRF_latest/experiments/3d_consistency/dataset/girl7_makeup_512/ori_imgs","/data/hanxinyang/MuNeRF_latest/experiments/3d_consistency/dataset/girl7_nerface_512/ori_imgs"]

for target in targets:
    things = os.listdir(target)
    for thing in things:
        if thing.endswith("lms"):
            continue
        else:
            img_path = os.path.join(target, thing)
            img = cv2.imread(img_path)
            
            img = cv2.resize(img,(512,512))
            cv2.imwrite(img_path, img)