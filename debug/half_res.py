import os
import cv2
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--name", type=str, required=True, help="Path to (.yml) config file."
)
configargs = parser.parse_args()
name = configargs.name
root = "/data/hanxinyang/MuNeRF_latest/dataset/%s/ori_imgs"%name
save_path = "/data/hanxinyang/MuNeRF_latest/dataset/%s/half_res"%name

things = os.listdir(root)
for thing in things:
    if not thing.endswith("png"):
        continue
    else:
        img = cv2.imread(os.path.join(root, thing))
        img = cv2.resize(img, (256,256))
        cv2.imwrite(os.path.join(save_path, thing),img )
