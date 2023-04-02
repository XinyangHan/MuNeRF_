import numpy as np
import os
import cv2
import argparse
import pdb
# def change(i):
#     if i == 1:
#         return 4
#     elif i == 2:
#         return 7
#     elif i == 3:
#         return 2
#     elif i == 4:
#         return 1
#     elif i == 5:
#         return 6
#     elif i == 6:
#         return 4
#     elif i == 7:
#         return 5
#     elif i == 8:
#         return 11
#     elif i == 10:
#         return 8
#     elif i == 11:
#         return 3
#     elif i == 12:
#         return 9
#     elif (i == 14) or (i == 15) or (i == 16): 
#         return 10
#     elif i == 17:
#         return 0
#     else:
#         return i


d = {0:0, 1:4, 2:7, 3:2, 4:1, 5:6, 6:4, 7:5, 8:11, 9:9, 10:8, 11:3, 12:9, 13:13, 14:10, 15:10, 16:10, 17:0, 18:13}

def handle(img, save_path):
    
    save_img = np.vectorize(d.get)(img)
    # pdb.set_trace()
    cv2.imwrite(save_path, save_img)


parser = argparse.ArgumentParser()

parser.add_argument(
    "--target1",
    type=str, 
    default='', 
    help="whether to save depth images.",
)

parser.add_argument(
    "--target2",
    type=str, 
    default='', 
    help="whether to save depth images.",
)
args = parser.parse_args()


targets = [args.target1, args.target2]

for target in targets:
    current_path = target
    print("Handling %s"%current_path)
    if not current_path:
        continue
    things = os.listdir(current_path)
    for thing in things:
        try:
            img_path = os.path.join(current_path, thing)
            img = cv2.imread(img_path)
            handle(img, img_path)
        except Exception as e:
            print(e)
            
            pdb.set_trace()