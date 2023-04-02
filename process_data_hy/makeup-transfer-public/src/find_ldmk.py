from api_util import FacePPAPI
import cv2
import os
import numpy as np
targets = ['/data/hanxinyang/MuNeRF_latest/makeup/00005.jpg','/data/hanxinyang/MuNeRF_latest/makeup/00006.jpg']

for target in targets:
    while True:
        img = cv2.imread(target)
        api = FacePPAPI()
        result = api.faceLandmarkDetector(target)
        if "error_message" in result:
            print(result)
            continue
        else:
            break
    result = np.array(result)
    name = target.split('/')[-1][:-4]
    np.save(os.path.join("/data/hanxinyang/MuNeRF_latest/makeup/landmark", "%s.npy"%name), result)

