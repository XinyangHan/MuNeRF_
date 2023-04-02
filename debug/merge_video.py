import os
import cv2
import numpy as np
import pdb
paths = [ "/data/hanxinyang/MuNeRF_latest/rendering/girl7_makeup_1204", "/data/hanxinyang/MuNeRF_latest/rendering/girl7_makeup_0656", "/data/hanxinyang/MuNeRF_latest/rendering/girl7_makeup_4406"]

for path in paths:
    things = os.listdir(path)
    things.sort()
    # pdb.set_trace()
    trial = things[0] if things[0].endswith("png") else things[1]
    fps = 25 #视频每秒24帧
    test = cv2.imread(os.path.join(path, trial))
    shape = test.shape
    size = (shape[0], shape[1]) #需要转为视频的图片的尺寸
    name = path.split('/')[-1]
    # name = path.split('/')[-2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter("%s.mp4"%name, fourcc, fps, size)

    for thing in things:
        if not thing.endswith("png"):
            continue
        else:
            img = cv2.imread(os.path.join(path, thing))

            video.write(img)
    
    video.release()

    