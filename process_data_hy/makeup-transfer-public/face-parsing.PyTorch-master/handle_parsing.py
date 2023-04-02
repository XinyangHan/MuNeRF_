from PIL import Image
import numpy as np
import os 
import pdb
root="res/test_res1/"
roots="res/test_res1/"
paths = os.listdir(root)
for path in paths:
    path1=root+path
    pathsave=roots+path
    # pdb.set_trace()
    seg = np.array(Image.open(path1))
    new = np.zeros_like(seg)
    new[seg == 0] = 0
    new[seg == 1] = 4
    new[seg == 2] = 7
    new[seg == 3] = 2
    new[seg == 5] = 6
    new[seg == 4] = 1
    new[seg == 10] = 8
    new[seg == 12] = 9
    new[seg == 8] = 11
    new[seg == 13] = 13
    # new[seg == 6] = 12
    new[seg == 6] = 10
    new[seg == 11] = 3
    new[seg ==7] = 5
    # new[seg == 14] = 10
    new[seg == 14] = 2
    img = Image.fromarray(new)
    img.save(pathsave)
