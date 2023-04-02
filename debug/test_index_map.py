import os
import numpy as np
import pdb

paths = ["/data/hanxinyang/MuNeRF_latest/dataset/girl1/index_map.npy","/data/hanxinyang/MuNeRF_latest/dataset/girl7/index_map.npy"]
map = []
for path in paths:
    map.append(np.load(path))
    print(np.load(path))

info = (map[0]==map[1])
info = info.flatten()
list_info = info.tolist()
set_info = set(list_info)

print(set_info)
pdb.set_trace()


