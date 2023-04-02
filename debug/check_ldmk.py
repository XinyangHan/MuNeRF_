import numpy as np

path = "/data/hanxinyang/MuNeRF_latest/makeup/landmark/00001.npy"

content = np.load(path,allow_pickle=True)
print(content)
print(content.shape)