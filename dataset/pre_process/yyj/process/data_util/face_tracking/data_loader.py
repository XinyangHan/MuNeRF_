import torch
import numpy as np
import os


def load_dir(path, start, end):
    lmss = []
    imgs_paths = []
    for i in range(start, end):
        if os.path.isfile(os.path.join(path, '%04d'%i + '.lms')):
            lms = np.loadtxt(os.path.join(
                path, '%04d'%i + '.lms'), dtype=np.float32)
            lmss.append(lms)
            imgs_paths.append(os.path.join(path, '%04d'%i + '.png'))
    lmss = np.stack(lmss)
    lmss = torch.as_tensor(lmss).cuda()
    return lmss, imgs_paths
