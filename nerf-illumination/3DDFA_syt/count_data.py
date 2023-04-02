from tqdm import tqdm
from utils.io import _numpy_to_tensor, _load_cpu, _load_gpu, _load, _dump
from utils.ddfa import _parse_param
import os.path as osp
import os
import torch
import torchvision.transforms as transforms
import mobilenet_v1
import numpy as np
import cv2
import dlib
from utils.ddfa import ToTensorGjz, NormalizeGjz, str2bool
import scipy.io as sio
from utils.inference import get_suffix, parse_roi_box_from_landmark, crop_img, predict_68pts, dump_to_ply, dump_vertex, \
    draw_landmarks, predict_dense, parse_roi_box_from_bbox, get_colors, write_obj_with_colors
from utils.cv_plot import plot_pose_box
from utils.estimate_pose import parse_pose
from utils.render import get_depths_image, cget_depths_image, cpncc
from utils.paf import gen_img_paf
import argparse
#import torch.backends.cudnn as cudnn
from utils.params import *
#select a data and calc the param------------------------
#search
save_dir = "./train.configs/mesh"
npylist = open("./train.configs/train_aug_120x120.list.train")
param_fp = os.path.join(d2, 'param_all_norm.pkl')
params_data = _load_cpu(param_fp)[:,:]
print(params_data.shape)

roi_box = [0,0,120,120]

#a_all = np.zeros((636252,3,53215),dtype='float32')  
#b_all = np.zeros((636252,3,53215),dtype='float32')

a_all = np.zeros((1000,3,53215),dtype='float32')
aa_all_mean = np.zeros((640,3,53215),dtype='float32')
aa_all_var = np.zeros((640,3,53215),dtype='float32')

tri = sio.loadmat('visualize/tri.mat')['tri']

index_aa = -1
index = -1
for line in tqdm(npylist.readlines()):
    
    index +=1
    if index >=1000:
        break
    #if index >=3050:
    #    break
    '''
    if index%1000==0:
        index_aa = index//1000
        a1000_mean = np.mean(a_all, axis=0)
        a1000_var = np.var(a_all, axis=0)
        print(a1000_mean[:,0])
        aa_all_mean[index_aa,:,:] = a1000_mean
        aa_all_var[index_aa,:,:] = a1000_var
    '''
    
    param = params_data[index]
    vertices = predict_dense(param, roi_box) #(3, 53215)
    vertices[1, :] = 120 + 1 - vertices[1, :]
    print(vertices.shape)

    wfp = os.path.join(save_dir,os.path.basename(line)[:-5]+".ply")
    #_dump(os.path.join(save_dir,os.path.basename(line)[:-5]+".pkl"),vertices)
    dump_to_ply(vertices, tri, wfp)

    #a_all[index%1000,:,:] = vertices
'''
print(index_aa)

aaaa_mean = np.mean(aa_all_mean[1:index_aa+1,:,:], axis=0)
print(aaaa_mean[:,0])
aaaa_var = np.mean(aa_all_var[1:index_aa+1,:,:], axis=0)
print(aaaa_var[:,0])
print(np.sqrt(aaaa_var))
_dump("./aaaa_mean.pkl",aaaa_mean)
_dump("./aaaa_std.pkl",np.sqrt(aaaa_var))
'''