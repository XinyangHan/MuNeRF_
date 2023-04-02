
from utils.io import _numpy_to_tensor, _load_cpu, _load_gpu, _load, _dump
import numpy as np
from utils.inference import get_suffix, dump_to_ply
import scipy.io as sio
from utils.ddfa import _parse_param
import os.path as osp

from utils.render import get_depths_image, cget_depths_image, cpncc
import cv2
import os
# from .io import _load

# 检查-----------------------------------------------------------------
d = 'train.configs/new-60-29'
tri = sio.loadmat('visualize/tri.mat')['tri']
w_shp = _load(osp.join(d, 'w_shp_sim.npy'))
w_exp = _load(osp.join(d, 'w_exp_sim.npy'))  # simplified version
u_shp = _load(osp.join(d, 'u_shp.npy')) # base shape
u_exp = _load(osp.join(d, 'u_exp.npy')) # base exp
u = u_shp + u_exp

param_fp = osp.join(d, 'param_all_norm_12+89.pkl') # param for each image
param_wh = osp.join(d, 'param_whitening_12+89.pkl')# mean & std for parm
params_data = _load_cpu(param_fp)
meta = _load_cpu(param_wh)
param_mean = meta.get('param_mean') # mean
param_std = meta.get('param_std') # std 

#select a data and calc the param------------------------
#search
fileindex = 394493
npylist = open("./train.configs/train_aug_120x120.list.train")
index = -1
from tqdm import tqdm
for line in tqdm(npylist.readlines()):
    index +=1
    if line == "HELEN_HELEN_3227203804_1_1_1.jpg":
    	fileindex = index
    	print("\n",index,"\n")

#gen
param = params_data[fileindex,:] * param_std + param_mean
roi_box = [0, 0, 120, 120]

#landmarks
from utils.inference import get_suffix, parse_roi_box_from_landmark, crop_img, predict_68pts, dump_to_ply, dump_vertex, \
    draw_landmarks, predict_dense, parse_roi_box_from_bbox, get_colors, write_obj_with_colors
img_fp = "./test_data/HELEN_HELEN_3227203804_1_1_1.jpg"
suffix = get_suffix(img_fp)
img_ori = cv2.imread(img_fp)
pts68 = predict_68pts(param, roi_box)
draw_landmarks(img_ori, pts68, wfp=img_fp.replace(suffix, '_origin_ldmk.jpg'), show_flg=False)

#mesh
p_ = param[:12].reshape(3, -1)### 换了参数后这里的长度也要改
p = p_[:, :3]
offset = p_[:, -1].reshape(3, 1)
#alpha_shp = param[12:52].reshape(-1, 1)
alpha_shp = param[12:72].reshape(-1, 1)
#alpha_exp = param[52:].reshape(-1, 1)  
alpha_exp = param[72:].reshape(-1, 1)

vertex = p @ (u + w_shp @ alpha_shp + w_exp @ alpha_exp).reshape(3, -1, order='F') + offset
#vertex2 = (u + w_shp @ alpha_shp + w_exp @ alpha_exp).reshape(3, -1, order='F')
vertex[1, :] = 120 + 1 - vertex[1, :]
dump_to_ply(vertex, tri, img_fp.replace(suffix, '_origin.ply'))
#dump_to_ply(vertex2, tri, './test2.ply')

img = np.zeros((120, 120, 3))
# for i in range(len(vertices_lst)):

pncc_feature = cpncc(img, [vertex], tri - 1)
cv2.imwrite(img_fp.replace(suffix, '_origin.png'), pncc_feature[:, :, ::-1])  # cv2.imwrite will swap RGB -> BGR
