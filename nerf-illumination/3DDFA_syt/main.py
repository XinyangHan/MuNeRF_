#!/usr/bin/env python3
# coding: utf-8

__author__ = 'cleardusk'

"""
The pipeline of 3DDFA prediction: given one image, predict the 3d face vertices, 68 landmarks and visualization.

[todo]
1. CPU optimization: https://pmchojnacki.wordpress.com/2018/10/07/slow-pytorch-cpu-performance
"""
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
import json
from utils.ddfa import ToTensorGjz, NormalizeGjz, str2bool
import scipy.io as sio
from utils.inference import get_suffix, parse_roi_box_from_landmark, crop_img, predict_68pts, dump_to_ply, dump_vertex, \
    draw_landmarks, predict_dense, parse_roi_box_from_bbox, get_colors, write_obj_with_colors
from utils.cv_plot import plot_pose_box
from utils.estimate_pose import parse_pose
from utils.render import get_depths_image, cget_depths_image, cpncc
from utils.paf import gen_img_paf
import argparse
import torch.backends.cudnn as cudnn

import pickle
import glob

STD_SIZE = 120

input_dim = 62

def main(args):
    # 1. load pre-tained model
    if input_dim == 62 :
        checkpoint_fp = 'models/phase1_wpdc_vdc.pth.tar'
    else:
        checkpoint_fp = 'models/phase1_wpdc_checkpoint_epoch_50.pth.tar'
    arch = 'mobilenet_1'

    checkpoint = torch.load(checkpoint_fp, map_location=lambda storage, loc: storage)['state_dict']
    model = getattr(mobilenet_v1, arch)(num_classes=input_dim)  # 62 = 12(pose) + 40(shape) +10(expression)

    model_dict = model.state_dict()
    # because the model is trained by multiple gpus, prefix module should be removed
    for k in checkpoint.keys():
        model_dict[k.replace('module.', '')] = checkpoint[k]
    model.load_state_dict(model_dict)
    if args.mode == 'gpu':
        cudnn.benchmark = True
        model = model.cuda()
    model.eval()
    
    # 2. load dlib model for face detection and landmark used for face cropping
    if args.dlib_landmark:
        dlib_landmark_model = 'models/shape_predictor_68_face_landmarks.dat'
        face_regressor = dlib.shape_predictor(dlib_landmark_model)
    if args.dlib_bbox:
        face_detector = dlib.get_frontal_face_detector()
    jsonfile = '/mnt/d/heyue/nerf-illumination/jsons/cl/pose.json'
    # 3. forward
    tri = sio.loadmat('visualize/tri.mat')['tri']
    transform = transforms.Compose([ToTensorGjz(), NormalizeGjz(mean=127.5, std=128)])
    img_files = sorted(glob.glob(os.path.join(args.dir, '*.png')))
    img_files += sorted(glob.glob(os.path.join(args.dir, '*.JPG')))

    if args.save == None:
        args.save = args.dir.rstrip('/') + "_param"
    if not os.path.exists(args.save):
        os.mkdir(args.save)
    # for img_fp in args.files:
    posedict = {}
    for img_fp in img_files:
        filename = img_fp.split('/')[-1]
        print('filename here!!!!!!!', filename)
        img_ori = cv2.imread(img_fp)
        if args.dlib_bbox:
            rects = face_detector(img_ori, 1)
        else:
            rects = []

        if len(rects) == 0:
            continue
            rects = dlib.rectangles()
            rect_fp = img_fp + '.bbox'
            lines = open(rect_fp).read().strip().split('\n')[1:]
            for l in lines:
                l, r, t, b = [int(_) for _ in l.split(' ')[1:]]
                rect = dlib.rectangle(l, r, t, b)
                rects.append(rect)

        pts_res = []
        Ps = []  # Camera matrix collection
        poses = []  # pose collection, [todo: validate it]
        vertices_lst = []  # store multiple face vertices
        ind = 0
        suffix = get_suffix(img_fp)
        for rect in rects:
            # whether use dlib landmark to crop image, if not, use only face bbox to calc roi bbox for cropping
            if args.dlib_landmark:
                # - use landmark for cropping
                pts = face_regressor(img_ori, rect).parts()
                pts = np.array([[pt.x, pt.y] for pt in pts]).T
                roi_box = parse_roi_box_from_landmark(pts)
            else:
                # - use detected face bbox
                bbox = [rect.left(), rect.top(), rect.right(), rect.bottom()]
                roi_box = parse_roi_box_from_bbox(bbox)

            img = crop_img(img_ori, roi_box)

            # forward: one step
            img = cv2.resize(img, dsize=(STD_SIZE, STD_SIZE), interpolation=cv2.INTER_LINEAR)
            input = transform(img).unsqueeze(0)
            with torch.no_grad():
                if args.mode == 'gpu':
                    input = input.cuda()
                param = model(input)
                param = param.squeeze().cpu().numpy().flatten().astype(np.float32)

            # 68 pts
            pts68 = predict_68pts(param, roi_box)
            
            # two-step for more accurate bbox to crop face
            if args.bbox_init == 'two':
                roi_box = parse_roi_box_from_landmark(pts68)
                img_step2 = crop_img(img_ori, roi_box)
                img_step2 = cv2.resize(img_step2, dsize=(STD_SIZE, STD_SIZE), interpolation=cv2.INTER_LINEAR)
                input = transform(img_step2).unsqueeze(0)
                with torch.no_grad():
                    if args.mode == 'gpu':
                        input = input.cuda()
                    param = model(input)
                    param = param.squeeze().cpu().numpy().flatten().astype(np.float32)

                pts68 = predict_68pts(param, roi_box)

            pts_res.append(pts68)
            P, pose = parse_pose(param)
            trans = torch.from_numpy(P[:,3]).unsqueeze(0)
            rot = torch.from_numpy(P[:,:3]).unsqueeze(0)
            #trans= -torch.bmm(rot, trans.unsqueeze(2)) / 512.
            P1 = P
            rot = rot.squeeze().numpy()
            trans = trans.squeeze().numpy()
            #P1 = P.tolist()
            P1[:,:3] = rot
            P1[:,3] = trans
            a = [0.0,0.0,0.0,1.0]
            P1 = P1.tolist()
            P1.append(a)
            print('here for poses!!!!!!!!!!', P1, pose)
            posedict[filename] = P1
            Ps.append(P)
            poses.append(pose)

            vertices, info_dict = predict_dense(param, roi_box)
            #print('info_dict@@@@@@@@@@@@', info_dict)
            ### add image size ###
            info_dict['HEIGHT'] = img_ori.shape[0]
            info_dict['WIDTH'] = img_ori.shape[1]

            # wfp = '{}_{}.npy'.format(img_fp.replace(suffix, ''), ind)
            # np.save(wfp, vertices)

            new_fp = os.path.join(args.save, img_fp.split('/')[-1])
            #wfp = '{}_{}.pkl'.format(new_fp.replace(suffix, ''), ind)
            #print(wfp)
            #with open(wfp, 'wb') as f:
            #    pickle.dump(info_dict, f)
            ind += 1
            continue



            # dense face 3d vertices
            if args.dump_ply or args.dump_vertex or args.dump_depth or args.dump_pncc or args.dump_obj:
                vertices = predict_dense(param, roi_box)
                # vertices = np.dot(P[:3,:3].transpose(), vertices-P[:,3:])
                vertices_lst.append(vertices)
            if args.dump_ply:
                dump_to_ply(vertices, tri, '{}_{}.ply'.format(img_fp.replace(suffix, ''), ind))
            if args.dump_vertex:
                dump_vertex(vertices, '{}_{}.mat'.format(img_fp.replace(suffix, ''), ind))
            if args.dump_pts:
                wfp = '{}_{}.txt'.format(img_fp.replace(suffix, ''), ind)
                np.savetxt(wfp, pts68, fmt='%.3f')
                print('Save 68 3d landmarks to {}'.format(wfp))
            if args.dump_roi_box:
                wfp = '{}_{}.roibox'.format(img_fp.replace(suffix, ''), ind)
                np.savetxt(wfp, roi_box, fmt='%.3f')
                print('Save roi box to {}'.format(wfp))
            if args.dump_paf:
                wfp_paf = '{}_{}_paf.jpg'.format(img_fp.replace(suffix, ''), ind)
                wfp_crop = '{}_{}_crop.jpg'.format(img_fp.replace(suffix, ''), ind)
                paf_feature = gen_img_paf(img_crop=img, param=param, kernel_size=args.paf_size)

                cv2.imwrite(wfp_paf, paf_feature)
                cv2.imwrite(wfp_crop, img)
                print('Dump to {} and {}'.format(wfp_crop, wfp_paf))
            if args.dump_obj:
                wfp = '{}_{}.obj'.format(img_fp.replace(suffix, ''), ind)
                colors = get_colors(img_ori, vertices)
                write_obj_with_colors(wfp, vertices, tri, colors)
                print('Dump obj with sampled texture to {}'.format(wfp))

        continue
        if args.dump_pose:
            # P, pose = parse_pose(param)  # Camera matrix (without scale), and pose (yaw, pitch, roll, to verify)
            img_pose = plot_pose_box(img_ori, Ps, pts_res)
            wfp = img_fp.replace(suffix, '_pose.jpg')
            cv2.imwrite(wfp, img_pose)
            print('Dump to {}'.format(wfp))
        if args.dump_depth:
            wfp = img_fp.replace(suffix, '_depth.png')
            # depths_img = get_depths_image(img_ori, vertices_lst, tri-1)  # python version
            depths_img = cget_depths_image(img_ori, vertices_lst, tri - 1)  # cython version
            cv2.imwrite(wfp, depths_img)
            print('Dump to {}'.format(wfp))
        if args.dump_pncc:
            wfp = img_fp.replace(suffix, '_pncc.png')
            pncc_feature = cpncc(img_ori, vertices_lst, tri - 1)  # cython version
            cv2.imwrite(wfp, pncc_feature[:, :, ::-1])  # cv2.imwrite will swap RGB -> BGR
            print('Dump to {}'.format(wfp))
        if args.dump_res:
            draw_landmarks(img_ori, pts_res, wfp=img_fp.replace(suffix, '_3DDFA.jpg'), show_flg=args.show_flg)

        # 检查-----------------------------------------------------------------

        #select a data and calc the param------------------------
        #search
        fileindex = -1
        npylist = open("./train.configs/train_aug_120x120.list.train")
        index = -1
        for line in tqdm(npylist.readlines()):
            index +=1
            if line.replace("\n","") == os.path.basename(img_fp):
                fileindex = index
                print("FIND!!!!!!!!:",index)
                break
        
        if fileindex != -1 :
            roi_box = [0,0,120,120]
            param = params_data[fileindex,:]

            #landmark
            pts68 = predict_68pts(param, roi_box)
            draw_landmarks(img_ori, pts68, wfp=img_fp.replace(suffix, '_origin_ldmk.jpg'), show_flg=args.show_flg)
            
            #mesh
            vertices = predict_dense(param, roi_box) #p @ (u + w_shp @ alpha_shp + w_exp @ alpha_exp).reshape(3, -1, order='F') + offset
            #vertices[1, :] = 120 + 1 - vertices[1, :]
            dump_to_ply(vertices, tri, img_fp.replace(suffix, '_origin.ply'))

            img = np.zeros((120, 120, 3))
            pncc_feature = cpncc(img, [vertices], tri - 1)
            cv2.imwrite(img_fp.replace(suffix, '_origin.png'), pncc_feature[:, :, ::-1])
    
    with open(jsonfile,"w") as f:
            json.dump(posedict,f,indent=1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='3DDFA inference pipeline')
    parser.add_argument('-f', '--files', nargs='+',
                        help='image files paths fed into network, single or multiple images')
    parser.add_argument('--dir',
                        help='image files paths fed into network, single or multiple images')
    parser.add_argument('--save',
                        help='image files paths fed into network, single or multiple images')

    parser.add_argument('-m', '--mode', default='cpu', type=str, help='gpu or cpu mode')
    parser.add_argument('--show_flg', default='false', type=str2bool, help='whether show the visualization result')
    parser.add_argument('--bbox_init', default='one', type=str,
                        help='one|two: one-step bbox initialization or two-step')
    parser.add_argument('--dump_res', default='true', type=str2bool, help='whether write out the visualization image')
    parser.add_argument('--dump_vertex', default='false', type=str2bool,
                        help='whether write out the dense face vertices to mat')
    parser.add_argument('--dump_ply', default='true', type=str2bool)
    parser.add_argument('--dump_pts', default='true', type=str2bool)
    parser.add_argument('--dump_roi_box', default='false', type=str2bool)
    parser.add_argument('--dump_pose', default='true', type=str2bool)
    parser.add_argument('--dump_depth', default='true', type=str2bool)
    parser.add_argument('--dump_pncc', default='true', type=str2bool)
    parser.add_argument('--dump_paf', default='false', type=str2bool)
    parser.add_argument('--paf_size', default=3, type=int, help='PAF feature kernel size')
    parser.add_argument('--dump_obj', default='true', type=str2bool)
    parser.add_argument('--dlib_bbox', default='true', type=str2bool, help='whether use dlib to predict bbox')
    parser.add_argument('--dlib_landmark', default='true', type=str2bool,
                        help='whether use dlib landmark to crop image')

    args = parser.parse_args()
    main(args)
