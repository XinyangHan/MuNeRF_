import json
import os

import cv2
import imageio
import numpy as np
import torch
import time
from torchvision import utils
from utils.api_util import FacePPAPI
import pdb

def translate_by_t_along_z(t):
    tform = np.eye(4).astype(np.float32)
    tform[2][3] = t
    return tform


def rotate_by_phi_along_x(phi):
    tform = np.eye(4).astype(np.float32)
    tform[1, 1] = tform[2, 2] = np.cos(phi)
    tform[1, 2] = -np.sin(phi)
    tform[2, 1] = -tform[1, 2]
    return tform


def rotate_by_theta_along_y(theta):
    tform = np.eye(4).astype(np.float32)
    tform[0, 0] = tform[2, 2] = np.cos(theta)
    tform[0, 2] = -np.sin(theta)
    tform[2, 0] = -tform[0, 2]
    return tform


def pose_spherical(theta, phi, radius):
    c2w = translate_by_t_along_z(radius)
    c2w = rotate_by_phi_along_x(phi / 180.0 * np.pi) @ c2w
    c2w = rotate_by_theta_along_y(theta / 180 * np.pi) @ c2w
    c2w = np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]) @ c2w
    return c2w

# HXY 1009
# Purpose : test the expression and pose of nerface and munerf
def load_flame_data_H(basedir, half_res=False, scale=0.5, testskip=1, debug=False, expressions=True,load_frontal_faces=False, load_bbox=True, test=False):
    print("starting data loading")
    splits = ["train", "val", "test"]
    if test:
        splits = ["val"]
    metas = {}
    for s in splits:
        with open(os.path.join(basedir, f"transforms_{s}.json"), "r") as fp:
            metas[s] = json.load(fp)
    with open(os.path.join(basedir, 'box.json'), "r") as fp:
        metab = json.load(fp)

    all_frontal_imgs = []
    all_imgs = []
    all_poses = []
    all_expressions = []
    all_bboxs = []
    all_img_ids = []
    counts = [0]
    focal = 1100     # default value
    for s in splits:
        meta = metas[s]
        imgs = []
        poses = []
        expressions = []
        frontal_imgs = []
        bboxs = []
        img_ids = []
        if s == "train" or testskip == 0:
            skip = 1
        else:
            skip = testskip
        focal = meta['focal_len']

        # pdb.set_trace(s)
        for frame in meta["frames"][::skip]:
            fsname = (str(frame['img_id'])).zfill(4) + '.png'
            fname = os.path.join(basedir,s,fsname)
            
            # fname = os.path.join(basedir, frame["file_path"] + ".png")
            imgs.append(imageio.imread(fname))
            if load_frontal_faces:
                fname = os.path.join(basedir, (str(frame['img_id'])).zfill(4) + "_frontal" + ".png")
                # fname = os.path.join(basedir, frame["file_path"] + "_frontal" + ".png")
                frontal_imgs.append(imageio.imread(fname))
            # import matplotlib.pyplot as plt
            # plt.imshow(imgs[-1])
            # plt.show()
            # pdb.set_trace()
            poses.append(np.array(frame["transform_matrix"]))
            expressions.append(np.array(frame["expression"]))
            img_ids.append(fsname)
            if load_bbox:
                # if "bbox" not in frame.keys():
                #     bboxs.append(np.array([0.0,1.0,0.0,1.0]))
                # else:
                #     bboxs.append(np.array(frame["bbox"]))
                bboxs.append(np.array(metab[fsname]))

        imgs = (np.array(imgs) / 255.0).astype(np.float32)
        if load_frontal_faces:
            frontal_imgs = (np.array(frontal_imgs) / 255.0).astype(np.float32)

        poses = np.array(poses).astype(np.float32)
        expressions = np.array(expressions).astype(np.float32)
        bboxs = np.array(bboxs).astype(np.float32)

        counts.append(counts[-1] + imgs.shape[0])
        all_imgs.append(imgs)
        all_frontal_imgs.append(frontal_imgs)
        all_poses.append(poses)
        all_expressions.append(expressions)
        all_bboxs.append(bboxs)
        all_img_ids.append(img_ids)

    i_split = [np.arange(counts[i], counts[i + 1]) for i in range(len(splits))]

    imgs = np.concatenate(all_imgs, 0)
    frontal_imgs = np.concatenate(all_frontal_imgs, 0) if load_frontal_faces else None
    poses = np.concatenate(all_poses, 0)
    expressions = np.concatenate(all_expressions, 0)
    bboxs = np.concatenate(all_bboxs, 0)
    img_ids = np.concatenate(all_img_ids, 0)

    H, W = imgs[0].shape[:2]
    # camera_angle_x = float(meta["camera_angle_x"])
    # focal = 0.5 * W / np.tan(0.5 * camera_angle_x)
    # focal = 1100

    #focals = (meta["focals"])
    # intrinsics = meta["intrinsics"] if meta["intrinsics"] else None
    # if meta["intrinsics"]:
    #     intrinsics = np.array(meta["intrinsics"])
    # else:
    #     intrinsics = np.array([focal, focal, 0.5, 0.5]) # fx fy cx cy
    intrinsics = np.array([focal, focal, 0.5, 0.5]) # fx fy cx cy
    # if type(focals) is list:
    #     focal = np.array([W*focals[0], H*focals[1]]) # fx fy  - x is width
    # else:
    #     focal = np.array([focal, focal])


    # In debug mode, return extremely tiny images
    if debug:
        H = H // 32
        W = W // 32
        #focal = focal / 32.0
        intrinsics[:2] = intrinsics[:2] / 32.0
        imgs = [
            torch.from_numpy(
                cv2.resize(imgs[i], dsize=(25, 25), interpolation=cv2.INTER_AREA)
            )
            for i in range(imgs.shape[0])
        ]
        imgs = torch.stack(imgs, 0)
        if frontal_imgs:
            frontal_imgs = [
                torch.from_numpy(
                    cv2.resize(frontal_imgs[i], dsize=(25, 25), interpolation=cv2.INTER_AREA)
                )
                for i in range(frontal_imgs.shape[0])
            ]
            frontal_imgs = torch.stack(frontal_imgs, 0)

        poses = torch.from_numpy(poses)

        return imgs, poses, render_poses, [H, W, intrinsics], i_split, frontal_imgs


    if half_res:
        # TODO: resize images using INTER_AREA (cv2)
        H = int(H * scale)
        W = int(W * scale)
        #focal = focal / 2.0
        intrinsics[:2] = intrinsics[:2] * scale
        imgs = [
            torch.from_numpy(
                #cv2.resize(imgs[i], dsize=(400, 400), interpolation=cv2.INTER_AREA)
                cv2.resize(imgs[i], dsize=(H, W), interpolation=cv2.INTER_AREA)
            )
            for i in range(imgs.shape[0])
        ]
        imgs = torch.stack(imgs, 0)
        if load_frontal_faces:
            frontal_imgs = [
                torch.from_numpy(
                    #cv2.resize(imgs[i], dsize=(400, 400), interpolation=cv2.INTER_AREA)
                    cv2.resize(frontal_imgs[i], dsize=(H, W), interpolation=cv2.INTER_AREA)
                )
                for i in range(frontal_imgs.shape[0])
            ]
            frontal_imgs = torch.stack(frontal_imgs, 0)

    else:
        imgs = [
            torch.from_numpy(imgs[i]
                # cv2.resize(imgs[i], dsize=(400, 400), interpolation=cv2.INTER_AREA)
                #cv2.resize(imgs[i], dsize=(H, W), interpolation=cv2.INTER_AREA)
            )
            for i in range(imgs.shape[0])
        ]
        imgs = torch.stack(imgs, 0)
        if load_frontal_faces:
            frontal_imgs = [
                torch.from_numpy(frontal_imgs[i]
                                 # cv2.resize(imgs[i], dsize=(400, 400), interpolation=cv2.INTER_AREA)
                                 # cv2.resize(imgs[i], dsize=(H, W), interpolation=cv2.INTER_AREA)
                                 )
                for i in range(frontal_imgs.shape[0])
            ]
            frontal_imgs = torch.stack(frontal_imgs, 0)

    poses = torch.from_numpy(poses)
    expressions = torch.from_numpy(expressions)
    bboxs[:,0:2] *= H
    bboxs[:,2:4] *= W
    bboxs = np.floor(bboxs)
    bboxs = torch.from_numpy(bboxs).int()
    print("Done with data loading")
    
    render_poses = torch.stack(
        [
            torch.from_numpy(pose_spherical(angle, -90, 0.99))
            for angle in np.linspace(-30, 30, 40+1)[:-1]
        ],
        0,
    )
    # pdb.set_trace()
    return imgs, poses, render_poses, [H, W, intrinsics], i_split, expressions, frontal_imgs, bboxs, img_ids

def load_flame_data(basedir, half_res=False, scale=0.5, testskip=1, debug=False, expressions=True,load_frontal_faces=False, load_bbox=True, test=False):
    print("starting data loading")
    splits = ["train", "val", "test"]
    if test:
        splits = ["val"]
    metas = {}
    for s in splits:
        with open(os.path.join(basedir, f"transforms_{s}.json"), "r") as fp:
            metas[s] = json.load(fp)
    with open(os.path.join(basedir, 'box.json'), "r") as fp:
        metab = json.load(fp)

    all_frontal_imgs = []
    all_imgs = []
    all_poses = []
    all_expressions = []
    all_bboxs = []
    all_img_ids = []
    counts = [0]
    focal = 1100     # default value
    for s in splits:
        meta = metas[s]
        imgs = []
        poses = []
        expressions = []
        frontal_imgs = []
        bboxs = []
        img_ids = []
        if s == "train" or testskip == 0:
            skip = 1
        else:
            skip = testskip
        focal = meta['focal_len']

        # pdb.set_trace(s)
        for frame in meta["frames"][::skip]:
            fsname = (str(frame['img_id'])).zfill(4) + '.png'
            fname = os.path.join(basedir,s,fsname)
            
            # fname = os.path.join(basedir, frame["file_path"] + ".png")
            imgs.append(imageio.imread(fname))
            if load_frontal_faces:
                fname = os.path.join(basedir, (str(frame['img_id'])).zfill(4) + "_frontal" + ".png")
                # fname = os.path.join(basedir, frame["file_path"] + "_frontal" + ".png")
                frontal_imgs.append(imageio.imread(fname))
            # import matplotlib.pyplot as plt
            # plt.imshow(imgs[-1])
            # plt.show()
            # pdb.set_trace()
            poses.append(np.array(frame["transform_matrix"]))
            expressions.append(np.array(frame["expression"]))
            img_ids.append(fsname)
            if load_bbox:
                # if "bbox" not in frame.keys():
                #     bboxs.append(np.array([0.0,1.0,0.0,1.0]))
                # else:
                #     bboxs.append(np.array(frame["bbox"]))
                bboxs.append(np.array(metab[fsname]))

        imgs = (np.array(imgs) / 255.0).astype(np.float32)
        if load_frontal_faces:
            frontal_imgs = (np.array(frontal_imgs) / 255.0).astype(np.float32)

        poses = np.array(poses).astype(np.float32)
        expressions = np.array(expressions).astype(np.float32)
        bboxs = np.array(bboxs).astype(np.float32)

        counts.append(counts[-1] + imgs.shape[0])
        all_imgs.append(imgs)
        all_frontal_imgs.append(frontal_imgs)
        all_poses.append(poses)
        all_expressions.append(expressions)
        all_bboxs.append(bboxs)
        all_img_ids.append(img_ids)

    i_split = [np.arange(counts[i], counts[i + 1]) for i in range(len(splits))]

    imgs = np.concatenate(all_imgs, 0)
    frontal_imgs = np.concatenate(all_frontal_imgs, 0) if load_frontal_faces else None
    poses = np.concatenate(all_poses, 0)
    expressions = np.concatenate(all_expressions, 0)
    bboxs = np.concatenate(all_bboxs, 0)
    img_ids = np.concatenate(all_img_ids, 0)

    H, W = imgs[0].shape[:2]
    # camera_angle_x = float(meta["camera_angle_x"])
    # focal = 0.5 * W / np.tan(0.5 * camera_angle_x)
    # focal = 1100

    #focals = (meta["focals"])
    # intrinsics = meta["intrinsics"] if meta["intrinsics"] else None
    # if meta["intrinsics"]:
    #     intrinsics = np.array(meta["intrinsics"])
    # else:
    #     intrinsics = np.array([focal, focal, 0.5, 0.5]) # fx fy cx cy
    intrinsics = np.array([focal, focal, 0.5, 0.5]) # fx fy cx cy
    # if type(focals) is list:
    #     focal = np.array([W*focals[0], H*focals[1]]) # fx fy  - x is width
    # else:
    #     focal = np.array([focal, focal])


    # In debug mode, return extremely tiny images
    if debug:
        H = H // 32
        W = W // 32
        #focal = focal / 32.0
        intrinsics[:2] = intrinsics[:2] / 32.0
        imgs = [
            torch.from_numpy(
                cv2.resize(imgs[i], dsize=(25, 25), interpolation=cv2.INTER_AREA)
            )
            for i in range(imgs.shape[0])
        ]
        imgs = torch.stack(imgs, 0)
        if frontal_imgs:
            frontal_imgs = [
                torch.from_numpy(
                    cv2.resize(frontal_imgs[i], dsize=(25, 25), interpolation=cv2.INTER_AREA)
                )
                for i in range(frontal_imgs.shape[0])
            ]
            frontal_imgs = torch.stack(frontal_imgs, 0)

        poses = torch.from_numpy(poses)

        return imgs, poses, render_poses, [H, W, intrinsics], i_split, frontal_imgs


    if half_res:
        # TODO: resize images using INTER_AREA (cv2)
        H = int(H * scale)
        W = int(W * scale)
        #focal = focal / 2.0
        intrinsics[:2] = intrinsics[:2] * scale
        imgs = [
            torch.from_numpy(
                #cv2.resize(imgs[i], dsize=(400, 400), interpolation=cv2.INTER_AREA)
                cv2.resize(imgs[i], dsize=(H, W), interpolation=cv2.INTER_AREA)
            )
            for i in range(imgs.shape[0])
        ]
        imgs = torch.stack(imgs, 0)
        if load_frontal_faces:
            frontal_imgs = [
                torch.from_numpy(
                    #cv2.resize(imgs[i], dsize=(400, 400), interpolation=cv2.INTER_AREA)
                    cv2.resize(frontal_imgs[i], dsize=(H, W), interpolation=cv2.INTER_AREA)
                )
                for i in range(frontal_imgs.shape[0])
            ]
            frontal_imgs = torch.stack(frontal_imgs, 0)

    else:
        imgs = [
            torch.from_numpy(imgs[i]
                # cv2.resize(imgs[i], dsize=(400, 400), interpolation=cv2.INTER_AREA)
                #cv2.resize(imgs[i], dsize=(H, W), interpolation=cv2.INTER_AREA)
            )
            for i in range(imgs.shape[0])
        ]
        imgs = torch.stack(imgs, 0)
        if load_frontal_faces:
            frontal_imgs = [
                torch.from_numpy(frontal_imgs[i]
                                 # cv2.resize(imgs[i], dsize=(400, 400), interpolation=cv2.INTER_AREA)
                                 # cv2.resize(imgs[i], dsize=(H, W), interpolation=cv2.INTER_AREA)
                                 )
                for i in range(frontal_imgs.shape[0])
            ]
            frontal_imgs = torch.stack(frontal_imgs, 0)

    poses = torch.from_numpy(poses)
    expressions = torch.from_numpy(expressions)
    bboxs[:,0:2] *= H
    bboxs[:,2:4] *= W
    bboxs = np.floor(bboxs)
    bboxs = torch.from_numpy(bboxs).int()
    print("Done with data loading")
    
    render_poses = torch.stack(
        [
            torch.from_numpy(pose_spherical(angle, -90, 0.99))
            for angle in np.linspace(-30, 30, 40+1)[:-1]
        ],
        0,
    )
    # pdb.set_trace()
    return imgs, poses, render_poses, [H, W, intrinsics], i_split, expressions, frontal_imgs, bboxs, img_ids

def load_flame_data_color_multi(basedir, style_id, half_res=False, scale=0.5, testskip=1, debug=False, expressions=True,load_frontal_faces=False, load_bbox=True, test=False, consistent=False):
    print("starting data loading")
    splits = ["train", "val", "test"]
    if test:
        splits = ["val"]
    
    with open(os.path.join(basedir, 'box.json'), "r") as fp:
        metab = json.load(fp)
    
    metas = {}
    for s in splits:
        with open(os.path.join(basedir, f"transforms_{s}.json"), "r") as fp:
            metas[s] = json.load(fp)
    
    style_id_lip = str(style_id[0])
    style_id_eyes = str(style_id[1])
    style_id_skin = str(style_id[2])
    base_makeup_dir = os.path.join(basedir, 'makeup')
    #lip
    base_warp_dir_lip = os.path.join(basedir, style_id_lip, 'warp_makeup'+style_id)
    warp_dir_lip = 'warp_makeup_' + style_id_lip
    base_mask_dir_lip = os.path.join(basedir, style_id_lip, 'mask')
    
    base_warp_dir_eyes = os.path.join(basedir, style_id_eyes, 'warp_makeup_'+style_id)
    warp_dir_eyes = 'warp_makeup_' + style_id_eyes
    base_mask_dir_eyes = os.path.join(basedir, style_id_eyes, 'mask')
    
    base_warp_dir_skin = os.path.join(basedir, style_id_skin, 'warp_makeup_'+style_id)
    warp_dir_skin = 'warp_makeup_' + style_id_skin
    base_mask_dir_skin = os.path.join(basedir, style_id_skin, 'mask')
    
    base_depth_dir = os.path.join(basedir, 'depth')
    os.makedirs(os.path.join(basedir,'half_res','landmark'), exist_ok=True)   

    all_frontal_imgs = []
    all_imgs = []
    all_wimgs_lip = []
    all_wimgs_eyes = []
    all_wimgs_skin = []
    
    all_mimgs_lip = []
    all_mimgs_eyes = []
    all_mimgs_skin = []
    all_poses = []
    all_expressions = []
    all_bboxs = []
    all_depths = []
    all_masks = []
    all_masks_lip = []
    all_masks_eyes = []
    all_masks_skin = []
    all_img_ids = []
    counts = [0]
    paths = {}
    paths['warped'] = []
    paths['makeup'] = []
    api = FacePPAPI()
    focal = 1100     # default value
    for s in splits:
        print('now process',s)
        meta = metas[s]
        imgs = []
        
        warped_imgs_lip = []
        warped_imgs_eyes = []
        warped_imgs_skin = []
        
        makeup_styles_lip = []
        makeup_styles_eyes = []
        makeup_styles_skin = []
        poses = []
        expressions = []
        frontal_imgs = []
        bboxs = []
        masks = []
        masks_lip = []
        masks_eyes = []
        masks_skin = []
        depths = []
        img_ids = []
        if s == "train" or testskip == 0:
            skip = 1
        else:
            skip = testskip
        focal = meta['focal_len']

        for frame in meta["frames"][::skip]:
            fsname = (str(frame['img_id'])).zfill(4) + '.png' # frame["file_path"].split('/')[-1] + ".png"
            fname = os.path.join(basedir, s, fsname) # nomakeup image
            
            fwname_lip = os.path.join(base_warp_dir_lip, fsname) # warped makeup image
            fwname_eyes = os.path.join(base_warp_dir_eyes, fsname) 
            fwname_skin = os.path.join(base_warp_dir_skin, fsname) 
            
            fwname_ = os.path.join(basedir,'half_res',fsname)
            fwname_landmark = os.path.join(basedir,'half_res','landmark',(str(frame['img_id'])).zfill(4) + '.npy')
            fmname = os.path.join(basedir, 'mask', fsname) # nomakeup image mask
            
            fmname_lip = os.path.join(base_mask_dir_lip, warp_dir_lip, fsname) # warped makeup image mask
            fmname_eyes = os.path.join(base_mask_dir_eyes, warp_dir_eyes, fsname) # warped makeup image mask
            fmname_skin = os.path.join(base_mask_dir_skin, warp_dir_skin, fsname) # warped makeup image mask
            
            fdname = os.path.join(base_depth_dir, fsname) # depth image
            # load makeup image
            fname_makeup_lip = os.path.join(base_makeup_dir, style_id_lip+'.jpg')# reference makeup image
            fname_makeup_eyes = os.path.join(base_makeup_dir, style_id_eyes+'.jpg')# reference makeup image
            fname_makeup_skin = os.path.join(base_makeup_dir, style_id_skin+'.jpg')# reference makeup image
            fname_makeup_landmark_lip = os.path.join(base_makeup_dir,'landmark',style_id_lip + '.npy')
            #fname_makeup_landmark_eyes = os.path.join(base_makeup_dir,'landmark',style_id_lip + '.npy')
            #fname_makeup_landmark_skin = os.path.join(base_makeup_dir,'landmark',style_id_lip + '.npy')
            # pdb.set_trace()
            print("fwname : %s"%fwname)
            if os.path.exists(fwname):
                warped_imgs_lip = imageio.imread(fwname_lip)
                warped_imgs_eyes = imageio.imread(fwname_eyes)
                warped_imgs_skin = imageio.imread(fwname_skin)
                if os.path.exists(fwname_landmark):
                    landmark_B_api = np.load(fwname_landmark, allow_pickle=True)
                    landmark_B_api = landmark_B_api.item()
                else:
                    landmark_B_api = api.faceLandmarkDetector(fwname_)
                    np.save(fwname_landmark, landmark_B_api)
                    time.sleep(0.5)
                paths['warped'].append(landmark_B_api)
                # load makeup image
                makeup_styles_lip.append(imageio.imread(fname_makeup_lip))
                makeup_styles_eyes.append(imageio.imread(fname_makeup_eyes))
                makeup_styles_skin.append(imageio.imread(fname_makeup_skin))
                if os.path.exists(fname_makeup_landmark_lip):
                    landmark_M_api = np.load(fname_makeup_landmark_lip, allow_pickle=True)
                    landmark_M_api = landmark_M_api.item()
                else:
                    landmark_M_api = api.faceLandmarkDetector(fname_makeup_lip)
                    np.save(fname_makeup_landmark_lip, landmark_M_api)
                    time.sleep(0.5)

                paths['makeup'].append(landmark_M_api)
                imgs.append(imageio.imread(fname))
                masks.append(imageio.imread(fmname))
                masks_lip = imageio.imread(fmname_lip)
                masks_eyes = imageio.imread(fmname_eyes)
                masks_skin = imageio.imread(fmname_skin)
                
                if os.path.exists(fdname):
                    depths.append(imageio.imread(fdname))
                poses.append(np.array(frame["transform_matrix"]))
                bboxs.append(np.array(metab[fsname]))
                expressions.append(np.array(frame["expression"]))#.append(expression) #
                img_ids.append(fsname)
            else:
                # pdb.set_trace()
                print("skip ", fsname)
                # continue

            if load_frontal_faces:
                fname = os.path.join(basedir, (str(frame['img_id'])).zfill(4) + "_frontal" + ".png")
                frontal_imgs.append(imageio.imread(fname))
            light_flag = False
            '''if load_bbox:
                if "bbox" not in frame.keys():
                    bboxs.append(np.array([0.0,1.0,0.0,1.0]))
                else:
                    bboxs.append(np.array(frame["bbox"]))
           '''
        imgs = (np.array(imgs) / 255.0).astype(np.float32)
        warped_imgs_lip = (np.array(warped_imgs_lip) / 255.0).astype(np.float32) 
        warped_imgs_eyes = (np.array(warped_imgs_eyes) / 255.0).astype(np.float32) 
        warped_imgs_skin = (np.array(warped_imgs_skin) / 255.0).astype(np.float32) 
        all_wimgs_lip.append(warped_imgs_lip)
        all_wimgs_eyes.append(warped_imgs_eyes)
        all_wimgs_skin.append(warped_imgs_skin)
        
        makeup_styles_lip = (np.array(makeup_styles_lip) / 255.0).astype(np.float32)
        all_mimgs_lip.append(makeup_styles_lip)
        makeup_styles_eyes = (np.array(makeup_styles_eyes) / 255.0).astype(np.float32)
        all_mimgs_eyes.append(makeup_styles_eyes)
        makeup_styles_skin = (np.array(makeup_styles_skin) / 255.0).astype(np.float32)
        all_mimgs_skin.append(makeup_styles_skin)

        masks = (np.array(masks) / 255.0).astype(np.float32)
        all_masks.append(masks)
        masks_lip = (np.array(masks_lip) / 255.0).astype(np.float32)
        all_masks_lip.append(masks_lip)
        masks_eyes = (np.array(masks_eyes) / 255.0).astype(np.float32)
        all_masks_eyes.append(masks_eyes)
        masks_skin = (np.array(masks_skin) / 255.0).astype(np.float32)
        all_masks_skin.append(masks_skin)

        if len(depths) > 0:
            depths = (np.array(depths) / 255.0).astype(np.float32)
            all_depths.append(depths)
        
        if load_frontal_faces:
            frontal_imgs = (np.array(frontal_imgs) / 255.0).astype(np.float32)

        poses = np.array(poses).astype(np.float32)
        expressions = np.array(expressions).astype(np.float32)
        bboxs = np.array(bboxs).astype(np.float32)
        counts.append(counts[-1] + imgs.shape[0])
        all_imgs.append(imgs)
        all_frontal_imgs.append(frontal_imgs)
        all_poses.append(poses)
        all_expressions.append(expressions)
        all_bboxs.append(bboxs)
        all_img_ids.append(img_ids)

    i_split = [np.arange(counts[i], counts[i + 1]) for i in range(len(splits))]
    
    warped_imgs_lip = np.concatenate(all_wimgs_lip, 0)
    warped_imgs_eyes = np.concatenate(all_wimgs_eyes, 0)
    warped_imgs_skin = np.concatenate(all_wimgs_skin, 0)
    
    makeup_imgs_lip = np.concatenate(all_mimgs_lip, 0)
    makeup_imgs_eyes = np.concatenate(all_mimgs_eyes, 0)
    makeup_imgs_skin = np.concatenate(all_mimgs_skin, 0)
    
    frontal_imgs = np.concatenate(all_frontal_imgs, 0) if load_frontal_faces else None
    poses = np.concatenate(all_poses, 0)
    expressions = np.concatenate(all_expressions, 0)
    bboxs = np.concatenate(all_bboxs, 0)
    masks = np.concatenate(all_masks, 0)
    masks_lip = np.concatenate(all_masks_lip, 0)
    masks_eyes = np.concatenate(all_masks_eyes, 0)
    masks_skin = np.concatenate(all_masks_skin, 0)
    if len(all_depths) > 0:
        depths = np.concatenate(all_depths, 0)
    else:
        depths = []
    imgs = np.concatenate(all_imgs, 0)
    #lights = np.concatenate(all_lights, 0)
    #details = np.concatenate(all_details, 0)
    img_ids = np.concatenate(all_img_ids, 0)
    H, W = imgs[0].shape[:2]
    #camera_angle_x = float(meta["camera_angle_x"])
    # focal = 1100 #0.5 * W / np.tan(0.5 * camera_angle_x)
    #print('focal,', camera_angle_x, focal)
    #focals = (meta["focal_len"])
    #if_intrinsics = False
    #intrinsics = meta["intrinsics"] if meta["intrinsics"] else None
    #if if_intrinsics:
    #    intrinsics = np.array(meta["intrinsics"])
   # else:
    intrinsics = np.array([focal, focal, 0.5, 0.5]) # fx fy cx cy
    
    # if type(focals) is list:
    #     focal = np.array([W*focals[0], H*focals[1]]) # fx fy  - x is width
    # else:
    #     focal = np.array([focal, focal])


    render_poses = torch.stack(
        [
            torch.from_numpy(pose_spherical(angle, -30.0, 4.0))
            for angle in np.linspace(-180, 180, 40 + 1)[:-1]
        ],
        0,
    )

    # In debug mode, return extremely tiny images
    #debug = True
    if debug:
        H = H // 2
        W = W // 2
        #focal = focal / 32.0
        intrinsics[:2] = intrinsics[:2] / 32.0
        imgs = [
            torch.from_numpy(
                cv2.resize(imgs[i], dsize=(25, 25), interpolation=cv2.INTER_AREA)
            )
            for i in range(imgs.shape[0])
        ]
        imgs = torch.stack(imgs, 0)
        if frontal_imgs:
            frontal_imgs = [
                torch.from_numpy(
                    cv2.resize(frontal_imgs[i], dsize=(25, 25), interpolation=cv2.INTER_AREA)
                )
                for i in range(frontal_imgs.shape[0])
            ]
            frontal_imgs = torch.stack(frontal_imgs, 0)

        poses = torch.from_numpy(poses)

        return imgs, poses, render_poses, [H, W, intrinsics], i_split, frontal_imgs,


    if half_res:
        # TODO: resize images using INTER_AREA (cv2)
        H = int(H * scale)
        W = int(W * scale)
        #focal = focal / 2.0
        intrinsics[:2] = intrinsics[:2] * scale
        imgs = [
            torch.from_numpy(
                cv2.resize(imgs[i], dsize=(H, W), interpolation=cv2.INTER_AREA)
            )
            for i in range(imgs.shape[0])
        ]
        imgs = torch.stack(imgs, 0)
        # warped_img
        warped_imgs_lip = [
            torch.from_numpy(
                cv2.resize(warped_imgs_lip[i], dsize=(H, W), interpolation=cv2.INTER_AREA)
            )
            for i in range(warped_imgs_lip.shape[0])
        ]
        warped_imgs_lip = torch.stack(warped_imgs_lip, 0)
        warped_imgs_eyes = [
            torch.from_numpy(
                cv2.resize(warped_imgs_eyes[i], dsize=(H, W), interpolation=cv2.INTER_AREA)
            )
            for i in range(warped_imgs_eyes.shape[0])
        ]
        warped_imgs_eyes = torch.stack(warped_imgs_eyes, 0)
        
        warped_imgs_skin = [
            torch.from_numpy(
                cv2.resize(warped_imgs_skin[i], dsize=(H, W), interpolation=cv2.INTER_AREA)
            )
            for i in range(warped_imgs_skin.shape[0])
        ]
        warped_imgs_skin = torch.stack(warped_imgs_skin, 0)
        
        makeup_imgs_lip = [
            torch.from_numpy(
                cv2.resize(makeup_imgs_lip[i], dsize=(H, W), interpolation=cv2.INTER_AREA)
            )
            for i in range(makeup_imgs_lip.shape[0])
        ]
        makeup_imgs_lip = torch.stack(makeup_imgs_lip, 0)
        makeup_imgs_eyes = [
            torch.from_numpy(
                cv2.resize(makeup_imgs_eyes[i], dsize=(H, W), interpolation=cv2.INTER_AREA)
            )
            for i in range(makeup_imgs_eyes.shape[0])
        ]
        makeup_imgs_eyes = torch.stack(makeup_imgs_eyes, 0)
        makeup_imgs_skin = [
            torch.from_numpy(
                cv2.resize(makeup_imgs_skin[i], dsize=(H, W), interpolation=cv2.INTER_AREA)
            )
            for i in range(makeup_imgs_skin.shape[0])
        ]
        makeup_imgs_skin = torch.stack(makeup_imgs_skin, 0)

        
        masks = [
            torch.from_numpy(
                cv2.resize(masks[i], dsize=(H, W), interpolation=cv2.INTER_AREA)
            )
            for i in range(masks.shape[0])
        ]
        masks = torch.stack(masks, 0)
        masks_lip = [
            torch.from_numpy(
                cv2.resize(masks_lip[i], dsize=(H, W), interpolation=cv2.INTER_AREA)
            )
            for i in range(masks_lip.shape[0])
        ]
        masks_lip = torch.stack(masks_lip, 0)
        masks_eyes = [
            torch.from_numpy(
                cv2.resize(masks_eyes[i], dsize=(H, W), interpolation=cv2.INTER_AREA)
            )
            for i in range(masks_eyes.shape[0])
        ]
        masks_eyes = torch.stack(masks_eyes, 0)
        masks_skin = [
            torch.from_numpy(
                cv2.resize(masks_skin[i], dsize=(H, W), interpolation=cv2.INTER_AREA)
            )
            for i in range(masks_skin.shape[0])
        ]
        masks_skin = torch.stack(masks_skin, 0)
        if len(depths) > 0:
            depths = [
                torch.from_numpy(
                    cv2.resize(depths[i], dsize=(H, W), interpolation=cv2.INTER_AREA)
                )
                for i in range(depths.shape[0])
            ]
            depths = torch.stack(depths, 0)
        else:
            depths = None
        if load_frontal_faces:
            frontal_imgs = [
                torch.from_numpy(
                    cv2.resize(frontal_imgs[i], dsize=(H, W), interpolation=cv2.INTER_AREA)
                )
                for i in range(frontal_imgs.shape[0])
            ]
            frontal_imgs = torch.stack(frontal_imgs, 0)

    else:
        imgs = [
            torch.from_numpy(imgs[i]
            )
            for i in range(imgs.shape[0])
        ]
        imgs = torch.stack(imgs, 0)
        if load_frontal_faces:
            frontal_imgs = [
                torch.from_numpy(frontal_imgs[i]
                                 )
                for i in range(frontal_imgs.shape[0])
            ]
            frontal_imgs = torch.stack(frontal_imgs, 0)

    poses = torch.from_numpy(poses)
    expressions = torch.from_numpy(expressions)
    
    bboxs[:,0:2] *= H #ai,bi,aj,bj
    bboxs[:,2:4] *= W
    bboxs = np.floor(bboxs)
    bboxs = torch.from_numpy(bboxs).int()
    print("Done with data loading")
    return imgs, poses, render_poses, [H, W, intrinsics], i_split, expressions, depths, bboxs, [warped_imgs_lip,warped_imgs_eyes,warped_imgs_skin], makeup_imgs_lip, paths, [masks,[masks_lip, masks_eyes, masks_skin]], img_ids


def load_flame_data_color(basedir, style_id, half_res=False, scale=0.5, testskip=1, debug=False, expressions=True,load_frontal_faces=False, load_bbox=True, test=False):
    print("starting data loading")
    splits = ["train", "val", "test"]
    if test:
        splits = ["test"]
    # 1009 HXY 为了让测试视频连续，将所有数据都拿进来渲染
    # pdb.set_trace()
    with open(os.path.join(basedir, 'box.json'), "r") as fp:
        metab = json.load(fp)
    
    metas = {}
    for s in splits:
        with open(os.path.join(basedir, f"transforms_{s}.json"), "r") as fp:
            metas[s] = json.load(fp)
    
    style_id = str(style_id)
    base_warp_dir = os.path.join(basedir, style_id, 'warp_makeup_'+style_id)
    base_makeup_dir = os.path.join('/data/hanxinyang/MuNeRF_latest/', 'makeup')
    warp_dir = 'warp_makeup_' + style_id
    base_mask_dir = os.path.join(basedir, style_id, 'mask')
    base_depth_dir = os.path.join(basedir, 'depth')
    os.makedirs(os.path.join(basedir,'half_res','landmark'), exist_ok=True)   

    all_frontal_imgs = []
    all_imgs = []
    all_wimgs = []
    all_mimgs = []
    all_poses = []
    all_expressions = []
    all_bboxs = []
    all_depths = []
    all_masks = []
    all_masks_ = []
    all_img_ids = []
    counts = [0]
    paths = {}
    paths['warped'] = []
    paths['makeup'] = []
    api = FacePPAPI()
    focal = 1100     # default value
    for s in splits:
        print('now process',s)
        meta = metas[s]
        imgs = []
        warped_imgs = []
        makeup_styles = []
        poses = []
        expressions = []
        frontal_imgs = []
        bboxs = []
        masks = []
        masks_ = []
        depths = []
        img_ids = []
        if s == "train" or testskip == 0:
            skip = 1
        else:
            skip = testskip
        focal = meta['focal_len']

        # Load all kinds of informations like masks, depths, poses, expressions, etc.
        
        for frame in meta["frames"][::skip]:
            fsname = (str(frame['img_id'])).zfill(4) + '.png' # frame["file_path"].split('/')[-1] + ".png"
            fname = os.path.join(basedir, s, fsname) # nomakeup image
            fwname = os.path.join(base_warp_dir, fsname) # warped makeup image
            fwname_ = os.path.join(basedir,'half_res',fsname)
            fwname_landmark = os.path.join(basedir,'half_res','landmark',(str(frame['img_id'])).zfill(4) + '.npy')
            fmname = os.path.join(basedir,style_id, 'mask', 'nonmakeup', fsname) # nomakeup image mask
            fmname_ = os.path.join(base_mask_dir, warp_dir, fsname) # warped makeup image mask
            fdname = os.path.join(base_depth_dir, fsname) # depth image
            # load makeup image
            fname_makeup = os.path.join(base_makeup_dir, style_id+'.jpg')# reference makeup image
            fname_makeup_landmark = os.path.join(base_makeup_dir,'landmark',style_id + '.npy')
            # pdb.set_trace()
            if os.path.exists(fname_makeup_landmark):
                landmark_M_api = np.load(fname_makeup_landmark, allow_pickle=True)
                landmark_M_api = landmark_M_api.item()

            paths['makeup'].append(landmark_M_api)
            if os.path.exists(fwname):
                # check here
                warped_imgs.append(imageio.imread(fwname))
                if os.path.exists(fwname_landmark):
                    landmark_B_api = np.load(fwname_landmark, allow_pickle=True)
                    landmark_B_api = landmark_B_api.item()
                else:
                    landmark_B_api = api.faceLandmarkDetector(fwname_)
                    np.save(fwname_landmark, landmark_B_api)
                    time.sleep(0.5)
                paths['warped'].append(landmark_B_api)
                # load makeup image
                makeup_styles.append(imageio.imread(fname_makeup))
                if os.path.exists(fname_makeup_landmark):
                    landmark_M_api = np.load(fname_makeup_landmark, allow_pickle=True)
                    landmark_M_api = landmark_M_api.item()
                else:
                    landmark_M_api = api.faceLandmarkDetector(fname_makeup)
                    np.save(fname_makeup_landmark, landmark_M_api)
                    time.sleep(0.5)

                paths['makeup'].append(landmark_M_api)
                imgs.append(imageio.imread(fname))
                # pdb.set_trace()
                masks.append(imageio.imread(fmname, as_gray=True))   # nonmakeup
                masks_.append(imageio.imread(fmname_, as_gray=True)) # warp
                if os.path.exists(fdname):
                    depths.append(imageio.imread(fdname))
                poses.append(np.array(frame["transform_matrix"]))
                bboxs.append(np.array(metab[fsname]))
                expressions.append(np.array(frame["expression"]))#.append(expression) #
                img_ids.append(fsname)
            else:
                # pdb.set_trace()
                print("skip ", fsname)
                # continue

            if load_frontal_faces:
                fname = os.path.join(basedir, (str(frame['img_id'])).zfill(4) + "_frontal" + ".png")
                frontal_imgs.append(imageio.imread(fname))
            light_flag = False
            '''if load_bbox:
                if "bbox" not in frame.keys():
                    bboxs.append(np.array([0.0,1.0,0.0,1.0]))
                else:
                    bboxs.append(np.array(frame["bbox"]))
           '''
        
        # Normalization
        imgs = (np.array(imgs) / 255.0).astype(np.float32)
        warped_imgs = (np.array(warped_imgs) / 255.0).astype(np.float32)
        # pdb.set_trace()
        all_wimgs.append(warped_imgs)
        
        makeup_styles = (np.array(makeup_styles) / 255.0).astype(np.float32)
        all_mimgs.append(makeup_styles)

        masks = (np.array(masks) / 255.0).astype(np.float32)
        all_masks.append(masks)
        masks_ = (np.array(masks_) / 255.0).astype(np.float32)
        all_masks_.append(masks_)

        if len(depths) > 0:
            depths = (np.array(depths) / 255.0).astype(np.float32)
            all_depths.append(depths)
        
        if load_frontal_faces:
            frontal_imgs = (np.array(frontal_imgs) / 255.0).astype(np.float32)

        poses = np.array(poses).astype(np.float32)
        expressions = np.array(expressions).astype(np.float32)
        bboxs = np.array(bboxs).astype(np.float32)
        counts.append(counts[-1] + imgs.shape[0])
        all_imgs.append(imgs)
        all_frontal_imgs.append(frontal_imgs)
        all_poses.append(poses)
        all_expressions.append(expressions)
        all_bboxs.append(bboxs)
        all_img_ids.append(img_ids)

    i_split = [np.arange(counts[i], counts[i + 1]) for i in range(len(splits))]
    
    # Concatenate Vectors
    # pdb.set_trace()
    warped_imgs = np.concatenate(all_wimgs, 0)
    makeup_imgs = np.concatenate(all_mimgs, 0)
    frontal_imgs = np.concatenate(all_frontal_imgs, 0) if load_frontal_faces else None
    poses = np.concatenate(all_poses, 0)
    expressions = np.concatenate(all_expressions, 0)
    bboxs = np.concatenate(all_bboxs, 0)
    masks = np.concatenate(all_masks, 0)
    masks_ = np.concatenate(all_masks_, 0)
    
    # pdb.set_trace()
    if len(all_depths) > 0:
        depths = np.concatenate(all_depths, 0)
    else:
        depths = []
    imgs = np.concatenate(all_imgs, 0)
    #lights = np.concatenate(all_lights, 0)
    #details = np.concatenate(all_details, 0)
    img_ids = np.concatenate(all_img_ids, 0)
    H, W = imgs[0].shape[:2]
    #camera_angle_x = float(meta["camera_angle_x"])
    # focal = 1100 #0.5 * W / np.tan(0.5 * camera_angle_x)
    #print('focal,', camera_angle_x, focal)
    #focals = (meta["focal_len"])
    #if_intrinsics = False
    #intrinsics = meta["intrinsics"] if meta["intrinsics"] else None
    #if if_intrinsics:
    #    intrinsics = np.array(meta["intrinsics"])
   # else:
    intrinsics = np.array([focal, focal, 0.5, 0.5]) # fx fy cx cy
    
    # if type(focals) is list:
    #     focal = np.array([W*focals[0], H*focals[1]]) # fx fy  - x is width
    # else:
    #     focal = np.array([focal, focal])

    if not test:
        render_poses = torch.stack(
            [
                torch.from_numpy(pose_spherical(angle, -30.0, 4.0))
                for angle in np.linspace(-180, 180, 2204 + 1)[:-1]
            ],
            0,
        )
    else:
        render_poses = torch.stack(
            [
                torch.from_numpy(pose_spherical(angle, -30.0, 4.0))
                for angle in np.linspace(-180, 180, 40 + 1)[:-1]
            ],
            0,
        )

    # In debug mode, return extremely tiny images
    #debug = True
    if debug:
        H = H // 2
        W = W // 2
        #focal = focal / 32.0
        intrinsics[:2] = intrinsics[:2] / 32.0
        imgs = [
            torch.from_numpy(
                cv2.resize(imgs[i], dsize=(25, 25), interpolation=cv2.INTER_AREA)
            )
            for i in range(imgs.shape[0])
        ]
        imgs = torch.stack(imgs, 0)
        if frontal_imgs:
            frontal_imgs = [
                torch.from_numpy(
                    cv2.resize(frontal_imgs[i], dsize=(25, 25), interpolation=cv2.INTER_AREA)
                )
                for i in range(frontal_imgs.shape[0])
            ]
            frontal_imgs = torch.stack(frontal_imgs, 0)

        poses = torch.from_numpy(poses)

        return imgs, poses, render_poses, [H, W, intrinsics], i_split, frontal_imgs,

    # Transform for half_res

    if half_res:
        # TODO: resize images using INTER_AREA (cv2)
        H = int(H * scale)
        W = int(W * scale)
        #focal = focal / 2.0
        intrinsics[:2] = intrinsics[:2] * scale
        imgs = [
            torch.from_numpy(
                cv2.resize(imgs[i], dsize=(H, W), interpolation=cv2.INTER_AREA)
            )
            for i in range(imgs.shape[0])
        ]
        imgs = torch.stack(imgs, 0)
        # warped_img
        warped_imgs = [
            torch.from_numpy(
                cv2.resize(warped_imgs[i], dsize=(H, W), interpolation=cv2.INTER_AREA)
            )
            for i in range(warped_imgs.shape[0])
        ]
        warped_imgs = torch.stack(warped_imgs, 0)
        makeup_imgs = [
            torch.from_numpy(
                cv2.resize(makeup_imgs[i], dsize=(H, W), interpolation=cv2.INTER_AREA)
            )
            for i in range(makeup_imgs.shape[0])
        ]
        makeup_imgs = torch.stack(makeup_imgs, 0)
        
        masks = [
            torch.from_numpy(
                cv2.resize(masks[i], dsize=(H, W), interpolation=cv2.INTER_AREA)
            )
            for i in range(masks.shape[0])
        ]
        masks = torch.stack(masks, 0)
        masks_ = [
            torch.from_numpy(
                cv2.resize(masks_[i], dsize=(H, W), interpolation=cv2.INTER_AREA)
            )
            for i in range(masks_.shape[0])
        ]
        masks_ = torch.stack(masks_, 0)
        if len(depths) > 0:
            depths = [
                torch.from_numpy(
                    cv2.resize(depths[i], dsize=(H, W), interpolation=cv2.INTER_AREA)
                )
                for i in range(depths.shape[0])
            ]
            depths = torch.stack(depths, 0)
        else:
            depths = None
        if load_frontal_faces:
            frontal_imgs = [
                torch.from_numpy(
                    cv2.resize(frontal_imgs[i], dsize=(H, W), interpolation=cv2.INTER_AREA)
                )
                for i in range(frontal_imgs.shape[0])
            ]
            frontal_imgs = torch.stack(frontal_imgs, 0)

    else:
        imgs = [
            torch.from_numpy(imgs[i]
            )
            for i in range(imgs.shape[0])
        ]
        imgs = torch.stack(imgs, 0)
        if load_frontal_faces:
            frontal_imgs = [
                torch.from_numpy(frontal_imgs[i]
                                 )
                for i in range(frontal_imgs.shape[0])
            ]
            frontal_imgs = torch.stack(frontal_imgs, 0)

    poses = torch.from_numpy(poses)
    expressions = torch.from_numpy(expressions)
    
    bboxs[:,0:2] *= H #ai,bi,aj,bj
    bboxs[:,2:4] *= W
    bboxs = np.floor(bboxs)
    bboxs = torch.from_numpy(bboxs).int()
    print("Done with data loading")
    return imgs, poses, render_poses, [H, W, intrinsics], i_split, expressions, depths, bboxs, warped_imgs, makeup_imgs, paths, [masks,masks_], img_ids

def load_flame_data_colorH(basedir, style_id, half_res=False, scale=0.5, testskip=1, debug=False, expressions=True,load_frontal_faces=False, load_bbox=True, test=False, consistent=False, fix_pose=False, nerface_dir = "/data/hanxinyang/MuNeRF_latest/rendering/finals/girl7_fix_expr_success"):
    print("starting data loading")
    splits = ["train", "val", "test"]
    # pdb.set_trace()
    if test and not consistent:
        splits = ["test"]
    # 1009 HXY 为了让测试视频连续，将所有数据都拿进来渲染
    
    with open(os.path.join(basedir, 'box.json'), "r") as fp:
        metab = json.load(fp)
    
    metas = {}
    for s in splits:
        with open(os.path.join(basedir, f"transforms_{s}.json"), "r") as fp:
            metas[s] = json.load(fp)
    
    style_id = str(style_id)
    base_warp_dir = os.path.join(basedir, style_id, 'warp_makeup_'+style_id)
    base_makeup_dir = os.path.join('/data/hanxinyang/MuNeRF_latest/', 'makeup')
    warp_dir = 'warp_makeup_' + style_id
    base_mask_dir = os.path.join(basedir, style_id, 'mask')
    base_depth_dir = os.path.join(basedir, 'depth')
    os.makedirs(os.path.join(basedir,'half_res','landmark'), exist_ok=True)   

    all_frontal_imgs = []
    all_imgs = []
    all_wimgs = []
    all_mimgs = []
    all_poses = []
    all_expressions = []
    all_bboxs = []
    all_depths = []
    all_masks = []
    all_masks_ = []
    all_img_ids = []
    counts = [0]
    paths = {}
    paths['warped'] = []
    paths['makeup'] = []
    api = FacePPAPI()
    focal = 1100     # default value
    for s in splits:
        print('now process',s)
        meta = metas[s]
        imgs = []
        warped_imgs = []
        makeup_styles = []
        poses = []
        expressions = []
        frontal_imgs = []
        bboxs = []
        masks = []
        masks_ = []
        depths = []
        img_ids = []
        if s == "train" or testskip == 0:
            skip = 1
        else:
            skip = testskip
        focal = meta['focal_len']

        # Load all kinds of informations like masks, depths, poses, expressions, etc.
        
        for frame in meta["frames"][::skip]:
            fsname = (str(frame['img_id'])).zfill(4) + '.png' # frame["file_path"].split('/')[-1] + ".png"
            fname = os.path.join(basedir, s, fsname) # nomakeup image
            fwname = os.path.join(base_warp_dir, fsname) # warped makeup image
            fwname_ = os.path.join(basedir,'half_res',fsname)
            fwname_landmark = os.path.join(basedir,'half_res','landmark',(str(frame['img_id'])).zfill(4) + '.npy')
            fmname = os.path.join(basedir,style_id, 'mask', 'nonmakeup', fsname) # nomakeup image mask
            fmname_ = os.path.join(base_mask_dir, warp_dir, fsname) # warped makeup image mask
            fdname = os.path.join(base_depth_dir, fsname) # depth image
            # load makeup image
            fname_makeup = os.path.join(base_makeup_dir, style_id+'.jpg')# reference makeup image
            fname_makeup_landmark = os.path.join(base_makeup_dir,'landmark',style_id + '.npy')
            # pdb.set_trace()
            if os.path.exists(fname_makeup_landmark):
                landmark_M_api = np.load(fname_makeup_landmark, allow_pickle=True)
                landmark_M_api = landmark_M_api.item()

            paths['makeup'].append(landmark_M_api)
            if os.path.exists(fwname):
                # check here
                warped_imgs.append(imageio.imread(fwname))
                if os.path.exists(fwname_landmark):
                    landmark_B_api = np.load(fwname_landmark, allow_pickle=True)
                    landmark_B_api = landmark_B_api.item()
                else:
                    landmark_B_api = api.faceLandmarkDetector(fwname_)
                    np.save(fwname_landmark, landmark_B_api)
                    time.sleep(0.5)
                paths['warped'].append(landmark_B_api)
                # load makeup image
                makeup_styles.append(imageio.imread(fname_makeup))
                if os.path.exists(fname_makeup_landmark):
                    landmark_M_api = np.load(fname_makeup_landmark, allow_pickle=True)
                    landmark_M_api = landmark_M_api.item()
                else:
                    landmark_M_api = api.faceLandmarkDetector(fname_makeup)
                    np.save(fname_makeup_landmark, landmark_M_api)
                    time.sleep(0.5)

                paths['makeup'].append(landmark_M_api)
                imgs.append(imageio.imread(fname))
                # pdb.set_trace()
                masks.append(imageio.imread(fmname))   # nonmakeup
                masks_.append(imageio.imread(fmname_)) # warp
                if os.path.exists(fdname):
                    depths.append(imageio.imread(fdname))
                poses.append(np.array(frame["transform_matrix"]))
                bboxs.append(np.array(metab[fsname]))
                expressions.append(np.array(frame["expression"]))#.append(expression) #
                img_ids.append(fsname)
            else:
                # pdb.set_trace()
                print("skip ", fsname)
                # continue

            if load_frontal_faces:
                fname = os.path.join(basedir, (str(frame['img_id'])).zfill(4) + "_frontal" + ".png")
                frontal_imgs.append(imageio.imread(fname))
            light_flag = False
            '''if load_bbox:
                if "bbox" not in frame.keys():
                    bboxs.append(np.array([0.0,1.0,0.0,1.0]))
                else:
                    bboxs.append(np.array(frame["bbox"]))
           '''
        
        # Normalization
        imgs = (np.array(imgs) / 255.0).astype(np.float32)
        warped_imgs = (np.array(warped_imgs) / 255.0).astype(np.float32)
        # pdb.set_trace()
        all_wimgs.append(warped_imgs)
        
        makeup_styles = (np.array(makeup_styles) / 255.0).astype(np.float32)
        all_mimgs.append(makeup_styles)

        masks = (np.array(masks) / 255.0).astype(np.float32)
        all_masks.append(masks)
        masks_ = (np.array(masks_) / 255.0).astype(np.float32)
        all_masks_.append(masks_)

        if len(depths) > 0:
            depths = (np.array(depths) / 255.0).astype(np.float32)
            all_depths.append(depths)
        
        if load_frontal_faces:
            frontal_imgs = (np.array(frontal_imgs) / 255.0).astype(np.float32)

        poses = np.array(poses).astype(np.float32)
        expressions = np.array(expressions).astype(np.float32)
        bboxs = np.array(bboxs).astype(np.float32)
        counts.append(counts[-1] + imgs.shape[0])
        all_imgs.append(imgs)
        all_frontal_imgs.append(frontal_imgs)
        all_poses.append(poses)
        all_expressions.append(expressions)
        all_bboxs.append(bboxs)
        all_img_ids.append(img_ids)

    i_split = [np.arange(counts[i], counts[i + 1]) for i in range(len(splits))]
    
    # Concatenate Vectors
    # pdb.set_trace()
    warped_imgs = np.concatenate(all_wimgs, 0)
    makeup_imgs = np.concatenate(all_mimgs, 0)
    frontal_imgs = np.concatenate(all_frontal_imgs, 0) if load_frontal_faces else None
    poses = np.concatenate(all_poses, 0)
    expressions = np.concatenate(all_expressions, 0)
    bboxs = np.concatenate(all_bboxs, 0)
    masks = np.concatenate(all_masks, 0)
    masks_ = np.concatenate(all_masks_, 0)

    if fix_pose:
        nerface_imgs_list = []
        pdb.set_trace()
        nerface_imgs = []
        things = os.listdir(nerface_dir)
        things.sort()
        for thing in things:
            nerface_img_dir = os.path.join(nerface_dir, thing)
            nerface_imgs_list.append(imageio.imread(nerface_img_dir))
        nerface_imgs_list = (np.array(nerface_imgs_list) / 255.0).astype(np.float32)
        nerface_imgs.append(nerface_imgs_list)
        nerface_imgs = np.concatenate(nerface_imgs, 0)

    # pdb.set_trace()
    if len(all_depths) > 0:
        depths = np.concatenate(all_depths, 0)
    else:
        depths = []
    imgs = np.concatenate(all_imgs, 0)
    #lights = np.concatenate(all_lights, 0)
    #details = np.concatenate(all_details, 0)
    img_ids = np.concatenate(all_img_ids, 0)
    H, W = imgs[0].shape[:2]
    #camera_angle_x = float(meta["camera_angle_x"])
    # focal = 1100 #0.5 * W / np.tan(0.5 * camera_angle_x)
    #print('focal,', camera_angle_x, focal)
    #focals = (meta["focal_len"])
    #if_intrinsics = False
    #intrinsics = meta["intrinsics"] if meta["intrinsics"] else None
    #if if_intrinsics:
    #    intrinsics = np.array(meta["intrinsics"])
   # else:
    intrinsics = np.array([focal, focal, 0.5, 0.5]) # fx fy cx cy
    
    # if type(focals) is list:
    #     focal = np.array([W*focals[0], H*focals[1]]) # fx fy  - x is width
    # else:
    #     focal = np.array([focal, focal])

    if not test:
        render_poses = torch.stack(
            [
                torch.from_numpy(pose_spherical(angle, -30.0, 4.0))
                for angle in np.linspace(-180, 180, 2204 + 1)[:-1]
            ],
            0,
        )
    else:
        render_poses = torch.stack(
            [
                torch.from_numpy(pose_spherical(angle, -30.0, 4.0))
                for angle in np.linspace(-180, 180, 40 + 1)[:-1]
            ],
            0,
        )

    # In debug mode, return extremely tiny images
    #debug = True
    if debug:
        H = H // 2
        W = W // 2
        #focal = focal / 32.0
        intrinsics[:2] = intrinsics[:2] / 32.0
        imgs = [
            torch.from_numpy(
                cv2.resize(imgs[i], dsize=(25, 25), interpolation=cv2.INTER_AREA)
            )
            for i in range(imgs.shape[0])
        ]
        imgs = torch.stack(imgs, 0)
        if frontal_imgs:
            frontal_imgs = [
                torch.from_numpy(
                    cv2.resize(frontal_imgs[i], dsize=(25, 25), interpolation=cv2.INTER_AREA)
                )
                for i in range(frontal_imgs.shape[0])
            ]
            frontal_imgs = torch.stack(frontal_imgs, 0)

        poses = torch.from_numpy(poses)

        return imgs, poses, render_poses, [H, W, intrinsics], i_split, frontal_imgs,

    # Transform for half_res

    if half_res:
        # TODO: resize images using INTER_AREA (cv2)
        H = int(H * scale)
        W = int(W * scale)
        #focal = focal / 2.0
        intrinsics[:2] = intrinsics[:2] * scale
        imgs = [
            torch.from_numpy(
                cv2.resize(imgs[i], dsize=(H, W), interpolation=cv2.INTER_AREA)
            )
            for i in range(imgs.shape[0])
        ]
        imgs = torch.stack(imgs, 0)

        if fix_pose:
            nerface_imgs = [
                torch.from_numpy(
                    #cv2.resize(nerface_imgs[i], dsize=(400, 400), interpolation=cv2.INTER_AREA)
                    cv2.resize(nerface_imgs[i], dsize=(H, W), interpolation=cv2.INTER_AREA)
                )
                for i in range(nerface_imgs.shape[0])
            ]
            nerface_imgs = torch.stack(nerface_imgs, 0)

        # warped_img
        warped_imgs = [
            torch.from_numpy(
                cv2.resize(warped_imgs[i], dsize=(H, W), interpolation=cv2.INTER_AREA)
            )
            for i in range(warped_imgs.shape[0])
        ]
        warped_imgs = torch.stack(warped_imgs, 0)
        makeup_imgs = [
            torch.from_numpy(
                cv2.resize(makeup_imgs[i], dsize=(H, W), interpolation=cv2.INTER_AREA)
            )
            for i in range(makeup_imgs.shape[0])
        ]
        makeup_imgs = torch.stack(makeup_imgs, 0)
        
        masks = [
            torch.from_numpy(
                cv2.resize(masks[i], dsize=(H, W), interpolation=cv2.INTER_AREA)
            )
            for i in range(masks.shape[0])
        ]
        masks = torch.stack(masks, 0)
        masks_ = [
            torch.from_numpy(
                cv2.resize(masks_[i], dsize=(H, W), interpolation=cv2.INTER_AREA)
            )
            for i in range(masks_.shape[0])
        ]
        masks_ = torch.stack(masks_, 0)
        if len(depths) > 0:
            depths = [
                torch.from_numpy(
                    cv2.resize(depths[i], dsize=(H, W), interpolation=cv2.INTER_AREA)
                )
                for i in range(depths.shape[0])
            ]
            depths = torch.stack(depths, 0)
        else:
            depths = None
        if load_frontal_faces:
            frontal_imgs = [
                torch.from_numpy(
                    cv2.resize(frontal_imgs[i], dsize=(H, W), interpolation=cv2.INTER_AREA)
                )
                for i in range(frontal_imgs.shape[0])
            ]
            frontal_imgs = torch.stack(frontal_imgs, 0)

    else:
        imgs = [
            torch.from_numpy(imgs[i]
            )
            for i in range(imgs.shape[0])
        ]
        imgs = torch.stack(imgs, 0)
        if load_frontal_faces:
            frontal_imgs = [
                torch.from_numpy(frontal_imgs[i]
                                 )
                for i in range(frontal_imgs.shape[0])
            ]
            frontal_imgs = torch.stack(frontal_imgs, 0)

    poses = torch.from_numpy(poses)
    expressions = torch.from_numpy(expressions)
    
    bboxs[:,0:2] *= H #ai,bi,aj,bj
    bboxs[:,2:4] *= W
    bboxs = np.floor(bboxs)
    bboxs = torch.from_numpy(bboxs).int()
    print("Done with data loading")
    
    if not fix_pose:
        nerface_imgs = None
    
    return imgs, poses, render_poses, [H, W, intrinsics], i_split, expressions, depths, bboxs, warped_imgs, makeup_imgs, paths, [masks,masks_], img_ids, nerface_imgs


def load_flame_dataH(basedir, half_res=False, scale=0.5, testskip=1, debug=False, expressions=True,load_frontal_faces=False, load_bbox=True, test=False, consistent=False, fix_pose=False):
    print("starting data loading")
    splits = ["train", "val", "test"]
    if test and not consistent:
        splits = ["test"]
    metas = {}
    for s in splits:
        with open(os.path.join(basedir, f"transforms_{s}.json"), "r") as fp:
            metas[s] = json.load(fp)
    with open(os.path.join(basedir, 'box.json'), "r") as fp:
        metab = json.load(fp)

    all_frontal_imgs = []
    all_imgs = []
    all_poses = []
    all_expressions = []
    all_bboxs = []
    all_img_ids = []
    counts = [0]
    focal = 1100     # default value
    for s in splits:
        meta = metas[s]
        imgs = []
        poses = []
        expressions = []
        frontal_imgs = []
        bboxs = []
        img_ids = []
        if s == "train" or testskip == 0:
            skip = 1
        else:
            skip = testskip
        focal = meta['focal_len']

        for frame in meta["frames"][::skip]:
            fsname = (str(frame['img_id'])).zfill(4) + '.png'
            fname = os.path.join(basedir,s,fsname)
            
            # fname = os.path.join(basedir, frame["file_path"] + ".png")
            # pdb.set_trace()
            imgs.append(imageio.imread(fname))
            if load_frontal_faces:
                fname = os.path.join(basedir, (str(frame['img_id'])).zfill(4) + "_frontal" + ".png")
                # fname = os.path.join(basedir, frame["file_path"] + "_frontal" + ".png")
                frontal_imgs.append(imageio.imread(fname))
            # import matplotlib.pyplot as plt
            # plt.imshow(imgs[-1])
            # plt.show()
            # pdb.set_trace()
            poses.append(np.array(frame["transform_matrix"]))
            expressions.append(np.array(frame["expression"]))
            img_ids.append(fsname)
            if load_bbox:
                # if "bbox" not in frame.keys():
                #     bboxs.append(np.array([0.0,1.0,0.0,1.0]))
                # else:
                #     bboxs.append(np.array(frame["bbox"]))
                bboxs.append(np.array(metab[fsname]))
        # pdb.set_trace()
        imgs = (np.array(imgs) / 255.0).astype(np.float32)
        if load_frontal_faces:
            frontal_imgs = (np.array(frontal_imgs) / 255.0).astype(np.float32)

        poses = np.array(poses).astype(np.float32)
        expressions = np.array(expressions).astype(np.float32)
        bboxs = np.array(bboxs).astype(np.float32)

        counts.append(counts[-1] + imgs.shape[0])
        all_imgs.append(imgs)
        all_frontal_imgs.append(frontal_imgs)
        all_poses.append(poses)
        all_expressions.append(expressions)
        all_bboxs.append(bboxs)
        all_img_ids.append(img_ids)

    i_split = [np.arange(counts[i], counts[i + 1]) for i in range(len(splits))]

    imgs = np.concatenate(all_imgs, 0)
    frontal_imgs = np.concatenate(all_frontal_imgs, 0) if load_frontal_faces else None
    poses = np.concatenate(all_poses, 0)
    expressions = np.concatenate(all_expressions, 0)
    bboxs = np.concatenate(all_bboxs, 0)
    img_ids = np.concatenate(all_img_ids, 0)

    H, W = imgs[0].shape[:2]
    # camera_angle_x = float(meta["camera_angle_x"])
    # focal = 0.5 * W / np.tan(0.5 * camera_angle_x)
    # focal = 1100

    #focals = (meta["focals"])
    # intrinsics = meta["intrinsics"] if meta["intrinsics"] else None
    # if meta["intrinsics"]:
    #     intrinsics = np.array(meta["intrinsics"])
    # else:
    #     intrinsics = np.array([focal, focal, 0.5, 0.5]) # fx fy cx cy
    intrinsics = np.array([focal, focal, 0.5, 0.5]) # fx fy cx cy
    # if type(focals) is list:
    #     focal = np.array([W*focals[0], H*focals[1]]) # fx fy  - x is width
    # else:
    #     focal = np.array([focal, focal])

    

        


    # In debug mode, return extremely tiny images
    if debug:
        H = H // 32
        W = W // 32
        #focal = focal / 32.0
        intrinsics[:2] = intrinsics[:2] / 32.0
        imgs = [
            torch.from_numpy(
                cv2.resize(imgs[i], dsize=(25, 25), interpolation=cv2.INTER_AREA)
            )
            for i in range(imgs.shape[0])
        ]
        imgs = torch.stack(imgs, 0)
        if frontal_imgs:
            frontal_imgs = [
                torch.from_numpy(
                    cv2.resize(frontal_imgs[i], dsize=(25, 25), interpolation=cv2.INTER_AREA)
                )
                for i in range(frontal_imgs.shape[0])
            ]
            frontal_imgs = torch.stack(frontal_imgs, 0)

        poses = torch.from_numpy(poses)

        return imgs, poses, render_poses, [H, W, intrinsics], i_split, frontal_imgs


    if half_res:
        # TODO: resize images using INTER_AREA (cv2)
        H = int(H * scale)
        W = int(W * scale)
        #focal = focal / 2.0
        intrinsics[:2] = intrinsics[:2] * scale
        imgs = [
            torch.from_numpy(
                #cv2.resize(imgs[i], dsize=(400, 400), interpolation=cv2.INTER_AREA)
                cv2.resize(imgs[i], dsize=(H, W), interpolation=cv2.INTER_AREA)
            )
            for i in range(imgs.shape[0])
        ]
        imgs = torch.stack(imgs, 0)
        
        
        
        if load_frontal_faces:
            frontal_imgs = [
                torch.from_numpy(
                    #cv2.resize(imgs[i], dsize=(400, 400), interpolation=cv2.INTER_AREA)
                    cv2.resize(frontal_imgs[i], dsize=(H, W), interpolation=cv2.INTER_AREA)
                )
                for i in range(frontal_imgs.shape[0])
            ]
            frontal_imgs = torch.stack(frontal_imgs, 0)

    else:
        imgs = [
            torch.from_numpy(imgs[i]
                # cv2.resize(imgs[i], dsize=(400, 400), interpolation=cv2.INTER_AREA)
                #cv2.resize(imgs[i], dsize=(H, W), interpolation=cv2.INTER_AREA)
            )
            for i in range(imgs.shape[0])
        ]
        imgs = torch.stack(imgs, 0)
        if load_frontal_faces:
            frontal_imgs = [
                torch.from_numpy(frontal_imgs[i]
                                 # cv2.resize(imgs[i], dsize=(400, 400), interpolation=cv2.INTER_AREA)
                                 # cv2.resize(imgs[i], dsize=(H, W), interpolation=cv2.INTER_AREA)
                                 )
                for i in range(frontal_imgs.shape[0])
            ]
            frontal_imgs = torch.stack(frontal_imgs, 0)

    poses = torch.from_numpy(poses)
    expressions = torch.from_numpy(expressions)
    bboxs[:,0:2] *= H
    bboxs[:,2:4] *= W
    bboxs = np.floor(bboxs)
    bboxs = torch.from_numpy(bboxs).int()
    print("Done with data loading")
    
    render_poses = torch.stack(
        [
            torch.from_numpy(pose_spherical(angle, -90, 0.99))
            for angle in np.linspace(-30, 30, 40+1)[:-1]
        ],
        0,
    )
    # pdb.set_trace()
    return imgs, poses, render_poses, [H, W, intrinsics], i_split, expressions, frontal_imgs, bboxs, img_ids
