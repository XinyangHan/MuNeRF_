import json
import os

import cv2
import imageio
import numpy as np
import torch


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


def load_flame_data(basedir, half_res=False, testskip=1, debug=False, expressions=True,load_frontal_faces=False, load_bbox=True, test=False):
    print("starting data loading")
    splits = ["train", "val", "test"]
    if test:
        splits = ["test"]
    jsondir = '/mnt/4T/heyue/nerf-illumination/4D-Facial-Avatars-main/nerface_code/nerf-pytorch/dave_dvp/'
    lightjson = jsondir + 'dave_w_detail.json'
    boxjson = jsondir  + 'box.json'
    posejson = jsondir + 'pose.json'
    metas = {}
    metal = {}
    metap = {}
    with open(os.path.join(lightjson), "r") as fp:
        metal = json.load(fp)
    with open(os.path.join(boxjson), "r") as fp:
        metab = json.load(fp)
    with open(os.path.join(posejson), "r") as fp:
        metap = json.load(fp)
    #failed_frame = metap['filters']
    for s in splits:
        with open(os.path.join(basedir, f"transforms_{s}.json"), "r") as fp:
            metas[s] = json.load(fp)

    all_frontal_imgs = []
    all_imgs = []
    all_poses = []
    all_expressions = []
    all_bboxs = []
    all_lights = []
    all_details = []
    counts = [0]
    for s in splits:
        print('now process',s)
        meta = metas[s]
        imgs = []
        poses = []
        lights = [] #
        expressions = []
        frontal_imgs = []
        bboxs = []
        details = []
        if s == "train" or testskip == 0:
            skip = 1
        else:
            skip = testskip

        for frame in meta["frames"][::skip]:
            fsname = frame["file_path"].split('/')[-1] + ".png"
            #if fsname in failed_frame: continue
            fname = os.path.join(basedir, frame["file_path"] + ".png")
            imgs.append(imageio.imread(fname))
            if load_frontal_faces:
                fname = os.path.join(basedir, frame["file_path"] + "_frontal" + ".png")
                frontal_imgs.append(imageio.imread(fname))
            test = False
            
            light = np.array(metal[fsname]['light'])
            expression = np.array(metal[fsname]['exp'])
            detail = np.array(metal[fsname]['detail'])
            poses.append(np.array(metap[fsname]))
            #poses.append(np.array(frame["transform_matrix"]))
            bboxs.append(np.array(metab[fsname]))
            expressions.append(expression) #np.array(frame["expression"]))
            details.append(detail)
            testlight = [0,0,0,0,0,0,0,0,0]
            
            #give a test light code
            if test:
                lights.append(testlight)
            else:
                lights.append(np.array(light))
            '''if load_bbox:
                if "bbox" not in frame.keys():
                    bboxs.append(np.array([0.0,1.0,0.0,1.0]))
                else:
                    bboxs.append(np.array(frame["bbox"]))
           '''
        imgs = (np.array(imgs) / 255.0).astype(np.float32)
        if load_frontal_faces:
            frontal_imgs = (np.array(frontal_imgs) / 255.0).astype(np.float32)

        poses = np.array(poses).astype(np.float32)
        expressions = np.array(expressions).astype(np.float32)
        bboxs = np.array(bboxs).astype(np.float32)
        lights = np.array(lights).astype(np.float32)
        details = np.array(details).astype(np.float32)

        counts.append(counts[-1] + imgs.shape[0])
        all_imgs.append(imgs)
        all_frontal_imgs.append(frontal_imgs)
        all_poses.append(poses)
        all_expressions.append(expressions)
        all_bboxs.append(bboxs)
        all_lights.append(lights)
        all_details.append(details)

    i_split = [np.arange(counts[i], counts[i + 1]) for i in range(len(splits))]

    imgs = np.concatenate(all_imgs, 0)
    frontal_imgs = np.concatenate(all_frontal_imgs, 0) if load_frontal_faces else None
    poses = np.concatenate(all_poses, 0)
    expressions = np.concatenate(all_expressions, 0)
    bboxs = np.concatenate(all_bboxs, 0)
    lights = np.concatenate(all_lights, 0)
    details = np.concatenate(all_details, 0)

    H, W = imgs[0].shape[:2]
    camera_angle_x = float(meta["camera_angle_x"])
    focal = 512 #0.5 * W / np.tan(0.5 * camera_angle_x)
    print('focal,', camera_angle_x, focal)
    #focals = (meta["focals"])
    if_intrinsics = True
    intrinsics = meta["intrinsics"] if meta["intrinsics"] else None
    if if_intrinsics:
        intrinsics = np.array(meta["intrinsics"])
    else:
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
        H = H // 2
        W = W // 2
        #focal = focal / 2.0
        intrinsics[:2] = intrinsics[:2] * 0.5
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
    lights = torch.from_numpy(lights)
    details = torch.from_numpy(details)
    
    bboxs[:,0:2] *= H #ai,bi,aj,bj
    bboxs[:,2:4] *= W
    bboxs = np.floor(bboxs)
    bboxs = torch.from_numpy(bboxs).int()
    print("Done with data loading")
    return imgs, poses, render_poses, [H, W, intrinsics], i_split, expressions, lights, details, frontal_imgs, bboxs
