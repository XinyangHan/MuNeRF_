import argparse
import os
import time
import sys

from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import json
import numpy as np
import pdb
from numpyencoder import NumpyEncoder
sys.path.insert(1, './nerf')
from train_helper import create_part, create_module
# import subprocess
# proc1 = subprocess.Popen(['scontrol', 'show', 'job', os.environ['SLURM_JOBID'], '-d'], stdout=subprocess.PIPE)
# process = subprocess.run(['grep', '-oP', 'GRES=.*IDX:\K\d'], stdin=proc1.stdout, capture_output=True, text=True)
# os.environ['EGL_DEVICE_ID'] = process.stdout.rstrip()
# proc1.stdout.close()
import model.Transformer as trans
from model.model import PatchGAN, Transfromer_PatchGAN, get_pairs
from utils.train_util import facePartsCoordinatesAPI
import torchvision.transforms as transforms
from PIL import Image
import PIL
import imageio
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")

#matplotlib.use("TkAgg")

import numpy as np
import torch
import torchvision
import yaml
from tqdm import tqdm
#from nerf-pytorch import


from nerf import (
    CfgNode,
    get_ray_bundle,
    load_flame_data,
    load_llff_data,
    models,
    get_embedding_function,
    run_one_iter_of_nerf,
    meshgrid_xy
)
from nerf.load_flame import load_flame_data, load_flame_data_color, load_flame_data_colorH, load_flame_dataH

def mask_preprocess(mask_A, mask_B):
    mask_A = mask_A.unsqueeze(0)
    mask_B = mask_B.unsqueeze(0)
    index_tmp = torch.nonzero(mask_A, as_tuple=False)
    x_A_index = index_tmp[:, 2]
    y_A_index = index_tmp[:, 3]
    index_tmp = torch.nonzero(mask_B, as_tuple=False)
    x_B_index = index_tmp[:, 2]
    y_B_index = index_tmp[:, 3]
    # mask_A = to_var(mask_A, requires_grad=False)
    # mask_B = to_var(mask_B, requires_grad=False)
    index = [x_A_index, y_A_index, x_B_index, y_B_index]
    index_2 = [x_B_index, y_B_index, x_A_index, y_A_index]
    mask_A = mask_A.squeeze(0)
    mask_B = mask_B.squeeze(0)
    return mask_A, mask_B, index, index_2
    
def save_plt_image(im1, outname):
    fig = plt.figure()
    fig.set_size_inches((6.4,6.4))
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    #plt.set_cmap('jet')
    ax.imshow(im1, aspect='equal')
    plt.savefig(outname, dpi=80)
    plt.close(fig)

def cast_to_image2(tensor, dataset_type):
    # Input tensor is (H, W, 3). Convert to (3, H, W).
    tensor = tensor.permute(2, 0, 1)
    tensor = tensor.clamp(0.0,1.0)
    # Convert to PIL Image and then np.array (output shape: (H, W, 3))
    img = None#np.array(torchvision.transforms.ToPILImage()(tensor.detach().cpu()))
    return tensor, img

def normal_map_from_depth_map(depthmap):
    h, w = np.shape(depthmap)
    normals = np.zeros((h, w, 3))
    phong = np.zeros((h, w, 3))
    for x in range(1, h - 1):
        for y in range(1, w - 1):
            dzdx = (float((depthmap[x + 1, y])) - float((depthmap[x - 1, y]))) / 2.0
            dzdy = (float((depthmap[x, y + 1])) - float((depthmap[x, y - 1]))) / 2.0

            n = np.array([-dzdx, -dzdy, 0.005])

            n = n * 1/np.linalg.norm(n)
            dir = np.array([x,y,1.0])
            dir = dir *1/np.linalg.norm(dir)

            normals[x, y] = (n*0.5 + 0.5)
            phong[x, y] = np.dot(dir,n)*0.5+0.5

    normals *= 255
    normals = normals.astype('uint8')
    #plt.imshow(depthmap, cmap='gray')
    #plt.show()
    plt.imshow(normals)
    plt.show()
    plt.imshow(phong)
    plt.show()
    return normals

def torch_normal_map(depthmap,focal,weights=None,clean=True, central_difference=False):
    W,H = depthmap.shape
    #normals = torch.zeros((H,W,3), device=depthmap.device)
    cx = focal[2]*W
    cy = focal[3]*H
    fx = focal[0]
    fy = focal[1]
    ii, jj = meshgrid_xy(torch.arange(W, device=depthmap.device),
                         torch.arange(H, device=depthmap.device))
    points = torch.stack(
        [
            ((ii - cx) * depthmap) / fx,
            -((jj - cy) * depthmap) / fy,
            depthmap,
        ],
        dim=-1)
    difference = 2 if central_difference else 1
    dx = (points[difference:,:,:] - points[:-difference,:,:])
    dy = (points[:,difference:,:] - points[:,:-difference,:])
    normals = torch.cross(dy[:-difference,:,:],dx[:,:-difference,:],2)
    normalize_factor = torch.sqrt(torch.sum(normals*normals,2))
    normals[:,:,0]  /= normalize_factor
    normals[:,:,1]  /= normalize_factor
    normals[:,:,2]  /= normalize_factor
    normals = normals * 0.5 +0.5

    if clean and weights is not None: # Use volumetric rendering weights to clean up the normal map
        mask = weights.repeat(3,1,1).permute(1,2,0)
        mask = mask[:-difference,:-difference]
        where = torch.where(mask > 0.22)
        normals[where] = 1.0
        normals = (1-mask)*normals + (mask)*torch.ones_like(normals)
    normals *= 255
    #plt.imshow(normals.cpu().numpy().astype('uint8'))
    #plt.show()
    return normals

def vis(tensor):
    plt.imshow((tensor*255).cpu().numpy().astype('uint8'))
    plt.show()
def normal_map_from_depth_map_backproject(depthmap):
    h, w = np.shape(depthmap)
    normals = np.zeros((h, w, 3))
    phong = np.zeros((h, w, 3))
    cx = cy = h//2
    fx=fy=500
    fx = fy = 1150
    for x in range(1, h - 1):
        for y in range(1, w - 1):
            #dzdx = (float((depthmap[x + 1, y])) - float((depthmap[x - 1, y]))) / 2.0
            #dzdy = (float((depthmap[x, y + 1])) - float((depthmap[x, y - 1]))) / 2.0

            p = np.array([(x*depthmap[x,y]-cx)/fx, (y*depthmap[x,y]-cy)/fy, depthmap[x,y]])
            py = np.array([(x*depthmap[x,y+1]-cx)/fx, ((y+1)*depthmap[x,y+1]-cy)/fy, depthmap[x,y+1]])
            px = np.array([((x+1)*depthmap[x+1,y]-cx)/fx, (y*depthmap[x+1,y]-cy)/fy, depthmap[x+1,y]])

            #n = np.array([-dzdx, -dzdy, 0.005])
            n = np.cross(px-p, py-p)
            n = n * 1/np.linalg.norm(n)
            dir = p#np.array([x,y,1.0])
            dir = dir *1/np.linalg.norm(dir)

            normals[x, y] = (n*0.5 + 0.5)
            phong[x, y] = np.dot(dir,n)*0.5+0.5

    normals *= 255
    normals = normals.astype('uint8')
    #plt.imshow(depthmap, cmap='gray')
    #plt.show()
    #plt.imshow(normals)
    #plt.show()
    #plt.imshow(phong)
    #plt.show()
    #print('a')
    return normals

def error_image(im1, im2):
    fig = plt.figure()
    diff = (im1 - im2)
    #gt_vs_theirs[total_mask, :] = 0
    #print("theirs ", np.sqrt(np.sum(np.square(gt_vs_theirs))), np.mean(np.square(gt_vs_theirs)))
    ax = plt.axes([0, 0, 1, 1], frameon=False)
    # Then we disable our xaxis and yaxis completely. If we just say plt.axis('off'),
    # they are still used in the computation of the image padding.
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # Even though our axes (plot region) are set to cover the whole image with [0,0,1,1],
    # by default they leave padding between the plotted data and the frame. We use tigher=True
    # to make sure the data gets scaled to the full extents of the axes.
    plt.autoscale(tight=True)
    plt.imshow(np.linalg.norm(diff, axis=2), cmap='jet')
    #ax.plt.axes('off')



    #ax = plt.Axes(fig, [0., 0., 1., 1.])
    #ax.set_axis_off()
    #plt.show()
    return fig

def interpolate_pose_new(pose1, pose2, inter_num=91):
    """ pose1 and pose2 are 3x4 matrix """
    def interpolate(a, b, idx):
        return a + idx*(b-a)/inter_num
    new_cameras = []
    

    rots = np.concatenate((pose1[np.newaxis,:3,:3], pose2[np.newaxis,:3,:3]), axis=0)
    key_rots = R.from_matrix(rots)
    key_times = [0, 1]
    slerp = Slerp(key_times, key_rots)
    times = np.arange(0,1,1/inter_num)
    interp_rots = slerp(times)
    interp_rots_mat = interp_rots.as_matrix()
    for idx in range(inter_num):
        inter_pos = interpolate(pose1[:3,3], pose2[:3,3], idx)
        inter_rot = interp_rots_mat[idx]
        
        new_camera = np.zeros_like(pose1)
        new_camera[:3,:3] = inter_rot
        new_camera[:3,3] = inter_pos
        new_camera[3,3] = 1

        new_cameras.append(new_camera)
    return np.stack(new_cameras)

def fix_to_render(id1=207, id2=230,num=100,name=''):

    with open('/data/hanxinyang/MuNeRF_latest/dataset/'+name+'/transforms_val.json', 'r') as fp:
        metab = json.load(fp)
        frames = metab.get("frames")

        for frame in frames:
            id = frame.get("img_id")
            if id == id1:
                pose1 = np.array(frame.get("transform_matrix"))
            if id == id2:
                pose2 = np.array(frame.get("transform_matrix"))
    # pdb.set_trace()
    re = interpolate_pose_new(pose1, pose2,num)
    print('attention!!!!!!', re)
    return re

def repeat_expre(exp1, inter_num=100):
    """ pose1 and pose2 are 3x4 matrix """
    def interpolate(a, b, idx):
        return a + idx*(b-a)/inter_num
    new_exp = []
    exp1_np = np.array(exp1)
    for idx in range(inter_num):
        new_exp.append(exp1_np)
    return np.stack(new_exp)
    
def interpolate_expre(exp1, exp2, inter_num=100):
    """ pose1 and pose2 are 3x4 matrix """
    def interpolate(a, b, idx):
        return a + idx*(b-a)/inter_num
    new_exp = []
    exp1_np = np.array(exp1)
    exp2_np = np.array(exp2)
    for idx in range(inter_num):
        exp_inter = interpolate(exp1_np, exp2_np, idx)
        new_exp.append(exp_inter)
    return np.stack(new_exp)

def fix_to_render_exp(id1=207, id2=230,num=100,name=''):
    pdb.set_trace()
    with open('/data/hanxinyang/MuNeRF_latest/dataset/'+name+'/transforms_val.json', 'r') as fp:
        metab = json.load(fp)
        frames = metab.get("frames")

        for frame in frames:
            id = frame.get("img_id")
            if id == id1:
                exp1 = np.array(frame.get("expression"))
            if id == id2:
                exp2 = np.array(frame.get("expression"))
    re = interpolate_expre(exp1, exp2,num)
    return re

def fix_to_render_exp_H(id1=207, num=100,name=''):
    with open('/data/hanxinyang/MuNeRF_latest/dataset/'+name+'/transforms_val.json', 'r') as fp:
        metab = json.load(fp)
        frames = metab.get("frames")

        for frame in frames:
            id = frame.get("img_id")
            # pdb.set_trace()
            if id == int(id1):
                exp1 = np.array(frame.get("expression"))
    re = repeat_expre(exp1,num)
    return re

def cast_to_image(tensor, dataset_type):
    # Input tensor is (H, W, 3). Convert to (3, H, W).
    tensor = tensor.permute(2, 0, 1)
    tensor = tensor.clamp(0.0,1.0)
    # Convert to PIL Image and then np.array (output shape: (H, W, 3))
    img = np.array(torchvision.transforms.ToPILImage()(tensor.detach().cpu()))
    return img
    # # Map back to shape (3, H, W), as tensorboard needs channels first.
    # return np.moveaxis(img, [-1], [0])


def cast_to_disparity_image(tensor):
    img = (tensor - tensor.min()) / (tensor.max() - tensor.min())
    img = img.clamp(0, 1) * 255
    return img.detach().cpu().numpy().astype(np.uint8)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, help="Path to (.yml) config file."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Checkpoint / pre-trained model to evaluate.",
    )
    parser.add_argument(
        "--savedir", type=str, default='./renders/', help="Save images to this directory, if specified."
    )
    parser.add_argument(
        "--save-disparity-image", action="store_true", help="Save disparity images too."
    )
    parser.add_argument(
        "--save-error-image", action="store_true", help="Save photometric error visualization"
    )
    parser.add_argument(
        "--no_makeup", action="store_true", help="If conduct a nerf on makeup project"
    )
    parser.add_argument(
        "--save_normal",
        action="store_true", 
        help="whether to save normal images.",
    )
    parser.add_argument(
        "--save_depth",
        action="store_true", 
        help="whether to save depth images.",
    )
    parser.add_argument(
        "--fix_pose",
        action="store_true", 
        help="whether to save depth images.",
    )
    parser.add_argument(
        "--fix_expre",
        action="store_true", 
        help="whether to save depth images.",
    )
    parser.add_argument(
        "--white_bg",
        action="store_true", 
        help="whether to save depth images.",
    )
    parser.add_argument(
        "--id",
        type=int, 
        default=0, 
        help="whether to save depth images.",
    )
    parser.add_argument(
        "--girl_name",
        type=str, 
        default='', 
        help="whether to save depth images.",
    )
    parser.add_argument(
        "--nerface_dir",
        type=str, 
        default="/data/hanxinyang/MuNeRF_latest/rendering/finals/girl7_fix_expr_success", 
        help="whether to save depth images.",
    )
    parser.add_argument(
        "--exp_id",
        type=str, 
        default='', 
        help="whether to save depth images.",
    )
    parser.add_argument(
        "--pose1",
        type=int, 
        default=0, 
        help="whether to save depth images.",
    )
    parser.add_argument(
        "--pose2",
        type=int, 
        default=0, 
        help="whether to save depth images.",
    )
    parser.add_argument(
        "--consistent",
        action="store_true", 
        help="whether to save depth images.",
    )
    configargs = parser.parse_args()
    if_mask = configargs.white_bg
    # Read config file.
    cfg = None
    
    if configargs.consistent:
        consistent = True
    else:
        consistent = False

    if configargs.fix_pose:
        fix_pose = True
    else:
        fix_pose = False
    
    with open(configargs.config, "r") as f:
        cfg_dict = yaml.load(f, Loader=yaml.FullLoader)
        cfg = CfgNode(cfg_dict)

    images, poses, render_poses, hwf = None, None, None, None
    i_train, i_val, i_test = None, None, None
    # pdb.set_trace()
    if cfg.dataset.type.lower() == "blender":
        # Load blender dataset
        if configargs.no_makeup:
            # pdb.set_trace()
            images, poses, render_poses, hwf, i_split, expressions, _, bboxs, img_ids = load_flame_dataH(
                    cfg.dataset.basedir,
                    half_res=cfg.dataset.half_res,
                    scale=cfg.dataset.scale,
                    testskip=cfg.dataset.testskip,
                    test=True,
                    consistent=consistent
                )
        elif fix_pose:
            images, poses, render_poses, hwf, i_split, expressions, depths, bboxs, warped_imgs, makeups, paths, masks, img_ids, nerface_imgs = load_flame_data_colorH(
                cfg.dataset.basedir,
                cfg.dataset.style_id,
                half_res=cfg.dataset.half_res,
                scale=cfg.dataset.scale,
                testskip=cfg.dataset.testskip,
                test=True, # HXY 1010 to include train and test images
                fix_pose = fix_pose,
                nerface_dir=configargs.nerface_dir
            )
        elif consistent:
            images, poses, render_poses, hwf, i_split, expressions, depths, bboxs, warped_imgs, makeups, paths, masks, img_ids, nerface_imgs = load_flame_data_colorH(
                cfg.dataset.basedir,
                cfg.dataset.style_id,
                half_res=cfg.dataset.half_res,
                scale=cfg.dataset.scale,
                testskip=cfg.dataset.testskip,
                test=True, # HXY 1010 to include train and test images
                consistent=consistent
            )
        else:
            images, poses, render_poses, hwf, i_split, expressions, depths, bboxs, warped_imgs, makeups, paths, masks, img_ids = load_flame_data_color(
                cfg.dataset.basedir,
                cfg.dataset.style_id,
                half_res=cfg.dataset.half_res,
                scale=cfg.dataset.scale,
                testskip=cfg.dataset.testskip,
                test=True, # HXY 1010 to include train and test images
            )
        #i_train, i_val, i_test = i_split
        i_test = i_split
        H, W, focal = hwf
        H, W = int(H), int(W)
    elif cfg.dataset.type.lower() == "llff":
        # Load LLFF dataset
        images, poses, bds, render_poses, i_test = load_llff_data(
            cfg.dataset.basedir, factor=cfg.dataset.downsample_factor,
        )
        hwf = poses[0, :3, -1]
        H, W, focal = hwf
        hwf = [int(H), int(W), focal]
        render_poses = torch.from_numpy(render_poses)

    # Device on which to run.
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda" + ":" + str(cfg.experiment.device)
        images.to(device)
        if (not configargs.no_makeup) and (fix_pose): # 在固定expression的munerf测试中需要用到nerface_imgs
            nerface_imgs.to(device)
    encode_position_fn = get_embedding_function(
        num_encoding_functions=cfg.models.coarse.num_encoding_fn_xyz,
        include_input=cfg.models.coarse.include_input_xyz,
        log_sampling=cfg.models.coarse.log_sampling_xyz,
    )

    encode_direction_fn = None
    if cfg.models.coarse.use_viewdirs:
        encode_direction_fn = get_embedding_function(
            num_encoding_functions=cfg.models.coarse.num_encoding_fn_dir,
            include_input=cfg.models.coarse.include_input_dir,
            log_sampling=cfg.models.coarse.log_sampling_dir,
        )

    # Initialize a coarse resolution model.
    add_convolution = cfg.models.remove.if_remove
    concat_global = cfg.experiment.concat_global
    densitytype = 'NeRFModelDensitymodule'
    colortype = 'NeRFModelColormodule'
    nerftype = 'NerfHY'
    
    # for j in range(nerface_imgs.shape[0]):
    #     # pdb.set_trace()
    #     test_img = nerface_imgs[j].squeeze()
    #     torchvision.utils.save_image(cast_to_image2(test_img[..., :3], cfg.dataset.type.lower())[0], "/data/hanxinyang/MuNeRF_latest/debug/H/test_img/%d.png"%j, nrow=1)


    
    if configargs.no_makeup:
        model_coarse = getattr(models, cfg.models.coarse.type)(
          num_encoding_fn_xyz=cfg.models.coarse.num_encoding_fn_xyz,
          num_encoding_fn_dir=cfg.models.coarse.num_encoding_fn_dir,
          include_input_xyz=cfg.models.coarse.include_input_xyz,
          include_input_dir=cfg.models.coarse.include_input_dir,
          use_viewdirs=cfg.models.coarse.use_viewdirs,
          num_layers=cfg.models.coarse.num_layers,
          hidden_size=cfg.models.coarse.hidden_size,
          include_expression=True
        )
        model_coarse.to(device)

        # If a fine-resolution model is specified, initialize it.
        model_fine = None
        if hasattr(cfg.models, "fine"):
          model_fine = getattr(models, cfg.models.fine.type)(
              num_encoding_fn_xyz=cfg.models.fine.num_encoding_fn_xyz,
              num_encoding_fn_dir=cfg.models.fine.num_encoding_fn_dir,
              include_input_xyz=cfg.models.fine.include_input_xyz,
              include_input_dir=cfg.models.fine.include_input_dir,
              use_viewdirs=cfg.models.fine.use_viewdirs,
              num_layers=cfg.models.coarse.num_layers,
              hidden_size=cfg.models.coarse.hidden_size,
              include_expression=True
          )
          model_fine.to(device)

        checkpoint = torch.load(configargs.checkpoint,map_location=device)
        coarse = checkpoint["model_coarse_state_dict"]
        keys = checkpoint["model_coarse_state_dict"].keys()
        shapes = [coarse[key].shape for key in keys]
        #print('checkpoint', list(zip(keys,shapes)))
        # pdb.set_trace()
        model_coarse.load_state_dict(checkpoint["model_coarse_state_dict"])
        # pdb.set_trace()
        if checkpoint["model_fine_state_dict"]:
          try:
              model_fine.load_state_dict(checkpoint["model_fine_state_dict"])
          except:
              print(
                  "The checkpoint has a fine-level model, but it could "
                  "not be loaded (possibly due to a mismatched config file."
              )
        if "height" in checkpoint.keys():
          hwf[0] = checkpoint["height"]
        if "width" in checkpoint.keys():
          hwf[1] = checkpoint["width"]
        if "focal_length" in checkpoint.keys():
          hwf[2] = checkpoint["focal_length"]
        if "background" in checkpoint.keys():
          background = checkpoint["background"]
          if background is not None:
              print("loaded background with shape ", background.shape)
              background.to(device)
        if "latent_codes" in checkpoint.keys():
          latent_codes = checkpoint["latent_codes"]
          use_latent_code = False
          if latent_codes is not None:
              use_latent_code = True
              latent_codes.to(device)
              print("loading index map for latent codes...")
              idx_map = np.load(cfg.dataset.basedir + "/index_map.npy").astype(int)
              print("loaded latent codes from checkpoint, with shape ", latent_codes.shape)
        model_coarse.eval()
        if model_fine:
          model_fine.eval()
    else:
    # ***------- for makeup test --------- *** #
    # load density --- fixed #
        use_patch_trans_gen = cfg.models.remove.transformer
        use_cross_trans = cfg.transformer.cross
        fix_density = cfg.experiment.fix_density
        if use_patch_trans_gen:
            transblock_patch = trans.define_G(cfg.transformer.embed_patch_dim, cfg.transformer.netG, cfg.transformer.init_type, cfg.transformer.init_gain, [cfg.experiment.device])
        else:
            transblock_patch = None
        if use_cross_trans:
            transblock_cross = trans.define_G(cfg.transformer.embed_cross_dim, cfg.transformer.netG, cfg.transformer.init_type, cfg.transformer.init_gain, [cfg.experiment.device])
        else:
            transblock_cross = None
        logdir = os.path.join(cfg.experiment.logdir, cfg.experiment.id)
        no_coarse_color = cfg.experiment.no_coarse_color
        model_coarse_density, model_coarse_color, model_fine_density, model_fine_color = create_part(cfg, device, no_coarse_color)
        # pdb.set_trace()
        model_coarse, model_fine, removecolor = create_module(cfg, device, model_coarse_density, model_coarse_color, model_fine_density, model_fine_color, transblock_patch, transblock_cross, fix_density)
        fixed_density =  False
        if fixed_density:
          checkpoint_density = torch.load(os.path.join(logdir, "checkpoint_density_fixed"+".ckpt"),map_location=device)
          model_coarse_density.load_state_dict(checkpoint_density['coarse'])
          model_fine_density.load_state_dict(checkpoint_density['fine'])

        # pdb.set_trace()
        checkpoint_color_all = torch.load(configargs.checkpoint,map_location=device)
        # ------ load checkpoint ---------#
        checkpoint_color_coarse = checkpoint_color_all['model_coarse_state_dict']
        checkpoint_color_fine = checkpoint_color_all['model_fine_state_dict']
        checkpoint_color_coarse_, checkpoint_density_coarse_ = {}, {}
        checkpoint_color_fine_, checkpoint_density_fine_ = {}, {}
        for item in checkpoint_color_coarse.keys():
          if item[:7]=='density': checkpoint_density_coarse_[item[8:]] = checkpoint_color_coarse[item]
          if item[:5]=='color': checkpoint_color_coarse_[item[6:]] = checkpoint_color_coarse[item]
        for item in checkpoint_color_fine.keys():
          if item[:7]=='density': checkpoint_density_fine_[item[8:]] = checkpoint_color_fine[item]
          if item[:5]=='color': checkpoint_color_fine_[item[6:]] = checkpoint_color_fine[item]

        if not no_coarse_color:
            model_coarse_color.load_state_dict(checkpoint_color_coarse_)
        model_fine_color.load_state_dict(checkpoint_color_fine_)
        model_coarse_density.load_state_dict(checkpoint_density_coarse_)
        model_fine_density.load_state_dict(checkpoint_density_fine_)

        if add_convolution:
            # 上采样
          removecolor.load_state_dict(checkpoint_color_all['removecolor_state_dict'])
        checkpoint = checkpoint_color_all
        if "height" in checkpoint.keys():
          hwf[0] = checkpoint["height"]
        if "width" in checkpoint.keys():
          hwf[1] = checkpoint["width"]
        if "focal_length" in checkpoint.keys():
          hwf[2] = checkpoint["focal_length"]
        # pdb.set_trace()
        if "background" in checkpoint.keys():
          background = checkpoint["background"]
        #   pdb.set_trace()
          if background is not None:
              print("loaded background with shape ", background.shape)
              background.to(device)
        if "latent_codes" in checkpoint.keys():
          latent_codes = checkpoint["latent_codes"]
          use_latent_code = False
          if latent_codes is not None:
              use_latent_code = True
              latent_codes.to(device)
              print("loading index map for latent codes...")
              idx_map = np.load(cfg.dataset.basedir + "/index_map.npy").astype(int)
              print("loaded latent codes from checkpoint, with shape ", latent_codes.shape)

        model_coarse.eval()
        if model_fine:
          model_fine.eval()
        if add_convolution:
            removecolor.eval()


    replace_background = True
    if replace_background:
        from PIL import Image
        #background = Image.open('./view.png')
        background = Image.open(cfg.dataset.basedir + '/bg/bc.jpg')
        #background = Image.open("./real_data/andrei_dvp/" + '/bg/00050.png')
        background.thumbnail((H,W))
        background = torch.from_numpy(np.array(background).astype(float)).to(device)
        background = background/255
        print('loaded custom background of shape', background.shape)

        #background = torch.ones_like(background)
        #background.permute(2,0,1)
    render_poses = render_poses.float().to(device)
    
    # Create directory to save images to.
    os.makedirs(configargs.savedir, exist_ok=True)
    if configargs.save_disparity_image:
        os.makedirs(os.path.join(configargs.savedir, "disparity"), exist_ok=True)
    if configargs.save_error_image:
        os.makedirs(os.path.join(configargs.savedir, "error"), exist_ok=True)
    if configargs.save_normal:
        os.makedirs(os.path.join(configargs.savedir, "normals"), exist_ok=True)
    if configargs.save_depth:
        os.makedirs(os.path.join(configargs.savedir, "depth"), exist_ok=True)
    # Evaluation loop
    times_per_image = []
    # print('i_test', i_test)
    transform_mask = transforms.Compose([
        transforms.Resize((256,256), interpolation=PIL.Image.NEAREST)])
        #transforms.ToTensor()])
    if if_mask:
        mask_no_makeup = masks[0][i_test].to(device)
        print(mask_no_makeup.shape)
        mask_bg  = (mask_no_makeup == 0).float() + (mask_no_makeup == 17).float()
        mask_bg = transform_mask(mask_bg)
        mask_bg, _, index_bg, _ = mask_preprocess(mask_bg, mask_bg)
    
    render_poses = render_poses.float().to(device)
    
    # for rendering both train and test
    both = consistent
    if both:
        final = i_test[0]
        for index, thing in enumerate(i_test):
            if index == 0:
                continue
            np.append(final, thing)
        i_test = final
    # pdb.set_trace()
    
    render_poses = poses[i_test].float().to(device)
    num = len(render_poses)
    print('inter num is: ', num)
    # pdb.set_trace()

    if configargs.no_makeup:
        flag = 0
    else:
        flag = 1

    if configargs.fix_pose:
        # 1008 HXY
        # 讲道理，要测consistency，在不用其他benchmark的情况，只能是说我们的和nerface几乎完全一致才行
        # 然后在选择pose上，其实朝向为上下左右的最好都要覆盖。这个仿照一下eg3d吧。可以先找左右两个做个实验，然后再改进一下。 
        print("Pose 1 : %d"%configargs.pose1)
        print("Pose 2 : %d"%configargs.pose2)

        if configargs.girl_name=='girl7':
            render_poses = fix_to_render(configargs.pose1,configargs.pose2,num,configargs.girl_name)
        if configargs.girl_name=='girl9':
            render_poses = fix_to_render(configargs.pose1,configargs.pose2,num,configargs.girl_name)
        render_poses = torch.tensor(render_poses).float().to(device)
        if not flag:
            torch.save(render_poses, os.path.join("/data/hanxinyang/MuNeRF_latest/debug/H", "density.pth"))
        else:
            torch.save(render_poses, os.path.join("/data/hanxinyang/MuNeRF_latest/debug/H", "style.pth"))

    #expressions = torch.arange(-6,6,0.5).float().to(device)
    # pdb.set_trace()
    render_expressions = expressions[i_test].float().to(device)
    if configargs.fix_expre:
        print("Fix expression")
        render_expressions = fix_to_render_exp_H(configargs.exp_id, num, configargs.girl_name)
        # if configargs.girl_name=='girl7':
        #     render_expressions = fix_to_render_exp_H(212,num,configargs.girl_name)
        # if configargs.girl_name=='girl9':
        #     render_expressions = fix_to_render_exp_H(115,num,configargs.girl_name)
        render_expressions = torch.tensor(render_expressions).float().to(device)
    
    #avg_img = torch.mean(images[i_train],axis=0)
    #avg_img = torch.ones_like(avg_img)
    #render_lights = lights[i_test].float().to(device)
    #render_details = details[i_test].float().to(device)
    

    #pose = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    #for i, pose in enumerate(tqdm(render_poses)):
    index_of_image_after_train_shuffle = 0
    if configargs.fix_expre:
        if not configargs.fix_pose:
            render_poses = render_poses[[configargs.id]*num]
    if configargs.fix_pose:
        render_expressions = render_expressions[[configargs.id]*num] ### TODO render specific expression
    #######################
    no_background = False
    no_expressions = False
    no_lcode = True
    nerf = False
    frontalize = False
    interpolate_mouth = False

    #######################
    if nerf:
        no_background = True
        no_expressions = True
        no_lcode = True
    if no_background: background=None
    if add_convolution: background = None
    if no_expressions: render_expressions = torch.zeros_like(render_expressions, device=render_expressions.device)
    if no_lcode:
        use_latent_code = True
        latent_codes = torch.zeros(5000,32,device=device)
    renders = render_expressions #list(zip(render_expressions, render_lights, render_details))
    
        
    
    for i, item in enumerate(tqdm(renders)):
        # if i < 266:
        #     continue
    #for i in range(75,151):
        torch.save(renders, os.path.join("/data/hanxinyang/MuNeRF_latest/debug/H", "expre.pth"))
        expression = item
        # reference = torch.load(os.path.join("/data/hanxinyang/MuNeRF_latest/debug/H", "expre.pth"))
        # pdb.set_trace()
        #lights = item[1]
        #details = item[2]
        #print('details shape', details.shape)
        #if i%25 != 0: ### TODO generate only every 25th im
        #if i != 511: ### TODO generate only every 25th im
        #    continue
        # if i > 20:
            # pdb.set_trace()
        start = time.time()
        rgb = None, None
        disp = None, None
        with torch.no_grad():
            pose = render_poses[i]

            # if flag:
            #     reference = torch.load(os.path.join("/data/hanxinyang/MuNeRF_latest/debug/H", "style.pth"))
            #     print("Density and style compare:")
            #     print(reference == render_poses)
                # pdb.set_trace()

            # pdb.set_trace()
            if interpolate_mouth:
                frame_id = 241
                num_images = 150
                pose = render_poses[241]
                expression = render_expressions[241].clone()
                expression[68] = torch.arange(-1, 1, 2 / 150, device=device)[i]

            if frontalize:
                pose = render_poses[0]
            #pose = render_poses[300] ### TODO fixes pose
            #expression = render_expressions[0] ### TODO fixes expr
            #expression = torch.zeros_like(expression).to(device)

            '''ablate = 'view_dir'

            if ablate == 'expression':
                pose = render_poses[100]
            elif ablate == 'latent_code':
                pose = render_poses[100]
                expression = render_expressions[100]
                if idx_map[100+i,1] >= 0:
                    #print("found latent code for this image")
                    index_of_image_after_train_shuffle = idx_map[100+i,1]
            elif ablate == 'view_dir':
                pose = render_poses[100]
                expression = render_expressions[100]
                _, ray_directions_ablation = get_ray_bundle(hwf[0], hwf[1], hwf[2], render_poses[240+i][:3, :4])'''

            pose = pose[:3, :4]

            #pose = torch.from_numpy(np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]))
            # if use_latent_code:
            #     if idx_map[i,1] >= 0:
            #         print("found latent code for this image")
            #         index_of_image_after_train_shuffle = idx_map[i,1]
            #index_of_image_after_train_shuffle = 10 ## TODO Fixes latent code
            #index_of_image_after_train_shuffle = idx_map[84,1] ## TODO Fixes latent code v2 for andrei
            index_of_image_after_train_shuffle = 10#idx_map[10,1] ## TODO Fixes latent code - USE THIS if not ablating!

            latent_code = latent_codes[index_of_image_after_train_shuffle].to(device) if use_latent_code else None
            #latent_code = None
            #latent_code = torch.mean(latent_codes)
            if add_convolution:
                feat_dim = cfg.experiment.feat_dim
                feat_scale = feat_dim / hwf[0]
                hwf[0] = feat_dim
                hwf[1] = feat_dim
                hwf[2] = np.array([hwf[2][0]*feat_scale, hwf[2][1]*feat_scale, hwf[2][2], hwf[2][3]])

            # 将pose转化成ray
            # 用expression来生成radience field
            ray_origins, ray_directions = get_ray_bundle(hwf[0], hwf[1], hwf[2], pose)
            # pdb.set_trace()
            rgb_coarse, disp_coarse, _, rgb_fine, disp_fine, _, weights = run_one_iter_of_nerf(
                hwf[0],
                hwf[1],
                hwf[2],
                model_coarse,
                model_fine,
                ray_origins,
                ray_directions,
                cfg,
                mode="validation",
                encode_position_fn=encode_position_fn,
                encode_direction_fn=encode_direction_fn,
                expressions = expression,
         #       lights = lights,
         #       details = details,
                # background_prior = background.view(-1,3) if (background is not None) else None, # 1007 HXY
                # background_prior = torch.ones_like(background).view(-1,3),  # White background
                background_prior = None,
                latent_code = latent_code,
                ray_directions_ablation = None,
                flag = flag
            )
            
            rgb = rgb_fine if rgb_fine is not None else rgb_coarse
            # pdb.set_trace()
            if configargs.save_normal:
                normals = torch_normal_map(disp_fine, focal, weights, clean=True)
                #normals = normal_map_from_depth_map_backproject(disp_fine.cpu().numpy())
                # save_plt_image(normals.cpu().numpy().astype('uint8'), os.path.join(configargs.savedir, 'normals', f"{i:04d}.png"))
                save_plt_image(normals.cpu().numpy().astype('uint8'), os.path.join(configargs.savedir, 'normals', img_ids[i]))
            
            if configargs.save_depth:
                #disp_fine = disp_fine*255.
                depth_fine = disp_fine.unsqueeze(-1)
                depth_fine =  torch.cat((depth_fine,depth_fine,depth_fine),dim=-1)
                depth_t, depth_fine = cast_to_image2(depth_fine[..., :3], cfg.dataset.type.lower())
                depth_t = depth_t.permute(1,2,0)
                depth_t = depth_t * 255.
                # savefile3 = os.path.join(configargs.savedir, 'depth', f"{i:04d}.png")
                savefile3 = os.path.join(configargs.savedir, 'depth', img_ids[i])
                save_plt_image(depth_t.cpu().numpy().astype('uint8'), os.path.join(configargs.savedir, 'depth', img_ids[i]))
                #imageio.imwrite(savefile3, depth_fine)
            
            #if configargs.save_disparity_image:
            if False:
                disp = disp_fine if disp_fine is not None else disp_coarse
                #normals = normal_map_from_depth_map_backproject(disp.cpu().numpy())
                normals = normal_map_from_depth_map_backproject(disp_fine.cpu().numpy())
                save_plt_image(normals.astype('uint8'), os.path.join(configargs.savedir,'normals', f"{i:04d}.png"))

        # pdb.set_trace()

        times_per_image.append(time.time() - start)
        if configargs.savedir:
            # pdb.set_trace()
            # savefile = os.path.join(configargs.savedir, f"{i:04d}.png")
            savefile = os.path.join(configargs.savedir, img_ids[i])
            if add_convolution:
                if not configargs.fix_pose:
                #prepare warp image(input image) `s rects
                    paths_warped = paths['warped']
                    paths_makeup = paths['makeup']
                    pose_target = poses[i].to(device)
                    expression_target = expressions[i].to(device)
                    pose_target = pose_target.view(-1,16)
                    expression_target = expression_target.unsqueeze(0)
                    pose_exp_target = torch.cat((pose_target,expression_target), dim=1)
                    pose_target_resize = pose_exp_target.unsqueeze(0)
                    pose_target_resize = pose_target_resize.unsqueeze(0)
                    n_local = cfg.loss.landmark.n_local
                    resize_scale = cfg.dataset.scale * 2.

                    img_target = warped_imgs[i].to(device)
                    makeup_img = makeups[i].to(device)
                    img_B_arr = np.array(torchvision.transforms.ToPILImage()((img_target.permute(2, 0, 1)).detach().cpu()))
                    landmark_B_api = paths_warped[i]
                    if landmark_B_api is not None:
                        img_target_rects = facePartsCoordinatesAPI(img_B_arr, landmark_B_api, n_local=n_local, scaling_factor=resize_scale)
                    else:
                        img_target_rects = None # b
                    img_M_arr = np.array(torchvision.transforms.ToPILImage()((makeup_img.permute(2, 0, 1)).detach().cpu()))
                    landmark_M_api = paths_makeup[i]
                    if landmark_M_api is not None:
                        makeup_target_rects = facePartsCoordinatesAPI(img_M_arr, landmark_M_api, n_local=n_local, general_ratio = 0.2, scaling_factor=resize_scale)  
                    else:
                        makeup_target_rects = None# a
                    if configargs.no_makeup:
                        rgb_t, _ = cast_to_image2(rgb[..., :3], cfg.dataset.type.lower())
                    else:
                        rgb_t, _ = cast_to_image2(rgb, cfg.dataset.type.lower())  
                    img = images[i].to(device)
                
                    warp_img = warped_imgs[i].to(device)
                    img_t, _ = cast_to_image2(img[..., :3], cfg.dataset.type.lower())
                    warp_img_t, _ = cast_to_image2(warp_img[..., :3], cfg.dataset.type.lower())
                    makeup_img_t, _ = cast_to_image2(warp_img[..., :3], cfg.dataset.type.lower())
                    debug_dir = ''
                    if concat_global:
                        if use_patch_trans_gen:
                            size_img = 512 * cfg.dataset.scale
                            patch_pairs = get_pairs(img_t.unsqueeze(0), makeup_img_t.unsqueeze(0), img_target_rects, makeup_target_rects, size_img, 64, debug_dir)
                            real_rgb,_ = removecolor(rgb_t.unsqueeze(0), img_t.unsqueeze(0), makeup_img_t.unsqueeze(0), patch_pairs, pose_target_resize, use_cross_trans)
                        else:
                            real_rgb = removecolor(rgb_t.unsqueeze(0), img_t.unsqueeze(0))
                    else:
                        real_rgb = removecolor(rgb_t.unsqueeze(0))
                    real_rgb = (real_rgb + 1) / 2
                    # real_rgb = real_rgb.squeeze().clamp(0.0,1.0)
                    real_rgb = real_rgb.squeeze()
                    img = np.array(torchvision.transforms.ToPILImage()(real_rgb.detach().cpu()))
                    if if_mask:
                        mask_bg = np.array(torchvision.transforms.ToPILImage()(mask_bg.detach().cpu()))
                        img = img*mask_bg
                    
                    imageio.imwrite(savefile, img)
                else:
                    paths_warped = paths['warped']
                    paths_makeup = paths['makeup']
                    pose_target = render_poses[i].to(device)
                    expression_target = render_expressions[i].to(device)
                    pose_target = pose_target.view(-1,16)
                    expression_target = expression_target.unsqueeze(0)
                    pose_exp_target = torch.cat((pose_target,expression_target), dim=1)
                    pose_target_resize = pose_exp_target.unsqueeze(0)
                    pose_target_resize = pose_target_resize.unsqueeze(0)
                    n_local = cfg.loss.landmark.n_local
                    resize_scale = cfg.dataset.scale * 2.

                    img_target = warped_imgs[i].to(device)
                    makeup_img = makeups[i].to(device)
                    img_B_arr = np.array(torchvision.transforms.ToPILImage()((img_target.permute(2, 0, 1)).detach().cpu()))
                    landmark_B_api = paths_warped[i]
                    if landmark_B_api is not None:
                        img_target_rects = facePartsCoordinatesAPI(img_B_arr, landmark_B_api, n_local=n_local, scaling_factor=resize_scale)
                    else:
                        img_target_rects = None # b
                    img_M_arr = np.array(torchvision.transforms.ToPILImage()((makeup_img.permute(2, 0, 1)).detach().cpu()))
                    landmark_M_api = paths_makeup[i]
                    if landmark_M_api is not None:
                        makeup_target_rects = facePartsCoordinatesAPI(img_M_arr, landmark_M_api, n_local=n_local, general_ratio = 0.2, scaling_factor=resize_scale)  
                    else:
                        makeup_target_rects = None# a
                    if configargs.no_makeup:
                        rgb_t, _ = cast_to_image2(rgb[..., :3], cfg.dataset.type.lower())
                    else:
                        rgb_t, _ = cast_to_image2(rgb, cfg.dataset.type.lower())  
                    img = images[i].to(device)
                    nerface_img = nerface_imgs[i].to(device)
                
                    warp_img = warped_imgs[i].to(device)
                    img_t, _ = cast_to_image2(img[..., :3], cfg.dataset.type.lower())
                    nerface_img_t, _ = cast_to_image2(nerface_img[..., :3], cfg.dataset.type.lower())
                    warp_img_t, _ = cast_to_image2(warp_img[..., :3], cfg.dataset.type.lower())
                    makeup_img_t, _ = cast_to_image2(warp_img[..., :3], cfg.dataset.type.lower())
                    debug_dir = ''
                    if concat_global:
                        if use_patch_trans_gen:
                            size_img = 512 * cfg.dataset.scale
                            patch_pairs = get_pairs(img_t.unsqueeze(0), makeup_img_t.unsqueeze(0), img_target_rects, makeup_target_rects, size_img, 64, debug_dir)
                            real_rgb,_ = removecolor(rgb_t.unsqueeze(0), img_t.unsqueeze(0), makeup_img_t.unsqueeze(0), patch_pairs, pose_target_resize, use_cross_trans)
                        else:
                            # real_rgb = removecolor(rgb_t.unsqueeze(0), img_t.unsqueeze(0)) # here !
                            # torchvision.utils.save_image(rgb_t, "/data/hanxinyang/MuNeRF_latest/debug/H/rbg_t%d.png"%i, nrow=1)
                            torchvision.utils.save_image(nerface_img_t, "/data/hanxinyang/MuNeRF_latest/debug/H/nerface_img_t%d.png"%i, nrow=1)
                            # pdb.set_trace()
                            real_rgb = removecolor(rgb_t.unsqueeze(0), nerface_img_t.unsqueeze(0)) # here !
                    else:
                        real_rgb = removecolor(rgb_t.unsqueeze(0))
                    real_rgb = (real_rgb + 1) / 2
                    # real_rgb = real_rgb.squeeze().clamp(0.0,1.0)
                    real_rgb = real_rgb.squeeze()
                    img = np.array(torchvision.transforms.ToPILImage()(real_rgb.detach().cpu()))
                    if if_mask:
                        mask_bg = np.array(torchvision.transforms.ToPILImage()(mask_bg.detach().cpu()))
                        img = img*mask_bg
                    
                    imageio.imwrite(savefile, img)
            else:
                # pdb.set_trace()

                print(savefile)
                imageio.imwrite(
                    savefile, cast_to_image(rgb[..., :3], cfg.dataset.type.lower())
                )
            if configargs.save_disparity_image:
                savefile = os.path.join(configargs.savedir, "disparity", f"{i:04d}.png")
                imageio.imwrite(savefile, cast_to_disparity_image(disp_fine))
            if configargs.save_error_image:
                savefile = os.path.join(configargs.savedir, "error", f"{i:04d}.png")
                GT = images[i_test][i]
                fig = error_image(GT, rgb.cpu().numpy())
                #imageio.imwrite(savefile, cast_to_disparity_image(disp))
                plt.savefig(savefile,pad_inches=0,bbox_inches='tight',dpi=54)
        # tqdm.write(f"Avg time per image: {sum(times_per_image) / (i + 1)}")


if __name__ == "__main__":
    main()
