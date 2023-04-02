import argparse
import glob
import os
import time
from unittest.mock import patch
import imageio
import sys
import model.Transformer as trans
import pdb
from numpy.core.fromnumeric import resize
from numpy.lib.function_base import disp
from vgg import VGG19
sys.path.insert(1, './nerf')
os.environ['GPU_DEBUG']='3'
import numpy as np
import torch
import torchvision
import yaml
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
from PIL import Image
from train_helper import create_part, create_module
from nerf.load_flame import load_flame_data, load_flame_data_color
from nerf.load_mask import load_mask_data2, mask_for_gt
from torchvision import transforms
from torchvision import utils as vtils
from helpers import FaceCropper
from model.model import PatchGAN, Transfromer_PatchGAN, get_pairs
from model.options import Options
from model.histogram_loss import HistogramLoss
from utils.train_util import facePartsCoordinatesAPI
from utils.api_util import FacePPAPI
from settings import DENSITY_TRAIN, COLOR_TRAIN
from torch.autograd import Variable
fc = FaceCropper(predictor_dir = './shape_predictor_68_face_landmarks.dat',)
sys.path.insert(1, './prnet')
sys.path.insert(1, './prnet/utils')
from prnet.api import PRN
import cv2
import numpy as np
from prnet.utils.cv_plot import plot_vertices
from nerf import (CfgNode, get_embedding_function, get_ray_bundle, img2mse, 
                   meshgrid_xy, models, landmark2d_loss, cal_histloss2, prnet, 
                  mse2psnr, run_one_iter_of_nerf, dump_rays, GaussianSmoothing)
local_parts = ['eye', 'eye_', 'mouth', 'nose', 'cheek', 'cheek_', 'eyebrow', 'eyebrow_', 'uppernose', 'forehead', 'sidemouth', 'sidemouth_']


def save_pairs(imgs,img_filename):
    row1 = torch.cat(imgs, 0)
    torchvision.utils.save_image(row1, img_filename, nrow=3)

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

def main():
    # parsing arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, help="Path to (.yml) config file."
    )
    parser.add_argument(
        "--density_nerf",
        action="store_true", 
        help="train a first nerf for density",
    )
    parser.add_argument(
        "--load_checkpoint",
        type=str,
        default="",
        help="Path to load saved checkpoint for finetune density and color to train from scratch.",
    )
    parser.add_argument(
        "--load_density_checkpoint",
        type=str,
        default="",
        help="Path to load saved checkpoint for finetune density and color to train from scratch.",
    )
    parser.add_argument(
        "--color_continue_train",
        action="store_true", 
        help="Path to load saved checkpoint from of new color module",
    )
    parser.add_argument(
        "--continue_train",
        type=str,
        default="",
        help="Path to load saved checkpoint from of non-makeup nerf.",
    )
    parser.add_argument(
        "--debug_dir",
        type=str,
        default="./debug/",
        help="Path to save debug files.",
    )
    parser.add_argument(
        "--make_gt",
        action="store_true", 
        help="Path to load saved checkpoint from of new color module",
    )
    configargs = parser.parse_args()    
    cfg = None

    # loading configs from .yaml files
    with open(configargs.config, "r") as f:
        cfg_dict = yaml.load(f, Loader=yaml.FullLoader)
        cfg = CfgNode(cfg_dict)
    
    # Device on which to run.
    if torch.cuda.is_available():
        # pdb.set_trace()
        device = "cuda" + ":" + str(cfg.experiment.device)
        #device_backup = "cuda" + ":" + str(cfg.experiment.device_backup)   
    else:
        device = "cpu"
        
    # pdb.set_trace()
    density = configargs.density_nerf
    
    # If a pre-cached dataset is available, skip the dataloader.
    USE_CACHED_DATASET = False
    train_paths, validation_paths = None, None
    images, poses, render_poses, hwf, i_split, expressions = None, None, None, None, None, None
    H, W, focal, i_train, i_val, i_test = None, None, None, None, None, None
    if hasattr(cfg.dataset, "cachedir") and os.path.exists(cfg.dataset.cachedir):
        train_paths = glob.glob(os.path.join(cfg.dataset.cachedir, "train", "*.data"))
        validation_paths = glob.glob(
            os.path.join(cfg.dataset.cachedir, "val", "*.data")
        )
        USE_CACHED_DATASET = True
    else:
        # Load dataset
        images, poses, render_poses, hwf, expressions = None, None, None, None, None
        # exp: 50, pose: 6
        if cfg.dataset.type.lower() == "blender":
            # pdb.set_trace()
            if density:
                images, poses, render_poses, hwf, i_split, expressions, _, bboxs, _ = load_flame_data(
                    cfg.dataset.basedir,
                    half_res=cfg.dataset.half_res,
                    scale=cfg.dataset.scale,
                    testskip=cfg.dataset.testskip,
                )
            else:
                # pdb.set_trace()
                images, poses, render_poses, hwf, i_split, expressions, depths, bboxs, warped_imgs, makeups, paths, masks, img_ids= load_flame_data_color(
                    cfg.dataset.basedir,
                    cfg.dataset.style_id,
                    half_res=cfg.dataset.half_res,
                    scale=cfg.dataset.scale,
                    testskip=cfg.dataset.testskip,
                )
            i_train, i_val, i_test = i_split
            H, W, focal = hwf
            H, W = int(H), int(W)
            hwf = [H, W, focal]
            if cfg.nerf.train.white_background:
                images = images[..., :3] * images[..., -1:] + (1.0 - images[..., -1:])
    # Seed experiment for repeatability
    seed = cfg.experiment.randomseed
    np.random.seed(seed)
    torch.manual_seed(seed)
    # pdb.set_trace()

    # Instantiation funtions and classes with specified params.

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

    # Initialize a coarse-resolution model.
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
    # pdb.set_trace()

    # If a fine-resolution model is specified, initialize it.
    model_fine = None
    if hasattr(cfg.models, "fine"):
        model_fine = getattr(models, cfg.models.fine.type)(
            num_encoding_fn_xyz=cfg.models.fine.num_encoding_fn_xyz,
            num_encoding_fn_dir=cfg.models.fine.num_encoding_fn_dir,
            include_input_xyz=cfg.models.fine.include_input_xyz,
            include_input_dir=cfg.models.fine.include_input_dir,
            use_viewdirs=cfg.models.fine.use_viewdirs,
            num_layers = cfg.models.coarse.num_layers,
            hidden_size =cfg.models.coarse.hidden_size,
            include_expression=True
        )
        model_fine.to(device)
    

    ###################################
    #   setting options
    ###################################
    train_background = False
    supervised_train_background = False
    blur_background = False

    train_latent_codes = True #here
    disable_expressions = False # True to disable expressions
    disable_latent_codes = False # True to disable latent codes
    disable_light = False # True to disable latent codes  #
    fixed_background = True # Do False to disable BG

    regularize_latent_codes = True # True to add latent code LOSS, false for most experiments
    use_patchgan_loss =  cfg.loss.patchgan.patchgan_loss
    use_new_patchgan = cfg.loss.patchgan.new_patchgan_loss
    use_patch_trans_gen = cfg.models.remove.transformer
    use_cross_trans = cfg.transformer.cross
    use_patchgan_content_loss = cfg.loss.patchgan.patchgan_content_loss
    use_remove_loss =  cfg.loss.remove.remove_loss
    use_landmark_loss = cfg.loss.landmark.landmark_loss
    use_histogram_loss = cfg.loss.histogram.histogram_loss
    use_old_histogram = cfg.loss.histogram.histogram_loss_old
    use_l2_loss = cfg.loss.l2.l2_loss
    use_depth_loss = cfg.loss.depth.depth_loss
    concat_vgg = cfg.experiment.concat_vgg
    concat_global = cfg.experiment.concat_global
    if_extreme = cfg.experiment.extreme_makeup
    ###################################
    ###################################


    if concat_vgg or use_new_patchgan:
        vggnet = VGG19(requires_grad=False).to(device)

    if concat_vgg:
        content_features = []
        for i in i_train:
            content_img = load_image(os.path.join(cfg.dataset.basedir,'train',img_ids[i])).to(device)
            content_feature = vggnet(content_img)
            content_features.append(content_feature['conv4_2']) # 512 -- 64


    if use_new_patchgan:
        transblock = trans.define_G(cfg.transformer.embed_dim, cfg.transformer.netG, cfg.transformer.init_type, cfg.transformer.init_gain, [cfg.experiment.device])
        if use_patchgan_content_loss:
            content_transblock = trans.define_G(cfg.transformer.embed_dim, cfg.transformer.netG, cfg.transformer.init_type, cfg.transformer.init_gain, [cfg.experiment.device])
    if use_patch_trans_gen:
        transblock_patch = trans.define_G(cfg.transformer.embed_patch_dim, cfg.transformer.netG, cfg.transformer.init_type, cfg.transformer.init_gain, [cfg.experiment.device])
    else:
        transblock_patch = None #, transblock_patch1, transblock_patch2 = None, None, None
    if use_cross_trans:
        transblock_cross = trans.define_G(cfg.transformer.embed_cross_dim, cfg.transformer.netG, cfg.transformer.init_type, cfg.transformer.init_gain, [cfg.experiment.device])
    else:
        transblock_cross = None #, transblock_cross1, transblock_cross2 = None, None, None
    supervised_train_background = train_background and supervised_train_background
    prn = PRN(is_dlib=False)
    # pdb.set_trace()
    if train_background:
        with torch.no_grad():
            avg_img = torch.mean(images[i_train],axis=0)
            print('want a avg iamge', avg_img)
            # Blur Background:
            if blur_background:
                avg_img = avg_img.permute(2,0,1)
                avg_img = avg_img.unsqueeze(0)
                smoother = GaussianSmoothing(channels=3, kernel_size=11, sigma=11)
                print("smoothed background initialization. shape ", avg_img.shape)
                avg_img = smoother(avg_img).squeeze(0).permute(1,2,0)
            background = torch.tensor(avg_img, device=device)
            #print('want a background iamge', background)
        background.requires_grad = True

    if fixed_background: # load GT background
        print("loading GT background to condition on")
        from PIL import Image
        background = Image.open(os.path.join(cfg.dataset.basedir,'bg','bc.jpg'))
        background.thumbnail((H,W))
        background = torch.from_numpy(np.array(background).astype(np.float32)).to(device)
        background = background/255
        print("bg shape", background.shape)
        print("should be ", images[i_train][0].shape)
        assert background.shape == images[i_train][0].shape
    else:
       background = None

    # Initialize optimizer.
    trainable_parameters = list(model_coarse.parameters())
    if model_fine is not None:
        trainable_parameters = trainable_parameters + list(model_fine.parameters())
    if train_background:
        #background.requires_grad = True
        #trainable_parameters.append(background) # add it later when init optimizer for different lr
        print("background.is_leaf " ,background.is_leaf, background.device)

    if train_latent_codes:
        latent_codes = torch.zeros(len(i_train),32, device=device)  #here
        #latent_codes = torch.zeros(2240,32, device=device)
        print("initialized latent codes with shape %d X %d" % (latent_codes.shape[0], latent_codes.shape[1]))
        if not disable_latent_codes:
            trainable_parameters.append(latent_codes)
            latent_codes.requires_grad = True

    if train_background:
        optimizer = getattr(torch.optim, cfg.optimizer.type)(
            [{'params':trainable_parameters},
             {'params':background, 'lr':cfg.optimizer.lr}],
            lr=cfg.optimizer.lr
        )
    else:
        print('debug',cfg.optimizer.lr)
        optimizer = getattr(torch.optim, cfg.optimizer.type)(
            [{'params':trainable_parameters},
             {'params': background, 'lr': cfg.optimizer.lr}], # this is obsolete but need for continuing training
            lr=cfg.optimizer.lr
        )
    logdir = os.path.join(cfg.experiment.logdir, cfg.experiment.id)
    os.makedirs(logdir, exist_ok=True)
    writer = SummaryWriter(logdir)
    with open(os.path.join(logdir, "config.yml"), "w") as f:
        f.write(cfg.dump())  # cfg, f, default_flow_style=False)

    start_iter = 0
    color = configargs.color_continue_train
    fix_density = cfg.experiment.fix_density
    no_coarse_color = cfg.experiment.no_coarse_color
    make_gt = configargs.make_gt
    # pdb.set_trace()

    if os.path.exists(configargs.load_checkpoint) and not density:
        # -- create models for training -- #
        model_coarse_density, model_coarse_color, model_fine_density, model_fine_color = create_part(cfg, device, no_coarse_color)
        model_coarse, model_fine, removecolor = create_module(cfg, device, model_coarse_density, model_coarse_color, model_fine_density, model_fine_color, transblock_patch, transblock_cross, fix_density)

        if use_new_patchgan:
            patchgan = Transfromer_PatchGAN(vggnet, transblock, cfg.models.patchgan.input_dim_a, cfg.models.patchgan.input_dim_a, 
                cfg.models.patchgan.dis_n_layer, cfg.experiment.device, cfg.loss.landmark.n_local)
            if use_patchgan_content_loss: content_patchgan = Transfromer_PatchGAN(vggnet, content_transblock, cfg.models.patchgan.input_dim_b, cfg.models.patchgan.input_dim_b, 
                cfg.models.patchgan.dis_n_layer, cfg.experiment.device, cfg.loss.landmark.n_local)
        else:
            patchgan = PatchGAN(cfg.models.patchgan.input_dim_a, cfg.models.patchgan.input_dim_b, 
                cfg.models.patchgan.dis_n_layer, cfg.experiment.device)
            if use_patchgan_content_loss: content_patchgan = PatchGAN(cfg.models.patchgan.input_dim_a, cfg.models.patchgan.input_dim_b, 
                cfg.models.patchgan.dis_n_layer, cfg.experiment.device)
        flag_density = False
        if flag_density:
            checkpoint_density = torch.load(configargs.load_density_checkpoint, map_location=device)

        # Load nerf density here
        checkpoint = torch.load(configargs.load_checkpoint, map_location=device)
        density_train, color_train = DENSITY_TRAIN, COLOR_TRAIN
        coarse_checkpoint_density, coarse_checkpoint_color, fine_checkpoint_density, fine_checkpoint_color, checkpoint_density = {}, {}, {}, {}, {}
        old_checkpoint_coarse = checkpoint["model_coarse_state_dict"]
        old_checkpoint_fine = checkpoint["model_fine_state_dict"]
        # pdb.set_trace()

        if not flag_density:
            for item in density_train:
                coarse_checkpoint_density[item] = old_checkpoint_coarse[item] if not color else old_checkpoint_coarse['density.'+item]
                fine_checkpoint_density[item] = old_checkpoint_fine[item] if not color else old_checkpoint_fine['density.'+item]
                checkpoint_density['coarse'] = coarse_checkpoint_density 
                checkpoint_density['fine'] = fine_checkpoint_density
        #torch.save(checkpoint_density, os.path.join(logdir, "checkpoint_density_fixed" + ".ckpt"))
        else:
            for item in density_train:
                old_checkpoint_coarse_d = checkpoint_density["model_coarse_state_dict"]
                old_checkpoint_fine_d = checkpoint_density["model_fine_state_dict"]
                coarse_checkpoint_density[item] =  old_checkpoint_coarse_d[item]
                fine_checkpoint_density[item] = old_checkpoint_fine_d[item]
                checkpoint_density['coarse'] = coarse_checkpoint_density 
                checkpoint_density['fine'] = fine_checkpoint_density
        for item in color_train:
            if not no_coarse_color:
                coarse_checkpoint_color[item] = old_checkpoint_coarse[item] if not color else old_checkpoint_coarse['color.'+item]
            fine_checkpoint_color[item] = old_checkpoint_fine[item] if not color else old_checkpoint_fine['color.'+item]
        
        # -- load coarse and fine model -- #
        # Load nerf color
        model_coarse_density.load_state_dict(coarse_checkpoint_density)
        if color and not no_coarse_color: model_coarse_color.load_state_dict(coarse_checkpoint_color)
        if checkpoint["model_fine_state_dict"]:
            model_fine_density.load_state_dict(fine_checkpoint_density)
            if color: model_fine_color.load_state_dict(fine_checkpoint_color)
        if color:
            removecolor.load_state_dict(checkpoint["removecolor_state_dict"])
            patchgan.load_state_dict(checkpoint["model_state_dict"])
        if checkpoint["background"] is not None:
            print("loaded bg from checkpoint")
            background = torch.nn.Parameter(checkpoint['background'].to(device))
            if fixed_background:
                background.requires_grad = False
        
        trainable_density = list(model_coarse_density.parameters())
        trainable_parameters = [] if no_coarse_color else list(model_coarse_color.parameters())
        if model_fine_color is not None:
            trainable_density = trainable_density + list(model_fine_density.parameters())
            trainable_parameters = trainable_parameters + list(model_fine_color.parameters())
            
        if checkpoint["latent_codes"] is not None:
            print("loaded latent codes from checkpoint")
            latent_codes = torch.nn.Parameter(checkpoint['latent_codes'].to(device))
            if not disable_latent_codes:
                trainable_parameters.append(latent_codes)
                latent_codes.requires_grad = True
        if use_remove_loss:
            trainable_parameters = trainable_parameters + list(removecolor.parameters())    
        if use_patchgan_loss:
            trainable_parameters_ = list(patchgan.parameters())
            if use_patchgan_content_loss:
                trainable_parameters_ = trainable_parameters_ + [param for param in content_patchgan.parameters() if param.requires_grad]
            # if use_new_patchgan: trainable_parameters_ = trainable_parameters_ + list(patchgan.transblock.parameters())
            optimizer_d = getattr(torch.optim, cfg.optimizer.type)(
                [{'params':trainable_parameters_}],
                lr=cfg.optimizer.lr
            )
        if not fix_density:
            
            optimizer_g = getattr(torch.optim, cfg.optimizer.type)(
                [{'params':trainable_parameters},
                {'params':trainable_density, 'lr': 0.001*cfg.optimizer.lr},
                {'params':background, 'lr':cfg.optimizer.lr}],
                lr=cfg.optimizer.lr
            )
        else:
            optimizer_g = getattr(torch.optim, cfg.optimizer.type)(
                trainable_parameters,
                lr=cfg.optimizer.lr
            )
            
        if color:
            opt_ckpt = checkpoint["optimizer_g_state_dict"]
            optimizer_g.load_state_dict(opt_ckpt)
            if use_patchgan_loss:
                opt_ckptd = checkpoint["optimizer_d_state_dict"]
                optimizer_d.load_state_dict(opt_ckptd)
        start_iter = checkpoint["iter"]
        images_gt = images
        images = warped_imgs
    

    # Load an existing checkpoint, if a path is specified.    
    if os.path.exists(configargs.continue_train):
        checkpoint = torch.load(configargs.continue_train, map_location=device)
        model_coarse.load_state_dict(checkpoint["model_coarse_state_dict"])
        if checkpoint["model_fine_state_dict"]:
            model_fine.load_state_dict(checkpoint["model_fine_state_dict"])
        if checkpoint["background"] is not None:
            print("loaded bg from checkpoint")
            background = torch.nn.Parameter(checkpoint['background'].to(device))
        if checkpoint["latent_codes"] is not None:
            print("loaded latent codes from checkpoint")
            latent_codes = torch.nn.Parameter(checkpoint['latent_codes'].to(device))
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_iter = checkpoint["iter"]
    # # TODO: Prepare raybatch tensor if batching random rays

    # pdb.set_trace()

    # Prepare importance sampling maps
    ray_importance_sampling_maps = []
    p = 0.9
    print("computing boundix boxes probability maps")
    for i in i_train:
        bbox = bboxs[i]
        probs = np.zeros((H,W))
        probs.fill(1-p)
        probs[bbox[0]:bbox[1],bbox[2]:bbox[3]] = p
        probs = (1/probs.sum()) * probs
        ray_importance_sampling_maps.append(probs.reshape(-1))
    print("Starting loop")
    
    if not density and use_remove_loss:
        removecolor.train()
    if not density and use_patchgan_loss:
        patchgan.train()

    debug_dir = configargs.debug_dir
    os.makedirs(debug_dir, exist_ok=True)

    train_iters = start_iter + cfg.experiment.train_iters
    
    for i in trange(start_iter, train_iters):
        model_coarse.train()
        if model_fine is not None:
            model_fine.train()

        rgb_coarse, rgb_fine = None, None
        target_ray_values = None
        background_ray_values = None
        if USE_CACHED_DATASET:
            datafile = np.random.choice(train_paths)
            cache_dict = torch.load(datafile)
            ray_bundle = cache_dict["ray_bundle"].to(device)
            ray_origins, ray_directions = (
                ray_bundle[0].reshape((-1, 3)),
                ray_bundle[1].reshape((-1, 3)),
            )
            target_ray_values = cache_dict["target"][..., :3].reshape((-1, 3))
            select_inds = np.random.choice(
                ray_origins.shape[0],
                size=(cfg.nerf.train.num_random_rays),
                replace=False,
            )
            ray_origins, ray_directions = (
                ray_origins[select_inds],
                ray_directions[select_inds],
            )
            target_ray_values = target_ray_values[select_inds].to(device)

            rgb_coarse, _, _, rgb_fine, _, _ = run_one_iter_of_nerf(
                cache_dict["height"],
                cache_dict["width"],
                cache_dict["focal_length"],
                model_coarse,
                model_fine,
                ray_origins,
                ray_directions,
                cfg,
                mode="train",
                encode_position_fn=encode_position_fn,
                encode_direction_fn=encode_direction_fn,
                expressions=expressions
                #lights = None
            )
        else:
            # pdb.set_trace()

            img_idx = np.random.choice(i_train)
            img_target = images[img_idx].to(device) # training target
            debug_img = img_target.detach().cpu().numpy()*255
            root_H = "/data/hanxinyang/MuNeRF_latest/debug/H/"
            debug_path_H = os.path.join(root_H, "img_target.jpg")
            # pdb.set_trace()
            cv2.imwrite(debug_path_H, debug_img)
            pose_target = poses[img_idx].to(device)
            if not disable_expressions:
                expression_target = expressions[img_idx].to(device) # vector
            else: # zero expr
                expression_target = torch.zeros(50, device=device)
            if not disable_latent_codes:
                latent_code = latent_codes[img_idx].to(device) if train_latent_codes else None  #here
            else:
                latent_codes = torch.zeros(32, device=device)
            # print('pase and expression', pose_target.shape, expression_target.shape)

            if not density:
                # parameters
                paths_warped = paths['warped']
                paths_makeup = paths['makeup']
                n_local = cfg.loss.landmark.n_local
                resize_scale = cfg.dataset.scale * 2.
                a = cfg.loss.coarse.loss_weight
                b = cfg.loss.fine.loss_weight
                img_gt = images_gt[img_idx].to(device) # no makeup
                if concat_vgg:
                    content_feature = content_features[img_idx]

                # B warped makeup image
                # img_target = images[img_idx].to(device)
                if depths is not None:
                    depth_target = depths[img_idx].to(device)
                else:
                    depth_target = None
                mask_no_makeup = masks[0][img_idx].to(device)
                img_B_arr = np.array(torchvision.transforms.ToPILImage()((img_target.permute(2, 0, 1)).detach().cpu()))
                landmark_B_api = paths_warped[img_idx]
                landmark_gt = fc.facePointsStasm(img_B_arr)
                if landmark_B_api is not None:
                    img_target_rects = facePartsCoordinatesAPI(img_B_arr, landmark_B_api, n_local=n_local, general_ratio = 0.1, scaling_factor=resize_scale)
                    img_target_rects_1 = facePartsCoordinatesAPI(img_B_arr, landmark_B_api, n_local=n_local, general_ratio = 0.2, scaling_factor=resize_scale)
                    img_target_rects_2 = facePartsCoordinatesAPI(img_B_arr, landmark_B_api, n_local=n_local, general_ratio = 0.4, scaling_factor=resize_scale)
                else:
                    img_target_rects = None # b
                
                # A dynamic makeup image

                makeup_target = makeups[img_idx].to(device)
                mask_makeup =  masks[1][img_idx].to(device)
                img_M_arr = np.array(torchvision.transforms.ToPILImage()((makeup_target.permute(2, 0, 1)).detach().cpu()))
                landmark_M_api = paths_makeup[img_idx]
                if landmark_M_api is not None:
                    makeup_target_rects = facePartsCoordinatesAPI(img_M_arr, landmark_M_api, n_local=n_local, general_ratio = 0.2, scaling_factor=resize_scale)  
                else:
                    makeup_target_rects = None# a
                size_gt = img_gt.shape[1]
                img_gt_t, _ = cast_to_image2(img_gt[..., :3])
                img_target_t, _ = cast_to_image2(img_target[..., :3])
                makeup_target_t, _ = cast_to_image2(makeup_target[..., :3])
            
            # sample rays
            ray_origins, ray_directions = get_ray_bundle(H, W, focal, pose_target)
            ray_origins_all, ray_directions_all = ray_origins, ray_directions

            # pdb.set_trace()
            # generate pixels
            coords = torch.stack(
                meshgrid_xy(torch.arange(H).to(device), torch.arange(W).to(device)),
                dim=-1,
            )
            coords = coords.reshape((-1, 2))
            # Use importance sampling to sample mainly in the bbox with prob p
            select_inds = np.random.choice(
                coords.shape[0], size=(cfg.nerf.train.num_random_rays), replace=False, p=ray_importance_sampling_maps[img_idx]
            )
            select_inds = coords[select_inds]
            
            ray_origins = ray_origins[select_inds[:, 0], select_inds[:, 1], :]
            ray_directions = ray_directions[select_inds[:, 0], select_inds[:, 1], :]
            target_s = img_target[select_inds[:, 0], select_inds[:, 1], :]
            
            if not density and depth_target is not None:
                #design depth for coarse and fine
                target_p = depth_target[select_inds[:, 0], select_inds[:, 1], :]
                target_p = target_p[...,:1]
                depth_target_ = depth_target[...,:1]

            background_ray_values = background[select_inds[:, 0], select_inds[:, 1], :] if (train_background or fixed_background) else None
            then = time.time()
            
            # pdb.set_trace()
            # train nerf
            psnr = 0.0
            loss_total = 0.0
            if density or use_l2_loss:
                rgb_coarse, _, _, rgb_fine, disp_fine, _, weights = run_one_iter_of_nerf(
                    H,
                    W,
                    focal,
                    model_coarse,
                    model_fine,
                    ray_origins,
                    ray_directions,
                    cfg,
                    mode="train",
                    encode_position_fn=encode_position_fn,
                    encode_direction_fn=encode_direction_fn,
                    expressions = expression_target,
                    background_prior= None,#background_ray_values,
                    latent_code = latent_code if not disable_latent_codes else torch.zeros(32,device=device)
                )
                target_ray_values = target_s
                
                # rgb loss
                coarse_loss = None
                if rgb_coarse is not None:
                    coarse_loss = torch.nn.functional.mse_loss(
                                    rgb_coarse[..., :3], target_ray_values[..., :3]
                                )
                print(f"Coarse loss : {coarse_loss}")
        
                fine_loss = None
                if rgb_fine is not None:
                    fine_loss = torch.nn.functional.mse_loss(
                                    rgb_fine[..., :3], target_ray_values[..., :3]
                                )
                l2_loss = (coarse_loss if coarse_loss is not None else 0.0) + (fine_loss if fine_loss is not None else 0.0)
                #loss_l2 = l2_loss*cfg.loss.l2.l2_weight1 if i < 350000 + cfg.loss.l2.l2_iter else l2_loss*cfg.loss.l2.l2_weight2
                loss_total = loss_total + l2_loss
                psnr = mse2psnr(l2_loss.item())
            
            
            latent_code_loss = torch.zeros(1, device=device)
            if train_latent_codes and not disable_latent_codes:
                latent_code_loss = torch.norm(latent_code) * 0.0005
            loss_total = loss_total + (latent_code_loss*10 if regularize_latent_codes else 0.0)

            background_loss = torch.zeros(1, device=device)
            if supervised_train_background:
                background_loss = torch.nn.functional.mse_loss(
                    background_ray_values[..., :3], target_ray_values[..., :3], reduction='none'
                ).sum(1)
                background_loss = torch.mean(background_loss*weights) * 0.001
            loss_total = loss_total + (background_loss if supervised_train_background is not None else 0.0)
            
            dis_back = ((i-start_iter)%cfg.experiment.g_step==0) and (i-start_iter>0) and i >cfg.loss.patchgan.patchgan_iter +start_iter
            if not density and cfg.loss.addloss.loss and i % cfg.loss.addloss.iter == 0:
                # generate rays
                feat_dim = cfg.experiment.feat_dim
                feat_scale = feat_dim / H
                feat_focal = np.array([focal[0]*feat_scale, focal[1]*feat_scale, focal[2], focal[3]])
                ray_origins_all, ray_directions_all = get_ray_bundle(feat_dim, feat_dim, feat_focal, pose_target)
                coarse_img, _, depth_coarse, fine_img, depth_fine, _ ,weights= run_one_iter_of_nerf(
                    feat_dim,
                    feat_dim,
                    feat_focal,
                    model_coarse,
                    model_fine,
                    ray_origins_all.to(device),
                    ray_directions_all.to(device),
                    cfg,
                    mode="validation",
                    encode_position_fn=encode_position_fn,
                    encode_direction_fn=encode_direction_fn,
                    expressions = expression_target,
                    background_prior = None, 
                    latent_code = torch.zeros(32).to(device) if train_latent_codes or disable_latent_codes else None,    #here
                    flag = 0 if density else 1
                )

                # save image results
                save_depth_debug = False
                save_res_debug = True
                img_fine_t_old, _ = cast_to_image2(fine_img) #!!!!!!!!!
                if coarse_img is None: coarse_img = fine_img
                img_coarse_t_old, _ = cast_to_image2(coarse_img)
                flag = ''
                    
                if cfg.loss.convolution.low_loss:
                    low_img_target_t = torch.nn.functional.interpolate(img_target_t.unsqueeze(0), size=(feat_dim, feat_dim)).squeeze(0)
                    low_loss = b*torch.nn.functional.mse_loss(img_fine_t_old.squeeze(), low_img_target_t.to(device)) + a*torch.nn.functional.mse_loss(img_coarse_t_old.squeeze(), low_img_target_t.to(device))
                    loss_total = loss_total + low_loss * cfg.loss.convolution.gen_weight

                # concat content vgg feature
                if concat_vgg:
                    content_feature = torch.sigmoid(content_feature)
                    img_fine_t_old_ = torch.cat((img_fine_t_old,content_feature.squeeze()), dim=0)
                    img_coarse_t_old_ = torch.cat((img_coarse_t_old,content_feature.squeeze()), dim=0)

                    real_fine = removecolor(img_fine_t_old_.unsqueeze(0))
                    real_coarse = removecolor(img_coarse_t_old_.unsqueeze(0))
                else:
                    if concat_global:
                        if use_patch_trans_gen:
                            if cfg.models.remove.single:
                                flag = 'makeup'
                                size_img = 512 * cfg.dataset.scale
                                pose_target = pose_target.view(-1,16)
                                expression_target = expression_target.unsqueeze(0)
                                pose_exp_target = torch.cat((pose_target,expression_target), dim=1)
                                pose_target_resize = pose_exp_target.unsqueeze(0)
                                pose_target_resize = pose_target_resize.unsqueeze(0)
                                if flag=='non': 
                                    patch_pairs = get_pairs(img_gt_t.unsqueeze(0), img_target_t.unsqueeze(0), img_target_rects_1, img_target_rects_1, size_img,50, debug_dir)
                                else: 
                                    patch_pairs = get_pairs(img_gt_t.unsqueeze(0), makeup_target_t.unsqueeze(0), img_target_rects_1, makeup_target_rects, size_img,64, debug_dir)
                                real_fine, patchs = removecolor(img_fine_t_old.unsqueeze(0), img_gt_t.unsqueeze(0), makeup_target_t.unsqueeze(0), patch_pairs, pose_target_resize, use_cross_trans)
                                real_coarse, _ = removecolor(img_coarse_t_old.unsqueeze(0), img_gt_t.unsqueeze(0), makeup_target_t.unsqueeze(0), patch_pairs, pose_target_resize, use_cross_trans)
                            else:
                                size_img = 512 * cfg.dataset.scale
                                patch_pair = get_pairs(img_gt_t.unsqueeze(0), img_target_t.unsqueeze(0), img_target_rects,size_img,50)
                                patch_pair1 = get_pairs(img_gt_t.unsqueeze(0), img_target_t.unsqueeze(0), img_target_rects_1,size_img,50)
                                patch_pair2 = get_pairs(img_gt_t.unsqueeze(0), img_target_t.unsqueeze(0), img_target_rects_2,size_img,50)
                                patch_pairs = [[patch_pair[0], patch_pair1[0], patch_pair2[0]], [patch_pair[1], patch_pair1[1], patch_pair2[1]]]
                                real_fine = removecolor(img_fine_t_old.unsqueeze(0), img_gt_t.unsqueeze(0), patch_pairs, use_cross_trans)
                                real_coarse = removecolor(img_coarse_t_old.unsqueeze(0), img_gt_t.unsqueeze(0), patch_pairs, use_cross_trans)
                        else:
                            real_fine = removecolor(img_fine_t_old.unsqueeze(0), img_gt_t.unsqueeze(0))
                            real_coarse = removecolor(img_coarse_t_old.unsqueeze(0), img_gt_t.unsqueeze(0))
                    else:
                        real_fine = removecolor(img_fine_t_old.unsqueeze(0))
                        real_coarse = removecolor(img_coarse_t_old.unsqueeze(0))     
                # additional operation! Important!!!!!
                real_fine = (real_fine + 1) / 2
                real_coarse = (real_coarse + 1) / 2
                gen_loss = b*torch.nn.functional.mse_loss(real_fine.squeeze(), img_target_t.to(device)) + a*torch.nn.functional.mse_loss(real_coarse.squeeze(), img_target_t.to(device))
                if cfg.loss.convolution.gen_loss:
                    loss_total = loss_total + gen_loss * cfg.loss.convolution.gen_weight
                psnr = mse2psnr(gen_loss.item())

                img_fine_t = real_fine.squeeze()
                img_coarse_t = real_coarse.squeeze()
                img_fine = np.array(torchvision.transforms.ToPILImage()(img_fine_t.clamp(0.0,1.0).detach().cpu()))
                img_coarse = np.array(torchvision.transforms.ToPILImage()(img_coarse_t.clamp(0.0,1.0).detach().cpu()))
                
                if save_res_debug and (i % 1 == 0):
                    savefile = os.path.join(debug_dir, '%d_debug_img.jpg'%i)
                    # imgs = (img_target_t.unsqueeze(0),img_fine_t_old.unsqueeze(0),real_fine,img_gt_t.unsqueeze(0))
                    imgs = (img_target_t.unsqueeze(0),real_fine,img_gt_t.unsqueeze(0))
                    os.makedirs(os.path.join(debug_dir,'res_rgb'), exist_ok=True)
                    save_res = os.path.join(debug_dir, 'res_rgb',str(img_idx)+'result_rgb.jpg')
                    #saveimg(save_res, real_fine)
                    try:
                        save_pairs(imgs, savefile)
                    except:
                        print('Save debug_img.jpg fail!')

                # save depth images
                if save_depth_debug:
                    depth_fine_t, _ = cast_to_depth(depth_fine.unsqueeze(-1))
                    depth_coarse_t, _ = cast_to_depth(depth_coarse.unsqueeze(-1))
                    depth_gt_t, depth_gt = cast_to_image2(depth_target[..., :3])
                    savefile3 = os.path.join(debug_dir, 'depth_img.jpg')
                    imgs = (depth_coarse_t.unsqueeze(0), depth_fine_t.unsqueeze(0), depth_gt_t.unsqueeze(0))
                    try:
                        save_pairs(imgs, savefile3)
                    except:
                        print('Save depth_img.jpg fail!')

                if use_depth_loss and not fix_density:
                    if depth_fine is not None:
                        depth_loss = b * torch.nn.functional.mse_loss(depth_fine.unsqueeze(-1), depth_target_) +\
                                        a * torch.nn.functional.mse_loss(depth_coarse.unsqueeze(-1), depth_target_)
                    else:
                        depth_loss = a * torch.nn.functional.mse_loss(depth_coarse.unsqueeze(-1), depth_target_)
                    loss_depth = depth_loss * cfg.loss.depth.depth_weight2
                    loss_total =  loss_total + loss_depth
                        
                if use_landmark_loss and i >= start_iter + cfg.loss.landmark.start_iter:
                    landmark_coarse = fc.facePointsStasm(img_coarse) #api.faceLandmarkDetector(savefile)
                    landmark_fine = fc.facePointsStasm(img_fine) #api.faceLandmarkDetector(savefile2)
                    
                    if not landmark_fine.any() or not landmark_coarse.any() or not landmark_gt.any():
                        geo_loss = torch.tensor(0.)
                    else:
                        geo_loss = landmark2d_loss(torch.from_numpy(landmark_gt).float(), torch.from_numpy(landmark_fine).float(), torch.from_numpy(landmark_coarse).float())   
                    loss_geo = geo_loss * cfg.loss.landmark.landmark_weight
                    loss_total = loss_total + loss_geo.to(device)
                
                size_img = 512 * cfg.dataset.scale
                criterionHis = HistogramLoss().to(device)
                fake_makeup1 = img_fine_t # 3xHxW
                fake_makeup2 = img_coarse_t
                img_target = img_target[..., :3]
                makeup = img_target.permute(2, 0, 1) # first order image style_target; warp image img_target
                makeup_gt = makeup_target.permute(2, 0, 1) # makeup image
                fake_makeup1, fake_makeup2, mask_no_makeup, mask_makeup, makeup, nonmakeup,makeup_gt = load_mask_data2(fake_makeup1,fake_makeup2,makeup,makeup_gt, img_gt_t, mask_no_makeup,mask_makeup, img_target_rects_1, resize_scale)
                if use_histogram_loss:
                    if use_old_histogram:
                        loss_total = loss_total
                    else:
                        if i>cfg.loss.histogram.histogram_iter + start_iter and if_extreme:
                            if_dense=True
                        else:
                            if_dense = False
                        #print('debug new dense', if_dense)
                        # 保存 face_makeup, makeup, makeup_gt，组合三个tesnor然后保存下来。
                        histoloss_fine = cal_histloss2(criterionHis, vggnet, prn, fake_makeup1,makeup,makeup_gt, mask_no_makeup,mask_makeup,nonmakeup,debug_dir, size_img,if_dense)
                        histoloss_coarse = cal_histloss2(criterionHis, vggnet, prn, fake_makeup2,makeup, makeup_gt, mask_no_makeup,mask_makeup,nonmakeup,debug_dir, size_img,if_dense)
                        ###
                        hist_loss = b*histoloss_fine + a*histoloss_coarse
                        loss_hist = hist_loss * cfg.loss.histogram.histogram_weight
                        #print('dense loss', hist_loss)
                        loss_total = loss_total + loss_hist.to(device)
                
                loss_d_total = 0
                
                if_points, if_texture, if_uv = False, True, True
                # pdb.set_trac
                _, uv_fake_t, uv_target_t = prnet(prn, debug_dir, fake_makeup1, makeup, makeup_gt, if_points, if_texture)
                uv_fixed_rects = [[52, 102, 56, 106], [52, 102, 145, 195], [123, 173, 100, 150], [84, 134, 110, 160], [80, 130, 56, 106], [80, 130, 145, 195], [29, 79, 56, 106], [29, 79, 145, 195], [52, 102, 110, 160], [7, 57, 106, 156], [123, 173, 56, 106], [123, 173, 143, 193]]
                
                if not if_dense:
                    if use_patchgan_loss and img_target_rects_1 is not None:
                        if_content = False
                        if not if_uv:
                            image_a_coarse = [img_coarse_t.unsqueeze(0), img_target_rects_1]      #GEN
                            image_b = [img_target.permute(2, 0, 1).unsqueeze(0), img_target_rects_1] #!!!!!!!!!!!! #STYLE
                            #image_b = [makeup_target.permute(2, 0, 1).unsqueeze(0), makeup_target_rects]
                        else:
                            image_a_coarse = [uv_fake_t.permute(2, 0, 1).unsqueeze(0), uv_fixed_rects]      #GEN
                            image_b = [uv_target_t.permute(2, 0, 1).unsqueeze(0), uv_fixed_rects] #!!!!!!!!!!!! #STYLE
                        coarsestyle_loss_g_dict = patchgan.local_style_g(image_a_coarse, image_b, size_img, if_content)
                        coarse_loss_g = sum(coarsestyle_loss_g_dict.values())
                        fine_loss_g = None
                        if not if_uv and fine_img is not None:
                            image_a_fine = [img_fine_t.unsqueeze(0), img_target_rects_1]
                            finestyle_loss_g_dict = patchgan.local_style_g(image_a_fine, image_b, size_img, if_content)
                            fine_loss_g = sum(finestyle_loss_g_dict.values())
                        patchgan_loss_g = coarse_loss_g + (fine_loss_g if fine_loss_g is not None else 0.0)
                        if i>cfg.loss.patchgan.patchgan_iter +start_iter:
                            loss_patchgan = patchgan_loss_g * cfg.loss.patchgan.patchgan_g_style_weight1 if i<cfg.loss.patchgan.patchgan_iter + start_iter + cfg.loss.patchgan.patchgan_iter_weight else patchgan_loss_g*cfg.loss.patchgan.patchgan_g_style_weight2
                            #print('patchgan g, loss:', loss_patchgan)
                        else:
                            loss_patchgan = torch.tensor(0.0)
                        # print('loss_patchgan_style', cfg.loss.patchgan.patchgan_weight1, ':', loss_patchgan)
                        loss_total = loss_total + loss_patchgan.to(device)
                        if dis_back:
                            coarsestyle_loss_d_dict = patchgan.local_style_d(image_a_coarse, image_b, size_img, if_content)
                            coarse_loss_d = sum(coarsestyle_loss_d_dict.values())
                            fine_loss_d = None
                            if not if_uv and fine_img is not None:
                                image_a_fine = [img_fine_t.unsqueeze(0), img_target_rects_1]
                                finestyle_loss_d_dict = patchgan.local_style_d(image_a_fine, image_b, size_img, if_content)
                                fine_loss_d = sum(finestyle_loss_d_dict.values())
                            patchgan_loss_d = coarse_loss_d + (fine_loss_d if fine_loss_d is not None else 0.0)
                            loss_d_total = loss_d_total + patchgan_loss_d.to(device)
                            #print('patchgan s d, loss:', patchgan_loss_d)
                else:
                    print('dense landmark dont use patchgan')
        
        if density:
            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()
        else:
            if not if_dense:
                if use_patchgan_loss and img_target_rects is not None and dis_back:
                    optimizer_d.zero_grad()
                    loss_d_total.backward()
                    optimizer_d.step()
                else:
                    optimizer_g.zero_grad()
                    loss_total.backward()
                    optimizer_g.step()
            else:
                print('!')
            
        # Learning rate updates
        num_decay_steps = cfg.scheduler.lr_decay * 1000
        lr_new = cfg.optimizer.lr * (
            cfg.scheduler.lr_decay_factor ** (i / num_decay_steps)
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr_new

        if i % cfg.experiment.print_every == 0 or i == train_iters - 1:
            tqdm.write(
                "[TRAIN] Iter: "
                + str(i)
                + " Loss: "
                + str(loss_total.item())
                + " BG Loss: "
                + str(background_loss.item())
                + " PSNR: "
                + str(psnr)
                + " LatentReg: "
                + str(latent_code_loss.item())
            )


        #gpu_profile(frame=sys._getframe(), event='line', arg=None)
        if train_latent_codes:
            writer.add_scalar("train/code_loss", latent_code_loss.item(), i)
        if supervised_train_background:
            writer.add_scalar("train/bg_loss", background_loss.item(), i)

        if rgb_coarse is not None:
            writer.add_scalar("train/coarse_loss", coarse_loss.item(), i)
        if rgb_fine is not None:
           writer.add_scalar("train/fine_loss", fine_loss.item(), i)
        #writer.add_scalar("train/psnr", psnr, i)

        # Validation
        if (
            i % cfg.experiment.validate_every == 0
            or i ==train_iters - 1 and False
        ):
            #torch.cuda.empty_cache()
            tqdm.write("[VAL] =======> Iter: " + str(i))
            model_coarse.eval()
            if model_fine:
                model_coarse.eval()

            start = time.time()
            with torch.no_grad():
                rgb_coarse, rgb_fine = None, None
                target_ray_values = None
                if USE_CACHED_DATASET:
                    datafile = np.random.choice(validation_paths)
                    cache_dict = torch.load(datafile)
                    rgb_coarse, _, _, rgb_fine, _, weights = run_one_iter_of_nerf(
                        cache_dict["height"],
                        cache_dict["width"],
                        cache_dict["focal_length"],
                        model_coarse,
                        model_fine,
                        cache_dict["ray_origins"].to(device),
                        cache_dict["ray_directions"].to(device),
                        cfg,
                        mode="validation",
                        encode_position_fn=encode_position_fn,
                        encode_direction_fn=encode_direction_fn,
                        expressions = expression_target,
                        latent_code = torch.zeros(32, device=device)
                    )
                    target_ray_values = cache_dict["target"].to(device)
                else:
                    # Do all validation set...
                    loss = 0
                    for img_idx in i_val[:2]:
                    # for img_idx in i_val[:]:

                        img_target = images[img_idx].to(device)

                        pose_target = poses[img_idx, :3, :4].to(device)
                        ray_origins, ray_directions = get_ray_bundle(
                            H, W, focal, pose_target
                        )
                        # pdb.set_trace()
                        rgb_coarse, _, _, rgb_fine, _, _ ,weights= run_one_iter_of_nerf(
                            H,
                            W,
                            focal,
                            model_coarse,
                            model_fine,
                            ray_origins,
                            ray_directions,
                            cfg,
                            mode="validation",
                            encode_position_fn=encode_position_fn,
                            encode_direction_fn=encode_direction_fn,
                            expressions = expression_target,
                            background_prior = background.view(-1,3) if (train_background or fixed_background) else None,
                            # background_prior =  None, # 1007 HXY
                            latent_code = torch.zeros(32).to(device) if train_latent_codes or disable_latent_codes else None,

                        )
                        #print("did one val")
                        debug_dir = configargs.debug_dir
                        # pdb.set_trace()
                        if density:
                            savefile = os.path.join(debug_dir, 'debug_val%d.jpg'%i)
                            
                        else:
                            save = os.path.join(debug_dir, 'val_feature','debug_fine_val.jpg')
                            savefile = os.path.join(debug_dir, str(img_idx)+'debug_val_makeup.jpg')
                        if rgb_coarse is None: rgb_coarse = rgb_fine
                        fine_t,_ = cast_to_image2(rgb_fine)
                        coarse_t,_ = cast_to_image2(rgb_coarse)
                        debug_f = os.path.join(root_H, "fine_t.jpg")
                        debug_c = os.path.join(root_H, "coarse_t.jpg")
                        torchvision.utils.save_image(fine_t, debug_f, nrow=1)
                        torchvision.utils.save_image(coarse_t, debug_c, nrow=1)
                        # pdb.set_trace()
                        imgs = (fine_t.unsqueeze(0), coarse_t.unsqueeze(0))
                        try:
                            save_pairs(imgs,savefile)
                            #saveimg(save, fine_t.unsqueeze(0))
                        except:
                            print('Save debug_val fail!')
                        target_ray_values = img_target
                        coarse_loss = img2mse(rgb_coarse[..., :3], target_ray_values[..., :3])
                        curr_loss, curr_fine_loss = 0.0, 0.0
                        if rgb_fine is not None:
                            curr_fine_loss = img2mse(rgb_fine[..., :3], target_ray_values[..., :3])
                            curr_loss = curr_fine_loss
                        else:
                            curr_loss = coarse_loss
                        loss += curr_loss + curr_fine_loss
                    
                loss /= len(i_val)
                psnr = mse2psnr(loss.item())
                writer.add_scalar("validation/loss", loss.item(), i)
                writer.add_scalar("validation/coarse_loss", coarse_loss.item(), i)
                writer.add_scalar("validation/psnr", psnr, i)
                writer.add_image(
                    "validation/rgb_coarse", cast_to_image(rgb_coarse[..., :3]), i
                )
                if rgb_fine is not None:
                    writer.add_image(
                        "validation/rgb_fine", cast_to_image(rgb_fine[..., :3]), i
                    )
                    # writer.add_scalar("validation/fine_loss", fine_loss.item(), i)

                writer.add_image(
                    "validation/img_target",
                    cast_to_image(target_ray_values[..., :3]),
                    i,
                )
                if train_background or fixed_background:
                    writer.add_image(
                        "validation/background", cast_to_image(background[..., :3]), i
                    )
                    # writer.add_image(
                    #     "validation/weights", (weights.detach().cpu().numpy()), i, dataformats='HW'
                    # )
                tqdm.write(
                    "Validation loss: "
                    + str(loss.item())
                    + " Validation PSNR: "
                    + str(psnr)
                    + " Time: "
                    + str(time.time() - start)
                )

        #gpu_profile(frame=sys._getframe(), event='line', arg=None)

        if i % cfg.experiment.save_every == 0 or i == train_iters - 1:
            checkpoint_dict = {
                "iter": i,
                "model_coarse_state_dict": model_coarse.state_dict(),
                "model_fine_state_dict": None if not model_fine else model_fine.state_dict(),
                # "removecolor_state_dict": removecolor.state_dict(),
                # "model_state_dict": patchgan.state_dict(),
                # "optimizer_state_dict": optimizer.state_dict(),
                # "optimizer_d_state_dict": optimizer_d.state_dict(),
                "loss": loss_total,
                "psnr": psnr,
                "background": None
                if not (train_background or fixed_background)
                else background.data,
                "latent_codes": None if not train_latent_codes else latent_codes.data
            }
            if density:
                checkpoint_dict['optimizer_state_dict'] = optimizer.state_dict()
            else:
                checkpoint_dict['optimizer_g_state_dict'] = optimizer_g.state_dict()
                if use_patchgan_loss:
                    checkpoint_dict['optimizer_d_state_dict'] = optimizer_d.state_dict()
                checkpoint_dict['removecolor_state_dict'] = removecolor.state_dict()
                checkpoint_dict['model_state_dict'] = patchgan.state_dict()
            torch.save(
                checkpoint_dict,
                os.path.join(logdir, "checkpoint" + str(i).zfill(5) + ".ckpt"),
            )
            keep_last_epochs = 2
            last_epoch = i - keep_last_epochs * cfg.experiment.save_every
            if last_epoch > 0 and os.path.exists(os.path.join(logdir,'checkpoint' + str(last_epoch).zfill(5) + '.ckpt')):
                os.remove(os.path.join(logdir,'checkpoint'+str(last_epoch).zfill(5)+'.ckpt'))
            tqdm.write("================== Saved Checkpoint =================")

    print("Done!")

def saveimg(save_file, file):
    #debug_dir = './debug2/'
    #savename = os.path.join(debug_dir, filename+'.jpg')
    vtils.save_image(file,save_file)

def save_pairs(imgs,img_filename):
    row1 = torch.cat(imgs, 3)
    torchvision.utils.save_image(row1, img_filename, nrow=1)

def cast_to_image(tensor):
    # Input tensor is (H, W, 3). Convert to (3, H, W).
    tensor = tensor.permute(2, 0, 1)
    tensor = tensor.clamp(0.0,1.0)
    # Conver to PIL Image and then np.array (output shape: (H, W, 3))
    img = np.array(torchvision.transforms.ToPILImage()(tensor.detach().cpu()))
    # Map back to shape (3, H, W), as tensorboard needs channels first.
    img = np.moveaxis(img, [-1], [0])
    return img

def cast_to_image2(tensor):
    # Input tensor is (H, W, 3). Convert to (3, H, W).
    tensor = tensor.permute(2, 0, 1)
    tensor = tensor.clamp(0.0,1.0)
    # Convert to PIL Image and then np.array (output shape: (H, W, 3))
    img = None#np.array(torchvision.transforms.ToPILImage()(tensor.detach().cpu()))
    return tensor, img

def cast_to_depth(depth_img):
    #epth_img = depth_img.unsqueeze(-1)
    depth_img = torch.cat((depth_img,depth_img,depth_img),dim=-1)
    depth_t, depth_img = cast_to_image2(depth_img[..., :3])
    return depth_t, depth_img

def handle_pdb(sig, frame):
    import pdb
    pdb.Pdb().set_trace(frame)

def normalize_batch(batch):
    # Normalize batch using ImageNet mean and std
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1)
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1)
    return (batch - mean) / std

def load_image(path):
    img = Image.open(path)
    trans = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    imgtensor = trans(img)
    return normalize_batch(imgtensor)



def to_var(x, requires_grad=False):
    if isinstance(x, list):
        return x
    if torch.cuda.is_available():
        x = x.cuda()
    if not requires_grad:
        return Variable(x, requires_grad=requires_grad)
    else:
        return Variable(x)

if __name__ == "__main__":
    import signal

    print("before signal registration")
    signal.signal(signal.SIGUSR1, handle_pdb)
    print("after registration")
    #sys.settrace(gpu_profile)

    main()
