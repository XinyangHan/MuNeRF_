import argparse
import glob
import os
import time
import imageio
import sys
import model.Transformer as trans

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
from nerf.load_mask import load_mask_data, load_mask_data2
from torchvision import transforms
from torchvision import utils as vtils
from helpers import FaceCropper
from model.model import PatchGAN
from model.options import Options
from model.histogram_loss import HistogramLoss
from utils.train_util import facePartsCoordinatesAPI
from utils.api_util import FacePPAPI
from settings import DENSITY_TRAIN, COLOR_TRAIN
from torch.autograd import Variable
fc = FaceCropper(predictor_dir = './shape_predictor_68_face_landmarks.dat',)

from nerf import (CfgNode, get_embedding_function, get_ray_bundle, img2mse,
                   meshgrid_xy, models, landmark2d_loss, cal_histloss, cal_histloss2,
                  mse2psnr, run_one_iter_of_nerf, dump_rays, GaussianSmoothing)
local_parts = ['eye', 'eye_', 'mouth', 'nose', 'cheek', 'cheek_', 'eyebrow', 'eyebrow_', 'uppernose', 'forehead', 'sidemouth', 'sidemouth_']


def main():

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
    configargs = parser.parse_args()    
    cfg = None
    with open(configargs.config, "r") as f:
        cfg_dict = yaml.load(f, Loader=yaml.FullLoader)
        cfg = CfgNode(cfg_dict)
    
    # Device on which to run.
    if torch.cuda.is_available():
        device = "cuda" + ":" + str(cfg.experiment.device)
        #device_backup = "cuda" + ":" + str(cfg.experiment.device_backup)   
    else:
        device = "cpu"
    # transblock = trans.define_G(cfg.transformer.use_resnet, cfg.transformer..if_global, cfg.transformer..input_nc, cfg.transformer.output_nc, cfg.transformer.ngf, 
    #                             cfg.transformer.netG, cfg.transformer.norm, not cfg.transformer.no_dropout, cfg.transformer.init_type, cfg.transformer.init_gain, device)

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
            if density:
                images, poses, render_poses, hwf, i_split, expressions, _, bboxs = load_flame_data(
                    cfg.dataset.basedir,
                    half_res=cfg.dataset.half_res,
                    scale=cfg.dataset.scale,
                    testskip=cfg.dataset.testskip,
                )
            else:
                images, poses, render_poses, hwf, i_split, expressions, depths, bboxs, warped_imgs, styles, paths, masks, img_ids= load_flame_data_color(
                    cfg.dataset.basedir,
                    cfg.dataset.warp_dir,
                    cfg.dataset.style_dir,
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
    
    # vggnet = VGG19(requires_grad=False)
    # styleimg = loadstyle(cfg.dataset.styledir)
    # style_vggfeature = vggnet(styleimg)
    #print('process style image done!', style_vggfeature)
    
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
    ###################################
    train_background = False
    supervised_train_background = False
    blur_background = False

    train_latent_codes = True
    disable_expressions = False # True to disable expressions
    disable_latent_codes = False # True to disable latent codes
    disable_light = False # True to disable latent codes  #
    fixed_background = True # Do False to disable BG
    regularize_latent_codes = True # True to add latent code LOSS, false for most experiments
    use_patchgan_loss =  cfg.loss.patchgan.patchgan_loss
    use_remove_loss =  cfg.loss.remove.remove_loss
    use_landmark_loss = cfg.loss.landmark.landmark_loss
    use_histogram_loss = cfg.loss.histogram.histogram_loss
    use_old_histogram = cfg.loss.histogram.histogram_loss_old
    use_l2_loss = cfg.loss.l2.l2_loss
    use_depth_loss = cfg.loss.depth.depth_loss
    ###################################
    ###################################

    supervised_train_background = train_background and supervised_train_background
    # Avg background
    #images[i_train]
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
            #avg_img = torch.zeros(H,W,3)
            #avg_img = torch.rand(H,W,3)
            #avg_img = 0.5*(torch.rand(H,W,3) + torch.mean(images[i_train],axis=0))
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
        latent_codes = torch.zeros(len(i_train),32, device=device)
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

    if os.path.exists(configargs.load_checkpoint) and not density:
        # -- create models for training -- #
        model_coarse_density, model_coarse_color, model_fine_density, model_fine_color = create_part(cfg, device)
        model_coarse, model_fine, removecolor = create_module(cfg, device, model_coarse_density, model_coarse_color, model_fine_density, model_fine_color)

        patchgan = PatchGAN(cfg.models.patchgan.input_dim_a, cfg.models.patchgan.input_dim_b, 
        cfg.models.patchgan.dis_n_layer, cfg.experiment.device)

        checkpoint = torch.load(configargs.load_checkpoint, map_location=device)
        density_train, color_train = DENSITY_TRAIN, COLOR_TRAIN
        coarse_checkpoint_density, coarse_checkpoint_color, fine_checkpoint_density, fine_checkpoint_color, checkpoint_density = {}, {}, {}, {}, {}
        old_checkpoint_coarse = checkpoint["model_coarse_state_dict"]
        old_checkpoint_fine = checkpoint["model_fine_state_dict"]
        for item in density_train:
            coarse_checkpoint_density[item] = old_checkpoint_coarse[item] if not color else old_checkpoint_coarse['density.'+item]
            fine_checkpoint_density[item] = old_checkpoint_fine[item] if not color else old_checkpoint_fine['density.'+item]
            checkpoint_density['coarse'] = coarse_checkpoint_density 
            checkpoint_density['fine'] = fine_checkpoint_density
        #torch.save(checkpoint_density, os.path.join(logdir, "checkpoint_density_fixed" + ".ckpt"))
        for item in color_train:
            coarse_checkpoint_color[item] = old_checkpoint_coarse[item] if not color else old_checkpoint_coarse['color.'+item]      
            fine_checkpoint_color[item] = old_checkpoint_fine[item] if not color else old_checkpoint_fine['color.'+item]
        
        # -- load coarse and fine model -- #
        model_coarse_density.load_state_dict(coarse_checkpoint_density)
        if color: model_coarse.color.load_state_dict(coarse_checkpoint_color)
        if checkpoint["model_fine_state_dict"]:
            model_fine_density.load_state_dict(fine_checkpoint_density)
            if color: model_fine.color.load_state_dict(fine_checkpoint_color)
        if color:
            removecolor.load_state_dict(checkpoint["removecolor_state_dict"])
            patchgan.load_state_dict(checkpoint["model_state_dict"])
        if checkpoint["background"] is not None:
            print("loaded bg from checkpoint")
            background = torch.nn.Parameter(checkpoint['background'].to(device))
            if fixed_background:
                background.requires_grad = False
        
        trainable_density = list(model_coarse_density.parameters())
        trainable_parameters = list(model_coarse.color.parameters())
        if model_fine_color is not None:
            trainable_parameters = trainable_parameters + list(model_fine.color.parameters())
            trainable_density = trainable_density + list(model_fine_density.parameters())
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
            optimizer_d = getattr(torch.optim, cfg.optimizer.type)(
                [{'params':trainable_parameters_}],
                lr=cfg.optimizer.lr
            )
        optimizer_g = getattr(torch.optim, cfg.optimizer.type)(
            [{'params':trainable_parameters},
            {'params':trainable_density, 'lr': 0.001*cfg.optimizer.lr},
            {'params':background, 'lr':cfg.optimizer.lr}],
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
        checkpoint = torch.load(configargs.continue_train)
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

    for i in trange(start_iter, cfg.experiment.train_iters):
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
            #target_ray_values = target_ray_values[select_inds].to(device)
            # ray_bundle = torch.stack([ray_origins, ray_directions], dim=0).to(device)

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
            img_idx = np.random.choice(i_train)
            img_target = images[img_idx].to(device) # training target
            pose_target = poses[img_idx].to(device)
            if not disable_expressions:
                expression_target = expressions[img_idx].to(device) # vector
            else: # zero expr
                expression_target = torch.zeros(50, device=device)
            if not disable_latent_codes:
                latent_code = latent_codes[img_idx].to(device) if train_latent_codes else None
            else:
                latent_codes = torch.zeros(32, device=device)

            if not density:
                # parameters
                paths_warped = paths['warped']
                paths_style = paths['styles']
                n_local = cfg.loss.landmark.n_local
                resize_scale = cfg.dataset.scale * 2.
                a = cfg.loss.coarse.loss_weight
                b = cfg.loss.fine.loss_weight
                img_gt = images_gt[img_idx].to(device) # no makeup

                # B warped makeup image
                # img_target = images[img_idx].to(device)
                depth_target = depths[img_idx].to(device)
                mask_no_makeup = masks[0][img_idx].to(device)
                img_B_arr = np.array(torchvision.transforms.ToPILImage()((img_target.permute(2, 0, 1)).detach().cpu()))
                landmark_B_api = paths_warped[img_idx]
                landmark_gt = fc.facePointsStasm(img_B_arr)
                if landmark_B_api is not None:
                    img_target_rects = facePartsCoordinatesAPI(img_B_arr, landmark_B_api, n_local=n_local, scaling_factor=resize_scale)
                else:
                    img_target_rects = None # b
                
                # A dynamic makeup image
                style_target = styles[img_idx].to(device) # a
                mask_makeup =  masks[1][img_idx].to(device)
                img_A_arr = np.array(torchvision.transforms.ToPILImage()((style_target.permute(2, 0, 1)).detach().cpu()))
                landmark_A_api = paths_style[img_idx]
                if landmark_A_api is not None:
                    style_target_rects = facePartsCoordinatesAPI(img_A_arr, landmark_A_api, n_local=n_local, scaling_factor=resize_scale)  
                else:
                    style_target_rects = None# a
            
            # sample rays
            ray_origins, ray_directions = get_ray_bundle(H, W, focal, pose_target)
            ray_origins_all, ray_directions_all = ray_origins, ray_directions

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
            
            if not density:
                #design depth for coarse and fine
                target_p = depth_target[select_inds[:, 0], select_inds[:, 1], :]
                target_p = target_p[...,:1]
                depth_target_ = depth_target[...,:1]

            background_ray_values = background[select_inds[:, 0], select_inds[:, 1], :] if (train_background or fixed_background) else None
            then = time.time()
            
            # train nerf
            rgb_coarse, _, _, rgb_fine, _, _, weights = run_one_iter_of_nerf(
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
                background_prior=background_ray_values,
                latent_code = latent_code if not disable_latent_codes else torch.zeros(32,device=device)
            )
            target_ray_values = target_s
            
            loss_total = 0.0
            # rgb loss
            coarse_loss =torch.nn.functional.mse_loss(
                            rgb_coarse[..., :3], target_ray_values[..., :3]
                        )
            fine_loss = None
            if rgb_fine is not None:
                fine_loss = torch.nn.functional.mse_loss(
                                rgb_fine[..., :3], target_ray_values[..., :3]
                            )
            l2_loss = coarse_loss + (fine_loss if fine_loss is not None else 0.0)
            #loss_l2 = l2_loss*cfg.loss.l2.l2_weight1 if i < 350000 + cfg.loss.l2.l2_iter else l2_loss*cfg.loss.l2.l2_weight2
            if density or use_l2_loss:
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
            
            if not density and cfg.loss.addloss.loss and i % cfg.loss.addloss.iter == 0:
                coarse_img, _, depth_coarse, fine_img, depth_fine, _ ,weights= run_one_iter_of_nerf(
                    H,
                    W,
                    focal,
                    model_coarse,
                    model_fine,
                    ray_origins_all.to(device),
                    ray_directions_all.to(device),
                    cfg,
                    mode="validation",
                    encode_position_fn=encode_position_fn,
                    encode_direction_fn=encode_direction_fn,
                    expressions = expression_target,
                    background_prior = background.view(-1,3).to(device) if (train_background or fixed_background) else None,
                    latent_code = torch.zeros(32).to(device) if train_latent_codes or disable_latent_codes else None,
                )

                # save image results
                img_fine_t, img_fine = cast_to_image2(fine_img[..., :3])
                img_coarse_t, img_coarse = cast_to_image2(coarse_img[..., :3])
                img_gt_t, _ = cast_to_image2(img_gt[..., :3])
                img_target_t, _ = cast_to_image2(img_target[..., :3])
                style_target_t, _ = cast_to_image2(style_target[..., :3])
                
                # save depth images
                depth_fine_t, _ = cast_to_depth(depth_fine.unsqueeze(-1))
                depth_coarse_t, _ = cast_to_depth(depth_coarse.unsqueeze(-1))
                depth_gt_t, depth_gt = cast_to_image2(depth_target[..., :3])
                
                save_depth_debug = True
                save_res_debug = True
                
                if save_depth_debug:
                    savefile3 = os.path.join(debug_dir, 'depth_img.jpg')
                    imgs = (depth_coarse_t.unsqueeze(0), depth_fine_t.unsqueeze(0), depth_gt_t.unsqueeze(0))
                    try:
                        save_pairs(imgs, savefile3)
                    except:
                        print('Save depth_img.jpg fail!')

                if not use_remove_loss and save_res_debug:
                    savefile = os.path.join(debug_dir, 'debug_img.jpg')
                    imgs = (img_target_t.unsqueeze(0),img_coarse_t.unsqueeze(0),img_fine_t.unsqueeze(0),img_gt_t.unsqueeze(0))
                    save_pairs(imgs, savefile)

                if use_depth_loss:
                    if depth_fine is not None:
                        depth_loss = b * torch.nn.functional.mse_loss(depth_fine.unsqueeze(-1), depth_target_) +\
                                        a * torch.nn.functional.mse_loss(depth_coarse.unsqueeze(-1), depth_target_)
                    else:
                        depth_loss = a * torch.nn.functional.mse_loss(depth_coarse.unsqueeze(-1), depth_target_)
                    loss_depth = depth_loss * cfg.loss.depth.depth_weight2
                    loss_total =  loss_total + loss_depth
                
                back = (i-start_iter)%cfg.experiment.g_step==0
                if use_patchgan_loss and style_target_rects is not None and img_target_rects is not None:
                    size_img = 512 * cfg.dataset.scale
                    image_a_coarse = [img_coarse_t.unsqueeze(0), img_target_rects]      #GEN
                    image_b = [style_target.permute(2, 0, 1).unsqueeze(0), style_target_rects]                  #STYLE
                    image_c = [img_target.permute(2, 0, 1).unsqueeze(0), img_target_rects]                      #WARP
                    coarsestyle_loss_dict = patchgan.local_style_d(image_a_coarse, image_b, image_c, size_img)
                    coarse_loss = sum(coarsestyle_loss_dict.values())
                    fine_loss = None
                    if fine_img is not None:
                        image_a_fine = [img_fine_t.unsqueeze(0), img_target_rects]
                        finestyle_loss_dict = patchgan.local_style_d(image_a_fine, image_b, image_c, size_img)
                        fine_loss = sum(finestyle_loss_dict.values())
                    patchgan_loss = coarse_loss + (fine_loss if fine_loss is not None else 0.0)
                    loss_patchgan = patchgan_loss*cfg.loss.patchgan.patchgan_weight1 if i<cfg.loss.patchgan.patchgan_iter +350000 else patchgan_loss*cfg.loss.patchgan.patchgan_weight2  
                    loss_patchgan_g = loss_patchgan.clone().detach()
                    loss_total =  loss_total + loss_patchgan_g.to(device)
                    loss_d_total = loss_patchgan.to(device)
                    if back:
                        optimizer_d.zero_grad()
                        loss_d_total.backward()
                        optimizer_d.step()

                if use_remove_loss: #and i >= start_iter + cfg.loss.remove.start_iter
                    cycle_fine = removecolor(img_fine_t.unsqueeze(0))
                    cycle_coarse = removecolor(img_coarse_t.unsqueeze(0))
                    
                    _,cycle_fine_numpy = cast_to_image2(cycle_fine.squeeze())
                    _,cycle_coarse_numpy = cast_to_image2(cycle_coarse.squeeze())
                    ldmk_fine = fc.facePointsStasm(cycle_fine_numpy)
                    ldmk_coarse = fc.facePointsStasm(cycle_coarse_numpy)
                    if not ldmk_fine.any() or not ldmk_coarse.any():
                        geo_loss_rm = torch.tensor(0.)
                    else:
                        geo_loss_rm = landmark2d_loss(torch.from_numpy(landmark_gt).float(), torch.from_numpy(ldmk_fine).float(), torch.from_numpy(ldmk_coarse).float())
                    if save_res_debug:
                        savefile = os.path.join(debug_dir, 'debug_img.jpg')
                        imgs = (img_target_t.unsqueeze(0),img_coarse_t.unsqueeze(0),img_fine_t.unsqueeze(0),cycle_fine,img_gt_t.unsqueeze(0))
                        try:
                            save_pairs(imgs, savefile)
                        except:
                            print('Save debug_img.jpg fail!')
                        
                    remove_loss = b*torch.nn.functional.mse_loss(cycle_fine.squeeze(), img_gt_t.to(device)) + a*torch.nn.functional.mse_loss(cycle_coarse.squeeze(), img_gt_t.to(device))
                    loss_remove = remove_loss * cfg.loss.remove.remove_weight + geo_loss_rm * cfg.loss.remove.ldmk_weight
                    loss_total = loss_total + loss_remove.to(device)
                
                if use_landmark_loss and i >= start_iter + cfg.loss.landmark.start_iter:
                    landmark_coarse = fc.facePointsStasm(img_coarse) #api.faceLandmarkDetector(savefile)
                    landmark_fine = fc.facePointsStasm(img_fine) #api.faceLandmarkDetector(savefile2)
                    #print('landmarks: ', landmark_gt, landmark_fine, landmark_coarse)
                    if not landmark_fine.any() or not landmark_coarse.any() or not landmark_gt.any():
                        geo_loss = torch.tensor(0.)
                    else:
                        #print('use landmark now')
                        geo_loss = landmark2d_loss(torch.from_numpy(landmark_gt).float(), torch.from_numpy(landmark_fine).float(), torch.from_numpy(landmark_coarse).float())   
                    loss_geo = geo_loss * cfg.loss.landmark.landmark_weight
                    loss_total = loss_total + loss_geo.to(device)
                
                if use_histogram_loss:
                    print('use histo')
                    if use_old_histogram:
                        criterionHis = HistogramLoss().to(device)
                        fake_makeup1 = img_fine_t
                        fake_makeup2 = img_coarse_t
                        makeup = img_target.permute(2, 0, 1) # first order image style_target; warp image img_target
                        eye_rects = [img_target_rects[0], img_target_rects[1],img_target_rects[2]]
                        fake_makeup1, fake_makeup2, mask_no_makeup, mask_makeup, makeup,style, nonmakeup = load_mask_data(fake_makeup1,fake_makeup2,makeup,style_target_t, img_gt_t,mask_no_makeup, mask_makeup, eye_rects)
                        histoloss_fine = cal_histloss(criterionHis, fake_makeup1,makeup,style, mask_no_makeup,mask_makeup,nonmakeup,debug_dir)
                        histoloss_coarse = cal_histloss(criterionHis, fake_makeup2,makeup,style, mask_no_makeup,mask_makeup,nonmakeup,debug_dir)
                        ###
                        hist_loss = b*histoloss_fine + a*histoloss_coarse
                        loss_hist = hist_loss * cfg.loss.histogram.histogram_weight
                        loss_total = loss_total + loss_hist.to(device)
                        #print('total_loss5,histo',loss_total)
                    else:
                        print('use new one')
                        criterionHis = HistogramLoss().to(device)
                        fake_makeup1 = img_fine_t
                        fake_makeup2 = img_coarse_t
                        makeup = img_target.permute(2, 0, 1) # first order image style_target; warp image img_target
                        fake_makeup1, fake_makeup2, mask_no_makeup, mask_makeup, makeup, nonmakeup = load_mask_data2(fake_makeup1,fake_makeup2,makeup, img_gt_t,mask_no_makeup, mask_makeup, img_target_rects, resize_scale)
                        histoloss_fine = cal_histloss2(criterionHis, fake_makeup1,makeup, mask_no_makeup,mask_makeup,nonmakeup,debug_dir)
                        histoloss_coarse = cal_histloss2(criterionHis, fake_makeup2,makeup, mask_no_makeup,mask_makeup,nonmakeup,debug_dir)
                        ###
                        hist_loss = b*histoloss_fine + a*histoloss_coarse
                        loss_hist = hist_loss * cfg.loss.histogram.histogram_weight
                        loss_total = loss_total + loss_hist.to(device)

        if density:
            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()
        else:
            print('before opt ',loss_total)
            optimizer_g.zero_grad()
            loss_total.backward()
            optimizer_g.step()
        
        # Learning rate updates
        num_decay_steps = cfg.scheduler.lr_decay * 1000
        lr_new = cfg.optimizer.lr * (
            cfg.scheduler.lr_decay_factor ** (i / num_decay_steps)
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr_new

        if i % cfg.experiment.print_every == 0 or i == cfg.experiment.train_iters - 1:
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

        writer.add_scalar("train/coarse_loss", coarse_loss.item(), i)
        if rgb_fine is not None:
           writer.add_scalar("train/fine_loss", fine_loss.item(), i)
        #writer.add_scalar("train/psnr", psnr, i)

        # Validation
        if (
            i % cfg.experiment.validate_every == 0
            or i == cfg.experiment.train_iters - 1 and False
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
                        img_target = images[img_idx].to(device)
                        #tqdm.set_description('val im %d' % img_idx)
                        #tqdm.refresh()  # to show immediately the update

                        pose_target = poses[img_idx, :3, :4].to(device)
                        ray_origins, ray_directions = get_ray_bundle(
                            H, W, focal, pose_target
                        )
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
                            latent_code = torch.zeros(32).to(device) if train_latent_codes or disable_latent_codes else None,

                        )
                        #print("did one val")
                        debug_dir = configargs.debug_dir
                        if density:
                            savefile = os.path.join(debug_dir, 'debug_val.jpg')
                        else:
                            savefile = os.path.join(debug_dir, 'debug_val_makeup.jpg')
                        fine_t,_ = cast_to_image2(rgb_fine[..., :3])
                        coarse_t,_ = cast_to_image2(rgb_coarse[..., :3])
                        imgs = (fine_t.unsqueeze(0), coarse_t.unsqueeze(0))
                        save_pairs(imgs,savefile)
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
                    writer.add_scalar("validation/fine_loss", fine_loss.item(), i)

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

        if i % cfg.experiment.save_every == 0 or i == cfg.experiment.train_iters - 1:
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

def saveimg(filename, file):
    debug_dir = './debug/'
    savename = os.path.join(debug_dir, filename+'.jpg')
    vtils.save_image(file,savename)

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
    img = np.array(torchvision.transforms.ToPILImage()(tensor.detach().cpu()))
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

def loadstyle(path):
    styleimg = Image.open(path)
    trans = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    styletensor = trans(styleimg)
    return normalize_batch(styletensor)



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

# 541 
# depth loss
            # if depth_fine is not None:
            #     depth_loss = b * torch.nn.functional.mse_loss(
            #                         depth_fine, target_p
            #                     ) +\
            #                  a * torch.nn.functional.mse_loss(
            #                         depth_coarse, target_p
            #                     )
            # else:
            #     depth_loss = torch.nn.functional.mse_loss(
            #                         depth_coarse, target_p
            #                     )
            # loss_depth = depth_loss * cfg.loss.depth.depth_weight1
            # loss_total = loss_total + loss_depth