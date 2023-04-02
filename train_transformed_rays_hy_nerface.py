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

    
    transblock_patch = None #, transblock_patch1, transblock_patch2 = None, None, None
    
    transblock_cross = None #, transblock_cross1, transblock_cross2 = None, None, None
    supervised_train_background = train_background and supervised_train_background
    prn = PRN(is_dlib=False)

    # load GT background
    print("loading GT background to condition on")
    from PIL import Image
    background = Image.open(os.path.join(cfg.dataset.basedir,'bg','bc.jpg'))
    background.thumbnail((H,W))
    background = torch.from_numpy(np.array(background).astype(np.float32)).to(device)
    background = background/255
    print("bg shape", background.shape)
    print("should be ", images[i_train][0].shape)
    assert background.shape == images[i_train][0].shape


    # Initialize optimizer.
    trainable_parameters = list(model_coarse.parameters())
    trainable_parameters = trainable_parameters + list(model_fine.parameters())

    latent_codes = torch.zeros(len(i_train),32, device=device)  #here
    #latent_codes = torch.zeros(2240,32, device=device)
    print("initialized latent codes with shape %d X %d" % (latent_codes.shape[0], latent_codes.shape[1]))
    trainable_parameters.append(latent_codes)
    latent_codes.requires_grad = True

        
   
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

        img_idx = np.random.choice(i_train)
        img_target = images[img_idx].to(device) # training target
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
        
        loss_total = loss_total + (background_loss if supervised_train_background is not None else 0.0)
        
        dis_back = ((i-start_iter)%cfg.experiment.g_step==0) and (i-start_iter>0) and i >cfg.loss.patchgan.patchgan_iter +start_iter
        
        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()
            
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
                            savefile = os.path.join(debug_dir, 'debug_val%d.jpg'%i)
                            
                        else:
                            save = os.path.join(debug_dir, 'val_feature','debug_fine_val.jpg')
                            savefile = os.path.join(debug_dir, str(img_idx)+'debug_val_makeup.jpg')
                        if rgb_coarse is None: rgb_coarse = rgb_fine
                        fine_t,_ = cast_to_image2(rgb_fine)
                        coarse_t,_ = cast_to_image2(rgb_coarse)
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
            checkpoint_dict['optimizer_state_dict'] = optimizer.state_dict()
            
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
