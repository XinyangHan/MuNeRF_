from pickle import NONE
import sys
from numpy.core.numeric import False_
import torch
import os
import copy
import torch.nn as nn
from torch.autograd import Variable
from .nerf_helpers import get_minibatches, ndc_rays
from .nerf_helpers import sample_pdf_2 as sample_pdf
from .nerf_helpers import dump_rays
from .nerf_helpers import cumprod_exclusive
import torchvision
from torchvision import utils as vtils
sys.path.insert(1, './prnet')
sys.path.insert(1, './prnet/utils')
from prnet.api import PRN
import cv2
import numpy as np
from prnet.utils.cv_plot import plot_vertices
import pdb

def saveimg(filename, file):
    debug_dir = './debug2/'
    savename = os.path.join(debug_dir, filename+'.jpg')
    vtils.save_image(file,savename)

def cal_hist(image):
    """
        cal cumulative hist for channel list
    """
    hists = []
    for i in range(0, 3):
        channel = image[i]
        # channel = image[i, :, :]
        channel = torch.from_numpy(channel)
        # hist, _ = np.histogram(channel, bins=256, range=(0,255))
        hist = torch.histc(channel, bins=256, min=0, max=256)
        hist = hist.numpy()
        # refHist=hist.view(256,1)
        sum = hist.sum()
        pdf = [v / sum for v in hist]
        for i in range(1, 256):
            pdf[i] = pdf[i - 1] + pdf[i]
        hists.append(pdf)
    return hists


def cal_trans(ref, adj):
    """
        calculate transfer function
        algorithm refering to wiki item: Histogram matching
    """
    table = list(range(0, 256))
    for i in list(range(1, 256)):
        for j in list(range(1, 256)):
            if ref[i] >= adj[j - 1] and ref[i] <= adj[j]:
                table[i] = j
                break
    table[255] = 255
    return table


def histogram_matching(dstImg, refImg, index):
    """
        perform histogram matching
        dstImg is transformed to have the same the histogram with refImg's
        index[0], index[1]: the index of pixels that need to be transformed in dstImg
        index[2], index[3]: the index of pixels that to compute histogram in refImg
    """
    index = [x.cpu().numpy() for x in index]
    device = refImg.device

    dstImg = dstImg.detach().cpu().numpy()
    refImg = refImg.detach().cpu().numpy()
    dst_align = [dstImg[i, index[0], index[1]] for i in range(0, 3)]
    ref_align = [refImg[i, index[2], index[3]] for i in range(0, 3)]
    hist_ref = cal_hist(ref_align)
    hist_dst = cal_hist(dst_align)
    tables = [cal_trans(hist_dst[i], hist_ref[i]) for i in range(0, 3)]

    mid = copy.deepcopy(dst_align)


    for i in range(0, 3):
        for k in range(0, len(index[0])):
            dst_align[i][k] = tables[i][int(mid[i][k])]

    for i in range(0, 3):
        dstImg[i, index[0], index[1]] = dst_align[i]

    dstImg = torch.FloatTensor(dstImg).cuda(device)
    return dstImg


def run_network(network_fn, pts, ray_batch, chunksize, embed_fn, embeddirs_fn, expressions = None, latent_code = None):
    pts_flat = pts.reshape((-1, pts.shape[-1]))
    embedded = embed_fn(pts_flat)
    if embeddirs_fn is not None:
        viewdirs = ray_batch[..., None, -3:]
        input_dirs = viewdirs.expand(pts.shape)
        input_dirs_flat = input_dirs.reshape((-1, input_dirs.shape[-1]))
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = torch.cat((embedded, embedded_dirs), dim=-1)

    batches = get_minibatches(embedded, chunksize=chunksize)
    if expressions is None:
        preds = [network_fn(batch) for batch in batches]
    elif latent_code is not None:
        preds = [network_fn(batch, expressions, latent_code) for batch in batches]
    else:
        preds = [network_fn(batch, expressions) for batch in batches]
    radiance_field = torch.cat(preds, dim=0)
    radiance_field = radiance_field.reshape(
        list(pts.shape[:-1]) + [radiance_field.shape[-1]]
    )
    del embedded, input_dirs_flat
    return radiance_field


def predict_and_render_radiance(
    ray_batch,
    model_coarse,
    model_fine,
    options,
    mode="train",
    encode_position_fn=None,
    encode_direction_fn=None,
    expressions = None,
    #lights = None,
    #details = None,
    background_prior = None,
    latent_code = None,
    ray_dirs_fake = None,
    flag = 0
):
    if flag==0:
        from .volume_rendering_utils import volume_render_radiance_field
    else:
        from .volume_rendering_utils2 import volume_render_radiance_field
    # TESTED
    num_rays = ray_batch.shape[0]
    ro, rd = ray_batch[..., :3], ray_batch[..., 3:6].clone() # TODO remove clone ablation rays
    bounds = ray_batch[..., 6:8].view((-1, 1, 2))
    near, far = bounds[..., 0], bounds[..., 1]
    # TODO: Use actual values for "near" and "far" (instead of 0. and 1.)
    # when not enabling "ndc".
    #print('latent codes!!!!predict_and_render_radiance', latent_code.shape)
    t_vals = torch.linspace(
        0.0,
        1.0,
        getattr(options.nerf, mode).num_coarse,
        dtype=ro.dtype,
        device=ro.device,
    )
    if not getattr(options.nerf, mode).lindisp:
        z_vals = near * (1.0 - t_vals) + far * t_vals
    else:
        z_vals = 1.0 / (1.0 / near * (1.0 - t_vals) + 1.0 / far * t_vals)
    z_vals = z_vals.expand([num_rays, getattr(options.nerf, mode).num_coarse])

    if getattr(options.nerf, mode).perturb:
        # Get intervals between samples.
        mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat((mids, z_vals[..., -1:]), dim=-1)
        lower = torch.cat((z_vals[..., :1], mids), dim=-1)
        # Stratified samples in those intervals.
        t_rand = torch.rand(z_vals.shape, dtype=ro.dtype, device=ro.device)
        z_vals = lower + (upper - lower) * t_rand
    # pts -> (num_rays, N_samples, 3)
    pts = ro[..., None, :] + rd[..., None, :] * z_vals[..., :, None]
    # Uncomment to dump a ply file visualizing camera rays and sampling points
    #dump_rays(ro.detach().cpu().numpy(), pts.detach().cpu().numpy())
    #ray_batch[...,3:6] = ray_dirs_fake[0][...,3:6] # TODO remove this this is for ablation of ray dir
    radiance_field = run_network(
        model_coarse,
        pts,
        ray_batch,
        getattr(options.nerf, mode).chunksize,
        encode_position_fn,
        encode_direction_fn,
        expressions,
        #lights,
        #details,
        latent_code
    )
    rgb_coarse, disp_coarse, depth_coarse = None, None, None
    if options.experiment.no_coarse_color:
        noise = 0.0
        radiance_field_noise_std = getattr(options.nerf, mode).radiance_field_noise_std
        output_alpha = radiance_field[..., 0]
        one_e_10 = torch.tensor(
            [1e10], dtype=rd.dtype, device=rd.device
        )
        dists = torch.cat(
            (
                z_vals[..., 1:] - z_vals[..., :-1],
                one_e_10.expand(z_vals[..., :1].shape),
            ),
            dim=-1,
        )
        dists = dists * rd[..., None, :].norm(p=2, dim=-1)
        if radiance_field_noise_std > 0.0:
            noise = (
                torch.randn(
                    output_alpha.shape,
                    dtype=radiance_field.dtype,
                    device=radiance_field.device,
                )
                * radiance_field_noise_std
            )
            # noise = noise.to(radiance_field)
        sigma_a = torch.nn.functional.relu(output_alpha + noise)
        
        sigma_a[:,-1] = sigma_a[:,-1] + 1e-6 # todo commented this for FCB demo !!!!!!
        alpha = 1.0 - torch.exp(-sigma_a * dists)
        weights = alpha * cumprod_exclusive(1.0 - alpha + 1e-10)

        depth_map = weights * z_vals
        depth_coarse = depth_map.sum(dim=-1)
    else:
        # make last RGB values of each ray, the background
        if background_prior is not None:
            print('back groud is not none')
            radiance_field[:,-1,:3] = background_prior

        # pdb.set_trace()
        
        (
            rgb_coarse,
            disp_coarse,
            acc_coarse,
            weights,
            depth_coarse,
        ) = volume_render_radiance_field(
            radiance_field,
            z_vals,
            rd,
            radiance_field_noise_std=getattr(options.nerf, mode).radiance_field_noise_std,
            white_background=getattr(options.nerf, mode).white_background,
            background_prior=background_prior
        )

    # pdb.set_trace()
    rgb_fine, disp_fine, acc_fine, depth_fine = None, None, None, None
    if getattr(options.nerf, mode).num_fine > 0 and model_fine is not None:
        # rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map

        z_vals_mid = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        z_samples = sample_pdf(
            z_vals_mid,
            weights[..., 1:-1],
            getattr(options.nerf, mode).num_fine,
            det=(getattr(options.nerf, mode).perturb == 0.0),
        )
        z_samples = z_samples.detach()

        z_vals, _ = torch.sort(torch.cat((z_vals, z_samples), dim=-1), dim=-1)
        # pts -> (N_rays, N_samples + N_importance, 3)
        pts = ro[..., None, :] + rd[..., None, :] * z_vals[..., :, None]

        radiance_field = run_network(
            model_fine,
            pts,
            ray_batch,
            getattr(options.nerf, mode).chunksize,
            encode_position_fn,
            encode_direction_fn,
            expressions,
            #lights,
            #details,
            latent_code
        )
        # make last RGB values of each ray, the background
        if background_prior is not None:
            radiance_field[:, -1, :3] = background_prior

        # Uncomment to dump a ply file visualizing camera rays and sampling points
        #dump_rays(ro.detach().cpu().numpy(), pts.detach().cpu().numpy(), radiance_field)

        #dump_rays(ro.detach().cpu().numpy(), pts.detach().cpu().numpy(), torch.softmax(radiance_field[:,:,-1],1).detach().cpu().numpy())

        #rgb_fine, disp_fine, acc_fine, _, depth_fine = volume_render_radiance_field(
        rgb_fine, disp_fine, acc_fine, weights, depth_fine = volume_render_radiance_field( # added use of weights
            radiance_field,
            z_vals,
            rd,
            radiance_field_noise_std=getattr(options.nerf, mode).radiance_field_noise_std,
            white_background=getattr(options.nerf, mode).white_background,
            background_prior=background_prior
        )
        # 找到问题了，问题在这：
        # 这里因为设置了no_coarse_color，所以没有rbg_coarse
        # 而且在model_fine中，rgb_fine最终的数值是和background一致的
        # 所以要么是model_coarse没有出现的问题，要么是model_fine渲染的问题
    # pdb.set_trace()
    #return rgb_coarse, disp_coarse, acc_coarse, rgb_fine, disp_fine, acc_fine, depth_fine #added depth fine
    return rgb_coarse, disp_coarse, depth_coarse, rgb_fine, depth_fine, acc_fine, weights[:,-1] #changed last return val to fine_weights


def run_one_iter_of_nerf(
    height,
    width,
    focal_length,
    model_coarse,
    model_fine,
    ray_origins,
    ray_directions,
    options,
    mode="train",
    encode_position_fn=None,
    encode_direction_fn=None,
    expressions = None,
    #lights = None,
    #details = None,
    background_prior=None,
    latent_code = None,
    ray_directions_ablation = None,
    flag = 0
):
    #print('in run one iter: ',background_prior)
    viewdirs = None
    if options.nerf.use_viewdirs:
        # Provide ray directions as input
        viewdirs = ray_directions
        viewdirs = viewdirs / viewdirs.norm(p=2, dim=-1).unsqueeze(-1)
        viewdirs = viewdirs.view((-1, 3))
    # Cache shapes now, for later restoration.
    if flag==0:
        restore_shapes = [
            ray_directions.shape, 
            ray_directions.shape[:-1],
            ray_directions.shape[:-1],
        ]
    else:
        restore_shapes = [
            torch.Size(list(ray_directions.shape[:2])+[64]),
            ray_directions.shape[:-1],
            ray_directions.shape[:-1],
        ]
    if model_fine:
        restore_shapes = restore_shapes + restore_shapes
        restore_shapes = restore_shapes + [ray_directions.shape[:-1]] # to return fine depth map
    if options.dataset.no_ndc is False:
        #print("calling ndc")
        ro, rd = ndc_rays(height, width, focal_length, 1.0, ray_origins, ray_directions)
        ro = ro.view((-1, 3))
        rd = rd.view((-1, 3))
    else:
        #print("calling ndc")
        #"caling normal rays (not NDC)"
        ro = ray_origins.view((-1, 3))
        rd = ray_directions.view((-1, 3))
        #rd_ablations = ray_directions_ablation.view((-1, 3))
    near = options.dataset.near * torch.ones_like(rd[..., :1])
    far = options.dataset.far * torch.ones_like(rd[..., :1])
    rays = torch.cat((ro, rd, near, far), dim=-1)
    #rays_ablation = torch.cat((ro, rd_ablations, near, far), dim=-1)
    # if options.nerf.use_viewdirs: # TODO uncomment
    #     rays = torch.cat((rays, viewdirs), dim=-1)
    #
    viewdirs = None  # TODO remove this paragraph
    '''if options.nerf.use_viewdirs:
        # Provide ray directions as input
        viewdirs = ray_directions_ablation
        viewdirs = viewdirs / viewdirs.norm(p=2, dim=-1).unsqueeze(-1)
        viewdirs = viewdirs.view((-1, 3))'''


    #batches_ablation = get_minibatches(rays_ablation, chunksize=getattr(options.nerf, mode).chunksize)
    batches_ablation = None
    batches = get_minibatches(rays, chunksize=getattr(options.nerf, mode).chunksize)
    assert(batches[0].shape == batches[0].shape)
    background_prior = get_minibatches(background_prior, chunksize=getattr(options.nerf, mode).chunksize) if\
        background_prior is not None else background_prior
    #print("predicting")
    #for i, batch in enumerate(batches):
    #    print('i:', i)
    #    print('batch: ', batch)
    #print('latent codes!!!!', latent_code.shape)
    # pdb.set_trace()

    # if background_prior is not None:
    #     pdb.set_trace()

    pred = [
        predict_and_render_radiance(
            batch,
            model_coarse,
            model_fine,
            options,
            mode,
            encode_position_fn=encode_position_fn,
            encode_direction_fn=encode_direction_fn,
            expressions = expressions,
            #lights = lights,
            #details = details,
            background_prior = background_prior[0] if background_prior is not None else background_prior,
            latent_code = latent_code,
            ray_dirs_fake = batches_ablation,
            flag = flag
        )
        for i,batch in enumerate(batches)
    ]
    #print("predicted")

    synthesized_images = list(zip(*pred))
    synthesized_images = [
        torch.cat(image, dim=0) if image[0] is not None else (None)
        for image in synthesized_images
    ]

    if mode == "validation":
        synthesized_images = [
            image.view(shape) if image is not None else None
            for (image, shape) in zip(synthesized_images, restore_shapes)
        ]

        # Returns rgb_coarse, disp_coarse, acc_coarse, rgb_fine, disp_fine, acc_fine
        # (assuming both the coarse and fine networks are used).
        if model_fine:
            return tuple(synthesized_images)
        else:
            # If the fine network is not used, rgb_fine, disp_fine, acc_fine are
            # set to None.
            return tuple(synthesized_images + [None, None, None, None])

    return tuple(synthesized_images)

def cal_histloss_multi(criterionHis, vggnet, prn, fake_makeup,makeup,makeup_gt,mask_nonmakeup,mask_makeup,nonmakeup,save_dir, size_img,if_dense):
    skin_local_parts = ['eye', 'eye_', 'mouth', 'nose', 'cheek', 'cheek_', 'eyebrow', 'eyebrow_', 'uppernose', 'forehead', 'sidemouth', 'sidemouth_']
    #skin_local_parts_weight = [1.0, 1.0,1.0,1.0,1.0,1.0,2.0,2.0,1.0,1.0,1.0,1.0]
    #lambda_his_lip, lambda_his_skin, lambda_his_nose, lambda_his_eyebrow, lambda_his_eye, lambda_his_bg, lambda_his_skin_parts, lambda_pr_points = 1.0, 1.0, 1.0, 2.0, 1.0, 4.0, 10.0, 1.0
    skin_local_parts_weight = [1.0, 1.0,1.0,1.0,1.0,1.0,2.0,2.0,1.0,1.0,1.0,1.0]
    lambda_his_lip, lambda_his_skin, lambda_his_nose, lambda_his_eyebrow, lambda_his_eye, lambda_his_bg, lambda_his_skin_parts, lambda_pr_points = 1.0, 0.1, 1.0, 2.0, 1.0, 4.0, 0.1, 1.0
    makeup_lip = makeup[0]
    makeup_eyes = makeup[1]
    makeup_skin = makeup[2]
    style = None
    his_loss = 0.0
    save_imgs = []
    save_imgs_skinpatches = []
    #print('process lip!!!!, lips use his_matching')
    imgs1, g_A_lip_loss_his = criterionHis(vggnet, fake_makeup, makeup_lip, style,
                                            mask_nonmakeup["mask_A_lip"],
                                            mask_makeup['mask_B_lip'],
                                            mask_nonmakeup["index_A_lip"],
                                            nonmakeup,part='lip_his')
    save_imgs.extend(imgs1)
    his_loss = his_loss + g_A_lip_loss_his*lambda_his_lip
    #print('process nose!!!!, nose use his_matching')
    imgs2, g_A_nose_loss_his = criterionHis(vggnet, fake_makeup, makeup_skin, style,
                                            mask_nonmakeup["mask_A_nose"],
                                            mask_makeup['mask_B_nose'],
                                            mask_nonmakeup["index_A_nose"],
                                            nonmakeup,part='nose_his')
    save_imgs.extend(imgs2)
    
    if not if_dense:
        his_loss = his_loss + g_A_nose_loss_his*lambda_his_nose 
    else:
        his_loss = his_loss
        
    #print('process whole skin!!!!,skin use warp')
    imgs3, g_A_skin_loss_his = criterionHis(vggnet, fake_makeup, makeup_skin, style,
                                            mask_nonmakeup["mask_A_skin"],
                                            mask_makeup['mask_B_skin'],
                                            mask_nonmakeup["index_A_skin"],
                                            nonmakeup,part='skin_warp')
    save_imgs.extend(imgs3)
    his_loss = his_loss + g_A_skin_loss_his*lambda_his_skin

    #print('process eyebrow!!!! eyebrow use his_matching')
    imgs4, g_A_eyebrow_loss_his = criterionHis(vggnet, fake_makeup, makeup_eyes, style,
                                            mask_nonmakeup["mask_A_eyebrow"],
                                            mask_makeup['mask_B_eyebrow'],
                                            mask_nonmakeup["index_A_eyebrow"],
                                            nonmakeup,part='eyebrow_his')
    imgs5, g_A_eyebrow_loss_his_ = criterionHis(vggnet, fake_makeup, makeup_eyes, style,
                                            mask_nonmakeup["mask_A_eyebrow_"],
                                            mask_makeup['mask_B_eyebrow_'],
                                            mask_nonmakeup["index_A_eyebrow_"],
                                            nonmakeup,part='eyebrow__his')
    save_imgs.extend(imgs4)
    his_loss = his_loss + (g_A_eyebrow_loss_his + g_A_eyebrow_loss_his_)*lambda_his_eyebrow

    #print('process eyebrow!!!! eyebrow use warp')
    imgs6, g_A_eye_loss_his = criterionHis(vggnet, fake_makeup, makeup_eyes, style,
                                            mask_nonmakeup["mask_A_eye"],
                                            mask_makeup['mask_B_eye'],
                                            mask_nonmakeup["index_A_eye"],
                                            nonmakeup,part='eye_warp')
    imgs7, g_A_eye_loss_his_ = criterionHis(vggnet, fake_makeup, makeup_eyes, style,
                                            mask_nonmakeup["mask_A_eye_"],
                                            mask_makeup['mask_B_eye_'],
                                            mask_nonmakeup["index_A_eye_"],
                                            nonmakeup,part='eye__warp')
    save_imgs.extend(imgs6)
    save_imgs.extend(imgs7)
    his_loss = his_loss + (g_A_eye_loss_his + g_A_eye_loss_his_)*lambda_his_eye

    imgs_G, g_A_bg_loss_ori = criterionHis(vggnet, fake_makeup, makeup_lip, style,
                                            mask_nonmakeup["mask_A_bg"],
                                            mask_nonmakeup["mask_A_bg"],
                                            mask_nonmakeup["index_A_bg"],
                                            nonmakeup,part='bg')
    his_loss = his_loss + g_A_bg_loss_ori*lambda_his_bg
    save_imgs.extend(imgs_G)

    his_loss_skin = 0.0
    skin_mask = mask_makeup['skin_patches']

    for i in range(len(skin_local_parts)):
        part = skin_local_parts[i]
        mask_info = skin_mask[i]
        imgs_skin_patch , g_A_skinpatch_loss_his = criterionHis(vggnet, fake_makeup, makeup_skin, style,
                                            mask_info[0],
                                            mask_info[1],
                                            mask_info[2],
                                            nonmakeup,part=part)
        save_imgs_skinpatches.extend(imgs_skin_patch)
        his_loss_skin = his_loss_skin +  g_A_skinpatch_loss_his*skin_local_parts_weight[i]
    
    his_loss = his_loss + his_loss_skin*lambda_his_skin_parts
    
    if_points, if_texture = True, False
    pr_points_loss,_,_ = prnet(prn, save_dir, fake_makeup, makeup,makeup_gt_lip, if_points, if_texture)
    if if_dense:
        his_loss_all = his_loss + lambda_pr_points*pr_points_loss
    else: his_loss_all = his_loss
    save_imgs = tuple(save_imgs)
    save_imgs_skinpatches = tuple(save_imgs_skinpatches)
    try:
        save_pairs(save_imgs, os.path.join(save_dir, 'parts_new.jpg'))
    except:
        print('Save parts_new.jpg fail!')
    try:
        save_pairs(save_imgs_skinpatches, os.path.join(save_dir, 'skin_parts_new.jpg'))
    except:
        print('Save skin_parts_new.jpg fail!')
    
    return his_loss_all

def de_norm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

def de_norm2(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

def prnet(prn, save_dir, fake, real, gt, points, texture):
    # to numpy 
    device = fake.device
    real = de_norm(real)
    fake_t = fake.unsqueeze(0)
    fake = fake.permute(1,2,0)
    fake = fake.cpu().detach().numpy()
    real_t = real.unsqueeze(0)
    real = real.permute(1,2,0)
    real = real.cpu().detach().numpy()
    #pos_fake = prn.net_forward(fake)
    #vertices_fake = prn.ge t_vertices(pos_fake)
    pos_real = prn.net_forward(real)
    loss =  None
    if points:
        vertices_real = prn.get_vertices(pos_real)         
        #real_points = plot_vertices(real, vertices_real)
        #fake_points = plot_vertices(fake, vertices_real)
        #real_points = real_points.copy()
        #fake_points = fake_points.copy()
        #fake_points_t = torch.from_numpy(fake_points)
        #real_points_t = torch.from_numpy(real_points)
        #four_point = (fake_t, fake_points_t.permute(2,0,1).unsqueeze(0).to(device), real_t, real_points_t.permute(2,0,1).unsqueeze(0).to(device))
        #save_pairs(four_point, os.path.join(save_dir, 'point_pair.jpg'))
        
        color_fake = torch.from_numpy(prn.get_colors(fake, vertices_real))
        color_real = torch.from_numpy(prn.get_colors(real, vertices_real))
        loss = torch.nn.functional.mse_loss(color_fake, color_real)
        return loss, None, None
        
    if texture:
        gt = de_norm(gt)
        gt_t = gt.unsqueeze(0)
        gt = gt.permute(1,2,0)
        gt = gt.cpu().detach().numpy()
        pos_gt = prn.net_forward(gt)
        texture_real = cv2.remap(gt, pos_gt[:,:,:2].astype(np.float32), None, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,borderValue=(0))
        texture_fake = cv2.remap(fake, pos_real[:,:,:2].astype(np.float32), None, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,borderValue=(0))
        texture_real = texture_real.copy()
        texture_fake = texture_fake.copy()
        texture_real_t = torch.from_numpy(texture_real)
        texture_fake_t = torch.from_numpy(texture_fake)
        #four_texture = (fake_t, texture_fake_t.permute(2,0,1).unsqueeze(0).to(device),gt_t, texture_real_t.permute(2,0,1).unsqueeze(0).to(device))
        #save_pairs(four_texture, os.path.join(save_dir, 'texture_pair.jpg'))       
        return loss, texture_real_t, texture_fake_t

def prnet_multi(prn, save_dir, fake, real, gt, points, texture):
    # to numpy 
    device = fake.device
    real = de_norm(real)
    fake_t = fake.unsqueeze(0)
    fake = fake.permute(1,2,0)
    fake = fake.cpu().detach().numpy()
    real_t = real.unsqueeze(0)
    real = real.permute(1,2,0)
    real = real.cpu().detach().numpy()
    #pos_fake = prn.net_forward(fake)
    #vertices_fake = prn.ge t_vertices(pos_fake)
    pos_real = prn.net_forward(real)
    loss =  None
    if points:
        vertices_real = prn.get_vertices(pos_real)         
        #real_points = plot_vertices(real, vertices_real)
        #fake_points = plot_vertices(fake, vertices_real)
        #real_points = real_points.copy()
        #fake_points = fake_points.copy()
        #fake_points_t = torch.from_numpy(fake_points)
        #real_points_t = torch.from_numpy(real_points)
        #four_point = (fake_t, fake_points_t.permute(2,0,1).unsqueeze(0).to(device), real_t, real_points_t.permute(2,0,1).unsqueeze(0).to(device))
        #save_pairs(four_point, os.path.join(save_dir, 'point_pair.jpg'))
        
        color_fake = torch.from_numpy(prn.get_colors(fake, vertices_real))
        color_real = torch.from_numpy(prn.get_colors(real, vertices_real))
        loss = torch.nn.functional.mse_loss(color_fake, color_real)
        return loss, None, None
        
    if texture:
        gt_lip = de_norm(gt[0])
        gt_eyes = de_norm(gt[1])
        gt_skin = de_norm(gt[2])
        gt_t_lip = gt_lip.unsqueeze(0)
        gt_t_eyes = gt_eyes.unsqueeze(0)
        gt_t_skin = gt_skin.unsqueeze(0)
        gt_lip = gt_lip.permute(1,2,0).cpu().detach().numpy()
        gt_eyes = gt_eyes.permute(1,2,0).cpu().detach().numpy()
        gt_skin = gt_skin.permute(1,2,0).cpu().detach().numpy()
        
        #pos_gt = prn.net_forward(gt)
        texture_real_lip = cv2.remap(gt_lip, prn.net_forward(gt_lip)[:,:,:2].astype(np.float32), None, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,borderValue=(0))
        texture_real_eyes = cv2.remap(gt_eyes, prn.net_forward(gt_lip)[:,:,:2].astype(np.float32), None, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,borderValue=(0))
        texture_real_skin = cv2.remap(gt_skin, prn.net_forward(gt_lip)[:,:,:2].astype(np.float32), None, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,borderValue=(0))
        
        texture_fake = cv2.remap(fake, pos_real[:,:,:2].astype(np.float32), None, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,borderValue=(0))
        #texture_real = texture_real.copy()
        texture_fake = texture_fake.copy()
        texture_real_lip_t = torch.from_numpy(texture_real_lip.copy())
        texture_real_eyes_t = torch.from_numpy(texture_real_eyes.copy())
        texture_real_skin_t = torch.from_numpy(texture_real_skin.copy())
        texture_fake_t = torch.from_numpy(texture_fake)
        #four_texture = (fake_t, texture_fake_t.permute(2,0,1).unsqueeze(0).to(device),gt_t, texture_real_t.permute(2,0,1).unsqueeze(0).to(device))
        #save_pairs(four_texture, os.path.join(save_dir, 'texture_pair.jpg'))       
        return loss, [texture_real_lip_t,texture_real_eyes_t,texture_real_skin_t], texture_fake_t

def cal_histloss2(criterionHis, vggnet, prn, fake_makeup,makeup,makeup_gt,mask_nonmakeup,mask_makeup,nonmakeup,save_dir, size_img,if_dense):

    # Important!
    skin_local_parts = ['eye', 'eye_', 'mouth', 'nose', 'cheek', 'cheek_', 'eyebrow', 'eyebrow_', 'uppernose', 'forehead', 'sidemouth', 'sidemouth_']
    skin_local_parts_weight = [1.0, 1.0,1.0,1.0,1.0,1.0,2.0,2.0,1.0,1.0,1.0,1.0]
    #skin_local_parts_weight = [1.0, 1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
    # lambda_his_lip, lambda_his_skin, lambda_his_nose, lambda_his_eyebrow, lambda_his_eye, lambda_his_bg, lambda_his_skin_parts, lambda_pr_points = 1.0, 0.5, 1.0, 2.0, 1.0, 4.0, 0.00, 1.0 #HXY 1014 
    lambda_his_lip, lambda_his_skin, lambda_his_nose, lambda_his_eyebrow, lambda_his_eye, lambda_his_bg, lambda_his_skin_parts, lambda_pr_points = 1.0, 1.0, 1.0, 2.0, 1.0, 4.0, 1.0, 1.0
    #skin_local_parts_weight = [1.0, 1.0,1.0,1.0,1.0,1.0,2.0,2.0,1.0,1.0,1.0,1.0]
    #lambda_his_lip, lambda_his_skin, lambda_his_nose, lambda_his_eyebrow, lambda_his_eye, lambda_his_bg, lambda_his_skin_parts, lambda_pr_points = 1.0, 1.0, 1.0, 2.0, 1.0, 4.0, 0.1, 1.0
    style = None
    his_loss = 0.0
    save_imgs = []
    save_imgs_skinpatches = []
    #print('process lip!!!!, lips use his_matching')
    # pdb.set_trace()
    imgs1, g_A_lip_loss_his = criterionHis(vggnet, fake_makeup, makeup, style,
                                            mask_nonmakeup["mask_A_lip"],
                                            mask_makeup['mask_B_lip'],
                                            mask_nonmakeup["index_A_lip"],
                                            nonmakeup,part='lip_his')
    save_imgs.extend(imgs1)
    his_loss = his_loss + g_A_lip_loss_his*lambda_his_lip
    #print('process nose!!!!, nose use his_matching')
    imgs2, g_A_nose_loss_his = criterionHis(vggnet, fake_makeup, makeup, style,
                                            mask_nonmakeup["mask_A_nose"],
                                            mask_makeup['mask_B_nose'],
                                            mask_nonmakeup["index_A_nose"],
                                            nonmakeup,part='nose_his')
    save_imgs.extend(imgs2)
    
    if not if_dense:
        his_loss = his_loss + g_A_nose_loss_his*lambda_his_nose 
    else:
        his_loss = his_loss
        
    #print('process whole skin!!!!,skin use warp')
    imgs3, g_A_skin_loss_his = criterionHis(vggnet, fake_makeup, makeup, style,
                                            mask_nonmakeup["mask_A_skin"],
                                            mask_makeup['mask_B_skin'],
                                            mask_nonmakeup["index_A_skin"],
                                            nonmakeup,part='skin_warp')
    save_imgs.extend(imgs3)
    his_loss = his_loss + g_A_skin_loss_his*lambda_his_skin

    #print('process eyebrow!!!! eyebrow use his_matching')
    imgs4, g_A_eyebrow_loss_his = criterionHis(vggnet, fake_makeup, makeup, style,
                                            mask_nonmakeup["mask_A_eyebrow"],
                                            mask_makeup['mask_B_eyebrow'],
                                            mask_nonmakeup["index_A_eyebrow"],
                                            nonmakeup,part='eyebrow_his')
    imgs5, g_A_eyebrow_loss_his_ = criterionHis(vggnet, fake_makeup, makeup, style,
                                            mask_nonmakeup["mask_A_eyebrow_"],
                                            mask_makeup['mask_B_eyebrow_'],
                                            mask_nonmakeup["index_A_eyebrow_"],
                                            nonmakeup,part='eyebrow__his')
    save_imgs.extend(imgs4)
    his_loss = his_loss + (g_A_eyebrow_loss_his + g_A_eyebrow_loss_his_)*lambda_his_eyebrow

    #print('process eyebrow!!!! eyebrow use warp')
    imgs6, g_A_eye_loss_his = criterionHis(vggnet, fake_makeup, makeup, style,
                                            mask_nonmakeup["mask_A_eye"],
                                            mask_makeup['mask_B_eye'],
                                            mask_nonmakeup["index_A_eye"],
                                            nonmakeup,part='eye_warp')
    imgs7, g_A_eye_loss_his_ = criterionHis(vggnet, fake_makeup, makeup, style,
                                            mask_nonmakeup["mask_A_eye_"],
                                            mask_makeup['mask_B_eye_'],
                                            mask_nonmakeup["index_A_eye_"],
                                            nonmakeup,part='eye__warp')
    save_imgs.extend(imgs6)
    save_imgs.extend(imgs7)
    his_loss = his_loss + (g_A_eye_loss_his + g_A_eye_loss_his_)*lambda_his_eye

    imgs_G, g_A_bg_loss_ori = criterionHis(vggnet, fake_makeup, makeup, style,
                                            mask_nonmakeup["mask_A_bg"],
                                            mask_nonmakeup["mask_A_bg"],
                                            mask_nonmakeup["index_A_bg"],
                                            nonmakeup,part='bg')
    his_loss = his_loss + g_A_bg_loss_ori*lambda_his_bg
    save_imgs.extend(imgs_G)

    his_loss_skin = 0.0
    skin_mask = mask_makeup['skin_patches']

    for i in range(len(skin_local_parts)):
        part = skin_local_parts[i]
        mask_info = skin_mask[i]
        imgs_skin_patch , g_A_skinpatch_loss_his = criterionHis(vggnet, fake_makeup, makeup, style,
                                            mask_info[0],
                                            mask_info[1],
                                            mask_info[2],
                                            nonmakeup,part=part)
        save_imgs_skinpatches.extend(imgs_skin_patch)
        his_loss_skin = his_loss_skin +  g_A_skinpatch_loss_his*skin_local_parts_weight[i]
    
    his_loss = his_loss + his_loss_skin*lambda_his_skin_parts
    
    if_points, if_texture = True, False
    pr_points_loss,_,_ = prnet(prn, save_dir, fake_makeup, makeup,makeup_gt, if_points, if_texture)
    if if_dense:
        his_loss_all = his_loss + lambda_pr_points*pr_points_loss
    else: his_loss_all = his_loss
    save_imgs = tuple(save_imgs)
    save_imgs_skinpatches = tuple(save_imgs_skinpatches)
    try:
        save_pairs(save_imgs, os.path.join(save_dir, 'parts_new.jpg'))
    except:
        print('Save parts_new.jpg fail!')
    try:
        save_pairs(save_imgs_skinpatches, os.path.join(save_dir, 'skin_parts_new.jpg'))
    except:
        print('Save skin_parts_new.jpg fail!')
    
    return his_loss_all

def save_pairs(imgs,img_filename):
    row1 = torch.cat(imgs, 0)
    torchvision.utils.save_image(row1, img_filename, nrow=4)


import math
import numbers
import torch
from torch import nn
from torch.nn import functional as F

class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """
    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / (2 * std)) ** 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight, groups=self.groups, padding=5)
