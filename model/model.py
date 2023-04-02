from model.networks import Dis_pair, Dis_single, get_scheduler
from model.networks import init_net
from torchvision import utils as vtils
import torch
import torch.nn as nn
import numpy as np
import utils
import os
import torchvision
import pdb
import numpy as np
import cv2

mode = ""

def save_pairs(imgs,img_filename):
    row1 = torch.cat(imgs, 0)
    torchvision.utils.save_image(row1, img_filename, nrow=2)


def get_pairs(contents, styles,rects,rects_style, size_img, targetsize, save_dir):
    contents_list = []
    styles_list = []

    for rect in rects[:6]:
        C = contents.size(1)
        H = rect[1]-rect[0]
        W = rect[3]-rect[2]
        device = contents.device
        content_patch = torch.empty((C,H,W)).to(device)
        x1_t, x2_t, y1_t, y2_t = rect
        if x2_t>size_img: x2_t = int(size_img)
        if y2_t>size_img: y2_t = int(size_img)
        if x1_t<0: x1_t=0
        if y1_t<0: y1_t=0

        content_patch = contents[:,:,x1_t:x2_t,y1_t:y2_t].clone()
        content_patch = torch.nn.functional.interpolate(content_patch, size=(targetsize, targetsize))
        contents_list.append(content_patch)
    save_imgc = tuple(contents_list) 
    try:
        save_pairs(save_imgc, os.path.join(save_dir, 'crop_content.jpg'))
    except:
        print('Save crop_content.jpg fail!')

    for rectm in rects_style[:6]:
        C = styles.size(1)
        H = rectm[1]-rectm[0]
        W = rectm[3]-rectm[2]
        device = styles.device
        style_patch = torch.empty((C,H,W)).to(device)
        x1_m, x2_m, y1_m, y2_m = rectm
        if x2_m>size_img: x2_m = int(size_img)
        if y2_m>size_img: y2_m = int(size_img)
        if x1_m<0: x1_m=0
        if y1_m<0: y1_m=0
        style_patch = styles[:,:,x1_m:x2_m,y1_m:y2_m].clone()
        style_patch = torch.nn.functional.interpolate(style_patch, size=(targetsize, targetsize))
        styles_list.append(style_patch)
    
    save_imgs = tuple(styles_list) 
    try:
        save_pairs(save_imgs, os.path.join(save_dir, 'crop_ref.jpg'))
    except:
        print('Save crop_ref.jpg fail!')

    return [contents_list, styles_list]

class PatchGAN(nn.Module):
    def __init__(self, input_dim_a=3, input_dim_b=3, dis_n_layer=5, device=0):
        super(PatchGAN, self).__init__()
        # options defined
        # input_dim_a = 3
        # input_dim_b = 3
        # dis_n_layer = 3
        self.local_style_dis = True
        n_local = 12
        self.n_local = n_local
        self.device = torch.device('cuda:{}'.format(device)) if device>=0 else torch.device('cpu')
        self.style_d_ls_weight = 2.0
        self.style_g_ls_weight = 2.0
        self.local_parts = ['eye', 'eye_','mouth', 'nose', 'cheek', 'cheek_', 'eyebrow', 'eyebrow_', 'uppernose', 'forehead', 'sidemouth', 'sidemouth_']
        self.local_parts_laplacian_weight = [4.0, 4.0, 2.0, 2.0,  4.0, 4.0, 3.0, 3.0, 2.0, 4.0, 2.0, 2.0]

        counter = 0
        for i in range(self.n_local):
            local_part = self.local_parts[i]
            counter += 1
        self.valid_n_local = counter
        
        if self.local_style_dis:
            for i in range(self.n_local):
                local_part = self.local_parts[i]
                if '_' in local_part:
                    continue
                setattr(self, 'dis'+local_part.capitalize(), init_net(Dis_pair(input_dim_a, input_dim_b, dis_n_layer), device, init_type='normal', gain=0.02))
        # Setup the loss function for training
        self.criterionL1 = nn.L1Loss()

    def backward_local_styleD(self, netD, rects_transfer, rects_after, rects_blend, name='', flip=False, size_img=256):
        C = self.input_B.size(1)
        H = rects_transfer[1]-rects_transfer[0]
        W = rects_transfer[3]-rects_transfer[2]

        transfer_crop = torch.empty((C,H,W)).to(self.device)
        after_crop = torch.empty((C,H,W)).to(self.device)
        blend_crop = torch.empty((C,H,W)).to(self.device)
        
        #print('rects_transfer, rects_after, rects_blend',rects_transfer, rects_after, rects_blend)
        x1_t, x2_t, y1_t, y2_t = rects_transfer
        x1_a, x2_a, y1_a, y2_a = rects_after
        x1_b, x2_b, y1_b, y2_b = rects_blend
        if x2_t>size_img or y2_t>size_img or x2_a>size_img or y2_a>size_img or x2_b>size_img or y2_b>size_img:
            return torch.tensor(0)
        if x1_t<0 or y1_t<0 or x1_a<0 or y1_a<0 or x1_b<0 or y1_b<0:
            return torch.tensor(0)

        if not flip:
            transfer_crop = self.input_A[:,:,x1_t:x2_t,y1_t:y2_t].clone()
            after_crop = self.input_B[:,:,x1_a:x2_a,y1_a:y2_a].clone()
            blend_crop = self.input_C[:,:,x1_b:x2_b,y1_b:y2_b].clone()
        else:
            transfer_crop = self.input_A[:,:,x1_t:x2_t,y1_t:y2_t].clone()
            after_crop = self.input_B[:,:,x1_a:x2_a,y1_a:y2_a].clone()
            blend_crop = self.input_C[:,:,x1_b:x2_b,y1_b:y2_b].clone()
        
        setattr(self, name+'_transfer', transfer_crop)
        setattr(self, name+'_after', after_crop)
        setattr(self, name+'_blend', blend_crop)

        pred_fake = netD.forward(after_crop.detach(), transfer_crop.detach())
        pred_real = netD.forward(after_crop.detach(), blend_crop.detach())
        loss_D = 0
        for it, (out_a, out_b) in enumerate(zip(pred_fake, pred_real)):
            out_fake = torch.sigmoid(out_a)
            out_real = torch.sigmoid(out_b)
            all1 = torch.ones((out_real.size(0))).to(self.device)
            all0 = torch.zeros((out_fake.size(0))).to(self.device)
            ad_true_loss = nn.functional.binary_cross_entropy(out_real, all1)
            ad_fake_loss = nn.functional.binary_cross_entropy(out_fake, all0)
            loss_D = loss_D + (ad_true_loss + ad_fake_loss)
        loss_D = loss_D * self.style_d_ls_weight / self.n_local
        return loss_D

    def backward_G_GAN_local_style(self, netD, rects_transfer, rects_after, flip=False, size_img=256):
        C = self.input_B.size(1)
        H = rects_transfer[1]-rects_transfer[0]
        W = rects_transfer[3]-rects_transfer[2]

        transfer_crop = torch.empty((C,H,W)).to(self.device)
        after_crop = torch.empty((C,H,W)).to(self.device)

        x1_t, x2_t, y1_t, y2_t = rects_transfer
        x1_a, x2_a, y1_a, y2_a = rects_after
        if x2_t>size_img or y2_t>size_img or x2_a>size_img or y2_a>size_img:
            return torch.tensor(0)
        if x1_t<0 or y1_t<0 or x1_a<0 or y1_a<0:
            return torch.tensor(0)

        if not flip:
            transfer_crop = self.input_A[:,:,x1_t:x2_t,y1_t:y2_t].clone()
            after_crop = self.input_B[:,:,x1_a:x2_a,y1_a:y2_a].clone()
        else:
            transfer_crop = self.input_A[:,:,x1_t:x2_t,y1_t:y2_t].clone()
            after_crop = self.input_B[:,:,x1_a:x2_a,y1_a:y2_a].clone()

        outs_fake = netD.forward(after_crop.detach(), transfer_crop)
        loss_G = 0
        for out_a in outs_fake:
            outputs_fake = torch.sigmoid(out_a)
            all_ones = torch.ones_like(outputs_fake).to(self.device)
            loss_G += nn.functional.binary_cross_entropy(outputs_fake, all_ones)
        loss_G = loss_G * self.style_g_ls_weight / self.n_local
        return loss_G

    def local_style_d(self, image_a, image_b, image_c, size_img=256):
        self.input_A = image_a[0].to(self.device).detach()
        self.input_B = image_b[0].to(self.device).detach()
        self.input_C = image_c[0].to(self.device).detach()
        self.rects_A = image_a[1]
        self.rects_B = image_b[1]
        self.rects_C = image_c[1]
        loss_style = {}
        for i in range(self.n_local):
            local_part = self.local_parts[i]
            tp = 0.0            
            if '_' not in local_part:
                loss_D_Style = self.backward_local_styleD(getattr(self, 'dis'+local_part.capitalize()), self.rects_A[i], self.rects_B[i], self.rects_C[i], name=local_part,size_img=size_img)
                tp = loss_D_Style
            else:
                local_part = local_part.split('_')[0]
                loss_D_Style_ = self.backward_local_styleD(getattr(self, 'dis'+local_part.capitalize()), self.rects_A[i], self.rects_B[i], self.rects_C[i], name=local_part+'2', flip=True,size_img=size_img)
                tp = loss_D_Style_
            loss_style[local_part] = tp                   
        return loss_style

    def local_style_g(self, image_a, image_b, size_img=256, ):
        self.input_A = image_a[0].to(self.device)
        self.input_B = image_b[0].to(self.device).detach()
        self.rects_A = image_a[1]
        self.rects_B = image_b[1]
        loss_style = {}
        n_local = len(self.rects_A)
        for i in range(n_local):
            local_part = self.local_parts[i]    
            if '_' not in local_part:
                loss_G_Style = self.backward_G_GAN_local_style(getattr(self, 'dis'+local_part.capitalize()), self.rects_A[i], self.rects_B[i], size_img=size_img)
            else:
                local_part = local_part.split('_')[0]
                loss_G_Style = self.backward_G_GAN_local_style(getattr(self, 'dis'+local_part.capitalize()), self.rects_A[i], self.rects_B[i], flip=True, size_img=size_img)
            loss_style[local_part] = loss_G_Style
        return loss_style


class Transfromer_PatchGAN(nn.Module):
    def __init__(self, vgg, transblock, input_dim_a=3, input_dim_b=3, dis_n_layer=5, device=0, n_local=12,if_uv=True):
        super(Transfromer_PatchGAN, self).__init__()
        # options defined
        self.local_style_dis = True
        # n_local = 6
        if not if_uv:
            self.n_local = n_local
            self.device = torch.device('cuda:{}'.format(device)) if device>=0 else torch.device('cpu')
            self.style_d_ls_weight = 2.0
            self.style_g_ls_weight = 2.0
            self.local_parts = ['eye', 'eye_','mouth', 'nose', 'cheek', 'cheek_', 'eyebrow', 'eyebrow_', 'uppernose', 'forehead', 'sidemouth', 'sidemouth_']
            self.local_parts_laplacian_weight = [1.0, 1.0, 1.0, 1.0,  1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        else:
            self.n_local = n_local
            self.device = torch.device('cuda:{}'.format(device)) if device>=0 else torch.device('cpu')
            self.style_d_ls_weight = 2.0
            self.style_g_ls_weight = 2.0
            self.local_parts = ['eye', 'eye_','mouth', 'nose', 'cheek', 'cheek_', 'eyebrow', 'eyebrow_', 'uppernose', 'forehead', 'sidemouth', 'sidemouth_']
            self.local_parts_eyes = ['eye', 'eye_','eyebrow', 'eyebrow_', ]
            self.local_parts_eyes = ['mouth', 'sidemouth', 'sidemouth_']
            self.local_parts_skin = ['nose', 'cheek', 'cheek_', 'uppernose', 'forehead']
            # !!!
            # 关于是否给uv看到皮肤等部分，在这里调整，置零即可。
            self.local_parts_laplacian_weight = [1.0, 1.0, 1.0, 1.0,  1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
            
        self.whole_trans= False
        self.if_vgg= False
        self.no_trans = True
        counter = 0
        for i in range(self.n_local):
            local_part = self.local_parts[i]
            counter += 1
        self.valid_n_local = counter
        
        if self.local_style_dis:
            for i in range(self.n_local):
                local_part = self.local_parts[i]
                if '_' in local_part:
                    continue
                setattr(self, 'dis'+local_part.capitalize(), init_net(Dis_single(input_dim_a, dis_n_layer), device, init_type='normal', gain=0.02))
        # Setup the loss function for training
        self.criterionL1 = nn.L1Loss()

        # vgg and transformer
        self.vgg = vgg.to(device)
        self.transblock = transblock.to(device)

    def backward_local_styleD(self, netD, rects_transfer, rects_after, name='', flip=False, size_img=256, if_content=True):
        C = self.input_B.size(1)
        H = rects_transfer[1]-rects_transfer[0]
        W = rects_transfer[3]-rects_transfer[2]
        size_img = int(size_img)

        transfer_crop = torch.empty((C,H,W)).to(self.device)
        after_crop = torch.empty((C,H,W)).to(self.device)
        
        x1_t, x2_t, y1_t, y2_t = rects_transfer
        x1_a, x2_a, y1_a, y2_a = rects_after

        if x2_t>size_img or y2_t>size_img or x2_a>size_img or y2_a>size_img:
            return torch.tensor(0)
        if x1_t<0 or y1_t<0 or x1_a<0 or y1_a<0:
            return torch.tensor(0)
        if not flip:
            transfer_crop = self.input_A[:,:,x1_t:x2_t,y1_t:y2_t].clone()
            after_crop = self.input_B[:,:,x1_a:x2_a,y1_a:y2_a].clone()
        else:
            transfer_crop = self.input_A[:,:,x1_t:x2_t,y1_t:y2_t].clone()
            after_crop = self.input_B[:,:,x1_a:x2_a,y1_a:y2_a].clone()

        # vgg style feature
        transfer_crop_resize = torch.nn.functional.interpolate(transfer_crop, size=(size_img, size_img))
        after_crop_resize = torch.nn.functional.interpolate(after_crop, size=(size_img, size_img))
        transfer_feat, after_feat = self.vgg(transfer_crop_resize), self.vgg(after_crop_resize)
        if self.whole_trans: # 625
            fake_trans_feat = self.transblock(transfer_crop)
            real_trans_feat = self.transblock(after_crop)
        if self.if_vgg:
            if if_content:
                fake_trans_feat = transfer_feat['conv2_2']
                real_trans_feat = after_feat['conv2_2']
            else:
                fake_trans_feat = transfer_feat['conv4_2']
                real_trans_feat = after_feat['conv4_2']
            
        if self.no_trans:
            fake_trans_feat = transfer_crop
            real_trans_feat = after_crop

        pred_fake = netD.forward(fake_trans_feat.detach())
        pred_real = netD.forward(real_trans_feat.detach())
        loss_D = 0
        for it, (out_a, out_b) in enumerate(zip(pred_fake, pred_real)):
            out_fake = torch.sigmoid(out_a)
            out_real = torch.sigmoid(out_b)
            all1 = torch.ones((out_real.size(0))).to(self.device)
            all0 = torch.zeros((out_fake.size(0))).to(self.device)
            ad_true_loss = nn.functional.binary_cross_entropy(out_real, all1)
            ad_fake_loss = nn.functional.binary_cross_entropy(out_fake, all0)
            loss_D = loss_D + (ad_true_loss + ad_fake_loss)
        loss_D = loss_D * self.style_d_ls_weight / self.n_local
        return loss_D

    def backward_G_GAN_local_style(self, netD, rects_transfer, rects_after, flip=False, size_img=256, if_content=True):
        C = self.input_B.size(1)
        H = rects_transfer[1]-rects_transfer[0]
        W = rects_transfer[3]-rects_transfer[2]
        size_img = int(size_img)

        transfer_crop = torch.empty((C,H,W)).to(self.device)
        after_crop = torch.empty((C,H,W)).to(self.device)

        x1_t, x2_t, y1_t, y2_t = rects_transfer
        x1_a, x2_a, y1_a, y2_a = rects_after

        if x2_t>size_img or y2_t>size_img or x2_a>size_img or y2_a>size_img:
            return torch.tensor(0)
        if x1_t<0 or y1_t<0 or x1_a<0 or y1_a<0:
            return torch.tensor(0)
        if not flip:
            transfer_crop = self.input_A[:,:,x1_t:x2_t,y1_t:y2_t].clone()
            after_crop = self.input_B[:,:,x1_a:x2_a,y1_a:y2_a].clone()
        else:
            transfer_crop = self.input_A[:,:,x1_t:x2_t,y1_t:y2_t].clone()
            after_crop = self.input_B[:,:,x1_a:x2_a,y1_a:y2_a].clone()
        
        # pdb.set_trace()

        # input_A_copy = self.input_A.cpu().numpy().squeeze()
        # debug_transfer_crop = transfer_crop.cpu().numpy().squeeze()
        # input_A_copy = np.transpose(input_A_copy, (1,2,0))
        
        # debug_transfer_crop = np.transpose(debug_transfer_crop, (1,2,0))
        # # pdb.set_trace()
        # debug_dir = "/data/hanxinyang/MuNeRF_latest/debug/debug_patch"
        # cv2.imwrite(os.path.join(debug_dir, "inputA" + ".jpg"), input_A_copy*255)
        # cv2.imwrite(os.path.join(debug_dir, mode + ".jpg"), debug_transfer_crop*255)
        # vgg style feature
        transfer_crop_resize = torch.nn.functional.interpolate(transfer_crop, size=(size_img, size_img))
        # after_crop_resize = torch.nn.functional.interpolate(after_crop, size=(size_img, size_img))
        transfer_feat = self.vgg(transfer_crop_resize)
        if self.whole_trans: # 625
            fake_trans_feat = self.transblock(transfer_crop)
        if self.if_vgg:
            if if_content:
                fake_trans_feat = transfer_feat['conv2_2']
            else:
                fake_trans_feat = transfer_feat['conv4_2']
        if self.no_trans:
            fake_trans_feat = transfer_crop
        
        outs_fake = netD.forward(fake_trans_feat)
        loss_G = 0
        for out_a in outs_fake:
            outputs_fake = torch.sigmoid(out_a)
            all_ones = torch.ones_like(outputs_fake).to(self.device)
            loss_G += nn.functional.binary_cross_entropy(outputs_fake, all_ones)
        loss_G = loss_G * self.style_g_ls_weight / self.n_local
        return loss_G

    def local_style_d(self, image_a, image_b, size_img=256, if_content=True):
        self.input_A = image_a[0].to(self.device).detach()
        self.input_B = image_b[0].to(self.device).detach()
        self.rects_A = image_a[1]
        self.rects_B = image_b[1]
        loss_style = {}
        for i in range(self.n_local):
            local_part = self.local_parts[i]
            tp = 0.0            
            if '_' not in local_part:
                loss_D_Style = self.backward_local_styleD(getattr(self, 'dis'+local_part.capitalize()), self.rects_A[i], self.rects_B[i], name=local_part,size_img=size_img, if_content=if_content)
                tp = loss_D_Style
            else:
                local_part = local_part.split('_')[0]
                loss_D_Style_ = self.backward_local_styleD(getattr(self, 'dis'+local_part.capitalize()), self.rects_A[i], self.rects_B[i],name=local_part+'2', flip=True,size_img=size_img, if_content=if_content)
                tp = loss_D_Style_
            loss_style[local_part] = tp * self.local_parts_laplacian_weight[i]                  
        return loss_style

    def local_style_g(self, image_a, image_b, size_img=256, if_content=True):
        # pdb.set_trace()
        self.input_A = image_a[0].to(self.device)
        self.input_B = image_b[0].to(self.device).detach()
        self.rects_A = image_a[1]
        self.rects_B = image_b[1]
        loss_style = {}
        for i in range(self.n_local):
            local_part = self.local_parts[i]    
            global mode 
            mode = local_part
            if '_' not in local_part:
                loss_G_Style = self.backward_G_GAN_local_style(getattr(self, 'dis'+local_part.capitalize()), self.rects_A[i], self.rects_B[i], size_img=size_img, if_content=if_content)
            else:
                local_part = local_part.split('_')[0]
                loss_G_Style = self.backward_G_GAN_local_style(getattr(self, 'dis'+local_part.capitalize()), self.rects_A[i], self.rects_B[i], flip=True, size_img=size_img, if_content=if_content)
            loss_style[local_part] = loss_G_Style * self.local_parts_laplacian_weight[i]
        return loss_style

    def local_style_d_multi(self, image_a, image_b, size_img=256, if_content=True):
        self.input_A = image_a[0].to(self.device).detach()
        self.input_B_lip = image_b[0][0].to(self.device).detach()
        self.input_B_eyes = image_b[0][1].to(self.device).detach()
        self.input_B_skin = image_b[0][2].to(self.device).detach()
        self.rects_A = image_a[1]
        self.rects_B = image_b[1]
        loss_style = {}
        for i in range(self.n_local):
            local_part = self.local_parts[i]
            tp = 0.0   
            if local_part in self.local_parts_eyes:
                self.input_B = self.input_B_lip
            elif local_part in self.local_parts_eyes:
                self.input_B = self.input_B_eyes
            else:
                self.input_B = self.input_B_skin
            if '_' not in local_part:
                loss_D_Style = self.backward_local_styleD(getattr(self, 'dis'+local_part.capitalize()), self.rects_A[i], self.rects_B[i], name=local_part,size_img=size_img, if_content=if_content)
                tp = loss_D_Style
            else:
                local_part = local_part.split('_')[0]
                loss_D_Style_ = self.backward_local_styleD(getattr(self, 'dis'+local_part.capitalize()), self.rects_A[i], self.rects_B[i],name=local_part+'2', flip=True,size_img=size_img, if_content=if_content)
                tp = loss_D_Style_
            loss_style[local_part] = tp              
        return loss_style

    def local_style_g_multi(self, image_a, image_b, size_img=256, if_content=True):
        self.input_A = image_a[0].to(self.device)
        #self.input_B = image_b[0].to(self.device).detach()
        self.input_B_lip = image_b[0][0].to(self.device).detach()
        self.input_B_eyes = image_b[0][1].to(self.device).detach()
        self.input_B_skin = image_b[0][2].to(self.device).detach()
        self.rects_A = image_a[1]
        self.rects_B = image_b[1]
        loss_style = {}
        for i in range(self.n_local):
            local_part = self.local_parts[i]    
            if local_part in self.local_parts_eyes:
                self.input_B = self.input_B_lip
            elif local_part in self.local_parts_eyes:
                self.input_B = self.input_B_eyes
            else:
                self.input_B = self.input_B_skin
            if '_' not in local_part:
                loss_G_Style = self.backward_G_GAN_local_style(getattr(self, 'dis'+local_part.capitalize()), self.rects_A[i], self.rects_B[i], size_img=size_img, if_content=if_content)
            else:
                local_part = local_part.split('_')[0]
                loss_G_Style = self.backward_G_GAN_local_style(getattr(self, 'dis'+local_part.capitalize()), self.rects_A[i], self.rects_B[i], flip=True, size_img=size_img, if_content=if_content)
            loss_style[local_part] = loss_G_Style
        return loss_style