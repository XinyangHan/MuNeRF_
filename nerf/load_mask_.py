import os.path
import torchvision.transforms as transforms

from PIL import Image
import PIL
import numpy as np
import torch
from torch.autograd import Variable
import torchvision
from torchvision import utils as vtils
import cv2

def ToTensor(pic):
    # handle PIL Image
    if pic.mode == 'I':
        img = torch.from_numpy(np.array(pic, np.int32, copy=False))
    elif pic.mode == 'I;16':
        img = torch.from_numpy(np.array(pic, np.int16, copy=False))
    else:
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
    # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
    if pic.mode == 'YCbCr':
        nchannel = 3
    elif pic.mode == 'I;16':
        nchannel = 1
    else:
        nchannel = len(pic.mode)
    img = img.view(pic.size[1], pic.size[0], nchannel)
    # put it from HWC to CHW format
    # yikes, this transpose takes 80% of the loading time/CPU
    img = img.transpose(0, 1).transpose(0, 2).contiguous()
    if isinstance(img, torch.ByteTensor):
        return img.float()
    else:
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
    depth_img =  torch.cat((depth_img,depth_img,depth_img),dim=-1)
    dp_t, dp_n = cast_to_image2(depth_img[..., :3])
    return dp_t, dp_n

def save_mask(filename, file):
    file,_ = cast_to_depth(file)
    debug_dir='./debug2'
    savename = os.path.join(debug_dir, filename+'.jpg')
    vtils.save_image(file,savename)

def mask_for_gt(mask_nonmakeup,mask_makeup,makeup_img, nonmakeup_img,hw, rects):
    device = mask_nonmakeup.device
    h, w = hw, hw
    transform = transforms.Compose([
        transforms.Resize((hw,hw)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    transform_mask = transforms.Compose([
        transforms.Resize((h,w), interpolation=PIL.Image.NEAREST),
        ToTensor])
    makeup_img = transforms.ToPILImage()(makeup_img)
    nonmakeup_img = transforms.ToPILImage()(nonmakeup_img) 
    
    makeup_img = transform(makeup_img).to(device)
    nonmakeup_img = transform(nonmakeup_img).to(device)   
    nonmakeup_seg_img = transforms.ToPILImage()(mask_nonmakeup)
    mask_A = transform_mask(nonmakeup_seg_img).to(device)  # nonmakeup
    makeup_seg_img = transforms.ToPILImage()(mask_makeup)
    mask_B = transform_mask(makeup_seg_img).to(device)  # makeup

    mask_A_bg  = (mask_A == 0).float()
    mask_A_bg, _, index_A_bg, _ = mask_preprocess(mask_A_bg, mask_A_bg)
    size = mask_A_bg.unsqueeze(0).size()

    #skin #warp
    mask_A_skin = (mask_A == 4).float() + (mask_A == 10).float() + (mask_A == 2).float() + (mask_A == 7).float() + (mask_A == 6).float() + (mask_A == 1).float()
    #_,An1 = cast_to_depth(mask_A_skin.permute(1,2,0))
    #mask_A_skin =  torch.from_numpy(open_demo(An1)).permute(2,0,1)[:1,...].float().to(device)
    mask_A_skin, _, index_A_skin, _ = mask_preprocess(mask_A_skin, mask_A_skin)

    mask_A_lip, _, index_A_lip, _ = make_crop_rect(size, rects, mask_A_skin, part='mouth',if_skin=False)
    #save_mask('mouth', mask_A_lip.permute(1,2,0))

    #nose #his
    mask_A_nose = (mask_A == 8).float()
    mask_B_nose = (mask_B == 8).float()
    _,An5 = cast_to_depth(mask_A_nose.permute(1,2,0))
    _,An3 = cast_to_depth(mask_A_nose.permute(1,2,0))
    mask_A_nose =  torch.from_numpy(open_demo(An5)).permute(2,0,1)[:1,...].float().to(device)
    mask_B_nose =  torch.from_numpy(open_demo(An3)).permute(2,0,1)[:1,...].float().to(device)
    mask_A_nose, mask_B_nose, index_A_nose, index_B_nose = mask_preprocess(mask_A_nose, mask_B_nose)

    changed_mask = mask_A_skin.to(device) + mask_A_nose.to(device) + mask_A_lip.to(device)

    unchanged_mask = (changed_mask == 0).float()

    mask_A = {}
    mask_A["mask_A_skin"] = mask_A_skin
    mask_A["index_A_skin"] = index_A_skin
    mask_A["mask_A_nose"] = mask_A_nose
    mask_A["index_A_nose"] = index_A_nose
    mask_A["mask_A_lip"] = mask_A_lip
    mask_A["index_A_lip"] = index_A_lip
    mask_A["mask_A_bg"] = mask_A_bg
    mask_A["index_A_bg"] = index_A_bg
    mask_A['mask_B_nose_w'] = mask_B_nose
    return mask_A, nonmakeup_img, makeup_img

def load_mask_data2(fake_makeup1,fake_makeup2,makeup_img, nonmakeup_img,mask_nonmakeup, mask_makeup, rects, resize_scale):
    device = fake_makeup1.device
    local_parts = ['eye', 'eye_', 'mouth', 'nose', 'cheek', 'cheek_', 'eyebrow', 'eyebrow_', 'uppernose', 'forehead', 'sidemouth', 'sidemouth_']
    h, w = fake_makeup1.shape[1], fake_makeup1.shape[2]
    transform = transforms.Compose([
        transforms.Resize((h,w)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    transform_mask = transforms.Compose([
        transforms.Resize((h,w), interpolation=PIL.Image.NEAREST),
        ToTensor])
    scale_factor = 1/resize_scale
    makeup_img = transforms.ToPILImage()(makeup_img)
    nonmakeup_img = transforms.ToPILImage()(nonmakeup_img) 
    makeup_seg_img = transforms.ToPILImage()(mask_makeup)
    nonmakeup_seg_img = transforms.ToPILImage()(mask_nonmakeup)
    # fake_makeup1= transforms.ToPILImage()(fake_makeup1)
    # fake_makeup2= transforms.ToPILImage()(fake_makeup2)
    # makeup_img = to_var(transform(makeup_img), requires_grad=False)
    # nonmakeup_img = to_var(transform(nonmakeup_img), requires_grad=False)
    # fake_makeup1 = to_var(transform(fake_makeup1), requires_grad=False)
    # fake_makeup2 = to_var(transform(fake_makeup2), requires_grad=False)
    makeup_img = transform(makeup_img).to(device)
    nonmakeup_img = transform(nonmakeup_img).to(device)
    # fake_makeup1 = transform(fake_makeup1).to(device)
    # fake_makeup2 = transform(fake_makeup2).to(device)
    # fake_makeup1 = torch.nn.functional.interpolate(fake_makeup1.unsqueeze(0), size=[256,256]).squeeze().to(device)
    # fake_makeup2 = torch.nn.functional.interpolate(fake_makeup2.unsqueeze(0), size=[256,256]).squeeze().to(device)
    mask_B = transform_mask(makeup_seg_img).to(device)  # makeup
    mask_A = transform_mask(nonmakeup_seg_img).to(device)  # nonmakeup
    
    #face
    mask_A_face = (mask_A == 4).float()
    mask_B_face = (mask_B == 4).float()
    _,An2 = cast_to_depth(mask_A_face.permute(1,2,0))
    _,Bn2 = cast_to_depth(mask_B_face.permute(1,2,0))
    mask_A_face =  torch.from_numpy(open_demo(An2)).permute(2,0,1)[:1,...].float().to(device)
    mask_B_face = torch.from_numpy(open_demo(Bn2)).permute(2,0,1)[:1,...].float().to(device)

    #skin #warp
    mask_A_skin = (mask_A == 4).float() + (mask_A == 10).float()
    mask_B_skin = (mask_B == 4).float() + (mask_B == 10).float()
    _,An1 = cast_to_depth(mask_A_skin.permute(1,2,0))
    _,Bn1 = cast_to_depth(mask_B_skin.permute(1,2,0))
    mask_A_skin =  torch.from_numpy(open_demo(An1)).permute(2,0,1)[:1,...].float().to(device)
    mask_B_skin = torch.from_numpy(open_demo(Bn1)).permute(2,0,1)[:1,...].float().to(device)
    mask_A_skin, mask_B_skin, index_A_skin, index_B_skin = mask_preprocess(mask_A_skin, mask_B_skin)
    
    #nose #his
    mask_A_nose = (mask_A == 8).float()
    mask_B_nose = (mask_B == 8).float()

    _,An5 = cast_to_depth(mask_A_nose.permute(1,2,0))
    _,Bn5 = cast_to_depth(mask_B_nose.permute(1,2,0))
    mask_A_nose =  torch.from_numpy(open_demo(An5)).permute(2,0,1)[:1,...].float().to(device)
    mask_B_nose = torch.from_numpy(open_demo(Bn5)).permute(2,0,1)[:1,...].float().to(device)
    mask_A_nose, mask_B_nose, index_A_nose, index_B_nose = mask_preprocess(mask_A_nose, mask_B_nose)
    
    size = mask_A_face.unsqueeze(0).size()
    #lip #his
    mask_A_lip, mask_B_lip, index_A_lip, index_B_lip = make_crop_rect(size, rects, mask_A_skin, part='mouth',if_skin=False, scale_factor=scale_factor)

    #eye #his
    mask_A_eye, mask_B_eye, index_A_eye, index_B_eye = make_crop_rect(size, rects, mask_A_skin, part='eye',if_skin=False, scale_factor=scale_factor)
    mask_A_eye_, mask_B_eye_, index_A_eye_, index_B_eye_ = make_crop_rect(size, rects, mask_A_skin, part='eye_',if_skin=False, scale_factor=scale_factor)


    #eyebrow #his
    mask_A_eyebrow, mask_B_eyebrow, index_A_eyebrow, index_B_eyebrow = make_crop_rect(size, rects, mask_A_skin, part='eyebrow',if_skin=False, scale_factor=scale_factor)
    mask_A_eyebrow_, mask_B_eyebrow_, index_A_eyebrow_, index_B_eyebrow_ = make_crop_rect(size, rects, mask_A_skin, part='eyebrow_',if_skin=False, scale_factor=scale_factor)

    # 12 skin parts include eyes
    skin_patches = []
    for item in local_parts:
        skin_patches.append(list(make_crop_rect(size, rects, mask_A_skin, part=item,if_skin=True,scale_factor=scale_factor)))

    mask_A = {}
    mask_A["skin_patches"] = skin_patches
    mask_A["mask_A_skin"] = mask_A_skin
    mask_A["index_A_skin"] = index_A_skin
    mask_A["mask_A_nose"] = mask_A_nose
    mask_A["index_A_nose"] = index_A_nose
    mask_A["mask_A_eyebrow"] = mask_A_eyebrow
    mask_A["index_A_eyebrow"] = index_A_eyebrow
    mask_A["mask_A_eyebrow_"] = mask_A_eyebrow_
    mask_A["index_A_eyebrow_"] = index_A_eyebrow_
    mask_A["mask_A_eye"] = mask_A_eye
    mask_A["index_A_eye"] = index_A_eye
    mask_A["mask_A_eye_"] = mask_A_eye_
    mask_A["index_A_eye_"] = index_A_eye_
    mask_A["mask_A_lip"] = mask_A_lip
    mask_A["index_A_lip"] = index_A_lip

    mask_B = {}
    mask_B["skin_patches"] = skin_patches
    mask_B["mask_B_skin"] = mask_B_skin
    mask_B["index_B_skin"] = index_B_skin
    mask_B["mask_B_nose"] = mask_B_nose
    mask_B["index_B_nose"] = index_B_nose
    mask_B["mask_B_eyebrow"] = mask_B_eyebrow
    mask_B["index_B_eyebrow"] = index_B_eyebrow
    mask_B["mask_B_eyebrow_"] = mask_B_eyebrow_
    mask_B["index_B_eyebrow_"] = index_B_eyebrow_
    mask_B["mask_B_eye"] = mask_B_eye
    mask_B["index_B_eye"] = index_B_eye
    mask_B["mask_B_eye_"] = mask_B_eye_
    mask_B["index_B_eye_"] = index_B_eye_
    mask_B["mask_B_lip"] = mask_B_lip
    mask_B["index_B_lip"] = index_B_lip

    return fake_makeup1, fake_makeup2, mask_A, mask_B, makeup_img, nonmakeup_img

def make_crop_rect(size, rects,skin, part='',if_skin=True, scale_factor=4):
    local_parts = ['eye', 'eye_', 'mouth', 'nose', 'cheek', 'cheek_', 'eyebrow', 'eyebrow_', 'uppernose', 'forehead', 'sidemouth', 'sidemouth_']
    if not if_skin:
        mask_A_lip = rebound_box2(size, rects[local_parts.index(part)], scale_factor=scale_factor)
        mask_B_lip = mask_A_lip
    else:
        mask_A_lip = rebound_box_skin(skin, rects[local_parts.index(part)], scale_factor=scale_factor)
        mask_B_lip = mask_A_lip
    mask_A_lip, mask_B_lip, index_A_lip, index_B_lip = mask_preprocess(mask_A_lip, mask_B_lip)
    return mask_A_lip, mask_B_lip, index_A_lip, index_B_lip




def rebound_box(mask_A_face,  left_eye_rect, right_eye_rect):
    mask_A_face = mask_A_face.unsqueeze(0)
    left_eye_rect= [item*4 for item in left_eye_rect]
    right_eye_rect= [item*4 for item in right_eye_rect]
    x1_l, x2_l, y1_l, y2_l = left_eye_rect
    x1_r, x2_r, y1_r, y2_r = right_eye_rect
    mask_A_temp = torch.zeros(mask_A_face.size())
    mask_B_temp = torch.zeros(mask_A_face.size())
    mask_A_temp[:, :, x1_l:x2_l, y1_l:y2_l] =  mask_A_face[:, :, x1_l:x2_l, y1_l:y2_l]
    mask_B_temp[:, :, x1_r:x2_r, y1_r:y2_r] = mask_A_face[:, :, x1_r:x2_r, y1_r:y2_r]
    mask_A_temp = mask_A_temp.squeeze(0)
    mask_B_temp = mask_B_temp.squeeze(0)
    return mask_A_temp, mask_B_temp

def rebound_box2(mask_size, crop_rect, scale_factor=4):
    #mask_A = mask_A.unsqueeze(0)
    crop_rect= [int(item) for item in crop_rect]
    x1, x2, y1, y2 = crop_rect
    mask_A_temp = torch.zeros(mask_size)
    mask_crop_temp = torch.ones(mask_size)*255.
    mask_A_temp[:, :, x1:x2, y1:y2] =  mask_crop_temp[:, :, x1:x2, y1:y2]
    mask_A_temp = mask_A_temp.squeeze(0)
    return mask_A_temp

def rebound_box_skin(skin, crop_rect, scale_factor=4):
    skin = skin.unsqueeze(0)
    crop_rect= [int(item) for item in crop_rect]
    x1, x2, y1, y2 = crop_rect
    mask_A_temp = torch.zeros(skin.size())
    mask_A_temp[:, :, x1:x2, y1:y2] = skin[:, :, x1:x2, y1:y2]
    mask_A_temp = mask_A_temp.squeeze(0)
    return mask_A_temp

def rebound_box_lip(mask_A,  lip_rect):
    mask_A = mask_A.unsqueeze(0)
    lip_rect= [item*4 for item in lip_rect]
    x1_l, x2_l, y1_l, y2_l = lip_rect
    mask_A_temp = torch.zeros(mask_A.size())
    mask_A_temp[:, :, x1_l:x2_l, y1_l:y2_l] =  mask_A[:, :, x1_l:x2_l, y1_l:y2_l]
    mask_A_temp = mask_A_temp.squeeze(0)
    return mask_A_temp

def verifty_left_right(eye_left, eye_right):
    eye_left = eye_left.unsqueeze(0)
    eye_right = eye_right.unsqueeze(0)
    index_left = torch.nonzero(eye_left, as_tuple=False)
    y_left_index = index_left[:, 3]
    index_right = torch.nonzero(eye_right, as_tuple=False)
    y_right_index = index_right[:, 3]
    if y_left_index.numel() and y_right_index.numel():
        if min(y_left_index)<min(y_right_index): return eye_left.squeeze(0), eye_right.squeeze(0)
        else: return eye_right.squeeze(0), eye_left.squeeze(0)
    else:
        return eye_left.squeeze(0), eye_right.squeeze(0)

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

def to_var(x, requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    if not requires_grad:
        return Variable(x, requires_grad=requires_grad)
    else:
        return Variable(x)

def open_demo(img):#开操作=腐蚀+膨胀  去外边白点
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    result = cv2.dilate(cv2.erode(img,kernel),kernel)
    return result

'''index_tmp = torch.nonzero(mask_A, as_tuple=False)
    x_A_index = index_tmp[:, 2]
    y_A_index = index_tmp[:, 3]
    index_tmp = torch.nonzero(mask_B, as_tuple=False)
    x_B_index = index_tmp[:, 2]
    y_B_index = index_tmp[:, 3]
    mask_A_temp = mask_A.copy_(mask_A)
    mask_B_temp = mask_B.copy_(mask_B)
    if x_A_index.numel() and y_A_index.numel():
        mid_A_index = max(y_A_index)+8
        if int(max(y_A_index))-int(min(y_A_index))>59:
            mid_A_index = int((min(y_A_index) + max(y_A_index) + 1)/2)-10
        mask_A_temp[:, :, min(x_A_index) - 7:max(x_A_index) + 8, min(y_A_index) - 7:mid_A_index] = \
        mask_A_face[:, :, min(x_A_index) - 7:max(x_A_index) + 8, min(y_A_index) - 7:mid_A_index]
    if x_B_index.numel() and y_B_index.numel():
        mid_B_index = min(y_B_index) - 7
        if int(max(y_B_index))-int(min(y_B_index))>59:
            mid_B_index = int((min(y_B_index) + max(y_B_index) + 1)/2)+10
        mask_B_temp[:, :, min(x_B_index) - 7:max(x_B_index) + 8, mid_B_index:max(y_B_index)+8] = \
        mask_A_face[:, :, min(x_B_index) - 7:max(x_B_index) + 8, mid_B_index:max(y_B_index)+8]

    # mask_A_temp = self.to_var(mask_A_temp, requires_grad=False)
    # mask_B_temp = self.to_var(mask_B_temp, requires_grad=False)
    mask_A_temp = mask_A_temp.squeeze(0)
    mask_A = mask_A.squeeze(0)
    mask_B = mask_B.squeeze(0)
    mask_A_face = mask_A_face.squeeze(0)
    mask_B_temp = mask_B_temp.squeeze(0)'''
'''
    def load_mask_data(fake_makeup1, fake_makeup2, makeup_img, style_target,nonmakeup_img, mask_nonmakeup, mask_makeup, eye_rects):
    transform = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    transform_mask = transforms.Compose([
        transforms.Resize((256,256), interpolation=PIL.Image.NEAREST),
        ToTensor])


    makeup_img = transforms.ToPILImage()(makeup_img)
    style_target = transforms.ToPILImage()(style_target)
    nonmakeup_img = transforms.ToPILImage()(nonmakeup_img) 
    makeup_seg_img = transforms.ToPILImage()(mask_makeup)
    nonmakeup_seg_img = transforms.ToPILImage()(mask_nonmakeup)
    fake_makeup1= transforms.ToPILImage()(fake_makeup1)
    fake_makeup2= transforms.ToPILImage()(fake_makeup2)

    left_eye_rect = eye_rects[0]
    right_eye_rect = eye_rects[1]
    lip_rect = eye_rects[2]

    makeup_img = to_var(transform(makeup_img), requires_grad=False)
    style_target = to_var(transform(style_target), requires_grad=False)
    nonmakeup_img = to_var(transform(nonmakeup_img), requires_grad=False)
    fake_makeup1 = to_var(transform(fake_makeup1), requires_grad=False)
    fake_makeup2 = to_var(transform(fake_makeup2), requires_grad=False)
    mask_B = transform_mask(makeup_seg_img)  # makeup
    mask_A = transform_mask(nonmakeup_seg_img)  # nonmakeup
    makeup_seg = torch.zeros([3, 256, 256], dtype=torch.float)
    nonmakeup_seg = torch.zeros([3, 256, 256], dtype=torch.float)

    #face
    mask_A_face = (mask_A == 4).float().float()
    mask_B_face = (mask_B == 4).float().float()
    _,An2 = cast_to_depth(mask_A_face.permute(1,2,0))
    _,Bn2 = cast_to_depth(mask_B_face.permute(1,2,0))
    mask_A_face =  torch.from_numpy(open_demo(An2)).permute(2,0,1)[:1,...].float()
    mask_B_face = torch.from_numpy(open_demo(Bn2)).permute(2,0,1)[:1,...].float()

    #lip
    mask_A_face_lip = (mask_A == 9).float() + (mask_A == 13).float() +(mask_A == 4).float().float()
    mask_B_face_lip = (mask_B == 9).float() + (mask_B == 13).float() +(mask_B == 4).float().float()
    _,An = cast_to_depth(mask_A_face_lip.permute(1,2,0))
    _,Bn = cast_to_depth(mask_B_face_lip.permute(1,2,0))
    mask_A_face_lip =  torch.from_numpy(open_demo(An)).permute(2,0,1)[:1,...].float()
    mask_B_face_lip = torch.from_numpy(open_demo(Bn)).permute(2,0,1)[:1,...].float()

    mask_A_lip= rebound_box_lip(mask_A_face_lip, lip_rect)
    mask_B_lip= rebound_box_lip(mask_B_face_lip, lip_rect)
    mask_A_lip, mask_B_lip, index_A_lip, index_B_lip = mask_preprocess(mask_A_lip, mask_B_lip)

    makeup_seg[0] = mask_B_lip[0]
    nonmakeup_seg[0] = mask_A_lip[0]
    mask_A_skin = (mask_A == 4).float() + (mask_A == 10).float()
    mask_B_skin = (mask_B == 4).float() + (mask_B == 10).float()
    _,An1 = cast_to_depth(mask_A_skin.permute(1,2,0))
    _,Bn1 = cast_to_depth(mask_B_skin.permute(1,2,0))
    mask_A_skin =  torch.from_numpy(open_demo(An1)).permute(2,0,1)[:1,...].float()
    mask_B_skin = torch.from_numpy(open_demo(Bn1)).permute(2,0,1)[:1,...].float()
    mask_A_skin, mask_B_skin, index_A_skin, index_B_skin = mask_preprocess(mask_A_skin, mask_B_skin)

    mask_A_nose = (mask_A == 8).float()
    mask_B_nose = (mask_B == 8).float()
    _,An5 = cast_to_depth(mask_A_nose.permute(1,2,0))
    _,Bn5 = cast_to_depth(mask_B_nose.permute(1,2,0))
    mask_A_nose =  torch.from_numpy(open_demo(An5)).permute(2,0,1)[:1,...].float()
    mask_B_nose = torch.from_numpy(open_demo(Bn5)).permute(2,0,1)[:1,...].float()
    mask_A_nose, mask_B_nose, index_A_nose, index_B_nose = mask_preprocess(mask_A_nose, mask_B_nose)

    mask_A_eyebrow = (mask_A == 2).float() + (mask_A == 7).float()
    mask_B_eyebrow = (mask_B == 2).float() + (mask_B == 7).float()
    _,An6 = cast_to_depth(mask_A_eyebrow.permute(1,2,0))
    _,Bn6 = cast_to_depth(mask_B_eyebrow.permute(1,2,0))
    mask_A_eyebrow =  torch.from_numpy(open_demo(An6)).permute(2,0,1)[:1,...].float()
    mask_B_eyebrow = torch.from_numpy(open_demo(Bn6)).permute(2,0,1)[:1,...].float()
    mask_A_eyebrow, mask_B_eyebrow, index_A_eyebrow, index_B_eyebrow = mask_preprocess(mask_A_eyebrow, mask_B_eyebrow)
    
    makeup_seg[1] = mask_B_skin[0]
    nonmakeup_seg[1] = mask_A_skin[0]
    mask_A_eye_left = (mask_A == 6).float()
    mask_A_eye_right = (mask_A == 1).float()
    mask_B_eye_left = (mask_B == 1).float()
    mask_B_eye_right = (mask_B == 6).float()
    _,An3 = cast_to_depth(mask_A_eye_left.permute(1,2,0))
    _,Bn3 = cast_to_depth(mask_B_eye_left.permute(1,2,0))
    mask_A_eye_left =  torch.from_numpy(open_demo(An3)).permute(2,0,1)[:1,...].float()
    mask_B_eye_left = torch.from_numpy(open_demo(Bn3)).permute(2,0,1)[:1,...].float()
    _,An4 = cast_to_depth(mask_A_eye_right.permute(1,2,0))
    _,Bn4 = cast_to_depth(mask_B_eye_right.permute(1,2,0))
    mask_A_eye_right =  torch.from_numpy(open_demo(An4)).permute(2,0,1)[:1,...].float()
    mask_B_eye_right = torch.from_numpy(open_demo(Bn4)).permute(2,0,1)[:1,...].float()
    mask_A_eye_left, mask_A_eye_right = verifty_left_right(mask_A_eye_left, mask_A_eye_right)
    mask_B_eye_left, mask_B_eye_right = verifty_left_right(mask_B_eye_left, mask_B_eye_right)

    mask_A_eye_left, mask_A_eye_right = rebound_box(mask_A_face, left_eye_rect, right_eye_rect)
    mask_B_eye_left, mask_B_eye_right = rebound_box(mask_B_face, left_eye_rect, right_eye_rect)
    
    mask_A_eye_left, mask_B_eye_left, index_A_eye_left, index_B_eye_left = \
        mask_preprocess(mask_A_eye_left, mask_B_eye_left)
    mask_A_eye_right, mask_B_eye_right, index_A_eye_right, index_B_eye_right = \
        mask_preprocess(mask_A_eye_right, mask_B_eye_right)
    makeup_eye_B = mask_B_eye_left + mask_B_eye_right
    makeup_seg[2] = makeup_eye_B[0]
    makeup_eye_A = mask_A_eye_left + mask_A_eye_right
    nonmakeup_seg[2] = makeup_eye_A[0]

    mask_A = {}
    mask_A["mask_A_eye_left"] = mask_A_eye_left
    mask_A["mask_A_eye_right"] = mask_A_eye_right
    mask_A["index_A_eye_left"] = index_A_eye_left
    mask_A["index_A_eye_right"] = index_A_eye_right
    mask_A["mask_A_skin"] = mask_A_skin
    mask_A["index_A_skin"] = index_A_skin
    mask_A["mask_A_nose"] = mask_A_nose
    mask_A["index_A_nose"] = index_A_nose
    mask_A["mask_A_eyebrow"] = mask_A_eyebrow
    mask_A["index_A_eyebrow"] = index_A_eyebrow
    mask_A["mask_A_lip"] = mask_A_lip
    mask_A["index_A_lip"] = index_A_lip

    mask_B = {}
    mask_B["mask_B_eye_left"] = mask_B_eye_left
    mask_B["mask_B_eye_right"] = mask_B_eye_right
    mask_B["index_B_eye_left"] = index_B_eye_left
    mask_B["index_B_eye_right"] = index_B_eye_right
    mask_B["mask_B_skin"] = mask_B_skin
    mask_B["index_B_skin"] = index_B_skin
    mask_B["mask_B_nose"] = mask_B_nose
    mask_B["index_B_nose"] = index_B_nose
    mask_B["mask_B_eyebrow"] = mask_B_eyebrow
    mask_B["index_B_eyebrow"] = index_B_eyebrow
    mask_B["mask_B_lip"] = mask_B_lip
    mask_B["index_B_lip"] = index_B_lip
    
    return fake_makeup1, fake_makeup2, mask_A, mask_B, makeup_img, style_target, nonmakeup_img
    '''