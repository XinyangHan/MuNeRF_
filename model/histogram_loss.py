from nerf.load_mask import save_mask
from scripts import mask
import torch
import torch.nn as nn
from torch.autograd import Variable
import copy
import os
from torchvision import utils as vtils
import numpy as np
import cv2
import torchvision

class HistogramLoss(nn.Module):
    def __init__(self):
        super(HistogramLoss, self).__init__()
        self.criterionL1 = torch.nn.L1Loss()
    
    def de_norm(self, x):
        out = (x + 1) / 2
        return out.clamp(0, 1)

    def to_var(self, x, device, requires_grad=True):
        x = x.to(device)
        if not requires_grad:
            return Variable(x, requires_grad=requires_grad)
        else:
            return Variable(x)

    def forward(self, vggnet, input_data, target_data,target_data_eye, mask_src, mask_tar, index, ref, part):
        #we directly use skin of warped result and for other face parts, we use result after matching histogram
        device = input_data.device
        skin_local_parts = ['eye', 'eye_', 'mouth', 'nose', 'cheek', 'cheek_', 'eyebrow', 'eyebrow_', 'uppernose', 'forehead', 'sidemouth', 'sidemouth_']
        u = ['nose_his','eye_warp','eye__warp']
        warp_local_parts = ['skin_warp','eyebrow_his', 'eyebrow__his','lip_his']
        non_bg = ['bg']
        # input_data = (self.de_norm(input_data) * 255).squeeze() # gen makeup
        target_data = (self.de_norm(target_data) * 255).squeeze() # warp
        input_data = (input_data.clamp(0, 1) * 255).squeeze() # gen makeup
        # target_data = (target_data.clamp(0, 1) * 255).squeeze() # warp
        #target_data_eye = (self.de_norm(target_data_eye) * 255).squeeze()
        ref = (self.de_norm(ref) * 255).squeeze() # gen makeup
        # ref = (ref.clamp(0, 1) * 255).squeeze() # gen makeup
        mask_src = mask_src/255.
        mask_tar = mask_tar/255.
        mask_src = mask_src.expand(1, 3, mask_src.size(2), mask_src.size(2)).squeeze().to(device)
        mask_tar = mask_tar.expand(1, 3, mask_tar.size(2), mask_tar.size(2)).squeeze().to(device)
        input_masked = input_data * mask_src
        target_masked = target_data * mask_tar
        #target_masked_eye = target_data_eye * mask_tar
        ref_masked = ref * mask_src
    
        if part in warp_local_parts:
            input_match = target_data * mask_src
        elif part in skin_local_parts:
            input_match = target_data * mask_src
        elif part in non_bg:
            input_match = ref_masked
        else:
            input_match = histogram_matching(ref_masked, target_masked, index)
        input_match = self.to_var(input_match, device, requires_grad=False)

        imgs = [(target_masked/255.).unsqueeze(0),(ref_masked/255.).unsqueeze(0),(input_masked/255.).unsqueeze(0),(input_match/255.).unsqueeze(0)]
        use_vgg = False
        if use_vgg:
          if part=='bg':
              loss = torch.nn.functional.mse_loss(input_masked/255., input_match/255.)
          elif part in skin_local_parts:
              loss = self.criterionL1(input_masked/255., input_match/255.)
          else:
              input_masked_vgg = vggnet(input_masked.unsqueeze(0))
              input_masked_vgg = input_masked_vgg['conv4_2'] 
              input_match_vgg = vggnet(input_match.unsqueeze(0))
              input_match_vgg = input_match_vgg['conv4_2'] 
              loss = torch.nn.functional.mse_loss(input_masked, input_match)
        else:
          if part=='bg':
              loss = torch.nn.functional.mse_loss(input_masked/255., input_match/255.)
          else:
              loss = self.criterionL1(input_masked/255., input_match/255.)
        return imgs, loss

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
