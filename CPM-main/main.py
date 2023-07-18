import os
from parser import get_args

import cv2
import numpy as np
from makeup import Makeup
from PIL import Image


def color_makeup(A_txt, B_txt, alpha):
    color_txt = model.makeup(A_txt, B_txt)
    color = model.render_texture(color_txt)
    color = model.blend_imgs(model.face, color * 255, alpha=alpha)
    return color


def pattern_makeup(A_txt, B_txt, render_texture=False):
    mask = model.get_mask(B_txt)
    mask = (mask > 0.0001).astype("uint8")
    pattern_txt = A_txt * (1 - mask)[:, :, np.newaxis] + B_txt * mask[:, :, np.newaxis]
    pattern = model.render_texture(pattern_txt)
    pattern = model.blend_imgs(model.face, pattern, alpha=1)
    return pattern


if __name__ == "__main__":
    args = get_args()
    model = Makeup(args)
    input = args.input
    #input as dir
    inputlist = os.listdir(args.input)
    imgB = np.array(Image.open(args.style))
    import pdb
    pdb.set_trace()
    imgB = cv2.resize(imgB, (256, 256))
    B_txt = model.prn_process_target(imgB)
    mask = model.get_mask(B_txt)
    mask = (mask > 0.001).astype("uint8")
    for name in inputlist:
      input = name
      name = input.split('/')[-1]
      imgA = np.array(Image.open(os.path.join(args.input,name)))
      #imgB = np.array(Image.open(args.style))
      #imgB = cv2.resize(imgB, (256, 256))
  
      model.prn_process(imgA)
      A_txt = model.get_texture()  
  
      if args.color_only:
          output = color_makeup(A_txt, B_txt, args.alpha)
      elif args.pattern_only:
          output = pattern_makeup(A_txt, B_txt)
      else:
          color_txt = model.makeup(A_txt, B_txt) * 255
          #mask = model.get_mask(B_txt)
          #mask = (mask > 0.001).astype("uint8")
          new_txt = color_txt * (1 - mask)[:, :, np.newaxis] + B_txt * mask[:, :, np.newaxis]
          output = model.render_texture(new_txt)
          output = model.blend_imgs(model.face, output, alpha=1)
  
      x2, y2, x1, y1 = model.location_to_crop()
      #output = np.concatenate([imgB[x2:], model.face[x2:], output[x2:]], axis=1)
      save_path = os.path.join(args.savedir,name)
      if not os.path.exists(args.savedir):
          os.mkdir(args.savedir)
      Image.fromarray((output).astype("uint8")).save(save_path)
