import os 
import cv2
import argparse
from PIL import Image

import pdb
parser = argparse.ArgumentParser()
parser.add_argument(
    "--name", type=str, required=True, help="image dir."
)
parser.add_argument(
    "--style", type=str, required=True, help="image dir."
)
parser.add_argument(
    "--mask_path", type=str, required=True, help="image dir."
)
args = parser.parse_args()

ldmk_path = f"/home/yuanyujie/cvpr23/dataset/{args.name}/ori_imgs/landmark"
ori_warp_path = f"/home/yuanyujie/cvpr23/dataset/{args.name}/{args.style}/warp_makeup_{args.style}_ori"
warp_path = f"/home/yuanyujie/cvpr23/dataset/{args.name}/{args.style}/warp_makeup_{args.style}"

# os.system(f"mv {warp_path} {ori_warp_path}")
os.makedirs(warp_path, exist_ok=True)
for thing in os.listdir(ori_warp_path):
    # thing = thing.split("_")[0] + ".png"
    try:
        ori_style = os.path.join(ori_warp_path, thing[:-4]+".png")
        # ori_style = os.path.join(ori_warp_path, thing)
        # pdb.set_trace()
        # ori_img_path = os.path.join(f"/home/yuanyujie/cvpr23/dataset/{args.name}/ori_imgs", thing[:-4]+".png")
        ori_img_path = os.path.join(f"/home/yuanyujie/cvpr23/dataset/{args.name}/ori_imgs", thing)
        # mask_path = os.path.join(f"/home/yuanyujie/cvpr23/dataset/{args.name}/ori_imgs/ellipse_mask", thing)
        mask_path = os.path.join(args.mask_path, thing)
        image_mask = cv2.imread(mask_path)
        # raw_image = cv2.imread(ori_img_path)
        # style_image = cv2.imread(ori_style)
        
        image_mask = Image.fromarray(255*image_mask[:,:,0])
        raw_image = Image.open(ori_img_path)
        style_image = Image.open(ori_style)

        # raw_image =  Image.fromarray(raw_image)
        # style_image = Image.fromarray(style_image)
        pasted_image = raw_image.copy()

        #paste lip
        # pdb.set_trace()

        pasted_image.paste(style_image, (0,0,style_image.width, style_image.height), mask = image_mask)

        # pdb.set_trace()
        pasted_image.save(os.path.join(warp_path, f"{thing}"))
        # print(os.path.join(warp_path, f"{thing}"))
    except Exception as e:
        print(e)