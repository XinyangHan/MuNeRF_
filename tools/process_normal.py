import os, glob
import time
import argparse
import numpy as np
import cv2

def main(normal, warp, non):
    result_dir = './result/'
    result_non = './result/non/'
    result_warp = './result/warp/'
    
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(result_non, exist_ok=True)
    os.makedirs(result_warp, exist_ok=True)
    
    normal_list = glob.glob(os.path.join(normal,'*.png'))
    warp_list = glob.glob(os.path.join(warp, '*.png'))
    non_list = glob.glob(os.path.join(non, '*.png'))
    
    for i in range(len(normal_list)):
        normal_file = normal_list[i]
        warp_file = warp_list[i]
        non_file = non_list[i]
        normal = cv2.imread(normal_file)
        warp = cv2.imread(warp_file)
        non = cv2.imread(non_file)
        basename = normal_file.split('/')[-1]
        save_name_warp_normal = os.path.join(result_dir,'warp',basename)
        save_name_non_normal = os.path.join(result_dir,'non',basename)
        new_warp = warp * normal
        cv2.imwrite(save_name_warp_normal, new_warp)
        new_non = non * normal
        cv2.imwrite(save_name_non_normal, new_non)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--normal", type=str, required=True, help="image dir."
    )
    parser.add_argument(
	      "--warped", type=str, required=True, help="warped dir"
    )
    parser.add_argument(
        "--nonmakeup", type=str, required=True, help="non dir"
    )
	
    args = parser.parse_args()
    main(args.normal, args.warped, args.nonmakeup)
