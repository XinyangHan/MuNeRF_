import os, glob
import time
import argparse
import numpy as np
from utils.api_util import FacePPAPI

def main(img_dir, result_path):
    # save_dir = os.path.join(img_dir, 'landmark')
    save_dir = result_path
    
    os.makedirs(save_dir, exist_ok=True)
    img_list = glob.glob(os.path.join(img_dir,'*.png'))
    api = FacePPAPI()
    for i in range(len(img_list)):
        img_file = img_list[i]
        basename = os.path.basename(img_file)
        basename = basename.split('.')[0]
        if not os.path.exists(os.path.join(save_dir,basename+'.npy')):
            landmark_dict = api.faceLandmarkDetector(img_file)
            np.save(os.path.join(save_dir,basename+'.npy'), landmark_dict)
            if landmark_dict is not None:
                print(basename + ' done!')
            time.sleep(0.5)
        else:
            tmp_landmark = np.load(fwname_landmark, allow_pickle=True)
            if tmp_landmark is None:
                landmark_dict = api.faceLandmarkDetector(img_file)
                np.save(os.path.join(save_dir,basename+'.npy'), landmark_dict)
                if landmark_dict is not None:
                    print(basename + ' done!')
                time.sleep(0.5)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path", type=str, required=True, help="image dir."
    )
    parser.add_argument("--result_path", type=str, required=True, help="result path")
    args = parser.parse_args()
    main(args.path, args.result_path)