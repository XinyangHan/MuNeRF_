import os, glob
import time
import argparse
import numpy as np
from utils.api_util import FacePPAPI

def main(img_dir):
    save_dir = os.path.join(img_dir, 'landmark')
    os.makedirs(save_dir, exist_ok=True)
    #print(img_dir)
    img_list = glob.glob(os.path.join(img_dir,'*.png'))
    #print(img_list)
    api = FacePPAPI()
    for i in range(len(img_list)):#os.listdir(img_dir):
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
            fwname_landmark = os.path.join(save_dir,basename+'.npy')
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
    args = parser.parse_args()
    main(args.path)
