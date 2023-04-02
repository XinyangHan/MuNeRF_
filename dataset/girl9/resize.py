import os
import cv2
import argparse

def resize(id):
    #trainpath = '/data/heyue/104/4D-Facial-Avatars-main/nerface_code/nerf-pytorch/datasets/'+id+'/half_res'
    trainpath = '/data/hanxinyang/MuNeRF/dataset/girl9/half_res/new'
    for d in os.listdir(trainpath):
        path = os.path.join(trainpath, d)
        img = cv2.imread(path)
        img = cv2.resize(img, (256,256))
        cv2.imwrite(path, img)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=str, default='', help='path to the no_makeup dir')
    args = parser.parse_args()
    id = args.id
    resize(id)
