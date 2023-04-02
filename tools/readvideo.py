import cv2
from cv2 import VideoWriter, VideoWriter_fourcc, imread, resize
import os
from PIL import Image
import argparse


def Pic2Video(inputpath, outputpath):
    imgPath = inputpath
    videoPath = outputpath
    print('videoPath', videoPath)
    images = sorted(os.listdir(imgPath))
    print('images', images)
    fps = 10

    fourcc = VideoWriter_fourcc(*"mp4v")

    image = Image.open(imgPath + images[0])
    videoWriter = cv2.VideoWriter(videoPath, fourcc, fps, image.size)
    for im_name in range(len(images)):
        frame = cv2.imread(imgPath + images[im_name])
        #print(im_name)
        videoWriter.write(frame)
    print('finish~!')
    videoWriter.release()
    cv2.destroyAllWindows()


def Video2Pic():
    #videoPath = '/data/heyue/104/4D-Facial-Avatars-main/nerface_code/nerf-pytorch/chenlan/video_cl/train.mp4'
    #imgPath = '/data/heyue/104/4D-Facial-Avatars-main/nerface_code/nerf-pytorch/chenlan/test_warp/'
    videoPath =  '/data/heyue/first-order-model/result.mp4'
    imgPath = '/data/heyue/104/4D-Facial-Avatars-main/nerface_code/nerf-pytorch/chenlan/styles/'
    cap = cv2.VideoCapture(videoPath)
    suc = cap.isOpened()
    frame_count = 0
    while suc:
        frame_count += 1
        suc, frame = cap.read()
        print('write',imgPath + str(frame_count).zfill(4)+'.png')
        cv2.imwrite(imgPath + str(frame_count).zfill(4)+'.png', frame)
        cv2.waitKey(1)
    cap.release()
    print('finish~!')    
 
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=str, default='name', help='name of the project')
    parser.add_argument('--input', type=str, default='', help='path to the no_makeup dir')
    parser.add_argument('--output', type=str, default='', help='path to the result dir')
    args = parser.parse_args()
    inputpath = args.input
    outputpath = os.path.join(args.output, args.id+'.mp4')
    #Video2Pic()
    Pic2Video(inputpath, outputpath)
