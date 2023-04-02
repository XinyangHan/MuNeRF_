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
    fps = 10

    fourcc = VideoWriter_fourcc(*"mp4v")

    image = Image.open(imgPath + images[0])
    videoWriter = cv2.VideoWriter(videoPath, fourcc, fps, image.size)
    for im_name in range(len(images)):
        frame = cv2.imread(imgPath + images[im_name])
        #print(im_name)
        videoWriter.write(frame)
    print("finished！")
    videoWriter.release()
    cv2.destroyAllWindows()


def Video2Pic():
    videoPath = '/data/heyue/104/4D-Facial-Avatars-main/nerface_code/nerf-pytorch/chenlan/video_cl/test_warp.mp4'
    imgPath = '/data/heyue/104/4D-Facial-Avatars-main/nerface_code/nerf-pytorch/chenlan/test_warp/'
    cap = cv2.VideoCapture(videoPath)
    # fps = cap.get(cv2.CAP_PROP_FPS)  # 获取帧率
    # width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 获取宽度
    # height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 获取高度
    suc = cap.isOpened()
    frame_count = 0
    while suc:
        frame_count += 1
        suc, frame = cap.read()
        cv2.imwrite(imgPath + str(frame_count).zfill(4)+'.png', frame)
        cv2.waitKey(1)
    cap.release()
    print("finished！")
 
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=str, default='name', help='name of the project')
    parser.add_argument('--input', type=str, default='/data/heyue/104/4D-Facial-Avatars-main/nerface_code/nerf-pytorch/chenlan/test_warp/', help='path to the no_makeup dir')
    parser.add_argument('--output', type=str, default='/data/heyue/104/4D-Facial-Avatars-main/nerface_code/nerf-pytorch/chenlan/video_cl/', help='path to the result dir')
    args = parser.parse_args()
    inputpath = args.input
    outputpath = os.path.join(args.output, args.id+'.mp4')
    #Video2Pic()
    Pic2Video(inputpath, outputpath)