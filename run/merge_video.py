import os
import argparse
import cv2

parser = argparse.ArgumentParser()
parser.add_argument(
    "--path", type=str, required=True, help="image dir."
)

args = parser.parse_args()

root = args.path
things = os.listdir(root)
img = cv2.imread(os.path.join(root, things[0]))  #读取第一张图片
imgInfo = img.shape
size = (imgInfo[1],imgInfo[0])  #获取图片宽高度信息
print(size)



for thing in things:

videoWrite = cv2.VideoWriter('2.mp4',-1,5,size)# 根据图片的大小，创建写入对象 （文件名，支持的编码器，5帧，视频大小（图片大小））

for i in range(1,11):
    fileName = 'image'+str(i)+'.jpg'    #循环读取所有的图片
    img = cv2.imread(fileName)

    videoWrite.write(img      )# 将图片写入所创建的视频对象


print('end!')