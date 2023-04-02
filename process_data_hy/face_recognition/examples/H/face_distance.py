import face_recognition
import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import pdb
# Often instead of just checking if two faces match or not (True or False), it's helpful to see how similar they are.
# You can do that by using the face_distance function.

# The model was trained in a way that faces with a distance of 0.6 or less should be a match. But if you want to
# be more strict, you can look for a smaller face distance. For example, using a 0.55 cutoff would reduce false
# positive matches at the risk of more false negatives.

# Note: This isn't exactly the same as a "percent match". The scale isn't linear. But you can assume that images with a
# smaller distance are more similar to each other than ones with a larger distance.

# Load some images to compare against
root = "/data/hanxinyang/MuNeRF_latest/process_data_hy/face_recognition/examples/data"
raw_videos_path = "/data/hanxinyang/MuNeRF_latest/process_data_hy/face_recognition/examples/data/raw_videos"
raw_frames_path = "/data/hanxinyang/MuNeRF_latest/process_data_hy/face_recognition/examples/data/raw_frames"
selected_frames_path = "/data/hanxinyang/MuNeRF_latest/process_data_hy/face_recognition/examples/data/selected_frames"
selected_videos_path = "/data/hanxinyang/MuNeRF_latest/process_data_hy/face_recognition/examples/data/selected_videos"

# for  video_id in os.listdir(raw_videos_path):
#     # fetch video
#     # video_id = str(video_id + 1)
#     print("Dividing video : %s"%video_id)
#     video_path = os.path.join(raw_videos_path, video_id)

#     '''视频转图片'''
#     if not os.path.exists(os.path.join(raw_frames_path, video_id)):
#         os.mkdir(os.path.join(raw_frames_path, video_id))
#         os.mkdir(os.path.join(raw_frames_path, video_id, "frames"))
#         os.mkdir(os.path.join(raw_frames_path, video_id, "target"))

#     cap=cv.VideoCapture(video_path) #加载视频
#     isOpened=cap.isOpened()
#     i=0
#     while(isOpened):
#         i=i+1
#         flag,frame=cap.read()
#         fileName = '%05d'%i+".jpg"
#         if flag == True :
#             cv.imwrite(os.path.join(raw_frames_path, video_id, "frames", fileName),frame) # 命名 图片 图片质量，此处文件名必须以图片格式结尾命名
#             # pdb.set_trace()
#         else:
#             break
#     cap.release()
#     print("Finished dividing for video : %s"%video_id)

# pdb.set_trace()
# Selecting faces from frames
for video_id in os.listdir(raw_videos_path):
    target_img_path = os.path.join(raw_frames_path, video_id, 'target', '%s.png'%video_id[:-4])
    if not os.path.exists(target_img_path):
        continue
    # pdb.set_trace()

    target_img = face_recognition.load_image_file(target_img_path)

    # Get the face encodings for the known images
    target_face_encoding = face_recognition.face_encodings(target_img)[0]

    known_encodings = [
        target_face_encoding
    ]

    save_path = os.path.join(selected_frames_path, video_id)

    for count, img_name in enumerate(os.listdir(os.path.join(raw_frames_path, video_id, 'frames'))):
        try:
            # pdb.set_trace()

            img_path = os.path.join(raw_frames_path, video_id, 'frames', img_name)
            image_to_test = face_recognition.load_image_file(img_path)
            # pdb.set_trace()
            image_to_test_encoding = face_recognition.face_encodings(image_to_test)[0]

            # See how far apart the test image is from the known faces
            face_distances = face_recognition.face_distance(known_encodings, image_to_test_encoding)
            face_distance = face_distances[0]
            if face_distance < 0.6:
                cv.imwrite(os.path.join(save_path, "%05d.jpg"%count), image_to_test)
        except Exception as e:
            print(e)
            print("Video Id : %s, index : %d"%(video_id, count))
