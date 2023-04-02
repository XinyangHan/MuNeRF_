"""Demo code showing how to estimate human head pose.

There are three major steps:
1. Detect human face in the video frame.
2. Run facial landmark detection on the face image.
3. Estimate the pose by solving a PnP problem.

To find more details, please refer to:
https://github.com/yinguobing/head-pose-estimation
"""
from argparse import ArgumentParser

import cv2
import torch
import os
import json

from mark_detector import MarkDetector
from pose_estimator import PoseEstimator

print(__doc__)
print("OpenCV version: {}".format(cv2.__version__))

# Parse arguments from user input.
parser = ArgumentParser()
parser.add_argument("--video", type=str, default=None,
                    help="Video file to be processed.")
parser.add_argument("--frames", type=str, default="/mnt/d/heyue/nerf-illumination/input/",
                    help="Video file to be processed.")
parser.add_argument("--cam", type=int, default=None,
                    help="The webcam index.")
parser.add_argument("--size", type=int, default=512,
                    help="The frame size.")
parser.add_argument("--json", type=str, default='/mnt/d/heyue/nerf-illumination/jsons/pose.json', 
                    help="Path to save transform matrixes")
args = parser.parse_args()

def euler2rot(euler_angle):
    batch_size = euler_angle.shape[0]
    theta = euler_angle[:, 0].reshape(-1, 1, 1)
    phi = euler_angle[:, 1].reshape(-1, 1, 1)
    psi = euler_angle[:, 2].reshape(-1, 1, 1)
    one = torch.ones((batch_size, 1, 1), dtype=torch.float32,
                     device=euler_angle.device)
    zero = torch.zeros((batch_size, 1, 1), dtype=torch.float32,
                       device=euler_angle.device)
    rot_x = torch.cat((
        torch.cat((one, zero, zero), 1),
        torch.cat((zero, theta.cos(), theta.sin()), 1),
        torch.cat((zero, -theta.sin(), theta.cos()), 1),
    ), 2)
    rot_y = torch.cat((
        torch.cat((phi.cos(), zero, -phi.sin()), 1),
        torch.cat((zero, one, zero), 1),
        torch.cat((phi.sin(), zero, phi.cos()), 1),
    ), 2)
    rot_z = torch.cat((
        torch.cat((psi.cos(), -psi.sin(), zero), 1),
        torch.cat((psi.sin(), psi.cos(), zero), 1),
        torch.cat((zero, zero, one), 1)
    ), 2)
    return torch.bmm(rot_x, torch.bmm(rot_y, rot_z))

if __name__ == '__main__':
    # Before estimation started, there are some startup works to do.

    # 1. Setup the video source from webcam or video file.
    #video_src = args.cam if args.cam is not None else args.video
    #if video_src is None:
    #    print("Video source not assigned, default webcam will be used.")
    #    video_src = 0
    #input_dir = '/mnt/d/heyue/nerf-illumination/head-pose-estimation-master/dave/'

    #cap = cv2.VideoCapture(video_src)
    input_dir = args.frames
    #input_dir_train = input_dir + 'train/'
    #input_dir_test = input_dir + 'test/'
    #input_dir_val = input_dir + 'val/'
    frames = {}
    for image_path in sorted(os.listdir(input_dir)):
      if image_path.endswith('.jpg') or image_path.endswith('.png'):
          frames[image_path] = os.path.join(input_dir, image_path)
    '''for image_path in sorted(os.listdir(input_dir_test)):
      if image_path.endswith('.jpg') or image_path.endswith('.png'):
          frames[image_path] = os.path.join(input_dir_test, image_path)
    for image_path in sorted(os.listdir(input_dir_val)):
      if image_path.endswith('.jpg') or image_path.endswith('.png'):
          frames[image_path] = os.path.join(input_dir_val, image_path)'''
                
    # Get the frame size. This will be used by the pose estimator.
    #width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    #height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    height = width = args.size
    # 2. Introduce a pose estimator to solve pose.
    pose_estimator = PoseEstimator(img_size=(height, width))
    
    # 3. Introduce a mark detector to detect landmarks.
    mark_detector = MarkDetector()

    # 4. Measure the performance with a tick meter.
    tm = cv2.TickMeter()
    
    i = 0
    framekey =  list(frames.keys())
    maxnum = len(framekey)
    # Now, let the frames flow.
    posedict = {}
    filtering = []
    jsonfile_pose = args.json
    while i<maxnum:
        framename = framekey[i]
        framepath = frames[framename]
        frame = cv2.imread(framepath)
        # Read a frame.
        #frame_got, frame = cap.read()
        #if frame_got is False:
        #    break

        # If the frame comes from webcam, flip it so it looks like a mirror.
        #if video_src == 0:
        #    frame = cv2.flip(frame, 2)

        # Step 1: Get a face from current frame.
        facebox = mark_detector.extract_cnn_facebox(frame)

        # Any face found?
        if facebox is None:
            filtering.append(framename)
            print('framename0', framename)
        if facebox is not None:

            # Step 2: Detect landmarks. Crop and feed the face area into the
            # mark detector.
            x1, y1, x2, y2 = facebox
            face_img = frame[y1: y2, x1: x2]

            # Run the detection.
            tm.start()
            marks = mark_detector.detect_marks(face_img)
            tm.stop()

            # Convert the locations from local face area to the global image.
            marks *= (x2 - x1)
            marks[:, 0] += x1
            marks[:, 1] += y1

            # Try pose estimation with 68 points.
            euler_angle,  trans= pose_estimator.solve_pose_by_68_points(marks)
            # [1,3]
            trans = torch.from_numpy(trans) / int(frame.shape[1])
            euler_angle = torch.from_numpy(euler_angle).unsqueeze(2).unsqueeze(0) # [3,1,1]
            rot = euler2rot(euler_angle)    # ！
            rot_inv = rot.permute(0, 2, 1)   # ！

            trans_inv = -torch.bmm(rot_inv, trans.unsqueeze(0))   # ！
            pose = torch.eye(4, dtype=torch.float32)  # ！
            pose[:3, :3] = rot_inv
            pose[:3, 3] = trans_inv[:, 0]
            #print('process a frame pose',i, pose)
            posedict[framename] = pose.numpy().tolist()
        i+=1
            # All done. The best way to show the result would be drawing the
            # pose on the frame in realtime.

            # Do you want to see the pose annotation?
            #pose_estimator.draw_annotation_box(
            #    frame, pose[0], pose[1], color=(0, 255, 0))

            # Do you want to see the head axes?
            # pose_estimator.draw_axes(frame, pose[0], pose[1])

            # Do you want to see the marks?
            # mark_detector.draw_marks(frame, marks, color=(0, 255, 0))

            # Do you want to see the facebox?
            # mark_detector.draw_box(frame, [facebox])

        # Show preview.
        #cv2.imshow("Preview", frame)
        #if cv2.waitKey(1) == 27:
        #    break
    posedict['filters'] = filtering
    with open(jsonfile_pose, "w") as f:
        json.dump(posedict,f,indent=1)
    print('write pose json sucessfully')