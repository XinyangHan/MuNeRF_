import cv2, os, pickle
from glob import glob
import numpy as np
from shutil import copyfile
import time
import argparse
from pprint import pprint

from helpers import FaceCropper, cropFaceBBox, combineApiLandmark
from api_util import FacePPAPI
from constants import *

import tqdm
import pdb

fc = FaceCropper(predictor_dir = 'data/shape_predictor_68_face_landmarks.dat',)
api = FacePPAPI()

'''
arguments:
    --type:
        copy:
            Copy the images to a new location (for renaming)
        filter:
            Filter the images by landmark detection
        crop:
            Only to crop the before/after makeup images
        blend:
            Only to blend the images given the before/after makeup images without aligning/croping them
        crop_and_blend:
            First blend the before/after makeup images and crop every image
        mask:
            Generate the facial area mask for the images
        landmark:
            Detect the landmarks using API for the img files in the input root, and store the landmarks as a pickle file according to the given output path
            the output should be a path to the pickle file to store
        pcg:
            Process the all images in a dataroot in the same way as PairedCycleGAN
            input should be a dataroot containing sub-folders [before, after, blend] of cropped and blended filterImagesByLandmarks
            output will be structured as ['eye', 'mouth', 'skin', 'face']

usage:
    python handle_dataset.py --type copy --input ./DRIT/datasets/makeup_new/iccv_new/raw/ --output ./DRIT/datasets/makeup_new/iccv_new/rename --rename_starting 40000 --rename 1

    python handle_dataset.py --type crop --input ./DRIT/datasets/makeup_new/iccv_new/rename/ --output ./DRIT/datasets/makeup_new/iccv_new/crop
'''

def main():
    parser = argparse.ArgumentParser(description='To handle which dataset and how to handle it.')
    parser.add_argument("--name", type=str, default="filter", help="how to handle the dataset")

    # Basic arguments
    parser.add_argument("--type", type=str, choices=['pcg', 'copy', 'filter', 'crop', 'blend1', 'blend', 'crop_and_blend', 'mask', 'landmark'], default="filter", help="how to handle the dataset")
    parser.add_argument("--input", type=str, help="the root of the original dataset")
    parser.add_argument("--output", type=str, help="the root to store/find the filtered images by landmark detection")

    # arguments for filter and copy
    parser.add_argument("--rename", action="store_true", help="whether to rename the images files to a ordering numbers when type is filter (default 0)")
    parser.add_argument("--rename_starting", type=int, default=0, help="starting number of the renaming files (default 0)")

    # arguments for crop
    parser.add_argument("--expand", type=float, default=1.2, help="ratio to expand the croped facial region")

    # arguments for blend
    parser.add_argument("--detector", type=str, default="api", choices=["stasm", "dlib", "api"], help="which facial landmark detector to use")
    parser.add_argument("--landmark_input", type=str, help="the path to the pickle file storing the facial landmarks detected by API")
    parser.add_argument("--keep_eye_mouth", action="store_true", help="whether to keep the inner mouth and inner eye regions the same as the before makeup image")
    parser.add_argument("--include_forehead", action="store_true", help="whether to include the landmarks on the forehead from stasm landmark detector for wapring and blending")
    parser.add_argument("--adjust_color", action="store_true", help="whether to adjust the color by hue manipulation during blending")
    parser.add_argument("--adjust_lighting", action="store_true", help="whether to adjust the lighting by value manipulation during blending")

    # arguments for crop and blend
    parser.add_argument("--refer_result", type=int, default=0, choices=[0,1], help="Whether to refer to the existing results in crop folder when deciding which images to crop/blend (default 0)")
    parser.add_argument("--no_blend_overwrite", action="store_true", help="if set, the blended images that already exist will not be generated again")

    # arguments for mask
    parser.add_argument("--makeup_region_weight", type=float, default=0.5, help="weight of the major makeup region when generating mask")
    parser.add_argument("--facial_region_weight", type=float, default=1.0, help="weight of the facial region when generating mask")
    parser.add_argument("--background_region_weight", type=float, default=0.0, help="weight of the background region when generating mask")
    args = parser.parse_args()

    if args.type in ["filter", "copy"]:
        filterImagesByLandmarks(args)
    elif args.type == 'crop_and_blend':
        before_files, after_files = getFilesWithReference(args)
        print("before images: %d" % len(before_files))
        print("after images: %d" % len(after_files))
        cropAndBlend(before_files, after_files, args)
    elif args.type == 'crop':
        crop(args)
    elif args.type == 'blend1':
        print("blend")
        before_files, after_files = getFilesWithReference(args)
        print("before images: %d" % len(before_files))
        print("after images: %d" % len(after_files))
        # pdb.set_trace()
        # blend(before_files, after_files, args, True)
        # pdb.set_trace()
        blend(before_files, after_files, args, True, args.name)
        # color(before_files, args)
        # blend(before_files, after_files, args, 0)
        blend(before_files, after_files, args, 0, args.name)
    elif args.type == 'blend':
        before_files, after_files = getFilesWithReference(args)
        print("before images: %d" % len(before_files))
        print("after images: %d" % len(after_files))
        color(before_files, args)
        blend(before_files, after_files, args, True)
        blend(before_files, after_files, args, False)

    elif args.type == "mask":
        generateFaceMasks(args)
    elif args.type == "landmark":
        print("ldmk")
        facePPDetectAndStore(args.input, args.output)
    elif args.type == "pcg":
        processAsPCG(args)

# path is to a file
def assureFolderExist(path):
    folder = "/".join(path.split('/')[:-1])
    if not os.path.exists(folder):
        os.makedirs(folder)

def facecrop(img):
    #img = cv2.imread(path)
    cropimg = img[8:,150:510]
    resize_img = cv2.resize(cropimg, (512,512))
    #cv2.imwrite(path,cropimg)
    return resize_img

def filterImagesByLandmarks(args):
    input = args.input
    output = args.output
    # https://stackoverflow.com/questions/19309667/recursive-os-listdir
    input_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(input)) for f in fn]
    input_files = [f[len(input):] for f in input_files]

    for i, file in enumerate(input_files):
        src = os.path.join(input, file)

        if args.type == "filter":
            img = cv2.imread(src)
            if args.detector == "stasm":
                #print('test img', img.shape, img)
                landmark = fc.facePointsStasm(img)
                if np.array(landmark).size == 0:
                    print("landmarks fails on %s" % src)
                    continue
            else:
                landmark = fc.facePoints(img)
                raise NotImplementedError()

        if args.rename:
            dst = os.path.join(output, "/".join(file.split("/")[:-1]), "%05d.png" % (args.rename_starting + i))
        else:
            dst = os.path.join(output, file)

        assureFolderExist(dst)

        copyfile(src, dst)

def getFilesWithReference(args):
    input = args.input
    output = args.output
    # pdb.set_trace()
    input_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(input)) for f in fn]
    input_files = [f[len(input):] for f in input_files]
    before_files = [os.path.join(input, f[1:]) for f in input_files if "before" in f]
    after_files = [os.path.join(input, f[1:]) for f in input_files if "after" in f]
    if args.refer_result:
        bf = []
        af = []
        for f in before_files:
            o = os.path.join(output, "before", f.split("/")[-1])
            if os.path.exists(o):
                bf.append(f)
        for f in after_files:
            o = os.path.join(output, "after", f.split("/")[-1])
            if os.path.exists(o):
                af.append(f)
        before_files = bf
        after_files = af

    return before_files, after_files

def color(before_files, args):
    sum  = np.zeros(3)
# pdb.set_trace()

    after_source = "/data/hanxinyang/MuNeRF_latest/process_data_hy/makeup-transfer-public/dataH/crop/blend"
    color_store = "/data/hanxinyang/MuNeRF_latest/process_data_hy/makeup-transfer-public/dataH/color_store.png"
    # pdb.set_trace()
    
    count=0
    for bi, bd in enumerate(os.listdir(after_source)):
        count =  bi
        input = args.input

        api_landmarks = pickle.load(open(args.landmark_input, 'rb'))

        # pdb.set_trace()
        photo = bd.split('_')[0] + '.png'
        lm_path = input + 'before/' + photo
        pdb.set_trace()
        lm_api = api_landmarks[lm_path[len(input):]]
        img = cv2.imread(os.path.join(after_source, bd))
        
        # pdb.set_trace()
        left = lm_api['faces'][0]['landmark']['left_eyebrow_lower_right_corner']
        right =  lm_api['faces'][0]['landmark']['right_eyebrow_lower_left_corner']
        leftx, lefty = left['x'], left['y']
        rightx, righty = right['x'], right['y']
        targetx = int((leftx + rightx)/2)
        targety = int((lefty + righty)/2)
        sum += img[targetx][targety]

        # cv2.circle(img, (leftx, lefty), 10, (0, 0, 255), 8)
        # cv2.circle(img, (rightx, righty), 10, (0, 0, 255), 8)
        # cv2.circle(img, (targetx, targety), 10, (0, 0, 255), 8)

        # cv2.imwrite("/data/hanxinyang/MuNeRF_latest/process_data_hy/makeup-transfer-public/face-parsing.PyTorch-master/debugH/color.png", img)
        # cv2.imwrite("/data/hanxinyang/MuNeRF_latest/process_data_hy/makeup-transfer-public/face-parsing.PyTorch-master/debugH/color_little.png", img[targetx][targety])

        # pdb.set_trace()
    color = sum / (count + 1)

    # pdb.set_trace()

    cv2.imwrite(color_store, color)

def blend(before_files, after_files, args, first, name=""):
    input = args.input
    output = args.output

    if args.detector == "api":
        api_landmarks = pickle.load(open(args.landmark_input, 'rb'))
        # print(api_landmarks.keys())
        if len(api_landmarks.keys()) >= len(before_files) + len(after_files):
            print("API landmarks loaded successfully!")
    # pdb.set_trace()

    for bd in before_files:
        try:
            lm = api_landmarks[bd[len(input)+1:]]
            lm['faces'][0]
        except Exception as e:
            print(bd)
    for ad in after_files:
        try:
            lm = api_landmarks[bd[len(input)+1:]]
            lm['faces'][0]
        except Exception as e:
            print(ad)

    if first == True:
        color_store = "/data/hanxinyang/MuNeRF_latest/process_data_hy/makeup-transfer-public/dataH/color_raw.png"
    else:
        color_store = "/data/hanxinyang/MuNeRF_latest/process_data_hy/makeup-transfer-public/dataH/color_store.png"

    # color = cv2.imread(color_store).squeeze()[0]
    


    counter = 0
    total = len(before_files) * len(after_files)
    # pdb.set_trace()

    for bi, bd in enumerate(before_files):

        index_mask = bd.split("/")[-1]
        # pdb.set_trace()

        b = cv2.imread(bd)
        try:
            if args.detector == "stasm":
                landmark_b = fc.facePointsStasm(b)
            elif args.detector == "api":
                lm_stasm = fc.facePointsStasm(b)
                lm_api = api_landmarks[bd[len(input)+1:]]
                # pdb.set_trace()
                landmark_b = combineApiLandmark(lm_stasm, lm_api, include_forehead=args.include_forehead)
        except Exception as e:
            print(e)
            print("fail on %s" % bd)
            continue

        before_source = input + "/before"
        # before_source = "/data/hanxinyang/MuNeRF_latest/process_data_hy/makeup-transfer-public/dataH/crop/before"
        # after_source = "/data/hanxinyang/MuNeRF_latest/process_data_hy/makeup-transfer-public/dataH/crop/after"
        after_source = input + "/after"
        # color_store = input + "color_store.png"
        # color_store = "/data/hanxinyang/MuNeRF_latest/process_data_hy/makeup-transfer-public/dataH/color_store.png"

        if bi == 0:
            # photo = bd.split('_')[0] + '.png'
            # pdb.set_trace()
            # lm_path = input + 'before/' + photo
            lm_api = api_landmarks[bd[len(input)+1:]]
            if first == True:
                # pdb.set_trace()
                img = cv2.imread(os.path.join(before_source, bd[len(before_source)+1:]))
                # img = cv2.imread(os.path.join(before_source, bd[len(input)+1:]))
            else:
                img = cv2.imread(os.path.join((after_source),os.listdir(after_source)[0]))
                # pdb.set_trace()
            
            left = lm_api['faces'][0]['landmark']['left_eyebrow_lower_right_corner']
            right =  lm_api['faces'][0]['landmark']['right_eyebrow_lower_left_corner']
            leftx, lefty = left['x'], left['y']
            rightx, righty = right['x'], right['y']
            targetx = int((leftx + rightx)/2)
            targety = int((lefty + righty)/2)
            # sum = img[targetx][targety]
            # pdb.set_trace()

            # cv2.circle(img, (leftx, lefty), 10, (0, 0, 255), 8)
            # cv2.circle(img, (rightx, righty), 10, (0, 0, 255), 8)
            # cv2.circle(img, (targetx, targety), 10, (0, 0, 255), 8)

            # cv2.imwrite("/data/hanxinyang/MuNeRF_latest/process_data_hy/makeup-transfer-public/face-parsing.PyTorch-master/debugH/color.png", img)
            # cv2.imwrite(color_store, img[targetx][targety])
    
        # color = cv2.imread(color_store).squeeze()
    # pdb.set_trace()

        # color = [color[0][0], color[1][0], color[2][0]]
        # color_souce = cv2.imread(bd)
        # cv2.circle(color_souce, (leftx, lefty), 10, (0, 0, 255), 8)
        # cv2.circle(color_souce, (rightx, righty), 10, (0, 0, 255), 8)

        # cv2.imwrite("/data/hanxinyang/MuNeRF_latest/process_data_hy/makeup-transfer-public/face-parsing.PyTorch-master/debugH/color.png", color_souce)
        # cv2.imwrite("/data/hanxinyang/MuNeRF_latest/process_data_hy/makeup-transfer-public/face-parsing.PyTorch-master/debugH/color_little.png", color_souce[targetx][targety])


        # Get the mask for inner mouth and eye regions
        if args.keep_eye_mouth:
            before_inner_mask = fc.innerMouthEyeMask(b, args.detector, lm_api)
        # pdb.set_trace()
        for ai, ad in enumerate(after_files):
            counter += 1

            # dst_blended = os.path.join(output, "%s_%s.png" % (
            #     bd.split("/")[-1].split(".")[0],
            #     ad.split("/")[-1].split(".")[0]
            # ))
            # dst_blended = os.path.join(output, "%s_%s.png" % (ad.split("/")[-1].split(".")[0], name))
            dst_blended = os.path.join(output, "%s_%s.png" % (bd.split("/")[-1].split(".")[0], name))
            # pdb.set_trace()
            if args.no_blend_overwrite and os.path.isfile(dst_blended):
                continue

            if counter % 10 == 0:
                print("Working on %d/%d images" % (counter, total))

            a = cv2.imread(ad)
            try:
                if args.detector == "stasm":
                    landmark_a = fc.facePointsStasm(a)
                elif args.detector == "api":
                    lm_stasm = fc.facePointsStasm(a)
                    # pdb.set_trace()
                    lm_api = api_landmarks[ad[len(input)+1:]]
                    landmark_a = combineApiLandmark(lm_stasm, lm_api, include_forehead=args.include_forehead)
            except Exception as e:
                print(e)
                print("fail on %s" % ad)
                continue


            index = int(bd.split("/")[-1].split(".")[0])
            mask_path = os.path.join("/data/hanxinyang/MuNeRF_latest/process_data_hy/makeup-transfer-public/face-parsing.PyTorch-master/res/test_res1", "%d.png"%index)
            # mask = cv2.imread(mask_path)
            # pdb.set_trace()

            # print(b.shape)
            # for i in range(b.shape[0]):
            #     for j in range(b.shape[1]):
            #         for k in range(b.shape):
            #             if b[i][j][k] > 0:
            #                 b

            # pdb.set_trace()
            try:
                blended, _, flag = fc.warpFace(
                    b, a, 
                    before_points = landmark_b, after_points = landmark_a,
                    use_poisson = False,
                    use_stasm = True,
                    additional_gaussian = False,
                    clone_method = cv2.NORMAL_CLONE,
                    use_tps = False,
                    extra_blending_for_extreme = args.adjust_color,
                    hue_threshold = 0.15,
                    extra_blending_weight = 0.6,
                    adjust_value = args.adjust_lighting,
                    index=index_mask
                )
                # cv2.imwrite(dst_blended, b)
                # pdb.set_trace()
                
                if flag:
                    if args.keep_eye_mouth:
                        # blended = b * before_inner_mask + blended * (1 - before_inner_mask)
                        blended = b * before_inner_mask + blended * (1 - before_inner_mask)


                    assureFolderExist(dst_blended)
                    cv2.imwrite(dst_blended, blended)
            except Exception as e:
                print(e)
                continue
        if first == True:
            break
        

def cropAndBlend(before_files, after_files, args):
    input = args.input
    output = args.output

    for d in [os.path.join(output, "before"), os.path.join(output, "after"), os.path.join(output, "blend")]:
        if not os.path.exists(d):
            os.makedirs(d)

    if args.detector == "api":
        api_landmarks = pickle.load(open(args.landmark_input, 'rb'))
        if len(api_landmarks.keys()) == len(before_files) + len(after_files):
            print("API landmarks loaded successfully!")

    counter = 0

    total = len(before_files) * len(after_files)

    for bi, bd in enumerate(before_files):
        b = cv2.imread(bd)
        try:
            b = fc.alignFace(b)
            if args.detector == "stasm":
                landmark_b = fc.facePointsStasm(b)
            elif args.detector == "api":
                lm_stasm = fc.facePointsStasm(b)
                lm_api = api_landmarks[bd[len(input)+1:]]
                landmark_b = combineApiLandmark(lm_stasm, lm_api)
        except Exception as e:
            print(e)
            print("fail on %s" % bd)
            continue

        for ai, ad in enumerate(after_files):
            counter += 1
            if counter % 10 == 0:
                print("Working on %d/%d images" % (counter, total))

            a = cv2.imread(ad)

            try:
                a = fc.alignFace(a)
                if args.detector == "stasm":
                    landmark_a = fc.facePointsStasm(a)
                elif args.detector == "api":
                    lm_stasm = fc.facePointsStasm(a)
                    lm_api = api_landmarks[bd[len(input)+1:]]
                    landmark_a = combineApiLandmark(lm_stasm, lm_api)
            except Exception as e:
                print(e)
                print("fail on %s" % bd)
                continue

            blended, _ = fc.warpFace(
                b, a,
                use_poisson = True,
                use_stasm = True,
                additional_gaussian = False,
                clone_method = cv2.NORMAL_CLONE,
                use_tps = False
            )

            if bi == 0:
                dst_a = os.path.join(output, "after", ad.split("/")[-1])
                a = cropFaceBBox(a, landmark_a, expand=args.expand)
                cv2.imwrite(dst_a, a)

            dst_blended = os.path.join(output, "blend", "%s_%s.png" % (
                bd.split("/")[-1].split(".")[0],
                ad.split("/")[-1].split(".")[0]
            ))
            blended = cropFaceBBox(blended, landmark_b, expand=args.expand)
            cv2.imwrite(dst_blended, blended)

        dst_b = os.path.join(output, "before", bd.split("/")[-1])
        b = cropFaceBBox(b, landmark_b, expand=args.expand)
        cv2.imwrite(dst_b, b)

'''def crop(args):
    input = args.input
    output = args.output

    input_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(input)) for f in fn]
    input_files = [f[len(input):] for f in input_files]

    for f in input_files:
        src = os.path.join(input, f)
        dst1 = os.path.join(output,'debug', f)
        dst = os.path.join(output, f)
        img = cv2.imread(src)
        print(src)

        try:
            #img = fc.alignFace(img)
            #cv2.imwrite(dst1, img)
            landmark = fc.facePointsStasm(img)
        except:
            print("fail on %s" % src)
            continue
        if img is None:
            continue

        img = cropFaceBBox(img, landmark, expand=args.expand)
        assureFolderExist(dst)
        cv2.imwrite(dst, img)'''

def crop(args):
    input = args.input
    output = args.output
    print('!',os.path.expanduser(input))
    before_input_files, after_input_files = [], []
    for dp, dn ,fn in os.walk(os.path.expanduser(input)):
        if dp.split('/')[-1] == 'after':
            for f in fn:
              after_input_files.append(os.path.join(dp, f))
        else:
            for f in fn:
              before_input_files.append(os.path.join(dp, f))

    before_input_files = [f[len(input):] for f in before_input_files]
    after_input_files = [f[len(input):] for f in after_input_files]
    print('after!!!!!', after_input_files)

    #input_files = [f[len(input):] for f in input_files]

    for f in before_input_files:
        src = os.path.join(input, f)
        dst = os.path.join(output, f)
        img = cv2.imread(src)
        print(src)

        try:
            img = fc.alignFace(img)
            landmark = fc.facePointsStasm(img)
        except:
            print("fail on %s" % src)
            continue
        if img is None:
            continue

        #img = facecrop(img)#cropFaceBBox(img, landmark, expand=args.expand)
        assureFolderExist(dst)
        cv2.imwrite(dst, img)

    for f in after_input_files:
        src = os.path.join(input, f)
        dst = os.path.join(output, f)
        img = cv2.imread(src)
        print(src)

        try:
            img = fc.alignFace(img)
            landmark = fc.facePointsStasm(img)
        except:
            print("fail on %s" % src)
            continue
        if img is None:
            continue

        #img = facecrop(img)#cropFaceBBox(img, landmark, expand=args.expand)
        assureFolderExist(dst)
        cv2.imwrite(dst, img)

def generateFaceMasks(args):
    input = args.input
    output = args.output

    input_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(input)) for f in fn]
    input_files = [f[len(input):] for f in input_files]

    for f in input_files:
        src = os.path.join(input, f)
        dst = os.path.join(output, f)
        assureFolderExist(dst)
        img = cv2.imread(src)
        mask = fc.faceMask(img[:,:,::-1], white = args.makeup_region_weight, grey = args.facial_region_weight, black = args.background_region_weight)
        mask = (mask * 255).astype(np.uint8)
        cv2.imwrite(dst, mask)

def facePPDetectAndStore(input, output):
    input_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(input)) for f in fn]
    input_files = [f[len(input):] for f in input_files]
    # pdb.set_trace()
    api = FacePPAPI()

    results = {}

    for f in input_files:
        print(os.path.join(input, f))
        if f[0] == '/':
            f = f[1:]
        img = cv2.imread(os.path.join(input, f))
        if "landmark" in f:
            continue
        while True:
            

            result = api.faceLandmarkDetector(os.path.join(input, f[1:] if f[0] == '/' else f))
            # pdb.set_trace()
            if "error_message" in result:
                print(result)
                continue
            else:
                break
        # pdb.set_trace()
        results[f] = result
        # time.sleep()

    pickle.dump(results, open(output, 'wb'))
    print("Landmark finished!")

def incrementAndPrintCounter(counter):
    counter += 1
    if counter % 100 == 0:
        print("counter:", counter)
    return counter

def processAsPCG(args):
    input = args.input
    output = args.output

    dir_first_layer = ['eye', 'eye', 'mouth', 'skin', 'face']
    dir_second_laryer = ['before', 'after', "blend"]

    for d1 in dir_first_layer:
        for d2 in dir_second_laryer:
            d = os.path.join(output, d1, d2)
            if not os.path.exists(d):
                os.makedirs(d)

    if args.detector == "api":
        api_landmarks = pickle.load(open(args.landmark_input, 'rb'))

    # Get all input files and divide them
    input_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(input)) for f in fn]
    input_files = [f[len(input):] for f in input_files]
    # Path here are all partial path
    before_files = [f for f in input_files if "before" in f]
    after_files = [f for f in input_files if "after" in f]
    blend_files = [f for f in input_files if "blend" in f]

    print("before images: %d" % len(before_files))
    print("after images: %d" % len(after_files))
    print("blend images: %d" % len(blend_files))
    counter = 0

    # Handle the before-makeup and blend images
    for i, before_partial_path in enumerate(before_files):
        # Handle the path and read the image
        before_file_name = before_partial_path.split("/")[-1].split(".")[0]
        before_full_path = os.path.join(input, before_partial_path)
        before_img = cv2.imread(before_full_path)

        # Read/Detect landmarks
        if args.detector == "stasm":
            landmarks = fc.facePointsStasm(before_img)
        # Handle the landmark since now landmarks are from different sources
        elif args.detector == "api":
            landmarks = {}
            lm_stasm = fc.facePointsStasm(before_img)
            lm_api = api_landmarks[before_partial_path]
            landmarks['all'] = combineApiLandmark(lm_stasm, lm_api)
            landmarks['left_eye'] = api.getLeftEye(lm_api)
            landmarks['right_eye'] = api.getRightEye(lm_api)
            landmarks['mouth'] = api.getMouth(lm_api)

        # Process the before makeup image
        counter = incrementAndPrintCounter(counter)
        parts, rects = fc.generateTrainingData(before_img, landmarks, landmark_type=args.detector)
        parts[1] = cv2.flip(parts[1], 1)

        for x, part in enumerate(parts + [before_img]):
            d = '%s%s/%s/' % (output, dir_first_layer[x], dir_second_laryer[0])
            d += '%s_%d.png' % (before_file_name, x)
            # print("Saving:",d)
            cv2.imwrite(d, part)

        # Process the blend images
        this_blend_files = [f for f in blend_files if "%s_"%before_file_name in f]
        for blend_partial_path in this_blend_files:
            blend_file_name = blend_partial_path.split("/")[-1].split(".")[0]
            blend_full_path = os.path.join(input, blend_partial_path)
            blend_img = cv2.imread(blend_full_path)

            counter = incrementAndPrintCounter(counter)
            parts, rects = fc.generateTrainingData(blend_img, landmarks, landmark_type=args.detector)
            parts[1] = cv2.flip(parts[1], 1)

            for x, part in enumerate(parts + [blend_img]):
                d = '%s%s/%s/' % (output, dir_first_layer[x], dir_second_laryer[2])
                d += '%s_%d.png' % (blend_file_name, x)
                # print("Saving:",d)
                cv2.imwrite(d, part)

    # Handle the after-makeup and blend images
    for i, after_partial_path in enumerate(after_files):
        # Handle the path and read the image
        after_file_name = after_partial_path.split("/")[-1].split(".")[0]
        after_full_path = os.path.join(input, after_partial_path)
        after_img = cv2.imread(after_full_path)

        # Read/Detect landmarks
        if args.detector == "stasm":
            landmarks = fc.facePointsStasm(after_img)
        # Handle the landmark since now landmarks are from different sources
        elif args.detector == "api":
            landmarks = {}
            lm_stasm = fc.facePointsStasm(after_img)
            lm_api = api_landmarks[after_partial_path]
            landmarks['all'] = combineApiLandmark(lm_stasm, lm_api)
            landmarks['left_eye'] = api.getLeftEye(lm_api)
            landmarks['right_eye'] = api.getRightEye(lm_api)
            landmarks['mouth'] = api.getMouth(lm_api)

        # Process the after makeup image
        try:
            counter = incrementAndPrintCounter(counter)
            parts, rects = fc.generateTrainingData(after_img, landmarks, landmark_type=args.detector)
            parts[1] = cv2.flip(parts[1], 1)

            for x, part in enumerate(parts + [after_img]):
                d = '%s%s/%s/' % (output, dir_first_layer[x], dir_second_laryer[1])
                d += '%s_%d.png' % (after_file_name, x)
                # print("Saving:",d)
                cv2.imwrite(d, part)
        except:
            print("Failed on %s." % after_full_path)

    print("Complete!")

if __name__ == "__main__":
    # pdb.set_trace()
    main()


'''
rects
images[np.arange(2), [[rects[0,0]:rects[0,1], rects[0,2]:rects[0,3]], [rects[0,0]:rects[0,1], rects[0,2]:rects[0,3]]]]
'''
