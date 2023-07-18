import faceutils as futils
from smart_path import smart_path
from pathlib import Path
from PIL import Image
import fire
import numpy as np
import tqdm
from collections import defaultdict
from faceutils.mask import FaceParser
import pickle
from concern.visualize import mask2image
from concern.image import resize_by_max
import cv2
import argparse

import pdb
def main(
    image_dir="./MT-Dataset/hy/makeup",
    out_dir="./MT-Dataset/parsing/makeup",
    show=False):
    """
    dirs can also be S3 path such as s3://a/bc/
    """
    image_dir = smart_path(image_dir)
    out_dir = smart_path(out_dir)

    face_parser = FaceParser(device="cuda")

    for image_path in tqdm.tqdm(image_dir.rglob("*")):
        if not image_path.is_file():
            continue

        sub_dir = image_path.parent.name
        file_name = image_path.name
        out_file = out_dir.joinpath(sub_dir, file_name)

        if not out_file.parent.exists():
            out_file.parent.mkdir(parents=True, exist_ok=True)

        try:
            image = Image.open(image_path.open("rb"))
        except:
            continue
        np_image = np.asarray(image)

        with out_file.open("wb") as writer:
            mask = face_parser.parse(cv2.resize(np_image, (512, 512)))
            # pdb.set_trace()
            mask = mask.cpu().numpy().astype(np.uint8)

            if show:
                cv2.imshow(image_path.as_uri(), np.asarray(image))
                cv2.imshow(out_file.as_uri(), mask2image(mask))
                cv2.waitKey()

            Image.fromarray(mask).save(writer, "PNG")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source_dir",
        default="./MT-Dataset/hy/makeup",
        metavar="FILE",
        help="path to source image")
    parser.add_argument(
        "--save_dir",
        default="./MT-Dataset/parsing",
        metavar="FILE",
        help="path to source image")
    args =  parser.parse_args()
    main(args.source_dir, args.save_dir)

