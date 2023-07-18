import argparse
from pathlib import Path

from PIL import Image
from psgan import Inference
from fire import Fire
import numpy as np
import os
import faceutils as futils
from psgan import PostProcess
from setup import setup_config, setup_argparser


def main(save_path='transferred_image.png'):
    parser = setup_argparser()
    parser.add_argument(
        "--source_dir",
        default="./assets/images/non-makeup/",
        metavar="FILE",
        help="path to source image")
    parser.add_argument(
        "--reference_path",
        default="assets/images/makeup/1.png",
        help="path to reference images")
    parser.add_argument(
        "--save_path",
        default="assets/images/makeup/",
        help="path to reference images")
    parser.add_argument(
        "--speed",
        action="store_true",
        help="test speed")
    parser.add_argument(
        "--device",
        default="cpu",
        help="device used for inference")
    parser.add_argument(
        "--model_path",
        default="assets/models/G.pth",
        help="model for loading")

    args = parser.parse_args()
    config = setup_config(args)

    # Using the second cpu
    inference = Inference(
        config, args.device, args.model_path)
    postprocess = PostProcess(config)

    reference = Image.open(args.reference_path).convert("RGB")
    source_paths = sorted(os.listdir(args.source_dir))
    #np.random.shuffle(reference_paths)
    for source_path in source_paths:
        source = Image.open(os.path.join(args.source_dir,source_path)).convert("RGB")

        # Transfer the psgan from reference to source.
        image ,face= inference.transfer(source, reference, with_face=True)
        source_crop = source.crop(
            (face.left(), face.top(), face.right(), face.bottom()))
        #image = postprocess(source_crop, image)
        image.save(args.save_path+source_path)


if __name__ == '__main__':
    main()
