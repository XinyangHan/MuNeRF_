import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_pattern", default="./checkpoints/pattern.pth", type=str)
    parser.add_argument("--checkpoint_color", default="./checkpoints/color.pth", type=str)
    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument('--batch_size', default = '1', type = int)
    parser.add_argument("--prn", default=True, type=bool)
    parser.add_argument("--color_only", default=False, action="store_true")
    parser.add_argument("--pattern_only", default=False, action="store_true")

    parser.add_argument(
        "--input",
        type=str,
        default="./imgs/non-makeup.png",
        help="Path to input image (non-makeup)",
    )
    parser.add_argument(
        "--style",
        type=str,
        default="./imgs/style-2.png",
        help="Path to style image (makeup style | reference image)",
    )
    parser.add_argument("--alpha", type=float, default=0.5, help="opacity of color makeup")
    parser.add_argument("--savedir", type=str, default="./")

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    get_args()
