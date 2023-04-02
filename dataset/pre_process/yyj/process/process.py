import os

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--id', type=str, default='', help='path to the no_makeup dir')
args = parser.parse_args()
id = args.id
print("Start producing json.\n")
# os.system(f"cd /data/hanxinyang/MuNeRF_latest/process_data_hy/pre_process/yyj/process/process.py &&\
#            CUDA_VISIBLE_DEVICES=3 /data/hanxinyang/miniconda3/envs/nerf/bin/python data_util/process_data.py --dataset=/data/hanxinyang/MuNeRF_latest/dataset --id={id} --step=2 &&\
#            CUDA_VISIBLE_DEVICES=3 /data/hanxinyang/miniconda3/envs/nerf/bin/python data_util/process_data.py --dataset=/data/hanxinyang/MuNeRF_latest/dataset --id={id} --step=6 &&\
#            CUDA_VISIBLE_DEVICES=3 /data/hanxinyang/miniconda3/envs/nerf/bin/python data_util/process_data.py --dataset=/data/hanxinyang/MuNeRF_latest/dataset --id={id} --step=3 &&\
#            CUDA_VISIBLE_DEVICES=3 /data/hanxinyang/miniconda3/envs/nerf/bin/python data_util/process_data.py --dataset=/data/hanxinyang/MuNeRF_latest/dataset --id={id} --step=4 &&\
#            CUDA_VISIBLE_DEVICES=3 /data/hanxinyang/miniconda3/envs/nerf/bin/python data_util/process_data.py --dataset=/data/hanxinyang/MuNeRF_latest/dataset --id={id} --step=7")


os.system(f"cd /data/hanxinyang/MuNeRF_latest/process_data_hy/pre_process/yyj/process/process.py &&\
           CUDA_VISIBLE_DEVICES=3 /data/hanxinyang/miniconda3/envs/nerf/bin/python data_util/process_data.py --dataset=/data/hanxinyang/MuNeRF_latest/dataset --id={id} --step=6 &&\
           CUDA_VISIBLE_DEVICES=3 /data/hanxinyang/miniconda3/envs/nerf/bin/python data_util/process_data.py --dataset=/data/hanxinyang/MuNeRF_latest/dataset --id={id} --step=7")





