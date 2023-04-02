import os

targets = ['00005', '00030', '00053', '00160', '21013']
# targets = ['00005']


for target in targets:
    os.system(f"CUDA_VISIBLE_DEVICES=0 python ./train_transformed_rays_hy.py --config /data/hanxinyang/MuNeRF_latest/config/boy4/w_beautyloss_global_patchgan_{target}.yml  --load_checkpoint /data/hanxinyang/MuNeRF_latest/logs/boy4_density/checkpoint962100.ckpt --debug_dir ./debug/debug_boy4_{target} ")