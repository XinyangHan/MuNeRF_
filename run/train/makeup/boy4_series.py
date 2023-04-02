import os

targets = ['00005', '00030', '00053', '00160', '21013']

for target in targets:
    os.system(f"CUDA_VISIBLE_DEVICES=1 python ./train_transformed_rays_hy.py --config /home/yuanyujie/cvpr23/config/boy4/w_beautyloss_global_patchgan_{target}.yml  --load_checkpoint /home/yuanyujie/cvpr23/logs/finals/nerface/boy4_checkpoint962100.ckpt --debug_dir ./debug/debug_boy4_{target} ")