import os

targets = ['00005', '00030', '00053', '00160', '21013']

for target in targets:
    os.system(f"CUDA_VISIBLE_DEVICES=2 python /data/hanxinyang/MuNeRF_latest/eval_transformed_rays.py --config /data/hanxinyang/MuNeRF_latest/config/boy4/w_beautyloss_global_patchgan_{target}.yml  --checkpoint /data/hanxinyang/MuNeRF_latest/logs/boy4_style_{target}/boy4_{target}/checkpoint965099.ckpt --savedir ./rendering/boy4_maekup_{target} ")
    