# CUDA_VISIBLE_DEVICES=3 python ./train_transformed_rays_hy.py --config ./config/girl7/w_beautyloss_global_patchgan.yml  --load_checkpoint /data/hanxinyang/MuNeRF_latest/logs/girl7_style/girl7_10070_gengirl4/checkpoint838399.ckpt --color_continue_train --debug_dir ./debug/debug_girl7_crosscat_00120_big_new 

# CUDA_VISIBLE_DEVICES=3 python ./eval_transformed_rays.py --config ./config/girl7/w_beautyloss_global_patchgan.yml --checkpoint /data/hanxinyang/MuNeRF_latest/logs/girl7_style/girl7_10070_gengirl4/checkpoint838399.ckpt --savedir ./rendering/name/

cd /data/hanxinyang/MuNeRF_latest
CUDA_VISIBLE_DEVICES=2 python ./eval_transformed_rays.py --config /data/hanxinyang/MuNeRF_latest/config/girl7/w_beautyloss_global_patchgan.yml --checkpoint /data/hanxinyang/MuNeRF_latest/logs/girl7_style/girl7_10020_gengirl4/checkpoint320900.ckpt --savedir ./rendering/girl7_makeup/