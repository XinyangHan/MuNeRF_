CUDA_VISIBLE_DEVICES=0 python ./train_transformed_rays_hy.py --config ./config/girl22/w_beautyloss_global_patchgan_34016.yml  --load_checkpoint /home/yuanyujie/makeupnerf/logs/girl22_density/checkpoint512000.ckpt --debug_dir ./debug/debug_girl22_34016 