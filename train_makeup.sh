CUDA_VISIBLE_DEVICES=2 python ./train_transformed_rays_hy.py --config ./config/$1/w_beautyloss_global_patchgan.yml  --load_checkpoint $2 --debug_dir ./debug/debug_$1_style --color_continue_train
