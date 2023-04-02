
# Girl 10 extreme makeup 00005
CUDA_VISIBLE_DEVICES=2 python ./train_transformed_rays_hy.py --config ./config/girl10/w_beautyloss_global_patchgan.yml  --load_checkpoint /data/hanxinyang/MuNeRF_latest/logs/girl10_density/checkpoint71900.ckpt --debug_dir ./debug/debug_girl10_00005

# Boy4 extreme makeup 00005
CUDA_VISIBLE_DEVICES=2 python ./train_transformed_rays_hy.py --config /data/hanxinyang/MuNeRF_latest/config/boy4/w_beautyloss_global_patchgan.yml  --load_checkpoint /data/hanxinyang/MuNeRF_latest/logs/boy4_density/checkpoint962100.ckpt --debug_dir ./debug/debug_boy4_00005

# Girl 10 extreme makeup 00006

CUDA_VISIBLE_DEVICES=2 python ./train_transformed_rays_hy.py --config /data/hanxinyang/MuNeRF_latest/config/girl10/w_beautyloss_global_patchgan_00006.yml  --load_checkpoint /data/hanxinyang/MuNeRF_latest/logs/girl10_density/checkpoint71900.ckpt --debug_dir ./debug/debug_girl10_00006

# Boy4 extreme makeup 00006
CUDA_VISIBLE_DEVICES=2 python ./train_transformed_rays_hy.py --config /data/hanxinyang/MuNeRF_latest/config/boy4/w_beautyloss_global_patchgan_00006.yml  --load_checkpoint /data/hanxinyang/MuNeRF_latest/logs/boy4_density/checkpoint962100.ckpt --debug_dir ./debug/debug_boy4_00006

# Hold GPU
bash /data/hanxinyang/MuNeRF_latest/train_makeup_girl7.sh