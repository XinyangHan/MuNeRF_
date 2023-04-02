# CUDA_VISIBLE_DEVICES=2 python ./train_transformed_rays_hy.py --config ./config/girl9/w_beautyloss_global_patchgan.yml --density_nerf --debug_dir ./debug/girl9_density_white
cd /data/hanxinyang/MuNeRF_latest

CUDA_VISIBLE_DEVICES=2 python ./train_transformed_rays_hy.py --config /data/hanxinyang/MuNeRF_latest/config/$1/$1.yml --density_nerf --debug_dir ./debug/$1_density_beauty
# density nerf 训练原版nerface
# --color continue train 断点续训

# CUDA_VISIBLE_DEVICES=2 python ./train_transformed_rays_hy.py --config ./config/girl10/girl10.yml --density_nerf --debug_dir ./debug/girl10_density_white 

# CUDA_VISIBLE_DEVICES=2 python ./train_transformed_rays_hy.py --config ./config/$1/$1.yml --density_nerf --debug_dir ./debug/$1_density_white 


# CUDA_VISIBLE_DEVICES=2 python ./train_transformed_rays_hy.py --config /data/hanxinyang/MuNeRF_latest/config/girl4/w_beautyloss_global_patchgan1.yml --density_nerf --debug_dir ./debug/girl4_density_white 