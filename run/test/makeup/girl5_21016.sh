CUDA_VISIBLE_DEVICES=0 python ./eval_transformed_rays.py --config /home/yuanyujie/cvpr23/config/girl5/w_beautyloss_global_patchgan_21016.yml --checkpoint /home/yuanyujie/makeupnerf/logs/girl5_style/girl5_21016_dense/checkpoint568599.ckpt --savedir ./rendering/girl5_makeup_21016/ --consistent

# CUDA_VISIBLE_DEVICES=0 python ./eval_transformed_rays.py --config /home/yuanyujie/cvpr23/config/girl5/w_beautyloss_global_patchgan_21016.yml --checkpoint /home/yuanyujie/cvpr23/logs/girl5_21016/girl5_21016/checkpoint563599.ckpt --savedir ./rendering/girl5_makeup_21016_H/ --consistent