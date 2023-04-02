# CUDA_VISIBLE_DEVICES=2 python ./train_transformed_rays_hy.py --config ./config/girl7/w_beautyloss_global_patchgan.yml  --load_checkpoint /home/yuanyujie/cvpr23/logs/girl7_style/girl7_10070_gengirl4/checkpoint838399.ckpt --color_continue_train --debug_dir ./debug/debug_girl7_crosscat_00120_big_new 

# CUDA_VISIBLE_DEVICES=2 python ./eval_transformed_rays.py --config ./config/girl7/w_beautyloss_global_patchgan.yml --checkpoint /home/yuanyujie/cvpr23/logs/girl7_style/girl7_10070_gengirl4/checkpoint838399.ckpt --savedir ./rendering/name/

# CUDA_VISIBLE_DEVICES=2 python ./eval_transformed_rays_ori.py --config /home/yuanyujie/cvpr23/config/girl7/girl7_testH.yml --checkpoint /home/yuanyujie/cvpr23/logs/girl7_style/girl7_10020_gengirl4/checkpoint323200.ckpt --savedir ./rendering/girl7_makeup_1204/  

# CUDA_VISIBLE_DEVICES=2 python ./eval_transformed_rays.py --config /home/yuanyujie/cvpr23/config/girl7/girl7_testH.yml --checkpoint /home/yuanyujie/cvpr23/logs/girl7_style/girl7_10020_gengirl4/checkpoint323200.ckpt --savedir ./rendering/girl7_makeup_1204/  --fix_expre --girl_name girl7 --exp_id 1204 --pose1 4874 --pose2 2188 --fix_pose

cd /home/yuanyujie/cvpr23
CUDA_VISIBLE_DEVICES=2 python ./eval_transformed_rays.py --config /home/yuanyujie/cvpr23/config/girl7/girl7_testH.yml --checkpoint /home/yuanyujie/cvpr23/logs/girl7_style/girl7_10020_gengirl4/checkpoint323200.ckpt --savedir ./rendering/girl7_makeup_1204/  --fix_expre --girl_name girl7 --exp_id 1204 --pose1 4874 --pose2 2188 --fix_pose  --nerface_dir /home/yuanyujie/cvpr23/rendering/finals/girl7_fix_expr_1204

CUDA_VISIBLE_DEVICES=2 python ./eval_transformed_rays.py --config /home/yuanyujie/cvpr23/config/girl7/girl7_testH.yml --checkpoint /home/yuanyujie/cvpr23/logs/girl7_style/girl7_10020_gengirl4/checkpoint323200.ckpt --savedir ./rendering/girl7_makeup_0656/  --fix_expre --girl_name girl7 --exp_id 0656 --pose1 4874 --pose2 2188 --fix_pose    --nerface_dir /home/yuanyujie/cvpr23/rendering/finals/girl7_fix_expr_0656

CUDA_VISIBLE_DEVICES=2 python ./eval_transformed_rays.py --config /home/yuanyujie/cvpr23/config/girl7/girl7_testH.yml --checkpoint /home/yuanyujie/cvpr23/logs/girl7_style/girl7_10020_gengirl4/checkpoint323200.ckpt --savedir ./rendering/girl7_makeup_4406/  --fix_expre --girl_name girl7 --exp_id 4406 --pose1 4874 --pose2 2188 --fix_pose    --nerface_dir /home/yuanyujie/cvpr23/rendering/finals/girl7_fix_expr_4406