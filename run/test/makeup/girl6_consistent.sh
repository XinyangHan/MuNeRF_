CUDA_VISIBLE_DEVICES=2 python ./eval_transformed_rays.py --config /home/yuanyujie/cvpr23/config/girl6/w_beautyloss_global_patchgan.yml --checkpoint /home/yuanyujie/makeupnerf/logs/girl6_style/girl6_00120_dense/checkpoint841399.ckpt --savedir ./rendering/girl6_makeup_consistent/ --consistent