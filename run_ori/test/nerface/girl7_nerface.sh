CUDA_VISIBLE_DEVICES=2 python ./eval_transformed_rays.py --config /data/hanxinyang/MuNeRF_latest/config/girl7/girl7_test_nerface.yml --checkpoint /data/hanxinyang/MuNeRF_latest/logs/finals/nerface/girl7_checkpoint120700.ckpt --savedir ./rendering/girl7_nerface_consistent/ --no_makeup --consistent