# cd ..
CUDA_VISIBLE_DEVICES=2 python ./eval_transformed_rays.py --config /data/hanxinyang/MuNeRF_latest/config/girl9/girl9_test_nerface.yml --checkpoint /data/hanxinyang/MuNeRF_latest/logs/girl9_style/girl9_21015_dense_new/checkpoint09999.ckpt --savedir ./rendering/girl9_fix_expr/ --no_makeup --fix_expre --girl_name girl9