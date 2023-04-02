#CUDA_VISIBLE_DEVICES=2 python ./eval_transformed_rays.py --config ./config/girl9/boy1_test.yml  --savedir ./rendering/teaser_styled_girl9/no/00101/ --checkpoint ./logs/girl9_style/girl9_00101_dense/checkpoint289000.ckpt --fix_pose
#CUDA_VISIBLE_DEVICES=2 python ./eval_transformed_rays.py --config ./config/girl9/boy1_test.yml  --savedir ./rendering/teaser_styled_girl9/no/21015/ --checkpoint ./logs/girl9_style/girl9_21015_dense/checkpoint293499.ckpt --fix_pose
#CUDA_VISIBLE_DEVICES=2 python ./eval_transformed_rays.py --config ./config/girl9/boy1_test.yml  --savedir ./rendering/teaser_styled_girl9/no/10028/ --checkpoint ./logs/girl9_style/girl9_10028_dense/checkpoint293499.ckpt --fix_pose
#CUDA_VISIBLE_DEVICES=2 python ./eval_transformed_rays.py --config ./config/girl9/boy1_test.yml  --savedir ./rendering/teaser_styled_girl9/no/00197/ --checkpoint ./logs/girl9_style/girl9_00197_dense/checkpoint287000.ckpt --fix_pose
#21015
mkdir ./dataset/girl9/val
cp /data/hanxinyang/MuNeRF_latest/rendering/teaser_new/894_839_17/* ./dataset/girl9/val/
CUDA_VISIBLE_DEVICES=3 python ./eval_transformed_rays.py --config ./config/girl9/boy1_test.yml  --savedir ./rendering/teaser_new/21015_/894_839_17/ --checkpoint ./logs/girl9_style/girl9_21015_dense_new/checkpoint521000.ckpt --fix_pose --id 17 --girl_name girl9
rm -r ./dataset/girl9/val
#mv ./dataset/girl9/val ./dataset/girl9/val_pose1
mkdir ./dataset/girl9/val
cp /data/hanxinyang/MuNeRF_latest/rendering/teaser_new/614_894_3/* ./dataset/girl9/val/
CUDA_VISIBLE_DEVICES=3 python ./eval_transformed_rays.py --config ./config/girl9/boy1_test.yml  --savedir ./rendering/teaser_new/21015_/614_894_3/ --checkpoint ./logs/girl9_style/girl9_21015_dense_new/checkpoint521000.ckpt --fix_pose  --id 3 --girl_name girl9
rm -r ./dataset/girl9/val

#mv ./dataset/girl9/val ./dataset/girl9/val_pose2
mkdir ./dataset/girl9/val
cp /data/hanxinyang/MuNeRF_latest/rendering/teaser_new/275_385_15/* ./dataset/girl9/val/
CUDA_VISIBLE_DEVICES=3 python ./eval_transformed_rays.py --config ./config/girl9/boy1_test.yml  --savedir ./rendering/teaser_new/21015_/275_385_15/ --checkpoint ./logs/girl9_style/girl9_21015_dense_new/checkpoint521000.ckpt --fix_expre --id 15 --girl_name girl9
rm -r ./dataset/girl9/val
#mv ./dataset/girl9/val ./dataset/girl9/val_expre1
mkdir ./dataset/girl9/val
cp /home/yuanyujie/makeupnerf/rendering/teaser_new/385_315_21/* ./dataset/girl9/val/
CUDA_VISIBLE_DEVICES=3 python ./eval_transformed_rays.py --config ./config/girl9/boy1_test.yml  --savedir ./rendering/teaser_new/21015_/385_315_21/ --checkpoint ./logs/girl9_style/girl9_21015_dense_new/checkpoint521000.ckpt --fix_expre --id 21 --girl_name girl9
rm -r ./dataset/girl9/val
#mv ./dataset/girl9/val ./dataset/girl9/val_expre2
#mv ./dataset/girl9/val_ori ./dataset/girl9/val