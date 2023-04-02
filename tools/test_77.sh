#00245
#mv ./dataset/girl7/val./dataset/girl7/val_ori
mkdir ./dataset/girl7/val/
cp /home/yuanyujie/makeupnerf/rendering/girl7_video/252_626_55/* ./dataset/girl7/val/
CUDA_VISIBLE_DEVICES=1 python ./eval_transformed_rays.py --config ./config/girl7/girl7_test.yml  --savedir ./rendering/girl7_video/00245/252_626_55/ --checkpoint ./logs/girl7_style/girl7_000245_dense_uv/checkpoint316899.ckpt --fix_pose --id 55 --girl_name girl7
mv ./dataset/girl7/val ./dataset/girl7/val_pose1

mkdir ./dataset/girl7/val/
cp /home/yuanyujie/makeupnerf/rendering/girl7_video/252_1369_55/* ./dataset/girl7/val/
CUDA_VISIBLE_DEVICES=1 python ./eval_transformed_rays.py --config ./config/girl7/girl7_test.yml  --savedir ./rendering/girl7_video/00245/252_1369_55/ --checkpoint ./logs/girl7_style/girl7_000245_dense_uv/checkpoint316899.ckpt --fix_pose --id 55 --girl_name girl7
mv ./dataset/girl7/val ./dataset/girl7/val_pose2

mkdir ./dataset/girl7/val/
cp /home/yuanyujie/makeupnerf/rendering/girl7_video/252_416_55/* ./dataset/girl7/val/
CUDA_VISIBLE_DEVICES=1 python ./eval_transformed_rays.py --config ./config/girl7/girl7_test.yml  --savedir ./rendering/girl7_video/00245/252_416_55/ --checkpoint ./logs/girl7_style/girl7_000245_dense_uv/checkpoint316899.ckpt --fix_pose --id 55 --girl_name girl7
mv ./dataset/girl7/val ./dataset/girl7/val_pose3

mkdir ./dataset/girl7/val/
cp /home/yuanyujie/makeupnerf/rendering/girl7_video/516_252_55/* ./dataset/girl7/val/
CUDA_VISIBLE_DEVICES=1 python ./eval_transformed_rays.py --config ./config/girl7/girl7_test.yml  --savedir ./rendering/girl7_video/00245/516_252_55/ --checkpoint ./logs/girl7_style/girl7_000245_dense_uv/checkpoint316899.ckpt --fix_pose --id 55 --girl_name girl7
mv ./dataset/girl7/val ./dataset/girl7/val_pose4
#mv ./dataset/girl9/val_ori ./dataset/girl9/val

#10020
#mv ./dataset/girl7/val./dataset/girl7/val_ori
mkdir ./dataset/girl7/val/
cp /home/yuanyujie/makeupnerf/rendering/girl7_video/252_626_55/* ./dataset/girl7/val/
CUDA_VISIBLE_DEVICES=1 python ./eval_transformed_rays.py --config ./config/girl7/girl7_test.yml  --savedir ./rendering/girl7_video/10020/252_626_55/ --checkpoint ./logs/girl7_style/girl7_10020_dense/checkpoint314899.ckpt --fix_pose --id 55 --girl_name girl7
rm -r ./dataset/girl7/val
#mv ./dataset/girl7/val ./dataset/girl7/val_pose1

mkdir ./dataset/girl7/val/
cp /home/yuanyujie/makeupnerf/rendering/girl7_video/252_1369_55/* ./dataset/girl7/val/
CUDA_VISIBLE_DEVICES=1 python ./eval_transformed_rays.py --config ./config/girl7/girl7_test.yml  --savedir ./rendering/girl7_video/10020/252_1369_55/ --checkpoint ./logs/girl7_style/girl7_10020_dense/checkpoint314899.ckpt --fix_pose --id 55 --girl_name girl7
rm -r ./dataset/girl7/val
#mv ./dataset/girl7/val ./dataset/girl7/val_pose2

mkdir ./dataset/girl7/val/
cp /home/yuanyujie/makeupnerf/rendering/girl7_video/252_416_55/* ./dataset/girl7/val/
CUDA_VISIBLE_DEVICES=1 python ./eval_transformed_rays.py --config ./config/girl7/girl7_test.yml  --savedir ./rendering/girl7_video/10020/252_416_55/ --checkpoint ./logs/girl7_style/girl7_10020_dense/checkpoint314899.ckpt --fix_pose --id 55 --girl_name girl7
rm -r ./dataset/girl7/val/
#mv ./dataset/girl7/val ./dataset/girl7/val_pose3

mkdir ./dataset/girl7/val/
cp /home/yuanyujie/makeupnerf/rendering/girl7_video/516_252_55/* ./dataset/girl7/val/
CUDA_VISIBLE_DEVICES=1 python ./eval_transformed_rays.py --config ./config/girl7/girl7_test.yml  --savedir ./rendering/girl7_video/10020/516_252_55/ --checkpoint ./logs/girl7_style/girl7_10020_dense/checkpoint314899.ckpt --fix_pose --id 55 --girl_name girl7
rm -r ./dataset/girl7/val/
#mv ./dataset/girl7/val ./dataset/girl7/val_pose4

#10081
#mv ./dataset/girl7/val./dataset/girl7/val_ori
mkdir ./dataset/girl7/val/
cp /home/yuanyujie/makeupnerf/rendering/girl7_video/252_626_55/* ./dataset/girl7/val/
CUDA_VISIBLE_DEVICES=1 python ./eval_transformed_rays.py --config ./config/girl7/girl7_test.yml  --savedir ./rendering/girl7_video/10081/252_626_55/ --checkpoint ./logs/girl7_style/girl7_10081_dense/checkpoint316899.ckpt --fix_pose --id 55 --girl_name girl7
rm -r ./dataset/girl7/val/
#mv ./dataset/girl7/val ./dataset/girl7/val_pose1

mkdir ./dataset/girl7/val/
cp /home/yuanyujie/makeupnerf/rendering/girl7_video/252_1369_55/* ./dataset/girl7/val/
CUDA_VISIBLE_DEVICES=1 python ./eval_transformed_rays.py --config ./config/girl7/girl7_test.yml  --savedir ./rendering/girl7_video/10081/252_1369_55/ --checkpoint ./logs/girl7_style/girl7_10081_dense/checkpoint316899.ckpt --fix_pose --id 55 --girl_name girl7
rm -r ./dataset/girl7/val/
#mv ./dataset/girl7/val ./dataset/girl7/val_pose2

mkdir ./dataset/girl7/val/
cp /home/yuanyujie/makeupnerf/rendering/girl7_video/252_416_55/* ./dataset/girl7/val/
CUDA_VISIBLE_DEVICES=1 python ./eval_transformed_rays.py --config ./config/girl7/girl7_test.yml  --savedir ./rendering/girl7_video/10081/252_416_55/ --checkpoint ./logs/girl7_style/girl7_10081_dense/checkpoint316899.ckpt --fix_pose --id 55 --girl_name girl7
rm -r ./dataset/girl7/val
#mv ./dataset/girl7/val ./dataset/girl7/val_pose3

mkdir ./dataset/girl7/val/
cp /home/yuanyujie/makeupnerf/rendering/girl7_video/516_252_55/* ./dataset/girl7/val/
CUDA_VISIBLE_DEVICES=1 python ./eval_transformed_rays.py --config ./config/girl7/girl7_test.yml  --savedir ./rendering/girl7_video/10081/516_252_55/ --checkpoint ./logs/girl7_style/girl7_10081_dense/checkpoint316899.ckpt --fix_pose --id 55 --girl_name girl7
rm -r ./dataset/girl7/val/


















































































































