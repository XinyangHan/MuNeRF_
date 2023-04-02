import os

styles = ['']
for style in styles:
    os.system("python /home/yuanyujie/cvpr23/process_data_hy/makeup-transfer-public/10008_test_val.py --id {style} && python /home/yuanyujie/cvpr23/run/select_facial_part.py --name girl10 --style {style} && CUDA_VISIBLE_DEVICES=2 python ./train_transformed_rays_hy.py --config ./config/girl10/w_beautyloss_global_patchgan_{style}.yml  --load_checkpoint /home/yuanyujie/cvpr23/logs/finals/nerface/gril10_checkpoint71800.ckpt --debug_dir ./debug/debug_girl10_style")