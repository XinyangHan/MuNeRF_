# MuNeRF 预处理+训练脚本，如果遇到bug建议按命令分开运行。

# 重要的修改，重要的修改，重要的修改：：）
# 首先得用vscode文件夹内文本替换，替换掉所有的绝对路径，将/data/hanxinyang/MuNeRF_latest/ 替换成新的地址
# 值得注意的是脚本中指定GPU的命令需要根据需要进行批量替换CUDA_VISIBLE_DEVICES=3 


# 预处理，获取face ldmk, background ...
bash /data/hanxinyang/MuNeRF_latest/process_data_hy/pre_process/process_data.sh

# get the modnet running，然后也许下需要手动将类似这样的json文件移动到 ./dataset/boy1这样对应的文件夹下
bash /data/hanxinyang/MuNeRF_latest/process_data_hy/MODnet/run.sh girl21
bash /data/hanxinyang/MuNeRF_latest/process_data_hy/MODnet/run.sh girl22

# train the nerface, 可以在./debug对应目录下查看结果，也可以训练完成后使用./run/test下的脚本进行渲染
bash /data/hanxinyang/MuNeRF_latest/run/train/nerface/girl21.sh
bash /data/hanxinyang/MuNeRF_latest/run/train/nerface/girl22.sh

# get the pgt
bash /data/hanxinyang/MuNeRF_latest/process_data_hy/makeup-transfer-public/trial_warp.sh

# train the munerf
bash /data/hanxinyang/MuNeRF_latest/run/train/makeup/girl21_40008.sh
bash /data/hanxinyang/MuNeRF_latest/run/train/makeup/girl22_40005.sh
bash /data/hanxinyang/MuNeRF_latest/run/train/makeup/girl22_40008.sh

# render the results -> 这里需要根据最终训练的结果来替换ckpt路径
bash /data/hanxinyang/MuNeRF_latest/run/test/makeup/girl21_40008.sh
bash /data/hanxinyang/MuNeRF_latest/run/test/makeup/girl22_40005.sh
bash /data/hanxinyang/MuNeRF_latest/run/test/makeup/girl22_40008.sh

# 在/data/hanxinyang/MuNeRF_latest/rendering中挑选结果