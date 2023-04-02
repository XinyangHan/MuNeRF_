import os

targets = ['00197']
# targets = ['00002','00004','00197','00234']


for target in targets:
    path1 = f"/home/yuanyujie/cvpr23/logs/boy4_style_{target}/boy4_{target}/checkpoint964000.ckpt"
    path2 = f"/home/yuanyujie/cvpr23/logs/boy4_style_{target}/boy4_{target}/checkpoint965000.ckpt"

    os.system(f"rm {path1}")
    os.system(f"rm {path2}")

