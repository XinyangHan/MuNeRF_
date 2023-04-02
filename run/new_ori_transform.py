import os

targets = ['00002','00040','00197', '00234']

for target in targets:
    temp_path = f"/home/yuanyujie/cvpr23/dataset/girl10/{target}/warp_makeup_{target}"
    new_path = f"/home/yuanyujie/cvpr23/dataset/girl10/{target}/warp_makeup_{target}_new"
    ori_path = f"/home/yuanyujie/cvpr23/dataset/girl10/{target}/warp_makeup_{target}_ori"

    os.system(f"mv {temp_path} {new_path}")
    os.system(f"mv {ori_path} {temp_path}")