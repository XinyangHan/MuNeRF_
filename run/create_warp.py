import os

names = ['00002', '00004', '00197', '00234']
raw = "/home/yuanyujie/cvpr23/dataset/boy4/ori_imgs/0360.png"
for name in names:
    source = f"/home/yuanyujie/cvpr23/dataset/boy4/{name}/after"

    target_root = f"/home/yuanyujie/cvpr23/process_data_hy/makeup-transfer-public/dataH/trial_warp/boy4_{name}"
    target_before = os.path.join(target_root, "before")
    target_warp = os.path.join(target_root, "warp")

    os.makedirs(target_root, exist_ok=True)
    os.makedirs(target_before, exist_ok=True)
    os.makedirs(target_warp, exist_ok=True)

    os.system(f"cp -r {source} {target_root}")
    os.system(f"cp {raw} {target_before}")