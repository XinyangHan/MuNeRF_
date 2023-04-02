import os

save_root = "/home/yuanyujie/cvpr23/debug/H/1030"
targets = ['00002','00004','00197','00234']
for target in targets:
    source_path = f"/home/yuanyujie/cvpr23/rendering/boy4_makeup_{target}/0360.png"
    target_path = os.path.join(save_root, "munerf_"+target+'.png')
    os.system(f"cp {source_path} {target_path}")

    source_path_reference = f"/home/yuanyujie/cvpr23/dataset/boy4/{target}/after/{target}.jpg"
    target_path_reference = f"/home/yuanyujie/cvpr23/debug/H/1030/{target}.png"

    os.system(f"cp -rf {source_path_reference} {target_path_reference}")
