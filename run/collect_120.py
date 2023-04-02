import os

save_root = "/home/yuanyujie/cvpr23/debug/H/1030"
targets = ['00002','00005','00197','00234']
for target in targets:
    source_path = f"/home/yuanyujie/cvpr23/rendering/girl10_makeup_{target}/0120.png"
    target_path = os.path.join(save_root, "munerf_"+target+'.png')
    os.system(f"cp {source_path} {target_path}")

    source_path_reference = f"/home/yuanyujie/cvpr23/dataset/girl10/{target}/after/{target}.jpg"
    target_path_reference = f"/home/yuanyujie/cvpr23/debug/H/1104/girl10/{target}.png"

    os.system(f"cp -rf {source_path_reference} {target_path_reference}")
