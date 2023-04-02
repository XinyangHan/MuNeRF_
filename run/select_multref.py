import os

ids = ['2217', '2224','2334','3606']
parts = ['eyes', 'lip', 'skin', 'whole']
root = "/home/yuanyujie/makeupnerf/rendering/multiref/girl4"
target_root = "/home/yuanyujie/cvpr23/rendering/multiref_girl4"
for part in parts:
    id_root = os.path.join(root, part)
    id_target_root = os.path.join(target_root, part)
    os.makedirs(id_target_root, exist_ok=True)
    for id in ids:
        path = os.path.join(id_root, id + ".png")
        target_path = os.path.join(id_target_root, id +'.png')
        os.system(f"cp {path} {target_path}")