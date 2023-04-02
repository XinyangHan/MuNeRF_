import os

root = "/data/hanxinyang/MuNeRF_latest/process_data_hy/makeup-transfer-public/dataH/1008"

img_root = "/data/hanxinyang/MuNeRF_latest/process_data_hy/makeup-transfer-public/dataH/1008/girl4_00000/after"

for i in range(5,8):
    name = "girl"+"%d_00000"%i
    model_path = os.path.join(root, name)
    target_after = os.path.join(model_path, "after")
    os.system(f"rm -r {target_after}")
    os.system(f"cp -r {img_root} {target_after}")