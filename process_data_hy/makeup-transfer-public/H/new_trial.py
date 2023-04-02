import os

root = "/data/hanxinyang/MuNeRF_latest/process_data_hy/makeup-transfer-public/dataH/1008"

for i in range(4,8):
    name = "girl"+"%d_00000"%i
    model_path = os.path.join(root, name)
    removes = []
    remove_sub_dirs = ["after_ori", "warp"]
    for sub in remove_sub_dirs:
        removes.append(os.path.join(model_path, sub))
    for remove in removes:
        os.system(f"rm -r {remove}")