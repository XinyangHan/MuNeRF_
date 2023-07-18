import os

root = "/data/heyue/104/dataset"

makeups_root = "/data/heyue/104/dataset/girl1/makeup"
for i in range(4, 10):
    target_path = os.path.join(root, f"girl{i}", "makeup")
    # os.system(f"rm -r {target_path}")
    # os.system(f"cp -r {makeups_root} {target_path}")
    for j in range(37):
        index = 34000+j
        os.system(f"cd /data/heyue/makeup_related && bash ./fiveH.sh girl{i} {index}")