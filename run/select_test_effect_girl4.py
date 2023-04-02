import os
import pdb
root = "/home/yuanyujie/cvpr23/dataset/girl10/ori_imgs"
# targets = ['3252', '4753', '3252']
# targets = ['2121', '2335', '2823', '3400'] # girl4
# targets = ['0010', '0020', '0180', '0230', '0550', '0410'] #girl10

targets = ['0180', '0200', '0230', '0320', '0390', '0660', '1000'] #girl10 new

# targets = ['1906', '0718', '1918', '0870', '0612']

new_root = "/home/yuanyujie/cvpr23/debug/H/1110/girl10_5_raw"
os.makedirs(new_root, exist_ok=True)

for target in targets:
    path = os.path.join(root, "%s.png"%target)
    cp_path = os.path.join(new_root, "%s.png"%target)
    # pdb.set_trace()
    os.system(f"cp {path} {cp_path}")