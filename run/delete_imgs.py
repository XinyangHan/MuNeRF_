import os

path = "/home/yuanyujie/cvpr23/dataset/girl10/00234/warp_makeup_00234"
things = os.listdir(path)
for thing in things:
    if "filter" in thing:
        continue
    else:
        to_delete = os.path.join(path, thing)
        os.system(f"rm {to_delete}")