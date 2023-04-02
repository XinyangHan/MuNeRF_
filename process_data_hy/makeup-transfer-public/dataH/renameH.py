from calendar import c
import os

targets = ['/data/hanxinyang/MuNeRF_latest/dataset/boy4/00005/warp_makeup_00005','/data/hanxinyang/MuNeRF_latest/dataset/boy4/00006/warp_makeup_00006','/data/hanxinyang/MuNeRF_latest/dataset/girl10/00005/warp_makeup_00005','/data/hanxinyang/MuNeRF_latest/dataset/girl10/00006/warp_makeup_00006']

for target in targets:
    things = os.listdir(target)
    for thing in things:
        name = thing[:4]
        save_name = name + '.png'
        print(save_name)
        os.system(f"cd {target} && mv {thing} {save_name}")