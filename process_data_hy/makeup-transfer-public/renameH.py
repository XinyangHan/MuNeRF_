import os

targets = ['/data/hanxinyang/MuNeRF_latest/process_data_hy/makeup-transfer-public/dataH/boy4_0/warp_makeup_00000','/data/hanxinyang/MuNeRF_latest/process_data_hy/makeup-transfer-public/dataH/boy4_12/warp_makeup_00012','/data/hanxinyang/MuNeRF_latest/process_data_hy/makeup-transfer-public/dataH/girl10_0/warp_makeup_00000','/data/hanxinyang/MuNeRF_latest/process_data_hy/makeup-transfer-public/dataH/girl10_12/warp_makeup_00012']

for target in targets:
    things = os.listdir(target)
    for thing in things:
        name = thing.split('_')[0]
        save_name = name + '.png'
        os.system(f"cd {target} && mv {thing} {save_name}")