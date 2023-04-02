targets = ["/data/hanxinyang/MuNeRF_latest/dataset/boy4/00005/mask/warp_makeup_00005","/data/hanxinyang/MuNeRF_latest/dataset/boy4/00006/mask/warp_makeup_00006","/data/hanxinyang/MuNeRF_latest/dataset/girl10/00005/mask/warp_makeup_00005","/data/hanxinyang/MuNeRF_latest/dataset/girl10/00006/mask/warp_makeup_00006"]

import os
import pdb
for target in targets:
    try:
        save_path = target
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        things = os.listdir(target)
        # pdb.set_trace()
        for i, thing in enumerate(things):
            name = thing.split("_")[0]
            if name.endswith('png'):
                continue
            else:
                img_path = os.path.join(target, name + '.png')
            ori_path = os.path.join(target, thing)
            # pdb.set_trace()
            os.system("cd %s && mv %s %s"%(target, ori_path, img_path))

        # if len(os.listdir(target))== 0: 
        #     os.system("rm -r %s"%target)
        # else:
        #     pdb.set_trace()
    except Exception as e:
        print(e)

