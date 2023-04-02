targets = ["/home/yuanyujie/cvpr23/dataset/boy4/00005/mask/warp_makeup_00005","/home/yuanyujie/cvpr23/dataset/boy4/00005/warp_makeup_00005","/home/yuanyujie/cvpr23/dataset/boy4/00006/mask/warp_makeup_00006","/home/yuanyujie/cvpr23/dataset/boy4/00006/warp_makeup_00006", "/home/yuanyujie/cvpr23/dataset/boy4/00030/warp_makeup_00030", "/home/yuanyujie/cvpr23/dataset/boy4/00030/mask/warp_makeup_00030", "/home/yuanyujie/cvpr23/dataset/boy4/00053/warp_makeup_00053", "/home/yuanyujie/cvpr23/dataset/boy4/00053/mask/warp_makeup_00053","/home/yuanyujie/cvpr23/dataset/boy4/00160/mask/warp_makeup_00160","/home/yuanyujie/cvpr23/dataset/boy4/00160/warp_makeup_00160", "/home/yuanyujie/cvpr23/dataset/boy4/21013/mask/warp_makeup_21013", "/home/yuanyujie/cvpr23/dataset/boy4/21013/warp_makeup_21013","/home/yuanyujie/cvpr23/dataset/girl10/00005/warp_makeup_00005","/home/yuanyujie/cvpr23/dataset/girl10/00005/mask/warp_makeup_00005" ]

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

