import os

targets = ['4667','3613','2841','2334']


ori_root = "/home/yuanyujie/cvpr23/dataset/girl4/ori_imgs"
makeup_root = "/home/yuanyujie/cvpr23/rendering/girl4_makeup_00030"
save_root = "/home/yuanyujie/cvpr23/debug/H/1116"

for target in targets:
    ori_path = os.path.join(ori_root, target+'.png')
    makeup_path = os.path.join(makeup_root, target+'.png')
    save_ori_path = os.path.join(save_root, target+'_ori'+'.png')
    save_makeup_path = os.path.join(save_root, target+'_makeup'+'.png')

    os.system(f"cp {ori_path} {save_ori_path}")
    os.system(f"cp {makeup_path} {save_makeup_path}")

