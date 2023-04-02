import os
targets = ['34015']
# # targets = ['00011','00052','00116']
models = ['girl7']
for model in models:
    for target in targets:
        try:
            paths = [f"/home/yuanyujie/cvpr23/dataset/{model}/{target}/mask/warp_makeup_{target}", f"/home/yuanyujie/cvpr23/dataset/{model}/{target}/warp_makeup_{target}"]
            # paths = ['f"/home/yuanyujie/cvpr23/dataset/{model}/{target}/warp_makeup_{target}"']
            for path in paths:
                for thing in os.listdir(path):
                    if not ("_" in thing):
                        continue
                    else:
                        name = thing.split("_")[0]
                        ori_path = os.path.join(path, thing)
                        new_path = os.path.join(path, name+".png")

                        os.system(f"mv {ori_path} {new_path}")
        except:
            continue