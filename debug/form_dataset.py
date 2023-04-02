import os
import pdb

root = "/data/hanxinyang/MuNeRF_latest/dataset"
target_root = "/data/hanxinyang/makeup_compare/CPM/dataset/images/nonmakeup"
# notice : should be .png form

model_names = os.listdir(root)

for model_name in model_names:
    model_pictures_path = os.path.join(root, model_name, "train")

    # pdb.set_trace()
    if not os.path.exists(model_pictures_path):
        continue
    else:
        images = os.listdir(model_pictures_path)
        for image in images:
            image_path = os.path.join(model_pictures_path, image)

            os.system("cp %s %s"%(image_path, os.path.join(target_root, "%s_%s"%(model_name, image))))

            # pdb.set_trace()