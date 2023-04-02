
import os
import pathlib

# targets = ['00197']
# models = ['girl11','girl12','girl13','girl14']


targets =['34015']
# # targets = ['00011','00052','00116']
models = ['girl7']
# targets = ['00128']
# targets = ['00011','00052','00116']
# models = [ 'boy1']

# # nerface
for model in models:
    for target in targets:
        # new warp
        # os.system(f"python /home/yuanyujie/cvpr23/run/select_facial_part.py --name {model} --style {target}")

        # new config
        template_path = f"/home/yuanyujie/cvpr23/config/girl4/girl4.yml"
        target_config_path = template_path.replace("00003", target).replace("girl4", model)
        os.makedirs(target_config_path.split(model)[0]+model, exist_ok=True)
        pathlib.Path(target_config_path).touch()
        with open(template_path) as f:
            with open(target_config_path, 'w') as new:
                template = f.read()
                new_content = template.replace("00003", target).replace("girl4", model)
                new.write(new_content)

        # new train text
        template_path = f"/home/yuanyujie/cvpr23/run/train/nreface/girl4.sh"
        target_train_path = template_path.replace("girl4", model)
        os.makedirs(target_train_path.split(model)[0]+model, exist_ok=True)
        pathlib.Path(target_train_path).touch()

        with open(template_path) as f:
            with open(target_train_path, 'w') as new:
                template = f.read()
                new_content = template.replace("00003", target).replace("girl4", model)
                new.write(new_content)
        
        # new test text
        template_path = f"/home/yuanyujie/cvpr23/run/test/nerface/girl4.sh"
        target_test_path = template_path.replace("girl4", model)
        os.makedirs(target_test_path.split(model)[0]+model, exist_ok=True)
        pathlib.Path(target_test_path).touch()

        with open(template_path) as f:
            with open(target_test_path, 'w') as new:
                template = f.read()
                new_content = template.replace("00003", target).replace("girl4", model).replace("girl4", model)
                new.write(new_content)


# makeup
for model in models:
    for target in targets:
        # new warp
        # os.system(f"python /home/yuanyujie/cvpr23/run/select_facial_part.py --name {model} --style {target}")

        warp_root = f"/home/yuanyujie/cvpr23/dataset/{model}/{target}"
        after_path = os.path.join(warp_root, "after")
        os.makedirs(warp_root, exist_ok=True)
        os.makedirs(after_path, exist_ok=True)
        makeup_root = "/home/yuanyujie/cvpr23/makeup"
        makeup_source = os.path.join(makeup_root, target+".jpg")
        os.system(f"cp {makeup_source} {after_path}")

        # new config
        template_path = f"/home/yuanyujie/cvpr23/config/girl4/w_beautyloss_global_patchgan_00003.yml"
        target_config_path = template_path.replace("00003", target).replace("girl4", model)
        os.makedirs(target_config_path.split(model)[0]+model, exist_ok=True)
        pathlib.Path(target_config_path).touch()
        with open(template_path) as f:
            with open(target_config_path, 'w') as new:
                template = f.read()
                new_content = template.replace("00003", target).replace("girl4", model)
                new.write(new_content)

        # new train text
        template_path = f"/home/yuanyujie/cvpr23/run/train/makeup/girl4_00003.sh"
        target_train_path = template_path.replace("00003", target).replace("girl4", model)
        os.makedirs(target_train_path.split(model)[0]+model, exist_ok=True)
        pathlib.Path(target_train_path).touch()

        with open(template_path) as f:
            with open(target_train_path, 'w') as new:
                template = f.read()
                new_content = template.replace("00003", target).replace("girl4", model)
                new.write(new_content)
        
        # new test text
        template_path = f"/home/yuanyujie/cvpr23/run/test/makeup/girl4_00003.sh"
        target_test_path = template_path.replace("00003", target).replace("girl4", model)
        os.makedirs(target_test_path.split(model)[0]+model, exist_ok=True)
        pathlib.Path(target_test_path).touch()

        with open(template_path) as f:
            with open(target_test_path, 'w') as new:
                template = f.read()
                new_content = template.replace("00003", target).replace("girl4", model)
                new.write(new_content)