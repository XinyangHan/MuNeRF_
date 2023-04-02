import os
import pathlib

targets = ['34016']
# targets = ['00011','00052','00116']
models = [ 'girl8']

for model in models:
    for target in targets:
        # new warp
        # os.system(f"python /data/hanxinyang/MuNeRF_latest/run/select_facial_part.py --name {model} --style {target}")

        # new config
        template_path = f"/data/hanxinyang/MuNeRF_latest/config/{model}/w_beautyloss_global_patchgan_00003.yml"
        target_config_path = template_path.replace("00003", target)
        pathlib.Path(target_config_path).touch()
        with open(template_path) as f:
            with open(target_config_path, 'w') as new:
                template = f.read()
                new_content = template.replace("00003", target)
                new.write(new_content)

        # new train text
        template_path = f"/data/hanxinyang/MuNeRF_latest/run/train/makeup/{model}_00003.sh"
        target_train_path = template_path.replace("00003", target)
        pathlib.Path(target_train_path).touch()

        with open(template_path) as f:
            with open(target_train_path, 'w') as new:
                template = f.read()
                new_content = template.replace("00003", target)
                new.write(new_content)
        
        # new test text
        template_path = f"/data/hanxinyang/MuNeRF_latest/run/test/makeup/{model}_00003.sh"
        target_test_path = template_path.replace("00003", target)
        pathlib.Path(target_test_path).touch()

        with open(template_path) as f:
            with open(target_test_path, 'w') as new:
                template = f.read()
                new_content = template.replace("00003", target)
                new.write(new_content)