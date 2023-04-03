import os
import pathlib
import pdb

targets = ['34016']
# targets = ['00011','00052','00116']
models = [ 'girl20', 'girl21', 'girl22', 'girl23', 'girl24']
template_model = 'girl8'

for model in models:
    for target in targets:
        # new warp
        # os.system(f"python /data/hanxinyang/MuNeRF_latest/run/select_facial_part.py --name {model} --style {target}")

        # new config
        template_path = f"/data/hanxinyang/MuNeRF_latest/config/{template_model}/w_beautyloss_global_patchgan_00003.yml"
        
        target_config_path = template_path.replace("00003", target).replace(template_model, model)

        base_root = target_config_path.split('/w_beauty')[0]
        print("Make directory:", base_root)
        os.makedirs(base_root, exist_ok=True)
        pathlib.Path(target_config_path).touch()
        with open(template_path) as f:
            with open(target_config_path, 'w') as new:
                template = f.read()
                new_content = template.replace("00003", target).replace(template_model, model)
                new.write(new_content)

        # new train text
        template_path = f"/data/hanxinyang/MuNeRF_latest/run/train/makeup/{template_model}_00003.sh"
        target_train_path = template_path.replace("00003", target).replace(template_model, model)
        pathlib.Path(target_train_path).touch()

        with open(template_path) as f:
            with open(target_train_path, 'w') as new:
                template = f.read()
                new_content = template.replace("00003", target).replace(template_model, model)
                new.write(new_content)
        
        # new test text
        template_path = f"/data/hanxinyang/MuNeRF_latest/run/test/makeup/{template_model}_00003.sh"
        target_test_path = template_path.replace("00003", target).replace(template_model, model)
        pathlib.Path(target_test_path).touch()

        with open(template_path) as f:
            with open(target_test_path, 'w') as new:
                template = f.read()
                new_content = template.replace("00003", target).replace(template_model, model)
                new.write(new_content)
                
         # new train nerface
        template_path = f"/data/hanxinyang/MuNeRF_latest/run/train/nerface/{template_model}.sh"
        target_train_path = template_path.replace("00003", target).replace(template_model, model)
        pathlib.Path(target_train_path).touch()

        with open(template_path) as f:
            with open(target_train_path, 'w') as new:
                template = f.read()
                new_content = template.replace("00003", target).replace(template_model, model)
                new.write(new_content)
        
        # new test nerface
        template_path = f"/data/hanxinyang/MuNeRF_latest/run/test/nerface/{template_model}.sh"
        target_test_path = template_path.replace("00003", target).replace(template_model, model)
        pathlib.Path(target_test_path).touch()

        with open(template_path) as f:
            with open(target_test_path, 'w') as new:
                template = f.read()
                new_content = template.replace("00003", target).replace(template_model, model)
                new.write(new_content)