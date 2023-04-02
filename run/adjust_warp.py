import os

# source = "/home/yuanyujie/cvpr23/process_data_hy/makeup-transfer-public/dataH/1008/boy4_33331"
# target = "/home/yuanyujie/cvpr23/process_data_hy/makeup-transfer-public/dataH/trial_warp"

# os.system(f"mv {source} {target}")
# names = 
names = ['00002', '00004', '00197', '00234']

root = "/home/yuanyujie/cvpr23/process_data_hy/makeup-transfer-public/dataH/trial_warp"
for name in names:

    name_root = os.path.join(root, "boy4_%s"%name)
    temp_warp = os.path.join(name_root, "warp")
    temp_ldmk = os.path.join(name_root, "landmark.pk")

    # os.system(f"cd /home/yuanyujie/cvpr23/process_data_hy/makeup-transfer-public && \
    #                 ~/miniconda3/envs/heyue/bin/python src/handle_datasetH.py --type blend1 --input {name_root} --output {temp_warp} --landmark_input {temp_ldmk} --keep_eye_mouth --include_forehead --adjust_color --adjust_lighting")
    # for temp_warp in 


    os.system(f"cd /home/yuanyujie/cvpr23/process_data_hy/makeup-transfer-public && \
                    ~/miniconda3/envs/heyue/bin/python src/handle_datasetH.py --type landmark --input {name_root} --output {temp_ldmk}")


    os.system(f"cd /home/yuanyujie/cvpr23/process_data_hy/makeup-transfer-public && \
                    ~/miniconda3/envs/heyue/bin/python src/handle_datasetH.py --type blend1 --input {name_root} --output {temp_warp} --landmark_input {temp_ldmk} --keep_eye_mouth --include_forehead --name raw")
    
    os.system(f"cd /home/yuanyujie/cvpr23/process_data_hy/makeup-transfer-public && \
                    ~/miniconda3/envs/heyue/bin/python src/handle_datasetH.py --type blend1 --input {name_root} --output {temp_warp} --landmark_input {temp_ldmk} --keep_eye_mouth --include_forehead --adjust_color --name color")
    
    os.system(f"cd /home/yuanyujie/cvpr23/process_data_hy/makeup-transfer-public && \
                    ~/miniconda3/envs/heyue/bin/python src/handle_datasetH.py --type blend1 --input {name_root} --output {temp_warp} --landmark_input {temp_ldmk} --keep_eye_mouth --include_forehead --adjust_lighting --name lighting")
    
    os.system(f"cd /home/yuanyujie/cvpr23/process_data_hy/makeup-transfer-public && \
                    ~/miniconda3/envs/heyue/bin/python src/handle_datasetH.py --type blend1 --input {name_root} --output {temp_warp} --landmark_input {temp_ldmk} --keep_eye_mouth --include_forehead --adjust_color --adjust_lighting --name both")