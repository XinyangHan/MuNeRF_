import os
import pdb
import cv2
source_root = "/data/hanxinyang/MuNeRF_latest/dataset"
root = "/data/hanxinyang/MuNeRF_latest/process_data_hy/makeup-transfer-public/dataH"

names = ["boy4"]
styles = ["00197"]

for name in names:
    for style in styles:
        name_root = os.path.join(root, name + "_" + style)

        print("1: directories")
        ################
        # 1.directories
        ################
        # Temp directories
        os.makedirs(name_root, exist_ok=True)
        temp_raw = os.path.join(name_root, "before")
        temp_makeup = os.path.join(name_root, "after")
        temp_warp = os.path.join(name_root, "warp")
        temp_mask_makeup = os.path.join(name_root, "mask_makeup")
        os.makedirs(os.path.join(name_root, "raw"), exist_ok=True)
        os.makedirs(os.path.join(name_root, "makeup"), exist_ok=True)
         #/data/hanxinyang/MuNeRF_latest/dataset/boy4/00005/after

       

        
        
        # Load origin 
        name_source = os.path.join(source_root, name)
        test_source = os.path.join(name_source, "test")
        val_source = os.path.join(name_source, "val")
        train_source = os.path.join(name_source, "train")

        makeup_source = os.path.join(name_source, style, "after")
        # Target 
        # mask_target = "/data/hanxinyang/MuNeRF_latest/dataset/boy4/00005/mask/warp_makeup_00005"
        mask_target = os.path.join(name_source, style, "mask",  "warp_makeup_%s"%style)
        # /data/hanxinyang/MuNeRF_latest/dataset/boy4/00005/warp_makeup_00005
        warp_target = os.path.join(name_source, style, "warp_makeup_%s"%style)
        print("2:transmit")
        ################
        # 2. Transmit data : raw, makeup
        ################
        # For Raw
        test_imgs = os.listdir(test_source)
        val_imgs = os.listdir(val_source)
        train_imgs = os.listdir(train_source)
        for test_img in test_imgs:
            test_img_path = os.path.join(test_source, test_img)
            # pdb.set_trace()
            os.system(f"/bin/cp -rf  {test_img_path} {temp_raw}")
        for val_img in val_imgs:
            val_img_path = os.path.join(val_source, val_img)
            os.system(f"/bin/cp -rf  {val_img_path} {temp_raw}")
        for train_img in train_imgs:
            train_img_path = os.path.join(train_source, train_img)
            os.system(f"/bin/cp -rf  {train_img_path} {temp_raw}")

        # For makeup
        makeup_path = os.path.join(makeup_source, os.listdir(makeup_source)[0])
        os.system(f"/bin/cp -rf  {makeup_path} {temp_makeup}")      
        os.makedirs(os.path.join(name_root, "warp"), exist_ok=True)
        os.makedirs(os.path.join(name_root, "mask_makeup"), exist_ok=True)

for name in names:
    for style in styles:

        name_root = os.path.join(root, name + "_" + style)

        print("1: directories")
        ################
        # 1.directories
        ################
        # Temp directories
        os.makedirs(name_root, exist_ok=True)
        temp_raw = os.path.join(name_root, "before")
        temp_makeup = os.path.join(name_root, "after")
        temp_warp = os.path.join(name_root, "warp")
        temp_mask_makeup = os.path.join(name_root, "mask_makeup")
        os.makedirs(os.path.join(name_root, "raw"), exist_ok=True)
        os.makedirs(os.path.join(name_root, "makeup"), exist_ok=True)
        os.makedirs(os.path.join(name_root, "warp"), exist_ok=True)
        os.makedirs(os.path.join(name_root, "mask_makeup"), exist_ok=True)
        
        # Load origin 
        name_source = os.path.join(source_root, name)
        test_source = os.path.join(name_source, "test")
        val_source = os.path.join(name_source, "val")
        makeup_source = os.path.join(name_source, style, "after")
        #/data/hanxinyang/MuNeRF_latest/dataset/boy4/00005/after

        # Target 
        # mask_target = "/data/hanxinyang/MuNeRF_latest/dataset/boy4/00005/mask/warp_makeup_00005"
        mask_target = os.path.join(name_source, style, "mask",  "warp_makeup_%s"%style)
        # /data/hanxinyang/MuNeRF_latest/dataset/boy4/00005/warp_makeup_00005
        warp_target = os.path.join(name_source, style, "warp_makeup_%s"%style)

        print("3:warp")
        ################
        # 3.warp
        ################
        temp_ldmk = os.path.join(name_root, "landmark.pk")
        # pdb.set_trace()
        os.system(f"cd /data/hanxinyang/MuNeRF_latest/process_data_hy/makeup-transfer-public && \
                    /data/hanxinyang/miniconda3/envs/makeup-transfer/bin/python src/handle_datasetH.py --type landmark --input {name_root} --output {temp_ldmk}")

for name in names:

    for style in styles:
        name_root = os.path.join(root, name + "_" + style)

        print("1: directories")
        ################
        # 1.directories
        ################
        # Temp directories
        os.makedirs(name_root, exist_ok=True)
        temp_raw = os.path.join(name_root, "before")
        temp_makeup = os.path.join(name_root, "after")
        temp_warp = os.path.join(name_root, "warp")
        temp_mask_makeup = os.path.join(name_root, "mask_makeup")
        os.makedirs(os.path.join(name_root, "raw"), exist_ok=True)
        os.makedirs(os.path.join(name_root, "makeup"), exist_ok=True)
        os.makedirs(os.path.join(name_root, "warp"), exist_ok=True)
        os.makedirs(os.path.join(name_root, "mask_makeup"), exist_ok=True)
        
        # Load origin 
        name_source = os.path.join(source_root, name)
        test_source = os.path.join(name_source, "test")
        val_source = os.path.join(name_source, "val")
        makeup_source = os.path.join(name_source, style, "after")
        #/data/hanxinyang/MuNeRF_latest/dataset/boy4/00005/after

        # Target 
        # mask_target = "/data/hanxinyang/MuNeRF_latest/dataset/boy4/00005/mask/warp_makeup_00005"
        mask_target = os.path.join(name_source, style, "mask",  "warp_makeup_%s"%style)
        # /data/hanxinyang/MuNeRF_latest/dataset/boy4/00005/warp_makeup_00005
        temp_ldmk = os.path.join(name_root, "landmark.pk")
        warp_target = os.path.join(name_source, style, "warp_makeup_%s"%style)
        os.system(f"cd /data/hanxinyang/MuNeRF_latest/process_data_hy/makeup-transfer-public && \
                    /data/hanxinyang/miniconda3/envs/makeup-transfer/bin/python src/handle_datasetH.py --type blend1 --input {name_root} --output {temp_warp} --landmark_input {temp_ldmk} --keep_eye_mouth --include_forehead")


for name in names:
    for style in styles:
        name_root = os.path.join(root, name + "_" + style)

        print("1: directories")
        ################
        # 1.directories
        ################
        # Temp directories
        os.makedirs(name_root, exist_ok=True)
        temp_raw = os.path.join(name_root, "before")
        temp_makeup = os.path.join(name_root, "after")
        temp_warp = os.path.join(name_root, "warp")
        temp_mask_makeup = os.path.join(name_root, "mask_makeup")
        temp_mask_raw = os.path.join(name_root, "mask_raw")
        os.makedirs(os.path.join(name_root, "raw"), exist_ok=True)
        os.makedirs(os.path.join(name_root, "makeup"), exist_ok=True)
        os.makedirs(os.path.join(name_root, "warp"), exist_ok=True)
        os.makedirs(os.path.join(name_root, "mask_makeup"), exist_ok=True)
        os.makedirs(os.path.join(name_root, "mask_raw"), exist_ok=True)
        
        # Load origin 
        name_source = os.path.join(source_root, name)
        test_source = os.path.join(name_source, "test")
        val_source = os.path.join(name_source, "val")
        makeup_source = os.path.join(name_source, style, "after")
        #/data/hanxinyang/MuNeRF_latest/dataset/boy4/00005/after

        # Target 
        # mask_target = "/data/hanxinyang/MuNeRF_latest/dataset/boy4/00005/mask/warp_makeup_00005"
        mask_target = os.path.join(name_source, style, "mask",  "warp_makeup_%s"%style)
        # /data/hanxinyang/MuNeRF_latest/dataset/boy4/00005/warp_makeup_00005
        warp_target = os.path.join(name_source, style, "warp_makeup_%s"%style)

        print("4:mask")
        ################
        # 4.mask + handle mask
        ################
        print(f"Name : {name}, style : {style}")
        os.system(f"cd /data/hanxinyang/MuNeRF_latest/process_data_hy/makeup-transfer-public/face-parsing.PyTorch-master && \
                    CUDA_VISIBLE_DEVICES=2 /data/hanxinyang/miniconda3/envs/makeup-transfer/bin/python /data/hanxinyang/MuNeRF_latest/process_data_hy/makeup-transfer-public/face-parsing.PyTorch-master/testH.py --respth {temp_mask_makeup} --dspth {temp_warp}")

        os.system(f"cd /data/hanxinyang/MuNeRF_latest/process_data_hy/makeup-transfer-public/face-parsing.PyTorch-master && \
                    CUDA_VISIBLE_DEVICES=2 /data/hanxinyang/miniconda3/envs/makeup-transfer/bin/python /data/hanxinyang/MuNeRF_latest/process_data_hy/makeup-transfer-public/face-parsing.PyTorch-master/testH.py --respth {temp_mask_raw} --dspth {temp_raw}")
                    
# for name in ['girl10']:
# # for name in names:
#     for style in styles:
#         name_root = os.path.join(root, name + "_" + style)

#         print("1: directories")
#         ################
#         # 1.directories
#         ################
#         # Temp directories
#         os.makedirs(name_root, exist_ok=True)
#         temp_raw = os.path.join(name_root, "before")
#         temp_makeup = os.path.join(name_root, "after")
#         temp_warp = os.path.join(name_root, "warp")
#         temp_mask_makeup = os.path.join(name_root, "mask_makeup")
#         temp_mask_raw = os.path.join(name_root, "mask_raw")
#         os.makedirs(os.path.join(name_root, "raw"), exist_ok=True)
#         os.makedirs(os.path.join(name_root, "makeup"), exist_ok=True)
#         os.makedirs(os.path.join(name_root, "warp"), exist_ok=True)
#         os.makedirs(os.path.join(name_root, "mask_makeup"), exist_ok=True)
#         os.makedirs(os.path.join(name_root, "mask_raw"), exist_ok=True)
        
#         # Load origin 
#         name_source = os.path.join(source_root, name)
#         test_source = os.path.join(name_source, "test")
#         val_source = os.path.join(name_source, "val")
#         makeup_source = os.path.join(name_source, style, "after")
#         #/data/hanxinyang/MuNeRF_latest/dataset/boy4/00005/after

#         # Target 
#         # mask_target = "/data/hanxinyang/MuNeRF_latest/dataset/boy4/00005/mask/warp_makeup_00005"
#         mask_target = os.path.join(name_source, style, "mask",  "warp_makeup_%s"%style)
#         # /data/hanxinyang/MuNeRF_latest/dataset/boy4/00005/warp_makeup_00005
#         warp_target = os.path.join(name_source, style, "warp_makeup_%s"%style)
#         print("Changing indexs %s %s"%(name, style))
        
#         # Handle name
        
#         # for sub in [temp_mask_makeup, temp_mask_raw, temp_warp]:
#         #     things = os.listdir(sub)
#         #     for thing in things:
#         #         dir = os.path.join(sub, thing)
#         #         if len(thing) > 8:
#         #             prefix = thing[:4]
#         #             new_dir = os.path.join(sub, prefix + ".png")
#         #             os.system(f"/bin/cp -rf {dir} {new_dir}")
        
        
#         # Handle mask indexs
#         os.system(f"python /data/hanxinyang/MuNeRF_latest/process_data_hy/makeup-transfer-public/face-parsing.PyTorch-master/change_parsing.py --target1 {temp_mask_makeup}")
#         os.system(f"python /data/hanxinyang/MuNeRF_latest/process_data_hy/makeup-transfer-public/face-parsing.PyTorch-master/change_parsing.py --target1 {temp_mask_raw}")



# for name in names:
#     for style in styles:
#         name_root = os.path.join(root, name + "_" + style)

#         print("1: directories")
#         ################
#         # 1.directories
#         ################
#         # Temp directories
#         os.makedirs(name_root, exist_ok=True)
#         temp_raw = os.path.join(name_root, "before")
#         temp_makeup = os.path.join(name_root, "after")
#         temp_warp = os.path.join(name_root, "warp")
#         temp_mask_raw = os.path.join(name_root, "mask_raw")
#         temp_mask_makeup = os.path.join(name_root, "mask_makeup")
#         os.makedirs(os.path.join(name_root, "raw"), exist_ok=True)
#         os.makedirs(os.path.join(name_root, "makeup"), exist_ok=True)
#         os.makedirs(os.path.join(name_root, "warp"), exist_ok=True)
#         os.makedirs(os.path.join(name_root, "mask_makeup"), exist_ok=True)
        
#         # Load origin 
#         name_source = os.path.join(source_root, name)
#         test_source = os.path.join(name_source, "test")
#         val_source = os.path.join(name_source, "val")
#         makeup_source = os.path.join(name_source, style, "after")
#         #/data/hanxinyang/MuNeRF_latest/dataset/boy4/00005/after

#         # Target 
#         # mask_target = "/data/hanxinyang/MuNeRF_latest/dataset/boy4/00005/mask/warp_makeup_00005"
#         mask_target_makeup = os.path.join(name_source, style, "mask",  "warp_makeup_%s"%style)
#         mask_target_raw = os.path.join(name_source, style, "mask",  "nonmakeup")
#         # /data/hanxinyang/MuNeRF_latest/dataset/boy4/00005/warp_makeup_00005
#         warp_target = os.path.join(name_source, style, "warp_makeup_%s"%style)

#         print("5:to target")
#         ################
#         # move handled to target
#         ################
#         # warp
#         for thing in os.listdir(temp_warp):
#             warp_path = os.path.join(temp_warp, thing)
#             os.system(f"/bin/cp -rf {warp_path} {warp_target}")

#         # mask makeup
#         for thing in os.listdir(temp_mask_makeup):
#             mask_path = os.path.join(temp_mask_makeup, thing)
#             os.system(f"/bin/cp -rf  {mask_path} {mask_target_makeup}")

#         # mask raw
#         for thing in os.listdir(temp_mask_raw):
#             mask_path = os.path.join(temp_mask_raw, thing)
#             os.system(f"/bin/cp -rf  {mask_path} {mask_target_raw}")