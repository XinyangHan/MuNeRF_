#make warp_images
#--preprocess--#
mainpath=/104/4D-Facial-Avatars-main/nerface_code/nerf-pytorch/girl/
cd ../../../../makeup-transfer-public/
mkdir ./data/raw/
mkdir ./data/raw/before
mkdir ./data/raw/after
cp ../makeup/non-makeup/* ./data/raw/before/
cp ../makeup/makeup/* ./data/raw/after/
#--run--#
python src/handle_dataset.py --type filter --input data/raw/ --output data/filter/ --detector stasm
python src/handle_dataset.py --type crop --input data/filter/ --output data/crop/
python src/handle_dataset.py --type landmark --input data/crop/ --output data/crop/landmark.pk
python src/handle_dataset.py --type blend --input data/crop/ --output data/crop/blend/ --landmark_input data/crop/landmark.pk --keep_eye_mouth --include_forehead --adjust_color --adjust_lighting
mkdir ./data/crop/blend_color
python ./data/rename.py
#--move--#
mkdir .."${mainpath}"warp_makeup 
cp ./data/crop/blend_color/* .."${mainpath}"warp_makeup/

#make reference images
#--preprocess--#
cd ../
mkdir ."${mainpath}"frames
python ./104/tools/filter.py --input ."${mainpath}"whole --refer ."${mainpath}"warp_makeup --output ."${mainpath}"frames
python ./104/tools/readvideo.py --id train_ori --input ."${mainpath}"frames/ --output ./first-order-model/source/ --video
cd ./first-order-model
#--run--#
deactivate
source /mnt/d/heyue/first/venv/bin/activate
python demo.py  --config config/vox-256.yaml --driving_video ./source/train_ori.mp4 --source_image ../makeup/makeup/* --checkpoint ./ckpt/vox-cpk.pth --relative --adapt_scale
deactivate
source /mnt/d/heyue/mypro/venv/bin/activate
#--move--#
mainpath1=/4D-Facial-Avatars-main/nerface_code/nerf-pytorch/girl/
cd ..
cd ./104/tools/
mkdir .."${mainpath1}"styles_ori
python ./readvideo.py --id result --input .."${mainpath1}"styles_ori/ --output ../../first-order-model/
mkdir .."${mainpath1}"styles
python ./name_align.py --input .."${mainpath1}"styles_ori --refer .."${mainpath1}"warp_makeup --output .."${mainpath1}"styles/
cd /data/heyue/makeup-transfer-public
rm -r ./data/crop/
rm -r ./data/filter/
rm -r ./data/raw/
rm -r /data/heyue/104"${mainpath1}"frames
rm /data/heyue/first-order-model/source/train_ori.mp4






























