mainpath=/data/heyue/104/4D-Facial-Avatars-main/nerface_code/nerf-pytorch/datasets/$1
mkdir "${mainpath}"/related/
mkdir "${mainpath}"/related/video

# beautygan ref2
mkdir "${mainpath}"/related/beautygan
python3 ./BeautyGAN/main.py --no_makeup "${mainpath}"/frames --makeuped "${mainpath}"/makeup/$2.jpg --output "${mainpath}"/related/beautygan/
python /data/heyue/104/tools/readvideo.py --input "${mainpath}"/related/beautygan/$2 --output "${mainpath}"/related/video/ --id beautygan_$2 --video
# beautygan ref3
python3 ./BeautyGAN/main.py --no_makeup "${mainpath}"/frames --makeuped "${mainpath}"/makeup/$3.jpg --output "${mainpath}"/related/beautygan/
python /data/heyue/104/tools/readvideo.py --input "${mainpath}"/related/beautygan/$3 --output "${mainpath}"/related/video/ --id beautygan_$3 --video

# psgan ref2
mkdir "${mainpath}"/related/psgan
mkdir "${mainpath}"/related/psgan/$2
python3 ./PSGAN-master/demo.py --source_dir "${mainpath}"/frames --reference_path "${mainpath}"/makeup/$2.jpg --save_path "${mainpath}"/related/psgan/$2
python /data/heyue/104/tools/readvideo.py --input "${mainpath}"/related/psgan/$2 --output "${mainpath}"/related/video/ --id psgan$2 --video
# psgan ref3
mkdir "${mainpath}"/related/psgan/$3
python3 ./PSGAN-master/demo.py --source_dir "${mainpath}"/frames --reference_path "${mainpath}"/makeup/$3.jpg --save_path "${mainpath}"/related/psgan/$3
python /data/heyue/104/tools/readvideo.py --input "${mainpath}"/related/psgan/$3 --output "${mainpath}"/related/video/ --id psgan$3 --video

# ssat
mkdir "${mainpath}"/related/ssat
# -----make face mask for 
rm /data/heyue/face_mask/data/*
rm /data/heyue/face_mask/res/test_res1/* 
cp "${mainpath}"/frames/* /data/heyue/face_mask/data/
python ../face_mask/test.py
mkdir ./SSAT-master/testhy/images/non-makeup
mkdir ./SSAT-master/testhy/images/makeup
mkdir ./SSAT-master/testhy/seg1/non-makeup
mkdir ./SSAT-master/testhy/seg1/makeup
cp "${mainpath}"/frames/* ./SSAT-master/testhy/images/non-makeup/
cp "${mainpath}"/makeup/$2.jpg ./SSAT-master/testhy/images/makeup/
cp /data/heyue/face_mask/res/test_res1/* ./SSAT-master/testhy/seg1/non-makeup/
rm /data/heyue/face_mask/res/test_res1/*
rm /data/heyue/face_mask/data/*
cp "${mainpath}"/makeup/$2.jpg /data/heyue/face_mask/data/
python ../face_mask/test.py
cp /data/heyue/face_mask/res/test_res1/* ./SSAT-master/testhy/seg1/makeup/
python ./SSAT-master/test.py --dataroot "${mainpath}"/frames --phase test --name $2 --result_dir "${mainpath}"/related/ssat/
python /data/heyue/104/tools/readvideo.py --input "${mainpath}"/related/ssat/$2 --output "${mainpath}"/related/video/ --id ssat$2 --video
# -----make face mask for 
rm /data/heyue/face_mask/data/*
rm /data/heyue/face_mask/res/test_res1/*
rm ./SSAT-master/testhy/images/makeup/*
rm ./SSAT-master/testhy/seg1/makeup/*
cp "${mainpath}"/makeup/$3.jpg ./SSAT-master/testhy/images/makeup/
cp "${mainpath}"/makeup/$3.jpg /data/heyue/face_mask/data/
python ../face_mask/test.py
cp /data/heyue/face_mask/res/test_res1/* ./SSAT-master/testhy/seg1/makeup/
# ssat ref3
python ./SSAT-master/test.py --dataroot "${mainpath}"/frames --phase test --name $3 --result_dir "${mainpath}"/related/ssat/
python /data/heyue/104/tools/readvideo.py --input "${mainpath}"/related/ssat/$3 --output "${mainpath}"/related/video/ --id ssat$3 --video
rm /data/heyue/face_mask/res/test_res1/* 
rm -r ./SSAT-master/testhy/images/*
rm -r ./SSAT-master/testhy/seg/*

#SCGAN
mkdir "${mainpath}"/related/scgan
mkdir "${mainpath}"/related/scgan/$2
mkdir "${mainpath}"/related/scgan/$3
cp "${mainpath}"/makeup/$2.jpg ./SCGAN-master/MT-Dataset/hy/makeup/
cp "${mainpath}"/frames/* ./SCGAN-master/MT-Dataset/hy/non-makeup/
rm ./SCGAN-master/MT-Dataset/parsing/makeup/*
rm ./SCGAN-master/MT-Dataset/parsing/non-makeup/*
python ./SCGAN-master/get_masks.py
python ./SCGAN-master/get_masks.py --source_dir ./SCGAN-master/MT-Dataset/hy/non-makeup --save_dir ./SCGAN-master/MT-Dataset/parsing/non-makeup
CUDA_VISIBLE_DEVICES=2 python ./SCGAN-master/sc.py --phase test --dataroot ./MT-Dataset/hy --dirmap ./MT-Dataset/parsing --savedir 

