mainpath=/data/heyue/104/dataset/$1
mkdir "${mainpath}"/related/
mkdir "${mainpath}"/related/video

# beautygan ref2
cd ./BeautyGAN
mkdir "${mainpath}"/related/beautygan
python3 ./main.py --no_makeup "${mainpath}"/frames --makeuped "${mainpath}"/makeup/$2.jpg --output "${mainpath}"/related/beautygan/$2/
python /data/heyue/104/tools/readvideo.py --frame_dir "${mainpath}"/related/beautygan/$2/ --video_dir "${mainpath}"/related/video/ --id beautygan_$2 --video
# beautygan ref3
#python3 ./main.py --no_makeup "${mainpath}"/frames --makeuped "${mainpath}"/makeup/$3.jpg --output "${mainpath}"/related/beautygan/$3/
#python /data/heyue/104/tools/readvideo.py --frame_dir "${mainpath}"/related/beautygan/$3/ --video_dir "${mainpath}"/related/video/ --id beautygan_$3 --video
cd ..

cd ./PSGAN-master
## psgan ref2
mkdir "${mainpath}"/related/psgan
mkdir "${mainpath}"/related/psgan/$2
python3 ./demo.py --source_dir "${mainpath}"/frames --reference_path "${mainpath}"/makeup/$2.jpg --save_path "${mainpath}"/related/psgan/$2/
#python /data/heyue/104/tools/readvideo.py --frame_dir "${mainpath}"/related/psgan/$2/ --video_dir "${mainpath}"/related/video/ --id psgan_$2 --video
# # psgan ref3
#mkdir "${mainpath}"/related/psgan/$3
#python3 ./demo.py --source_dir "${mainpath}"/frames --reference_path "${mainpath}"/makeup/$3.jpg --save_path "${mainpath}"/related/psgan/$3/
#python /data/heyue/104/tools/readvideo.py --frame_dir "${mainpath}"/related/psgan/$3/ --video_dir "${mainpath}"/related/video/ --id psgan_$3 --video
cd ..

cd ./SSAT-master
# # ssat
mkdir "${mainpath}"/related/ssat
mkdir "${mainpath}"/related/ssat/$2
#mkdir "${mainpath}"/related/ssat/$3
# -----make face mask for 
rm /data/heyue/face_mask/data/*
rm /data/heyue/face_mask/res/test_res1/* 
cp "${mainpath}"/frames/* /data/heyue/face_mask/data/
CUDA_VISIBLE_DEVICES=3 python ../../face_mask/test.py --ckpt_path '/data/heyue/face_mask/res/cp' --respth '/data/heyue/face_mask/res/test_res' --dspth '/data/heyue/face_mask/data'
mkdir ./testhy/images/non-makeup
mkdir ./testhy/images/makeup
mkdir ./testhy/seg1/non-makeup
mkdir ./testhy/seg1/makeup
cp "${mainpath}"/frames/* ./testhy/images/non-makeup/
cp "${mainpath}"/makeup/$2.jpg ./testhy/images/makeup/
cp /data/heyue/face_mask/res/test_res1/* ./testhy/seg1/non-makeup/
rm /data/heyue/face_mask/res/test_res1/*
rm /data/heyue/face_mask/data/*
cp "${mainpath}"/makeup/$2.jpg /data/heyue/face_mask/data/
CUDA_VISIBLE_DEVICES=3 python ../../face_mask/test.py --ckpt_path '/data/heyue/face_mask/res/cp' --respth '/data/heyue/face_mask/res/test_res' --dspth '/data/heyue/face_mask/data'
cp /data/heyue/face_mask/res/test_res1/* ./testhy/seg1/makeup/
python ./test.py --dataroot ./testhy/images --phase test --name $2 --result_dir "${mainpath}"/related/ssat/
#python /data/heyue/104/tools/readvideo.py --frame_dir "${mainpath}"/related/ssat/$2/ --video_dir "${mainpath}"/related/video/ --id ssat_$2 --video
#make face mask for 
rm /data/heyue/face_mask/data/*
rm /data/heyue/face_mask/res/test_res1/*
rm ./testhy/images/makeup/*
rm ./testhy/seg1/makeup/*
#cp "${mainpath}"/makeup/$3.jpg ./testhy/images/makeup/
#cp "${mainpath}"/makeup/$3.jpg /data/heyue/face_mask/data/
#python ../../face_mask/test.py --ckpt_path '/data/heyue/face_mask/res/cp' --respth '/data/heyue/face_mask/res/test_res' --dspth '/data/heyue/face_mask/data'
#cp /data/heyue/face_mask/res/test_res1/* ./testhy/seg1/makeup/
# ssat ref3
#python ./test.py --dataroot ./testhy/images --phase test --name $3 --result_dir "${mainpath}"/related/ssat/
#python /data/heyue/104/tools/readvideo.py --frame_dir "${mainpath}"/related/ssat/$3/ --video_dir "${mainpath}"/related/video/ --id ssat_$3 --video
rm /data/heyue/face_mask/res/test_res1/* 
rm -r ./testhy/images/*
rm -r ./testhy/seg1/*
cd ..

cd ./SCGAN-master
#SCGAN
mkdir "${mainpath}"/related/scgan
mkdir "${mainpath}"/related/scgan/$2
#mkdir "${mainpath}"/related/scgan/$3
rm ./MT-Dataset/hy/makeup/*
rm ./MT-Dataset/hy/non-makeup/*
cp "${mainpath}"/makeup/$2.jpg ./MT-Dataset/hy/makeup/
cp "${mainpath}"/frames/* ./MT-Dataset/hy/non-makeup/
rm ./MT-Dataset/parsing/makeup/*
rm ./MT-Dataset/parsing/non-makeup/*
CUDA_VISIBLE_DEVICES=3 python ./get_masks.py
CUDA_VISIBLE_DEVICES=3 python ./get_masks.py --source_dir ./MT-Dataset/hy/non-makeup --save_dir ./MT-Dataset/parsing
rm ./test.txt
python ./writelines.py --id $2.jpg > ./test.txt
CUDA_VISIBLE_DEVICES=3 python ./sc.py --phase test --dataroot ./MT-Dataset/hy --dirmap ./MT-Dataset/parsing --save_path "${mainpath}"/related/scgan/$2
#python /data/heyue/104/tools/readvideo.py --frame_dir "${mainpath}"/related/scgan/$2/ --video_dir "${mainpath}"/related/video/ --id scgan_$2 --video
rm ./MT-Dataset/parsing/makeup/*
rm ./test.txt
#python ./writelines.py --id $3.jpg > ./test.txt
#rm ./MT-Dataset/hy/makeup/*
#cp "${mainpath}"/makeup/$3.jpg ./MT-Dataset/hy/makeup/
#python ./get_masks.py
#CUDA_VISIBLE_DEVICES=2 python ./sc.py --phase test --dataroot ./MT-Dataset/hy --dirmap ./MT-Dataset/parsing --save_path "${mainpath}"/related/scgan/$3
#python /data/heyue/104/tools/readvideo.py --frame_dir "${mainpath}"/related/scgan/$3/ --video_dir "${mainpath}"/related/video/ --id scgan_$3 --video
cd ..

cd ./CPM-main
# #CPM
mkdir "${mainpath}"/related/cpm
mkdir "${mainpath}"/related/cpm/$2
# #mkdir "${mainpath}"/related/cpm/$3
rm -r ./imgs/*
mkdir ./imgs/non
cp "${mainpath}"/makeup/$2.jpg ./imgs/
# #cp "${mainpath}"/makeup/$3.jpg ./imgs/
cp "${mainpath}"/frames/* ./imgs/non/
/home/heyue/miniconda3/envs/cpm/bin/python3 ./main.py --input ./imgs/non/ --style ./imgs/$2.jpg --savedir "${mainpath}"/related/cpm/$2
# /home/heyue/miniconda3/envs/cpm/bin/python3 /data/heyue/104/tools/readvideo.py --frame_dir "${mainpath}"/related/cpm/$2/ --video_dir "${mainpath}"/related/video/ --id cpm_$2 --video
#/home/heyue/miniconda3/envs/cpm/bin/python3 ./main.py --input ./imgs/non/ --style ./imgs/$3.jpg --savedir "${mainpath}"/related/cpm/$3
#/home/heyue/miniconda3/envs/cpm/bin/python3 /data/heyue/104/tools/readvideo.py --frame_dir "${mainpath}"/related/cpm/$3/ --video_dir "${mainpath}"/related/video/ --id cpm_$3 --video
cd ..

# #warp
#python /data/heyue/104/tools/readvideo.py --frame_dir "${mainpath}"/$2/warp_makeup_$2/ --video_dir "${mainpath}"/related/video/ --id warp_$2 --video
# #python /data/heyue/104/tools/readvideo.py --frame_dir "${mainpath}"/$3/warp_makeup_$3/ --video_dir "${mainpath}"/related/video/ --id warp_$3 --video

























































































































