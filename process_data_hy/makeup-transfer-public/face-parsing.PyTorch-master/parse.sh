# $1 model id
# $2 style id


cd /data/hanxinyang/MuNeRF_latest/process_data_hy/makeup-transfer-public/face-parsing.PyTorch-master
# CUDA_VISIBLE_DEVICES=2 python /data/hanxinyang/MuNeRF_latest/process_data_hy/makeup-transfer-public/face-parsing.PyTorch-master/testH.py --respth /data/hanxinyang/MuNeRF_latest/dataset/$1/$2/mask/nonmakeup --dspth /data/hanxinyang/MuNeRF_latest/dataset/$1/train 

# CUDA_VISIBLE_DEVICES=2 python /data/hanxinyang/MuNeRF_latest/process_data_hy/makeup-transfer-public/face-parsing.PyTorch-master/testH.py --respth /data/hanxinyang/MuNeRF_latest/dataset/$1/$2/mask/nonmakeup --dspth /data/hanxinyang/MuNeRF_latest/dataset/$1/test

# CUDA_VISIBLE_DEVICES=2 python /data/hanxinyang/MuNeRF_latest/process_data_hy/makeup-transfer-public/face-parsing.PyTorch-master/testH.py --respth /data/hanxinyang/MuNeRF_latest/dataset/$1/$2/mask/nonmakeup --dspth /data/hanxinyang/MuNeRF_latest/dataset/$1/val
echo "$1_$2"
CUDA_VISIBLE_DEVICES=2 python /data/hanxinyang/MuNeRF_latest/process_data_hy/makeup-transfer-public/face-parsing.PyTorch-master/testH.py --respth /data/hanxinyang/MuNeRF_latest/dataset/$1/$2/mask/warp_makeup_$2 --dspth /data/hanxinyang/MuNeRF_latest/dataset/$1/$2/warp_makeup_$2

# CUDA_VISIBLE_DEVICES=0 python /data/hanxinyang/MuNeRF_latest/process_data_hy/makeup-transfer-public/face-parsing.PyTorch-master/testH.py --respth /data/hanxinyang/makeup_compare/ssat-msp/dataset/seg1/makeup --dspth /data/hanxinyang/makeup_compare/ssat-msp/dataset/images/makeup 