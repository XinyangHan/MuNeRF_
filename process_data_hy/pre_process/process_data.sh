# transform the images in raw_imgs into 512*512 in file ori_imgs
# python /data/hanxinyang/MuNeRF_latest/dataset/pre_process/resize.py --name girl10

# box.json
# CUDA_VISIBLE_DEVICES=3 python /data/hanxinyang/MuNeRF_latest/process_data_hy/MODnet/run.py --inputdata /data/hanxinyang/MuNeRF_latest/dataset/girl10/ori_imgs/ --jsonfiles /data/hanxinyang/MuNeRF_latest/dataset/girl10/box.json

# train, test, val
python /data/hanxinyang/MuNeRF_latest/process_data_hy/pre_process/yyj/process/process.py --id girl15
python /data/hanxinyang/MuNeRF_latest/process_data_hy/pre_process/yyj/process/process.py --id 2
python /data/hanxinyang/MuNeRF_latest/process_data_hy/pre_process/yyj/process/process.py --id 3
python /data/hanxinyang/MuNeRF_latest/process_data_hy/pre_process/yyj/process/process.py --id 5
