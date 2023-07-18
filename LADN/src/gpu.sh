var=0
while [ $var -eq 0 ]
do
    count=0
    for i in $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    do
        if [ $i -lt 500 ]
        then
            echo 'GPU'$count' is avaiable'
             CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1,3 python3 run.py --backup_gpu 1 --dataroot ../datasets/makeup --name makeup --resize_size 576 --crop_size 512 --local_style_dis --n_local 12 --local_laplacian_loss --local_smooth_loss --no_extreme
            # 此处加上train 模型的语句
            var=1
            break
        fi
        count=$(($count+1))    
    done
done
