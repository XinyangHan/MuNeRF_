# python data_util/process_data.py --id=$1 --step=0 &
# CUDA_VISIBLE_DEVICES=2 python data_util/process_data.py --id=$1 --step=1
# CUDA_VISIBLE_DEVICES=2 python data_util/process_data.py --id=$1 --step=2
# python data_util/process_data.py --id=$1 --step=6 &
# python data_util/process_data.py --id=$1 --step=3
# python data_util/process_data.py --id=$1 --step=4
# python data_util/process_data.py --id=$1 --step=5
# wait
# python data_util/process_data.py --id=$1 --step=7

# python data_util/process_data.py --id=$1 --step=2
CUDA_VISIBLE_DEVICES=3 python data_util/process_data.py --id=$1 --step=6
