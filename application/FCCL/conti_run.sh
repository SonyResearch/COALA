# examples
python application/FCCL/main.py --gpus 4  --dataset cifar10 --split_type iid --num_tasks 5 --synthesis False
python application/FCCL/main.py --gpus 4  --dataset cifar10 --split_type iid --num_tasks 5 --synthesis True --kd_alpha 25