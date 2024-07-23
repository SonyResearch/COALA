python application/ClusterFL/main_fedavg.py --dataset  digits_five --split_type iid --num_of_clients 30 --clients_per_round 30 --local_epoch 5 --round 100 --gpus 4
python application/ClusterFL/main_fedavg.py --dataset  cifar10 --split_type dir --num_of_clients 30 --clients_per_round 30 --local_epoch 5 --round 150 --gpus 4

python application/ClusterFL/main_fedrep.py --dataset  cifar10 --split_type dir --num_of_clients 30 --clients_per_round 30 --local_epoch 5 --round 200 --gpus 4
python application/ClusterFL/main_fedrep.py --dataset  digits_five --split_type iid --num_of_clients 30 --clients_per_round 30 --local_epoch 5 --round 100 --gpus 4

python application/ClusterFL/main.py --dataset  digits_five --split_type iid --num_of_clients 30 --clients_per_round 30 --local_epoch 5 --round 100 --gpus 4
python application/ClusterFL/main.py --dataset  cifar10 --split_type dir --num_of_clients 30 --clients_per_round 30 --local_epoch 5 --round 150 --gpus 4