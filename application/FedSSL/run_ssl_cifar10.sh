python application/FedSSL/main.py --gpus 4 --dataset cifar10 --data_partition dir --num_of_clients 4 --clients_per_round 4 --update_encoder online --update_predictor global
python application/FedSSL/main.py --gpus 4 --dataset cifar10 --data_partition dir --num_of_clients 20 --clients_per_round 20 --update_encoder online --update_predictor global
python application/FedSSL/main.py --gpus 4 --dataset cifar10 --class_per_client 10 --num_of_clients 4 --clients_per_round 4 --update_encoder online --update_predictor global
python application/FedSSL/main.py --gpus 4 --dataset cifar10 --class_per_client 10 --num_of_clients 20 --clients_per_round 20 --update_encoder online --update_predictor global