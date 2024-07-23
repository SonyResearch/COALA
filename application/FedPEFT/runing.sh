# LoRA
python application/FedPEFT/main.py --dataset cifar10  --num_of_clients 100  --gpus 4 --rounds 50 --local_epoch 1 --clients_per_round 10 --r 1  --split_type iid --train_type lora
python application/FedPEFT/main.py --dataset cifar10  --num_of_clients 100  --gpus 4 --rounds 50 --local_epoch 1 --clients_per_round 10 --r 8 --split_type iid --train_type lora
python application/FedPEFT/main.py --dataset cifar10  --num_of_clients 100  --gpus 4 --rounds 50 --local_epoch 1 --clients_per_round 10 --r 32 --split_type iid --train_type lora
python application/FedPEFT/main.py --dataset cifar10  --num_of_clients 100  --gpus 4 --rounds 50 --local_epoch 1 --clients_per_round 10 --r 1  --split_type dir --train_type lora
python application/FedPEFT/main.py --dataset cifar10  --num_of_clients 100  --gpus 4 --rounds 50 --local_epoch 1 --clients_per_round 10 --r 8 --split_type dir --train_type lora
python application/FedPEFT/main.py --dataset cifar10  --num_of_clients 100  --gpus 4 --rounds 50 --local_epoch 1 --clients_per_round 10 --r 32 --split_type dir --train_type lora

python application/FedPEFT/main.py --dataset cifar100 --num_of_clients 100  --gpus 4 --rounds 50 --local_epoch 1 --clients_per_round 10 --r 1  --split_type iid --train_type lora --num_classes 100
python application/FedPEFT/main.py --dataset cifar100 --num_of_clients 100  --gpus 4 --rounds 50 --local_epoch 1 --clients_per_round 10 --r 8 --split_type iid --train_type lora --num_classes 100
python application/FedPEFT/main.py --dataset cifar100 --num_of_clients 100  --gpus 4 --rounds 50 --local_epoch 1 --clients_per_round 10 --r 32 --split_type iid --train_type lora --num_classes 100
python application/FedPEFT/main.py --dataset cifar100 --num_of_clients 100  --gpus 4 --rounds 50 --local_epoch 1 --clients_per_round 10 --r 1  --split_type dir --train_type lora --num_classes 100
python application/FedPEFT/main.py --dataset cifar100 --num_of_clients 100  --gpus 4 --rounds 50 --local_epoch 1 --clients_per_round 10 --r 8 --split_type dir --train_type lora --num_classes 100
python application/FedPEFT/main.py --dataset cifar100 --num_of_clients 100  --gpus 4 --rounds 50 --local_epoch 1 --clients_per_round 10 --r 32 --split_type dir --train_type lora --num_classes 100

python application/FedPEFT/main.py --dataset domainnet --num_of_clients 6  --gpus 3 --rounds 50 --local_epoch 1 --clients_per_round 6 --r 1  --split_type iid --train_type lora --test_mode test_in_client
python application/FedPEFT/main.py --dataset domainnet --num_of_clients 6  --gpus 3 --rounds 50 --local_epoch 1 --clients_per_round 6 --r 8 --split_type iid --train_type lora --test_mode test_in_client
python application/FedPEFT/main.py --dataset domainnet --num_of_clients 6  --gpus 3 --rounds 50 --local_epoch 1 --clients_per_round 6 --r 32 --split_type iid --train_type lora --test_mode test_in_client

# Head
python application/FedPEFT/main.py --dataset cifar10   --num_of_clients 100  --gpus 4 --rounds 50 --local_epoch 1 --clients_per_round 10  --split_type iid --train_type linear
python application/FedPEFT/main.py --dataset cifar10   --num_of_clients 100  --gpus 4 --rounds 50 --local_epoch 1 --clients_per_round 10  --split_type dir --train_type linear
python application/FedPEFT/main.py --dataset cifar100  --num_of_clients 100  --gpus 4 --rounds 50 --local_epoch 1 --clients_per_round 10  --split_type iid --train_type linear --num_classes 100
python application/FedPEFT/main.py --dataset cifar100  --num_of_clients 100  --gpus 4 --rounds 50 --local_epoch 1 --clients_per_round 10  --split_type dir --train_type linear --num_classes 100
python application/FedPEFT/main.py --dataset domainnet --num_of_clients 6    --gpus 3 --rounds 50 --local_epoch 1 --clients_per_round 6   --split_type iid --train_type linear --test_mode test_in_client

