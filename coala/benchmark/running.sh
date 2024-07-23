python coala/benchmark/cifar_benchmark.py --gpus 4  --dataset cifar10  --num_of_clients 100 --split_type iid 
python coala/benchmark/cifar_benchmark.py --gpus 4  --dataset cifar10  --num_of_clients 100 --split_type dir 
python coala/benchmark/cifar_benchmark.py --gpus 4  --dataset cifar10  --num_of_clients 100 --split_type class --class_per_client 2
python coala/benchmark/cifar_benchmark.py --gpus 4  --dataset cifar100  --num_of_clients 100 --split_type iid 
python coala/benchmark/cifar_benchmark.py --gpus 4  --dataset cifar100  --num_of_clients 100 --split_type dir 
python coala/benchmark/cifar_benchmark.py --gpus 4  --dataset cifar100  --num_of_clients 100 --split_type class --class_per_client 20
python coala/benchmark/digits5_benchmark.py --gpus 4 
python coala/benchmark/office_benchmark.py --gpus 4 
python coala/benchmark/domainnet_benchmark.py --gpus 4 
python coala/benchmark/cifar_semi_test.py --gpus 1  --semi_scenario label_in_server --num_labels_per_class 400 --split_type iid
python coala/benchmark/cifar_semi_test.py --gpus 1  --semi_scenario label_in_server --num_labels_per_class 400 --split_type dir
python coala/benchmark/cifar_semi_test.py --gpus 1  --semi_scenario label_in_client --num_labels_per_class 10 --split_type iid
python coala/benchmark/cifar_semi_test.py --gpus 1  --semi_scenario label_in_client --num_labels_per_class 10 --split_type dir