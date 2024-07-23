import coala
import argparse
import torch.multiprocessing as mp
from fl_client import FedAvgClient, ContrastiveClient
from models import SupConResNet, CNNDigit, AlexNet

def run(args, rank=0):
    assert args.dataset in ['cifar10', 'cifar10.1', 'digits_five', 'office_caltech', 'domainnet']
    # model initialization
    if args.dataset in ['cifar10', 'cifar10.1']:
        model = SupConResNet()
    elif args.dataset == 'digits_five':
        model = CNNDigit()
    elif args.dataset in ['office_caltech', 'domainnet']:
        model = AlexNet()
    else:
        raise NotImplementedError
    
    if args.shift_type == "f_shift":
        assert args.dataset == "cifar10.1"
    
    client = FedAvgClient if args.method == "fedavg" else ContrastiveClient

    # Define customized configurations.
    config = {
        "task_id": args.task_id,
        "data": {
            "root": "./data/",
            "dataset": args.dataset,
            "num_of_clients": args.num_of_clients,
            "split_type": args.split_type,
            "train_test_split": 0.7
        },
        "server": {
            "rounds": args.rounds,
            "clients_per_round": args.clients_per_round,
            "test_every": args.test_every,
            "test_all": True,  # Whether test all clients or only selected clients.
            "random_selection": True,
            "save_model_every": args.save_model_every
        },
        "client": {
            "local_epoch": args.local_epoch,
            "ft_epoch": args.ft_epoch,
            "rounds": args.rounds,
            "test_batch_size": args.test_batch_size,
            "batch_size": args.batch_size,
            "local_test": args.local_test,
            "optimizer": {
                "type": args.optimizer_type,
                "lr": args.lr,
                "lr_sc": args.lr_sc,
                "momentum": args.momentum,
                "weight_decay": args.weight_decay,
            },
            "temperature": args.temperature,
            "base_temperature": args.base_temperature,
            "shift_type": args.shift_type,
            "severity": args.severity,
            "seed": args.seed
        },

        "gpu": args.gpus,
        "test_mode": "test_in_client",

        "is_remote": False,
        "local_port": 22,
    }

    if args.gpus > 1:
        config.update({
            "distributed": {
                "rank": rank,
                "local_rank": rank,
                "world_size": args.gpus,
                "init_method": "tcp://127.0.0.1:8123"
            }
        })

    # Initialize with the new config.
    coala.register_model(model)
    coala.register_client(client)
    coala.init(config)
    # Execute federated learning training.
    coala.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test-Time Adaptation')
    parser.add_argument("--task_id", type=str, default="test_time")
    parser.add_argument('--dataset', type=str, default='digits_five', help="name of dataset, e.g., cifar10")
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--split_type', default="iid", type=str)
    parser.add_argument('--shift_type', type=str, default='original',
                        help="original, c_shift, f_shift, d_shift")
    parser.add_argument('--severity', type=int, default=5, help="severity of corruption")
    # 'c_shift' for image corruption, 'f_shift' for cifar10.1 natural shift, d_shift for domain shift

    parser.add_argument('--method', type=str, default="fedavg", help="fl algorithm, fedavg, fedicon")

    parser.add_argument('--gpus', default=1, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--test_batch_size', default=128, type=int)
    parser.add_argument('--local_epoch', default=5, type=int)
    parser.add_argument('--ft_epoch', default=5, type=int)
    parser.add_argument('--rounds', default=100, type=int)
    parser.add_argument('--num_of_clients', default=4, type=int)
    parser.add_argument('--clients_per_round', default=4, type=int)

    parser.add_argument('--optimizer_type', default='SGD', type=str, help='optimizer type')
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--lr_sc', default=0.005, type=float, help='learning rate for SupConLoss')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='sc: 1e-4, ce: 5e-4')
    parser.add_argument('--temperature', type=float, default=0.1,
                        help="temperature of contrastive loss.")
    parser.add_argument('--base_temperature', type=float, default=0.07,
                        help="base temperature of contrastive loss.")

    parser.add_argument('--local_test', action='store_true', help="Whether use test local models after training")
    parser.add_argument('--test_every', default=2, type=int, help='test every x rounds')
    parser.add_argument('--save_model_every', default=100, type=int, help='save models every x rounds')
    parser.add_argument('--seed', default=0, type=int, help='random seed')

    args = parser.parse_args()
    print("arguments: ", args)

    if args.gpus <= 1:
        run(args)
    else:
        mp.set_start_method("spawn")
        processes = []
        for rank in range(args.gpus):
            p = mp.Process(target=run, args=(args, rank))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
