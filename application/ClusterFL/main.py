import coala
import argparse
import torch.multiprocessing as mp
from models import CNNDigit, AlexNet, ResNet18
from fl_client import ClusterClient
from fl_server import ClusterServer


def run(args, rank=0):
    """
    Benchmark the performance of FedAvg on CIFAR-10 dataset with 10 clients.
    Expected test accuracy on IID:
        - 95.33% at round 200
    """
    if args.dataset in ['cifar10']:
        model = ResNet18()
    elif args.dataset == 'digits_five':
        model = CNNDigit()
    elif args.dataset == 'office_caltech':
        model = AlexNet()
    else:
        raise NotImplementedError

    config = {
        "data": {"dataset": args.dataset,
                 "num_of_clients": args.num_of_clients,
                 "split_type": args.split_type,
                 "alpha": args.alpha,
                 },
        "num_of_clusters": args.num_of_clusters,
        "client": {"local_epoch": args.local_epoch,
                   "batch_size": 32,
                   "optimizer": {
                       "type": args.optimizer_type,
                       "lr": args.lr,
                   }
                   },
        "server": {
            "rounds": args.round,
            "clients_per_round": args.clients_per_round,
            "test_every": 1,
            "save_model_every": 20
        },
        "test_mode": "test_in_client",
        "gpu": args.gpus,
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

    coala.register_model(model)
    coala.register_client(ClusterClient)
    coala.register_server(ClusterServer)
    coala.init(config)
    coala.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ClusterFL')
    parser.add_argument('--gpus', default=1, type=int)
    parser.add_argument('--num_of_clients', default=30, type=int)
    parser.add_argument('--clients_per_round', default=30, type=int)
    parser.add_argument('--num_of_clusters', default=5, type=int)
    parser.add_argument('--split_type', default="iid", type=str)
    parser.add_argument('--alpha', default=0.5, type=float)
    parser.add_argument('--dataset', default="digits_five", type=str)
    parser.add_argument('--model', default="CNNDigit", type=str)
    parser.add_argument('--optimizer_type', default='SGD', type=str, help='optimizer type')
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--local_epoch', default=1, type=int)
    parser.add_argument('--round', default=100, type=int)
    args = parser.parse_args()

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
