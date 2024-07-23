import coala
import argparse
import torch.multiprocessing as mp

def run(args, rank=0):
    """
    Benchmark the performance of FedAvg on CIFAR-10 dataset with 10 clients.
    Expected test accuracy on IID:
        - 95.33% at round 200
    """
    config = {
        "data": {"dataset": args.dataset,
                 "num_of_clients": args.num_of_clients,
                 "split_type": args.split_type,
                 "alpha": args.alpha,
                 "class_per_client": args.class_per_client,
                 },
        "model": "resnet18",
        "client": {"local_epoch": 5,
                   "batch_size": 32,
                   "optimizer": {
                        "type": args.optimizer_type,
                        "lr": args.lr,
                        }
                },
        "server": {
            "rounds": 200,
            "clients_per_round": 10,
            "test_every": 1,
            "save_model_every": 20
        },
        "test_mode": "test_in_server",
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

    coala.init(config)
    coala.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CIFAR')
    parser.add_argument('--gpus', default=1, type=int)
    parser.add_argument('--num_of_clients', default=100, type=int)
    parser.add_argument('--split_type', default="iid", type=str)
    parser.add_argument('--alpha', default=0.5, type=float)
    parser.add_argument('--class_per_client', default=2, type=int)
    parser.add_argument('--dataset', default="cifar10", type=str)
    parser.add_argument('--optimizer_type', default='SGD', type=str, help='optimizer type')
    parser.add_argument('--lr', default=0.01, type=float)
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
