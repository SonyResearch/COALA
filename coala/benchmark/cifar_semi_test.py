import coala
import argparse
import torch.multiprocessing as mp

def run(args, rank=0):
    """
    Benchmark the performance of semiFL on CIFAR-10 dataset with 100 clients.
    Expected test accuracy:
        - 69.33% at round 20
    """
    config = {
        "task_id": "cifar",
        "data": {
            "dataset": "cifar10",
            "split_type": args.split_type,
            "num_of_clients": args.num_of_clients,
            "fl_type": "semi_supervised",
            "semi_scenario": args.semi_scenario,
            "num_labels_per_class": args.num_labels_per_class,
        },
        "server": {
            "rounds": 500,
            "clients_per_round": 10,
            "batch_size": 256,
            "local_epoch": 5,
            "test_every": 1,
            "aggregation_strategy": "equal",
            "optimizer":{
                "type": "SGD",
                "lr": 0.03,
                "momentum":0.9,
                "weight_decay":5e-4
                   }
        },
        "client": {"local_epoch": 5,
                   "batch_size": 32,
                   "optimizer": {
                        "type": args.optimizer_type,
                        "lr": args.lr,
                        "momentum": args.momentum
                        }
                    },
        "model": "resnet18",
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
    parser = argparse.ArgumentParser(description='CIFAR Semi')
    parser.add_argument('--gpus', default=1, type=int)
    parser.add_argument('--num_of_clients', default=100, type=int)
    parser.add_argument('--num_labels_per_class', default=400, type=int)
    parser.add_argument('--semi_scenario', default="label_in_server", type=str)
    parser.add_argument('--split_type', default="iid", type=str)
    parser.add_argument('--optimizer_type', default='SGD', type=str, help='optimizer type')
    parser.add_argument('--lr', default=0.03, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
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
            
