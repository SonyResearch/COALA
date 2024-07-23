import coala
import argparse
import torch.multiprocessing as mp

def run(args, rank=0):
    """
    Benchmark the performance of Office_Caltech dataset with 4 clients.
    Expected test accuracy: 
        - 65.45 % at round 20
    """
    config = {
        "task_id": "office_caltech",
        "data": {
            "dataset": "office_caltech",
            "split_type": "iid",
            "num_of_clients": 4,
            "train_test_split": 0.7
        },
        "server": {
            "rounds": 100,
            "clients_per_round": 4,
            "test_every": 2,
            "test_all": True,
            "test_method": "average",
        },
        "client": {"local_epoch": 5,
                   "optimizer": {
                        "type": args.optimizer_type,
                        "lr": args.lr,
                        "momentum": 0.9,
                        "weight_decay": 0.0005,
                    },},
        "model": "alexnet",
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

    coala.init(config)
    coala.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Office')
    parser.add_argument('--gpus', default=4, type=int)
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
