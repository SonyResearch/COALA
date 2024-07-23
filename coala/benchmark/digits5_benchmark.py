import coala
import argparse
import torch.multiprocessing as mp

def run(args, rank=0):
    """
    Benchmark the performance of Digits_five dataset with 50 clients.
    Expected test accuracy: 
        - 91.75% at round 20
    """
    config = {
        "task_id": "digits_five",
        "data": {
            "dataset": "digits_five",
            "split_type": "iid",
            "num_of_clients": 5,
        },
        "server": {
            "rounds": 100,
            "clients_per_round": 5,
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
                    },
            },
        "model": "digits5cnn",
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
    parser = argparse.ArgumentParser(description='Digits5')
    parser.add_argument('--gpus', default=1, type=int)
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
