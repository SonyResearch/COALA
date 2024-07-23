import logging

logger = logging.getLogger(__name__)

import argparse
import coala
import torch.multiprocessing as mp


def run(rank, args):
    # Hyperparameters
    config = {
        "task_id": "cifar10",
        "data": {
            "dataset": "cifar10",
            "split_type": "iid",
            "num_of_clients": args.num_of_clients,
        },
        "server": {
            "rounds": args.rounds,
            "clients_per_round": args.clients_per_round,
            "test_every": args.test_every,
            "test_all": True,
            "random_selection": True,
        },
        "client": {
            "local_epoch": args.local_epoch,
            "rounds": args.rounds,
            "test_batch_size": 32,
            "batch_size": args.batch_size,
        },

        "model": "simple_cnn",
        "test_mode": "test_in_server",

        "is_remote": False,
        "local_port": 22,

        "distributed": {
            "world_size": args.gpus
        },

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
    parser = argparse.ArgumentParser(description='Application')
    parser.add_argument("--task_id", type=str, default="")
    parser.add_argument('--gpus', default=4, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--local_epoch', default=5, type=int)
    parser.add_argument('--rounds', default=100, type=int)
    parser.add_argument('--num_of_clients', default=50, type=int)
    parser.add_argument('--clients_per_round', default=10, type=int)
    parser.add_argument('--optimizer_type', default='Adam', type=str, help='optimizer type')
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--checkpoint_path', default="", type=str, help='The checkpoint to load initialized model.')
    parser.add_argument('--test_every', default=5, type=int, help='Test every x rounds')
    parser.add_argument('--save_model_every', default=100, type=int, help='save model every x rounds')
    args = parser.parse_args()
    print("arguments: ", args)
    if args.gpus <= 1:
        run(0, args)
    else:
        mp.set_start_method("spawn")
        processes = []
        for rank in range(args.gpus):
            p = mp.Process(target=run, args=(rank, args))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
