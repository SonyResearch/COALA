import argparse
import logging
import coala
from fl_client import FedReIDClient
from dataset import prepare_train_data, prepare_test_data
from model import Model
import torch.multiprocessing as mp

logger = logging.getLogger(__name__)


def run(rank, args):
    config = {
        "task_id": args.task_id,
        "gpu": args.gpus,
        "client": {
            "test_every": args.test_every,
            "local_epoch": 1,
            "track": False,
            "batch_size": 32,
            "optimizer": {
                "type": "SGD",
                "lr": 0.05,
                "momentum": 0.9
            }
        },

        "server": {
            "test_every": args.test_every,
            "test_all": False,
            "clients_per_round": 9,
            "rounds": 300,
            "batch_size": 32,
            "aggregation_content": "parameters"
        },

        "test_mode": "test_in_client",
        "test_method": "average"
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

    train_data = prepare_train_data(args.data_dir, args.datasets)
    test_data = prepare_test_data(args.data_dir, args.datasets)

    coala.register_dataset(train_data, test_data)
    coala.register_model(Model)
    coala.register_client(FedReIDClient)
    coala.init(config)

    coala.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FedReID Application')
    parser.add_argument('--task_id', type=str, default="")
    parser.add_argument('--data_dir', type=str, metavar='PATH', default="data/fedreid")
    parser.add_argument("--datasets", nargs="+", default=None, help="list of datasets, e.g., ['ilids']")
    parser.add_argument('--test_every', type=int, default=10)
    parser.add_argument("--gpus", type=int, default=0, help="default number of GPU")
    args = parser.parse_args()
    logger.info("arguments: ", args)

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
