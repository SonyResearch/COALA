import os
import coala
import argparse
from omegaconf import OmegaConf
import torch.multiprocessing as mp
from coala.datasets.cifar10.cifar10_conti import construct_cifar10_continual_datasets
from coala.datasets.cifar100.cifar100_conti import construct_cifar100_continual_datasets
from fl_server import ContinualServer
from fl_client import ContinualClient
from models import IncrementalNet


def run(args, rank=0):
    assert args.dataset in ['cifar10', 'cifar100']

    # Define customized configurations.
    config = {
        "task_id": args.task_id,
        "data": {
            "root": "./data/",
            "syn_dir": "./data/save_syn",
            "dataset": args.dataset,
            "num_of_clients": args.num_of_clients,
            "num_tasks": args.num_tasks,
            "split_type": args.split_type,
            "alpha": args.alpha,
            "synthetic_size": 8000,
            "synthesis":args.synthesis
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
            "task_index": 0,
            "local_epoch": args.local_epoch,
            "kd_alpha": args.kd_alpha,
            "rounds": args.rounds,
            "test_batch_size": args.test_batch_size,
            "batch_size": args.batch_size,
            "optimizer": {
                "type": args.optimizer_type,
                "lr": args.lr,
                "momentum": args.momentum,
            },
            "seed": args.seed
        },

        "gpu": args.gpus,
        "test_mode": "test_in_server",
        "model": args.model,

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

    # merge hypers for data synthesizer
    root_path = os.path.dirname(os.path.abspath(__file__))
    hyp_path = os.path.join(root_path, "syn_cfg.yaml")
    syn_hyp = OmegaConf.load(hyp_path)
    config.update(syn_hyp)
    conf = OmegaConf.create(config)

    # model initialization
    model = IncrementalNet(conf)
    if args.dataset == 'cifar10':
        train_data, test_data = construct_cifar10_continual_datasets(conf.data.root,
                                                                     conf.data.dataset,
                                                                     conf.data.num_of_clients,
                                                                     conf.data.split_type,
                                                                     alpha=conf.data.alpha,
                                                                     num_tasks=conf.data.num_tasks,
                                                                     )
    elif args.dataset == 'cifar100':
        train_data, test_data = construct_cifar100_continual_datasets(conf.data.root,
                                                                      conf.data.dataset,
                                                                      conf.data.num_of_clients,
                                                                      conf.data.split_type,
                                                                      alpha=conf.data.alpha,
                                                                      num_tasks=conf.data.num_tasks,
                                                                      )
    else:
        raise NotImplementedError

    # Initialize with the new config.
    coala.register_model(model)
    coala.register_dataset(train_data, test_data)
    coala.register_client(ContinualClient)
    coala.register_server(ContinualServer)
    coala.init(config)
    # Execute federated learning training.
    coala.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fed_Continual')
    parser.add_argument("--task_id", type=str, default="0")
    parser.add_argument('--dataset', type=str, default='cifar10', help="name of dataset, e.g., cifar10")
    parser.add_argument('--num_tasks', type=int, default=2, help="number of tasks")
    parser.add_argument('--split_type', type=str, default="iid", help="type of non-iid data")
    parser.add_argument('--alpha', type=float, default=0.5, help="parameter for Dirichlet distribution simulation")
    parser.add_argument('--model', type=str, default="resnet18", help="backbone")
    parser.add_argument('--synthesis', type=bool, default="True", help="whether to do data synthesis")

    parser.add_argument('--gpus', default=1, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--test_batch_size', default=128, type=int)
    parser.add_argument('--local_epoch', default=5, type=int)
    parser.add_argument('--kd_alpha', default=20, type=int)
    parser.add_argument('--rounds', default=100, type=int)
    parser.add_argument('--num_of_clients', default=4, type=int)
    parser.add_argument('--clients_per_round', default=4, type=int)

    parser.add_argument('--optimizer_type', default='SGD', type=str, help='optimizer type')
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')

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
