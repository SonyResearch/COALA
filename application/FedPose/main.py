import os
import argparse
import torch.multiprocessing as mp
from omegaconf import OmegaConf
from models.pose_resnet import get_pose_net
from fl_client import PoseClient
from fl_server import PoseServer
from dataset.mpii import construct_mpii_datasets
import coala


def run(args, rank=0):
    config = {
        "task_id": "mpii",
        "data": {
            "root": "./data/",
            "dataset": "mpii",
            "num_classes": args.num_classes,
            "split_type": args.split_type,
            "num_of_clients": args.num_of_clients,
        },
        "server": {
            "rounds": args.rounds,
            "clients_per_round": args.clients_per_round,
            "test_every": args.test_every,
            "test_all": False
        },
        "client": {
            "local_epoch": args.local_epoch,
            "optimizer": {
                "type": args.optimizer_type,
                "lr": args.lr,
                "momentum": args.momentum,
            },
            "num_classes": args.num_classes,
        },
        "model": {
            "name": "pose_resnet",
        },

        "test_mode": "test_in_server",

        "gpu": args.gpus,

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

    root_path = os.path.dirname(os.path.abspath(__file__))
    hyp_path = os.path.join(root_path, args.cfg)
    hyp = OmegaConf.load(hyp_path)

    conf = OmegaConf.create(config)
    conf = OmegaConf.merge(hyp, conf)
    train_data, test_data = construct_mpii_datasets(conf,
                                                    conf.data.root,
                                                    conf.data.dataset,
                                                    conf.data.num_of_clients,
                                                    conf.data.split_type)
    # using pretrained model from pytorch
    model = get_pose_net(cfg=conf, is_train=args.train)

    # Initialize with the new config.
    coala.register_dataset(train_data, test_data)
    coala.register_model(model)
    coala.register_client(PoseClient)
    coala.register_server(PoseServer)

    coala.init(conf)
    # Execute federated learning training.
    coala.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fed_Pose_Estimation')
    parser.add_argument("--task_id", type=int, default=0)
    parser.add_argument('--dataset', type=str, default='mpii', help="name of dataset, e.g., cifar10")
    parser.add_argument('--num_classes', type=int, default=16, help="number of classes")
    parser.add_argument('--split_type', type=str, default="iid", help="type of non-iid data")
    parser.add_argument('--alpha', type=float, default=0.5, help="parameter for Dirichlet distribution simulation")
    parser.add_argument('--model', type=str, default="pose_resnet", help="seg model")
    parser.add_argument('--cfg', type=str, default="configs/base_cfg.yaml", help="config file")

    parser.add_argument('--gpus', default=1, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--test_batch_size', default=32, type=int)
    parser.add_argument('--local_epoch', default=5, type=int)
    parser.add_argument('--rounds', default=100, type=int)
    parser.add_argument('--num_of_clients', default=20, type=int)
    parser.add_argument('--clients_per_round', default=4, type=int)

    parser.add_argument('--optimizer_type', default='SGD', type=str, help='optimizer type')
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')

    parser.add_argument('--train', default=True, type=bool, help='whether to train model')
    parser.add_argument('--test_every', default=5, type=int, help='test every x rounds')
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
