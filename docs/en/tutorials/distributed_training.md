# Tutorial 6: Distributed Training

COALA enables federated learning (FL) training over multiple GPUs. We define the following variables to further illustrate the idea:
* K: the number of clients who participated in training each round
* N: the number of available GPUs

When _K == N_, each selected client is allocated to a GPU to train.

When _K > N_, multiple clients are allocated to a GPU, then they execute training sequentially in the GPU.

When _K < N_, you can adjust to use fewer GPUs in training.

We make it easy to use distributed training. You just need to modify the configs, without changing the core implementations.
In particular, you need to set the number of GPUs in `gpu` and specific distributed settings in the `distributed` configs.

## Distributed Training with Pytorch Multiprocessing

The following is an example of distributed training on multiple gpus using [PyTorch Multiprocessing](https://pytorch.org/docs/stable/notes/multiprocessing.html).

```python
import logging

logger = logging.getLogger(__name__)

import argparse
import coala
import torch.multiprocessing as mp


def run(rank, args):
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
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--test_every', default=5, type=int, help='Test every x rounds')
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
```

## Distributed Training with Slurm

The following is an example of distributed training on a GPU cluster managed by _slurm_.

```python
import coala
from coala.distributed import slurm

# Get the distributed settings.
rank, local_rank, world_size, host_addr = slurm.setup()
# Set the distributed training settings.
config = {
    "gpu": world_size,
    "distributed": {
        "rank": rank, 
        "local_rank": local_rank, 
        "world_size": world_size, 
        "init_method": host_addr
    },
}
# Initialize COALA.
coala.init(config)
# Execute training with distributed training.
coala.run()
```

We will further provide scripts to set up distributed training using `multiprocess`. 
Pull requests are also welcomed.

