import os
import torch.multiprocessing as mp
import argparse
import sys
sys.path.append('')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sfl_coordinator import run

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split Federated Learning Example')
    parser.add_argument('--gpus', default=1, type=int)
    parser.add_argument('--rank', default=0, type=int)
    parser.add_argument("--num_of_clients", type=int, default=100)
    parser.add_argument("--num_of_rounds", type=int, default=400)
    parser.add_argument("--split_type", type=str, default="iid")
    parser.add_argument("--alpha", type=str, default=0.5)
    parser.add_argument("--class_per_client", type=str, default=2)
    parser.add_argument("--arch", type=str, default="resnet18_split")
    parser.add_argument("--cut_layer", type=int, default=1)
    args = parser.parse_args()
    print("args", args)

    '''Auto Adjust'''
    if args.num_of_clients <= 100:
        args.local_batch_size = 200 // args.num_of_clients
    else:
        args.local_batch_size = 1
    
    if args.num_of_clients <= 1000:
        args.avg_freq = 1000 // (args.local_batch_size * args.num_of_clients) # num_step per epoch.
    else:
        args.avg_freq = 1
    
    if args.num_of_clients >= 200:
        args.clients_per_round = 200
    else:
        args.clients_per_round = args.num_of_clients
    args.rounds = args.num_of_rounds * (args.num_of_clients // args.clients_per_round)


    config = {
        "gpu": args.gpus,
        "data": {"dataset": "cifar10", "num_of_clients": args.num_of_clients, 
                 "split_type": args.split_type, "class_per_client": args.class_per_client, "alpha": args.alpha},
        "model": args.arch,
        "cut_layer": args.cut_layer,
        "test_mode": "test_in_server",
        "server": {"rounds": args.rounds, "aggregation_strategy": "FedAvg", 
                   "aggregation_freq": args.avg_freq, 
                   "clients_per_round": args.clients_per_round,
                   "scheduler": "cos_anneal"},
        "client": {"batch_size": args.local_batch_size, "scheduler": "cos_anneal"},
        "fl_type": "self_supervised_SFL",
        "checkpoint_path": ''
    }
    

    if args.gpus <= 1:
        run(config, 0, "selfsupervised")
    else:
        mp.set_start_method("spawn")
        processes = []
        for rank in range(args.gpus):
            p = mp.Process(target=run, args=(config, rank, "selfsupervised"))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()