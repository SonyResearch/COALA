import coala
import argparse
import torch.nn as nn
import torch.multiprocessing as mp
import timm
from base_vit import ViT
from lora import LoRA_ViT,LoRA_ViT_timm
from fl_client import PEFTClient

def run(args, rank=0):
    """
    Benchmark the performance of FedAvg on CIFAR-10 dataset with 10 clients.
    """
    config = {
        "data": {"dataset": args.dataset,
                 "num_of_clients": args.num_of_clients,
                 "split_type": args.split_type,
                 "alpha": 0.5},
        "client": {"local_epoch": args.local_epoch,
                   "batch_size": 32,
                   "optimizer": {
                       "type": args.optimizer_type,
                       "lr": args.lr,
                   },
                   "local_test": args.local_test,
                   },
        "server": {
            "rounds": args.rounds,
            "clients_per_round": args.clients_per_round,
            "test_every": 20,
            "test_all": args.test_all,
            "save_model_every": 20
        },
        "test_mode": args.test_mode,
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

    # model = ViT('B_16_imagenet1k')
    model = timm.create_model("vit_base_patch16_224", pretrained=True)
    train_model = None
    if args.train_type == "lora":
        lora_model = LoRA_ViT_timm(model, r=args.r, num_classes=args.num_classes)
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        train_model = lora_model
    elif args.train_type == "full":
        model.reset_classifier(args.num_classes)
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        train_model = model
    elif args.train_type == "linear":
        model.reset_classifier(args.num_classes)
        for param in model.parameters():
            param.requires_grad = False
        for param in model.head.parameters():
            param.requires_grad = True
        num_params = sum(p.numel() for p in model.head.parameters())
        total_params = sum(p.numel() for p in model.parameters())
        train_model = model
    print(f"trainable parameters: {num_params/2**20:.4f}M")
    print(f"trainable percentage: {100*num_params/total_params:.4f}")

    coala.register_model(train_model)
    coala.register_client(PEFTClient)

    coala.init(config)
    coala.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DomainNet')
    parser.add_argument('--gpus', default=1, type=int)
    parser.add_argument('--num_of_clients', default=100, type=int)
    parser.add_argument('--clients_per_round', default=10, type=int)
    parser.add_argument('--split_type', default="iid", type=str)
    parser.add_argument('--dataset', default="cifar10", type=str)
    parser.add_argument('--num_classes', default=10, type=int)
    parser.add_argument('--optimizer_type', default='SGD', type=str, help='optimizer type')
    parser.add_argument('--lr', default=5e-3, type=float)
    parser.add_argument('--local_epoch', default=1, type=int)
    parser.add_argument('--rounds', default=10, type=int)
    parser.add_argument('--test_all', default=False, type=bool)
    parser.add_argument('--local_test', default=False, type=bool)
    parser.add_argument('--test_mode', default="test_in_server", type=str)
    parser.add_argument('--train_type', default="lora", type=str)
    parser.add_argument('--r', default=8, type=int)
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
