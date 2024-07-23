import os
from omegaconf import OmegaConf
from coala.datasets.bdd100k.bdd100k import construct_bdd100k_datasets
import coala

if __name__ == '__main__':

    config = {
        "task_id": "bdd100k",
        "data": {
            "root": "./data/",
            "dataset": "bdd100k",
            "split_type": "hdir",
            "num_of_clients": 10,
        },
        "server": {
            "rounds": 20,
            "clients_per_round": 10,
            "test_every": 10
        },
        "client": {"local_epoch": 5},
        "model": "cnn",
        "test_mode": "test_in_client",
    }
    conf = OmegaConf.create(config)
    train_data, test_data = construct_bdd100k_datasets(conf.data.root,
                                                       conf.data.dataset,
                                                       conf.data.num_of_clients,
                                                       conf.data.split_type)
