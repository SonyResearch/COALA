import os
from omegaconf import OmegaConf
from coala.datasets.landmarks.landmarks import construct_landmarks_datasets
import coala

if __name__ == '__main__':
    config = {
        "task_id": "landmarks",
        "data": {
            "root": "./data/",
            "dataset": "landmarks",  # gld23k, gld160k
            "num_of_clients": 233,  # 233, 1262
        },
    }
    conf = OmegaConf.create(config)
    train_data, test_data = construct_landmarks_datasets(conf.data.root,
                                                         conf.data.dataset,
                                                         conf.data.num_of_clients)
