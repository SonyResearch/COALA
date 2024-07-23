from omegaconf import OmegaConf
from coala.datasets.modelnet40.modelnet40 import construct_modelnet40_datasets

if __name__ == '__main__':
    config = {
        "task_id": "modelnet40",
        "data": {
            "root": "./data/",
            "dataset": "modelnet40",
            "num_of_clients": 10,
        },
    }
    conf = OmegaConf.create(config)
    train_data, test_data = construct_modelnet40_datasets(conf.data.root,
                                                          conf.data.dataset,
                                                          conf.data.num_of_clients)
