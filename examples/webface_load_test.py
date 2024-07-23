from omegaconf import OmegaConf
from coala.datasets.casia_webface.casia_webface import construct_casia_webface_datasets

if __name__ == '__main__':
    config = {
        "task_id": "casia_webface",
        "data": {
            "root": "./data/",
            "dataset": "casia_webface",
            "num_of_clients": 10,
        },
    }
    conf = OmegaConf.create(config)
    train_data, test_data = construct_casia_webface_datasets(conf.data.root,
                                                             conf.data.dataset,
                                                             conf.data.num_of_clients)
