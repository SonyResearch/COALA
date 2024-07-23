import argparse
import os
import yaml
import coala
from omegaconf import OmegaConf
import torch
import torch.multiprocessing as mp
# from models.yolo import Model
from models.split_models import load_model
from client import BaseSFLClient, MocoSFLClient
from server import BaseSFLServer, MocoSFLServer
from coala.datasets import construct_datasets
from sfl_datasets import construct_datasets_selfsupervised


def run(config, rank=0, style = "supervised"):
    # Define customized configurations.
    root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if style == "supervised":
        config_path = os.path.join(root_path, "splitFL/config/config_sfl.yaml")  # additional training parameters
    elif style == "selfsupervised":
        config_path = os.path.join(root_path, "splitFL/config/config_mocosfl.yaml")  # additional training parameters
    else:
        raise NotImplementedError
    
    if isinstance(config_path, str):
        with open(config_path, errors='ignore') as f:
            base_config = yaml.safe_load(f)  # load config dict
    
    conf = OmegaConf.create(base_config)
    
    # update distributed info in config
    config.update({
        "distributed": {
            "rank": rank,
            "local_rank": rank,
            "world_size": config["gpu"],
            "init_method": "tcp://127.0.0.1:8123"
        }
    })
    
    conf = OmegaConf.merge(conf, config)
    print(conf.distributed)

    if conf.client.scheduler == "multi_step":
        # set client rounds equal to server, to keep multi-step scheduler consistent
        conf.client.rounds = conf.server.rounds

    _model_class = load_model(conf.model)
    model = _model_class()  # create
    model.split(conf.cut_layer) # apply split
    if conf.checkpoint_path:
        print(f"Initializing models from checkpoint: {conf.checkpoint_path}")
        checkpoint = torch.load(conf.checkpoint_path)
        model.load_state_dict(checkpoint)
    
    # Prepare data
    if style == "supervised":
        train_data, test_data = construct_datasets(conf.data.root,
                                                   conf.data.dataset,
                                                   conf.data.num_of_clients,
                                                   conf.data.split_type,
                                                   conf.data.min_size,
                                                   conf.data.class_per_client,
                                                   conf.data.data_amount,
                                                   conf.data.iid_fraction,
                                                   conf.data.user,
                                                   conf.data.train_test_split,
                                                   conf.data.weights,
                                                   conf.data.alpha)
        val_data = None
        _client_class = BaseSFLClient
        _server_class = BaseSFLServer
    elif style == "selfsupervised":
        train_data, val_data, test_data = construct_datasets_selfsupervised(conf.data.root,
                                                                            conf.data.dataset,
                                                                            conf.data.num_of_clients,
                                                                            conf.data.split_type,
                                                                            conf.data.min_size,
                                                                            conf.data.class_per_client,
                                                                            conf.data.data_amount,
                                                                            conf.data.iid_fraction,
                                                                            conf.data.user,
                                                                            conf.data.train_test_split,
                                                                            conf.data.weights,
                                                                            conf.data.alpha,
                                                                            conf.moco.version)
        _client_class = MocoSFLClient
        _server_class = MocoSFLServer
    else:
        raise NotImplementedError
    print(f"Total labeled training data amount: {train_data.total_size()}")
    
    # Initialize with the new config.
    coala.register_dataset(train_data, test_data, val_data)
    coala.register_model(model)
    coala.register_client(_client_class)
    coala.register_server(_server_class)
    coala.init(conf)
    # Execute federated learning training.
    coala.run()