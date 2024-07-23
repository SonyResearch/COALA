import sys
import os
import yaml
from omegaconf import OmegaConf
sys.path.append('')
import coala
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.split_models import load_model
from sfl_datasets import construct_datasets_selfsupervised
from client import MocoSFLClient

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
config_path = os.path.join(root_path, "config/config_mocosfl.yaml")  # additional training parameters
if isinstance(config_path, str):
    with open(config_path, errors='ignore') as f:
        base_config = yaml.safe_load(f)  # load config dict
conf = OmegaConf.create(base_config)


config = {
        "gpu": 0,
        "data": {"dataset": "cifar10", "num_of_clients": 2},
        "model": "resnet18_split",
        "cut_layer": 1,
        "test_mode": "test_in_server",
        "server": {"rounds": 200, "aggregation_strategy": "FedAvg", 
                   "aggregation_freq": 5, 
                   "clients_per_round": 2,
                   "scheduler": "cos_anneal"},
        "client": {"batch_size": 100, "scheduler": "cos_anneal"},
        "fl_type": "self_supervised_SFL"
    }
conf = OmegaConf.merge(conf, config)
_model_class = load_model(conf.model)
model = _model_class()  # create
model.split(conf.cut_layer) # apply split

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

coala.start_remote_client(conf, train_data=train_data, test_data=None, model = model.client_cloud_copy[0], client = MocoSFLClient)
