import sys
sys.path.append('')

import os
import yaml
from omegaconf import OmegaConf

import coala
from coala.server import base as server_base
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.split_models import load_model
from sfl_datasets import construct_datasets_selfsupervised
from server import MocoSFLServer

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
config_path = os.path.join(root_path, "config/config_mocosfl.yaml")  # additional training parameters
if isinstance(config_path, str):
    with open(config_path, errors='ignore') as f:
        base_config = yaml.safe_load(f)  # load config dict
conf = OmegaConf.create(base_config)


config = {
        "gpu": 1,
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

parser = server_base.create_argument_parser()
parser.add_argument('--config', type=str, default="server_config.yaml", help='Server config file')
args = parser.parse_args()

if os.path.isfile(args.config):
    conf = coala.load_config(args.config, conf)

coala.register_dataset(None, test_data, val_data)

coala.register_model(model)

coala.register_server(MocoSFLServer)

coala.init(conf, init_all=False)
coala.start_server(args)