import argparse
import sys
sys.path.append('')
import coala
from coala.pb import common_pb2 as common_pb
from coala.pb import server_service_pb2 as server_pb
from coala.protocol import codec
from coala.communication import grpc_wrapper
from coala.registry import get_clients, SOURCES
import os
import yaml
from omegaconf import OmegaConf
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.split_models import load_model
from sfl_datasets import construct_datasets_selfsupervised

parser = argparse.ArgumentParser(description='MocoSFL Server')
parser.add_argument('--server-addr',
                    type=str,
                    default="172.18.0.1:23501",
                    help='Server address')
parser.add_argument('--etcd-addrs',
                    type=str,
                    default="172.17.0.1:2379",
                    help='Etcd address, or list of etcd addrs separated by ","')
parser.add_argument('--source',
                    type=str,
                    default="manual",
                    choices=SOURCES,
                    help='Source to get the clients')
args = parser.parse_args()


def send_run_request():


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

    print("Server address: {}".format(args.server_addr))
    print("Etcd address: {}".format(args.etcd_addrs))

    
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

    coala.register_dataset(train_data, test_data, val_data)
    
    _model_class = load_model(conf.model)
    model = _model_class()  # create
    model.split(conf.cut_layer) # a
    coala.register_model(model)
    
    coala.init(conf)
    model = coala.init_model()
    
    stub = grpc_wrapper.init_stub(grpc_wrapper.TYPE_SERVER, args.server_addr)

    request = server_pb.RunRequest(
        model=codec.marshal(model),
    )

    clients = get_clients(args.source, args.etcd_addrs)
    for c in clients:
        request.clients.append(server_pb.Client(client_id=c.id, index=c.index, address=c.address))

    response = stub.Run(request)
    if response.status.code == common_pb.SC_OK:
        print("Success")
    else:
        print(response)


if __name__ == '__main__':
    send_run_request()
