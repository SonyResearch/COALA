import logging
import os
import random
import sys
import time
from os import path

import numpy as np
import torch
from omegaconf import OmegaConf

from coala.client.base import BaseClient
from coala.client.base_semi import SemiFLClient
from coala.datasets import TEST_IN_SERVER
from coala.datasets.data import construct_datasets, construct_datasets_semi
from coala.distributed import dist_init, get_device
from coala.models.model import load_model
from coala.server.base import BaseServer
from coala.server.base_semi import SemiFLServer
from coala.simulation.system_hetero import resource_hetero_simulation

logger = logging.getLogger(__name__)


class Coordinator(object):
    """Coordinator manages federated learning server and client.
    A single instance of coordinator is initialized for each federated learning task
    when the package is imported.
    """

    def __init__(self):
        self.registered_model = False
        self.registered_dataset = False
        self.registered_server = False
        self.registered_client = False
        self.train_data = None
        self.test_data = None
        self.val_data = None
        self.conf = None
        self.model = None
        self._model_class = None
        self.server = None
        self._server_class = None
        self.clients = None
        self._client_class = None
        self.tracker = None
        self.s_train_data = None
        self.u_train_data = None

    def init(self, conf, init_all=True):
        """Initialize coordinator

        Args:
            conf (omegaconf.dictconfig.DictConfig): Internal configurations for federated learning.
            init_all (bool): Whether initialize dataset, model, server, and client other than configuration.
        """
        self.init_conf(conf)

        _set_random_seed(conf.seed)

        if init_all:
            self.init_dataset()

            self.init_model()

            self.init_server()

            self.init_clients()

    def run(self):
        """Run the coordinator and the federated learning process.
        Initialize `torch.distributed` if distributed training is configured.
        """
        start_time = time.time()

        if self.conf.is_distributed:
            dist_init(
                self.conf.distributed.backend,
                self.conf.distributed.init_method,
                self.conf.distributed.world_size,
                self.conf.distributed.rank,
                self.conf.distributed.local_rank,
            )
        self.server.start(self.model, self.clients)
        self.print_("Total training time {:.1f}s".format(time.time() - start_time))

    def init_conf(self, conf):
        """Initialize coordinator configuration.

        Args:
            conf (omegaconf.dictconfig.DictConfig): Configurations.
        """
        self.conf = conf
        self.conf.is_distributed = (self.conf.gpu > 1)
        if self.conf.gpu == 0:
            self.conf.device = "cpu"
        elif self.conf.gpu == 1:
            self.conf.device = 0
        else:
            self.conf.device = get_device(self.conf.gpu, self.conf.distributed.world_size,
                                          self.conf.distributed.local_rank)
        self.print_("Configurations: {}".format(self.conf))

    def init_dataset(self):
        """Initialize datasets. Use provided datasets if not registered."""
        if self.registered_dataset:
            return
        if self.conf.data.fl_type == "semi_supervised":
            self.s_train_data, self.u_train_data, self.test_data = construct_datasets_semi(self.conf.data.root,
                                                                                           self.conf.data.dataset,
                                                                                           self.conf.data.num_of_clients,
                                                                                           self.conf.data.split_type,
                                                                                           self.conf.data.min_size,
                                                                                           self.conf.data.class_per_client,
                                                                                           self.conf.data.data_amount,
                                                                                           self.conf.data.iid_fraction,
                                                                                           self.conf.data.user,
                                                                                           self.conf.data.train_test_split,
                                                                                           self.conf.data.weights,
                                                                                           self.conf.data.alpha,
                                                                                           self.conf.data.semi_scenario,
                                                                                           self.conf.data.num_labels_per_class)
            self.train_data = self.u_train_data
            print(f"Total labeled training data amount: {self.s_train_data.total_size()}")
            print(f"Total unlabeled training data amount: {self.u_train_data.total_size()}")
        else:
            self.train_data, self.test_data = construct_datasets(self.conf.data.root,
                                                                 self.conf.data.dataset,
                                                                 self.conf.data.num_of_clients,
                                                                 self.conf.data.split_type,
                                                                 self.conf.data.min_size,
                                                                 self.conf.data.class_per_client,
                                                                 self.conf.data.data_amount,
                                                                 self.conf.data.iid_fraction,
                                                                 self.conf.data.user,
                                                                 self.conf.data.train_test_split,
                                                                 self.conf.data.weights,
                                                                 self.conf.data.alpha)

            self.print_(f"Total training data amount: {self.train_data.total_size()}")
            self.print_(f"Total testing data amount: {self.test_data.total_size()}")

    def init_model(self):
        """Initialize model instance."""
        if not self.registered_model:
            self._model_class = load_model(self.conf.model)

        # model_class is None means model is registered as instance, no need initialization
        if self._model_class:
            self.model = self._model_class()

    def init_server(self):
        """Initialize a server instance."""
        if not self.registered_server:
            self._server_class = BaseServer
            if self.conf.data.fl_type == "semi_supervised":
                self._server_class = SemiFLServer

        kwargs = {
            "is_remote": self.conf.is_remote,
            "local_port": self.conf.local_port
        }

        if self.conf.data.fl_type == "semi_supervised" and self.conf.data.semi_scenario == 'label_in_server':
            kwargs["s_train_data"] = self.s_train_data

        if self.conf.test_mode == TEST_IN_SERVER:
            kwargs["test_data"] = self.test_data
            if self.val_data:
                kwargs["val_data"] = self.val_data

        self.server = self._server_class(self.conf, **kwargs)

    def init_clients(self):
        """Initialize client instances, each represents a federated learning client."""
        if not self.registered_client:
            self._client_class = BaseClient
            if self.conf.data.fl_type == "semi_supervised":
                self._client_class = SemiFLClient

        # Enforce system heterogeneity of clients.
        sleep_time = [0 for _ in self.train_data.users]
        if self.conf.resource_heterogeneous.simulate:
            sleep_time = resource_hetero_simulation(self.conf.resource_heterogeneous.fraction,
                                                    self.conf.resource_heterogeneous.hetero_type,
                                                    self.conf.resource_heterogeneous.sleep_group_num,
                                                    self.conf.resource_heterogeneous.level,
                                                    self.conf.resource_heterogeneous.total_time,
                                                    len(self.train_data.users))

        client_test_data = self.test_data
        if self.conf.test_mode == TEST_IN_SERVER:
            client_test_data = None

        if self.conf.data.fl_type == "semi_supervised":
            labeled_train_data = self.s_train_data if self.conf.data.semi_scenario == 'label_in_client' else None
            self.clients = [self._client_class(u,
                                               self.conf.client,
                                               labeled_train_data,
                                               self.u_train_data,
                                               client_test_data,
                                               self.conf.device,
                                               **{"sleep_time": sleep_time[i]})
                            for i, u in enumerate(self.u_train_data.users)]
        elif self.conf.data.fl_type == "self_supervised":
            raise NotImplementedError
        else:
            self.clients = [self._client_class(u,
                                               self.conf.client,
                                               self.train_data,
                                               client_test_data,
                                               self.conf.device,
                                               **{"sleep_time": sleep_time[i]})
                            for i, u in enumerate(self.train_data.users)]

        self.print_("Clients in total: {}".format(len(self.clients)))

    def init_client(self):
        """Initialize client instance.

        Returns:
            :obj:`BaseClient`: The initialized client instance.
        """
        if not self.registered_client:
            self._client_class = BaseClient

        # Get a random client if not specified
        if self.conf.index:
            user = self.train_data.users[self.conf.index]
        else:
            user = random.choice(self.train_data.users)

        return self._client_class(user,
                                  self.conf.client,
                                  self.train_data,
                                  self.test_data,
                                  self.conf.device,
                                  is_remote=self.conf.is_remote,
                                  local_port=self.conf.local_port,
                                  server_addr=self.conf.server_addr,
                                  tracker_addr=self.conf.tracker_addr)

    def start_server(self, args):
        """Start a server service for remote training.

        Server controls the model and testing dataset if configured to test in server.

        Args:
            args (argparse.Namespace): Configurations passed in as arguments, it is merged with configurations.
        """
        if args:
            self.conf = OmegaConf.merge(self.conf, args.__dict__)

        if self.conf.test_mode == TEST_IN_SERVER:
            self.init_dataset()

        self.init_model()

        self.init_server()

        self.server.start_service()

    def start_client(self, args):
        """Start a client service for remote training.

        Client controls training datasets.

        Args:
            args (argparse.Namespace): Configurations passed in as arguments, it is merged with configurations.
        """

        if args:
            self.conf = OmegaConf.merge(self.conf, args.__dict__)

        self.init_dataset()

        client = self.init_client()

        client.start_service()

    def register_dataset(self, train_data, test_data, val_data=None, s_train_data=None, u_train_data=None):
        """Register datasets.

        Datasets should inherit from :obj:`FederatedDataset`, e.g., :obj:`FederatedTensorDataset`.

        Args:
            train_data (:obj:`FederatedDataset`): Training dataset.
            test_data (:obj:`FederatedDataset`): Testing dataset.
            val_data (:obj:`FederatedDataset`): Validation dataset.
            s_train_data (:obj:`FederatedDataset`): Supervised training dataset (semi-supervised setting only).
            u_train_data (:obj:`FederatedDataset`): Unsupervised training dataset (semi-supervised setting only).
        """
        self.registered_dataset = True
        self.train_data = train_data
        self.test_data = test_data
        self.val_data = val_data
        self.s_train_data = s_train_data
        self.u_train_data = u_train_data

    def register_model(self, model):
        """Register customized model for federated learning.

        Args:
            model (nn.Module): PyTorch model, both class and instance are acceptable.
                Use model class when there is no specific arguments to initialize model.
        """
        self.registered_model = True
        if not isinstance(model, type):
            self.model = model
        else:
            self._model_class = model

    def register_server(self, server):
        """Register a customized federated learning server.

        Args:
            server (:obj:`BaseServer`): Customized federated learning server.
        """
        self.registered_server = True
        self._server_class = server

    def register_client(self, client):
        """Register a customized federated learning client.

        Args:
            client (:obj:`BaseClient`): Customized federated learning client.
        """
        self.registered_client = True
        self._client_class = client

    def print_(self, content):
        """Log the content only when the server is primary server.

        Args:
            content (str): The content to log.
        """
        if self._is_primary_server():
            logger.info(content)

    def _is_primary_server(self):
        """Check whether current running server is the primary server.

        In standalone or remote training, the server is primary.
        In distributed training, the server on `rank0` is primary.
        """
        return not self.conf.is_distributed or self.conf.distributed.rank == 0


def _set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# Initialize the global coordinator object
_global_coord = Coordinator()


def init_conf(conf=None):
    """Initialize configuration for COALA. It overrides and supplements default configuration loaded from config.yaml
    with the provided configurations.

    Args:
        conf (dict): Configurations.

    Returns:
        omegaconf.dictconfig.DictConfig: Internal configurations managed by OmegaConf.
    """
    here = path.abspath(path.dirname(__file__))
    config_file = path.join(here, 'config.yaml')
    return load_config(config_file, conf)


def load_config(file, conf=None):
    """Load and merge configuration from file and input

    Args:
        file (str): filename of the configuration.
        conf (dict): Configurations.

    Returns:
        omegaconf.dictconfig.DictConfig: Internal configurations managed by OmegaConf.
    """
    config = OmegaConf.load(file)
    if conf is not None:
        config = OmegaConf.merge(config, conf)
    return config


def init_logger(log_level):
    """Initialize internal logger of COALA.

    Args:
        log_level (int): Logger level, e.g., logging.INFO, logging.DEBUG
    """
    log_formatter = logging.Formatter("%(asctime)s [%(threadName)s] [%(levelname)-5.5s]  %(message)s")
    root_logger = logging.getLogger()

    log_level = logging.INFO if not log_level else log_level
    root_logger.setLevel(log_level)

    file_path = os.path.join(os.getcwd(), "logs")
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    file_path = path.join(file_path, "train" + time.strftime(".%m_%d_%H_%M_%S") + ".log")
    file_handler = logging.FileHandler(file_path)
    file_handler.setFormatter(log_formatter)
    root_logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)


def init(conf=None, init_all=True):
    """Initialize COALA.

    Args:
        conf (dict, optional): Configurations.
        init_all (bool, optional): Whether initialize dataset, model, server, and client other than configuration.
    """
    global _global_coord

    config = init_conf(conf)

    init_logger(config.tracking.log_level)

    _set_random_seed(config.seed)

    _global_coord.init(config, init_all)


def run():
    """Run federated learning process."""
    global _global_coord
    _global_coord.run()


def init_dataset():
    """Initialize dataset, either using registered dataset or out-of-the-box datasets set in config."""
    global _global_coord
    _global_coord.init_dataset()


def init_model():
    """Initialize model, either using registered model or out-of–the-box model set in config.

    Returns:
        nn.Module: Model used in federated learning.
    """
    global _global_coord
    _global_coord.init_model()

    return _global_coord.model


def start_server(args=None):
    """Start federated learning server service for remote training.

    Args:
        args (argparse.Namespace): Configurations passed in as arguments.
    """
    global _global_coord

    _global_coord.start_server(args)


def start_client(args=None):
    """Start federated learning client service for remote training.

    Args:
        args (argparse.Namespace): Configurations passed in as arguments.
    """
    global _global_coord

    _global_coord.start_client(args)


def get_coordinator():
    """Get the global coordinator instance.

    Returns:
        :obj:`Coordinator`: global coordinator instance.
    """
    return _global_coord


def register_dataset(train_data, test_data, val_data=None):
    """Register datasets for federated learning training.

    Args:
        train_data (:obj:`FederatedDataset` or dict[obj:`FederatedDataset`]): Training dataset.
        test_data (:obj:`FederatedDataset`): Testing dataset.
        val_data (:obj:`FederatedDataset`): Validation dataset.
        train_data.s_train_data (:obj:`FederatedDataset`): Supervised training dataset (semi-supervised setting only).
        train_data.u_train_data (:obj:`FederatedDataset`): Unsupervised training dataset (semi-supervised setting only).
    """
    global _global_coord
    if type(train_data) == dict:
        s_train_data = train_data['s_train_data']
        u_train_data = train_data['u_train_data']

        _global_coord.register_dataset(u_train_data, test_data, val_data, s_train_data, u_train_data)
    else:
        _global_coord.register_dataset(train_data, test_data, val_data)

def register_model(model):
    """Register model for federated learning training.

    Args:
        model (nn.Module): PyTorch model, both class and instance are acceptable.
    """
    global _global_coord
    _global_coord.register_model(model)


def register_server(server):
    """Register federated learning server.

    Args:
        server (:obj:`BaseServer`): Customized federated learning server.
    """
    global _global_coord
    _global_coord.register_server(server)


def register_client(client):
    """Register federated learning client.

    Args:
        client (:obj:`BaseClient`): Customized federated learning client.
    """
    global _global_coord
    _global_coord.register_client(client)
