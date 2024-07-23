import logging
import torch
import torch.nn as nn
import torch.distributed as dist
from omegaconf import OmegaConf
from coala.server.base import BaseServer
from coala.distributed.distributed import CPU, broadcast_model

logger = logging.getLogger(__name__)


class SemiFLServer(BaseServer):
    """Default implementation of federated semi-supervised learning server.

    Args:
        conf (omegaconf.dictconfig.DictConfig): Configurations of COALA.
        test_data (:obj:`FederatedDataset`): Test dataset for centralized testing in server, optional.
        val_data (:obj:`FederatedDataset`): Validation dataset for centralized validation in server, optional.
        is_remote (bool): A flag to indicate whether start remote training.
        local_port (int): The port of remote server service.
    """

    def __init__(self,
                 conf,
                 s_train_data=None,
                 test_data=None,
                 val_data=None,
                 is_remote=False,
                 local_port=22999):
        super(SemiFLServer, self).__init__(conf, test_data, val_data, is_remote, local_port)

        self.s_train_data = s_train_data
        self.s_train_loader = None
        self.semi_scenario = conf.data.semi_scenario

        self.is_training = False
        self.bs = conf.server.batch_size

    def pre_train(self):
        """Preprocessing after client training."""
        if self.semi_scenario == 'label_in_server' and self.current_round == 0:
            if self.conf.is_distributed:
                if self.is_primary_server():
                    self.server_update()
                dist.barrier()
                broadcast_model(self.model, src=0)
            else:
                self.server_update()

    def post_train(self):
        """Postprocessing after training."""
        if self.semi_scenario == 'label_in_server':
            if self.conf.is_distributed:
                if self.is_primary_server():
                    self.server_update()
                dist.barrier()
                broadcast_model(self.model, src=0)
            else:
                self.server_update()

    def server_update(self):
        # make sure all models are in train mode and on cuda for synchronization
        self.model.train()
        self.model.to(self.conf.device)
        self.train_in_server(self.conf.device)
        self.update_batch_norm_stats(self.conf.device)
        if self.current_round == 0:
            self.test()

    def train_in_server(self, device=CPU):
        """Server-side training on labeled data."""
        print("--- start server-side training ---")
        loss_fn, optimizer = self.pretrain_setup(self.conf, device)
        for i in range(self.conf.server.local_epoch):
            for batched_x, batched_y in self.s_train_loader:
                x, y = batched_x.to(device), batched_y.to(device)
                optimizer.zero_grad()
                out = self.model(x)
                loss = loss_fn(out, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)
                optimizer.step()

    def update_batch_norm_stats(self, device=CPU):
        data_loader = self.s_train_loader
        transform_train = self.s_train_data.transform
        transform_test = self.test_data.transform
        data_loader.dataset.transform_x = transform_test
        with torch.no_grad():
            self.model.train()
            self.model.apply(lambda m: reset_batch_norm(m, momentum=None, track_running_stats=True))
            for batched_x, batched_y in data_loader:
                x = batched_x.to(device)
                _ = self.model(x)
        data_loader.dataset.transform_x = transform_train

    def pretrain_setup(self, conf, device):
        """Setup loss function and optimizer before training."""
        self.model.train()
        self.model.to(device)
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = self.load_optimizer(conf)
        if self.s_train_loader is None:
            self.s_train_loader = self.s_train_data.loader(self.bs, shuffle=True, seed=conf.seed)

        return loss_fn, optimizer

    def load_optimizer(self, conf):
        """Load training optimizer. Implemented Adam and SGD."""
        if conf.server.optimizer.type == "Adam":
            optimizer = torch.optim.Adam(self.model.parameters(), lr=conf.server.optimizer.lr)
        else:
            # default using optimizer SGD
            optimizer = torch.optim.SGD(self.model.parameters(),
                                        lr=conf.server.optimizer.lr,
                                        momentum=conf.server.optimizer.momentum,
                                        weight_decay=conf.server.optimizer.weight_decay)
        return optimizer


def reset_batch_norm(m, momentum, track_running_stats):
    if isinstance(m, nn.BatchNorm2d):
        m.momentum = momentum
        m.track_running_stats = track_running_stats
        m.reset_running_stats()
    return m
