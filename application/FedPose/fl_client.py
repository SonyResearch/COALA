import logging
import time
import torch
import numpy as np
from coala.client import BaseClient
from coala.distributed.distributed import CPU
from utils.loss import JointsMSELoss

logger = logging.getLogger(__name__)


class PoseClient(BaseClient):
    def __init__(self, cid, conf, train_data, test_data, device, **kwargs):
        super(PoseClient, self).__init__(cid, conf, train_data, test_data, device, **kwargs)

    def load_loss_fn(self, conf):
        return JointsMSELoss()

    def train(self, conf, device=CPU):
        """Execute client training.

        Args:
            conf (omegaconf.dictconfig.DictConfig): Client configurations.
            device (str): Hardware device for training, cpu or cuda devices.
        """
        start_time = time.time()
        loss_fn, optimizer = self.pretrain_setup(conf, device)
        self.train_loss = []
        for i in range(conf.local_epoch):
            batch_loss = []
            for input, target, target_weight, meta in self.train_loader:
                x, y, w = input.to(device), target.to(device), target_weight.to(device)
                optimizer.zero_grad()
                out = self.model(x)
                loss = loss_fn(out, y, w)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            current_epoch_loss = sum(batch_loss) / len(batch_loss)
            self.train_loss.append(float(current_epoch_loss))
            logger.debug("Client {}, local epoch: {}, loss: {}".format(self.cid, i, current_epoch_loss))
        self.train_time = time.time() - start_time
        logger.debug("Client {}, Train Time: {}".format(self.cid, self.train_time))
