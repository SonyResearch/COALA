import copy
import logging
import time
import torch
from coala.client import BaseClient
from coala.distributed.distributed import CPU
from coala.tracking import metric

logger = logging.getLogger(__name__)


class PEFTClient(BaseClient):
    def __init__(self, cid, conf, train_data, test_data, device, **kwargs):
        super(PEFTClient, self).__init__(cid, conf, train_data, test_data, device, **kwargs)

    def test_local(self):
        """Test client local model after training."""
        self.test(self.conf, device=self.device)
        self.track(metric.TEST_METRIC, {"accuracy": float(self.test_accuracy), "loss": float(self.test_loss)})
        self.track(metric.TEST_TIME, self.test_time)

    def post_upload(self):
        """Postprocessing after uploading training/testing results."""
        self.model = None

    def load_optimizer(self, conf):
        """Load training optimizer. Implemented Adam and SGD."""
        if conf.optimizer.type == "Adam":
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                                         lr=conf.optimizer.lr)
        else:
            # default using optimizer SGD
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()),
                                        lr=conf.optimizer.lr,
                                        momentum=conf.optimizer.momentum,
                                        weight_decay=conf.optimizer.weight_decay)
        return optimizer


