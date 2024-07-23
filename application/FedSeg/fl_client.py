import copy
import logging
import time
import torch
from coala.client import BaseClient
from coala.distributed.distributed import CPU
from utils import FocalLoss
from evaluate import StreamSegMetrics

logger = logging.getLogger(__name__)


class SegClient(BaseClient):
    def __init__(self, cid, conf, train_data, test_data, device, **kwargs):
        super(SegClient, self).__init__(cid, conf, train_data, test_data, device, **kwargs)
        self.validate = StreamSegMetrics(self.conf.num_classes)

    def load_loss_fn(self, conf):
        criterion = torch.nn.CrossEntropyLoss(ignore_index=255, reduction='mean')
        if conf.optimizer.loss_type == 'focal_loss':
            criterion = FocalLoss(ignore_index=255, size_average=True)
        return criterion

    def load_optimizer(self, conf):
        optimizer = torch.optim.SGD(params=[
            {'params': self.model.backbone.parameters(), 'lr': 0.1 * conf.optimizer.lr},
            {'params': self.model.classifier.parameters(), 'lr': conf.optimizer.lr}, ], lr=conf.optimizer.lr,
            momentum=conf.optimizer.momentum, weight_decay=conf.optimizer.weight_decay)

        return optimizer
    
    def load_loader(self, conf):
        return self.train_data.loader(conf.batch_size, self.cid, shuffle=True, seed=conf.seed, drop_last=True)
    
    def test(self, conf, device=CPU):
        """Execute client testing.

        Args:
            conf (omegaconf.dictconfig.DictConfig): Client configurations.
            device (str): Hardware device for training, cpu or cuda devices.
        """
        begin_test_time = time.time()
        self.model.eval()
        self.model.to(device)
        self.validate.reset()
        loss_fn = self.load_loss_fn(conf)
        if self.test_loader is None:
            self.test_loader = self.test_data.loader(conf.test_batch_size, self.cid, shuffle=False, seed=conf.seed)
        self.test_loss = 0
        with torch.no_grad():
            for batched_x, batched_y in self.test_loader:
                images = batched_x.to(device)
                labels = batched_y.to(device)
                outputs = self.model(images)
                loss = loss_fn(outputs, labels)
                self.test_loss += loss.item()

                predicts = outputs.detach().max(dim=1)[1].cpu().numpy()
                targets = labels.cpu().numpy()
                self.validate.update(targets, predicts)

        self.test_loss /= len(self.test_loader)
        scores = self.validate.get_results()
        scores["loss"] = float(self.test_loss)
        self.test_time = time.time() - begin_test_time
        self.test_metric = scores
        self.model = self.model.cpu()
