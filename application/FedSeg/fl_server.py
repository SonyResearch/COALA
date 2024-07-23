import logging
from coala.distributed.distributed import CPU
import torch
from coala.tracking import metric
from coala.server.base import BaseServer
from utils import FocalLoss
from evaluate import StreamSegMetrics

logger = logging.getLogger(__name__)


class SegServer(BaseServer):
    def __init__(self, conf, test_data=None, val_data=None, is_remote=False, local_port=22999):
        super(SegServer, self).__init__(conf, test_data, val_data, is_remote, local_port)

        self.validate = StreamSegMetrics(self.conf.data.num_classes)

    def test_in_server(self, device=CPU):
        self.model.eval()
        self.model.to(device)
        self.validate.reset()
        test_loss = 0
        loss_fn = self.load_loss_fn(self.conf.client)
        test_loader = self.test_data.loader(self.conf.server.batch_size, shuffle=False, seed=self.conf.seed)
        with torch.no_grad():
            for batched_x, batched_y in test_loader:
                images = batched_x.to(device)
                labels = batched_y.to(device)
                outputs = self.model(images)
                loss = loss_fn(outputs, labels)
                test_loss += loss.item()

                predicts = outputs.detach().max(dim=1)[1].cpu().numpy()
                targets = labels.cpu().numpy()
                self.validate.update(targets, predicts)

            test_data_size = self.test_data.size()
            test_loss /= test_data_size
            scores = self.validate.get_results()
            scores["loss"] = float(test_loss)

            test_results = {metric.TEST_METRIC: scores}
            return test_results

    def load_loss_fn(self, conf):
        criterion = torch.nn.CrossEntropyLoss(ignore_index=255, reduction='mean')
        if conf.optimizer.loss_type == 'focal_loss':
            criterion = FocalLoss(ignore_index=255, size_average=True)
        return criterion
