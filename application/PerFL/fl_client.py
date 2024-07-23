import logging
import time
import torch
import copy
from coala.client import BaseClient
from coala.distributed.distributed import CPU
from coala.tracking import metric
from coala.tracking.evaluation import bit_to_megabyte

logger = logging.getLogger(__name__)


class FedRepClient(BaseClient):
    def __init__(self, cid, conf, train_data, test_data, device, **kwargs):
        super(FedRepClient, self).__init__(cid, conf, train_data, test_data, device, **kwargs)

    def train(self, conf, device=CPU):
        """Execute client training.

        Args:
            conf (omegaconf.dictconfig.DictConfig): Client configurations.
            device (str): Hardware device for training, cpu or cuda devices.
        """
        start_time = time.time()
        loss_fn, optimizer = self.pretrain_setup(conf, device)
        model = self.model
        if self.train_loader is None:
            self.train_loader = self.load_loader(conf)

        for name, param in model.named_parameters():
            if 'classifier' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        if conf.optimizer.type == "Adam":
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                         lr=conf.optimizer.lr)
        else:
            # default using optimizer SGD
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                                        lr=conf.optimizer.lr,
                                        momentum=conf.optimizer.momentum,
                                        weight_decay=conf.optimizer.weight_decay)
        self.train_loss = []
        for i in range(conf.local_epoch):
            batch_loss = []
            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                output = model(images)
                loss = loss_fn(output, labels)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            current_epoch_loss = sum(batch_loss) / len(batch_loss)
            self.train_loss.append(float(current_epoch_loss))
            logger.debug("Client {}, local epoch: {}, loss: {}".format(self.cid, i, current_epoch_loss))

        for name, param in model.named_parameters():
            if 'classifier' in name:
                param.requires_grad = False
            else:
                param.requires_grad = True

        if conf.optimizer.type == "Adam":
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                         lr=conf.optimizer.lr)
        else:
            # default using optimizer SGD
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                                        lr=conf.optimizer.lr,
                                        momentum=conf.optimizer.momentum,
                                        weight_decay=conf.optimizer.weight_decay)
        for i in range(1):
            batch_loss = []
            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                output = model(images)
                loss = loss_fn(output, labels)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            current_epoch_loss = sum(batch_loss) / len(batch_loss)
            self.train_loss.append(float(current_epoch_loss))
            logger.debug(
                "Client {}, local epoch: {}, loss: {}".format(self.cid, i + conf.local_epoch, current_epoch_loss))
        self.train_time = time.time() - start_time
        logger.debug("Client {}, Train Time: {}".format(self.cid, self.train_time))

        for name, param in model.named_parameters():
            param.requires_grad = True

    def set_model(self, model):
        """Set the given model as the client model.
        This method should be overwritten for different training backend, the default is PyTorch.

        Args:
            model (options: nn.Module, tf.keras.Model, ...): Global model distributed from the server.
        """
        if self.model:
            # only share global feature extractor for local model
            global_weight = model.state_dict()
            local_weight = self.model.state_dict()
            for k in local_weight.keys():
                if k not in ["classifier"]:
                    local_weight[k] = global_weight[k]
            self.model.load_state_dict(local_weight)

        else:
            self.model = copy.deepcopy(model)

    def run_test(self, model, conf):
        """Conduct testing on clients.

        Args:
            model (nn.Module): Model to test.
            conf (omegaconf.dictconfig.DictConfig): Client configurations.
        Returns:
            :obj:`UploadRequest`: Testing contents. Unify the interface for both local and remote operations.
        """
        self.conf = conf
        if conf.track:
            reset = not self._is_train
            self._tracker.set_client_context(conf.task_id, conf.round_id, self.cid, reset_client=reset)

        self._is_train = False

        self.pre_test()
        self.test(conf, self.device)
        self.post_test()

        self.track(metric.TEST_METRIC, {"accuracy": float(self.test_accuracy), "loss": float(self.test_loss)})
        self.track(metric.TEST_TIME, self.test_time)

        return self.upload()
