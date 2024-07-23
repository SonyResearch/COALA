import logging
import time
import torch
import copy
from coala.client import BaseClient
from coala.distributed.distributed import CPU
from torchvision import transforms
from losses import SupConLoss, softmax_entropy
from corruptions import RandCorruptions_Common

logger = logging.getLogger(__name__)


class FedAvgClient(BaseClient):
    def __init__(self, cid, conf, train_data, test_data, device, **kwargs):
        super(FedAvgClient, self).__init__(cid, conf, train_data, test_data, device, **kwargs)
        self.test_transform_orig = copy.deepcopy(self.test_data.transform)
        self.test_data.test_corruptted = False

    def reset_test(self):
        self.test_data.transform = copy.deepcopy(self.test_transform_orig)

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
            for batched_x, batched_y in self.train_loader:
                x, y = batched_x.to(device), batched_y.to(device)
                optimizer.zero_grad()
                _, out = self.model(x)
                loss = loss_fn(out, y)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            current_epoch_loss = sum(batch_loss) / len(batch_loss)
            self.train_loss.append(float(current_epoch_loss))
            logger.debug("Client {}, local epoch: {}, loss: {}".format(self.cid, i, current_epoch_loss))
        self.train_time = time.time() - start_time
        logger.debug("Client {}, Train Time: {}".format(self.cid, self.train_time))

    def set_test_corruption(self):
        if self.test_data is not None and self.conf.shift_type == 'c_shift':
            transform_original = self.test_data.transform
            corruption = RandCorruptions_Common(seed=self.conf.seed, severity=self.conf.severity)
            if isinstance(transform_original, dict):
                if isinstance(self.test_data.transform[self.cid].transforms[-1], transforms.Normalize):
                    self.test_data.transform[self.cid].transforms.insert(-1, corruption)
                else:
                    self.test_data.transform[self.cid].transforms.append(corruption)
            else:
                if not self.test_data.test_corruptted:
                    if isinstance(self.test_data.transform.transforms[-1], transforms.Normalize):
                        self.test_data.transform.transforms.insert(-1, corruption)
                    else:
                        self.test_data.transform.transforms.append(corruption)
                    self.test_data.test_corruptted = True

    def pre_test(self):
        """Preprocessing before testing."""

        if self.test_loader is None:
            self.test_loader = self.test_data.loader(self.conf.test_batch_size, self.cid, shuffle=False)

        if self.conf.round_id == self.conf.rounds - 1:
            if self.conf.shift_type == 'd_shift':  # exchange test set for simulating domain shift
                num_clients = len(self.test_data.data)
                dataset_id = str(num_clients - 1 - int(self.cid[-7:]))
                self.test_loader = self.test_data.loader(self.conf.test_batch_size, dataset_id, shuffle=False)

            elif self.conf.shift_type == 'c_shift':
                self.set_test_corruption()
            
            self.tent(self.conf, self.device)

            if self.conf.shift_type == 'c_shift':
                self.reset_test()
                self.set_test_corruption()

    def tent(self, conf, device=CPU):
        self.model.train()
        self.model.to(device)
        params, _ = collect_params(self.model)
        optimizer = set_optimizer(conf, params)
        for i in range(conf.ft_epoch):
            for images, labels in self.test_loader:
                if isinstance(images, tuple):
                    images = images[0]
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                _, logit = self.model(images)
                loss = softmax_entropy(logit).mean(0)
                loss.backward()
                optimizer.step()

    def test(self, conf, device=CPU):
        begin_test_time = time.time()
        self.model.eval()
        self.model.to(device)
        loss_fn = torch.nn.CrossEntropyLoss()
        self.test_loss = 0
        correct = 0
        with torch.no_grad():
            for batched_x, batched_y in self.test_loader:
                x = batched_x.to(device)
                y = batched_y.to(device)
                _, log_probs = self.model(x)
                loss = loss_fn(log_probs, y)
                _, y_pred = torch.max(log_probs, -1)
                correct += y_pred.eq(y.data.view_as(y_pred)).long().cpu().sum()
                self.test_loss += loss.item()
            test_size = self.test_data.size(self.cid)
            self.test_loss /= len(self.test_loader)
            self.test_accuracy = 100.0 * float(correct) / test_size

        logger.debug('Client {}, testing -- Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
            self.cid, self.test_loss, correct, test_size, self.test_accuracy))

        self.test_time = time.time() - begin_test_time
        self.test_metric = {"accuracy": self.test_accuracy, "loss": self.test_loss}
        self.model = self.model.cpu()


class TwoCropTransform(object):
    """Create two crops of the same image"""

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]
    

class ContrastiveClient(BaseClient):
    def __init__(self, cid, conf, train_data, test_data, device, **kwargs):
        super(ContrastiveClient, self).__init__(cid, conf, train_data, test_data, device, **kwargs)
        self.temperature = conf.temperature
        self.base_temperature = conf.base_temperature
        self.local_model = copy.deepcopy(self.model) # local personalized model
        self.test_transform_orig = copy.deepcopy(self.test_data.transform)
        self.test_data.test_corruptted = False
        self.set_train_augmentation()

    def set_train_augmentation(self):
        transform_original = self.train_data.transform
        if isinstance(transform_original, dict):
            transform_original = transform_original[self.cid]
            self.train_data.transform[self.cid] = copy.deepcopy(TwoCropTransform(transform_original))
        else:
            if not isinstance(self.train_data.transform, TwoCropTransform):
                self.train_data.transform = copy.deepcopy(TwoCropTransform(transform_original))

    def set_test_corruption(self):
        if self.test_data is not None and self.conf.shift_type == 'c_shift':
            transform_original = self.test_data.transform
            corruption = RandCorruptions_Common(seed=self.conf.seed, severity=self.conf.severity)
            if isinstance(transform_original, dict):
                transform_original = transform_original[self.cid]
                if isinstance(transform_original.transforms[-1], transforms.Normalize):
                    transform_original.transforms.insert(-1, corruption)
                else:
                    transform_original.transforms.append(corruption)
                self.test_data.transform[self.cid] = copy.deepcopy(TwoCropTransform(transform_original))
            else:
                if not self.test_data.test_corruptted:
                    if isinstance(transform_original.transforms[-1], transforms.Normalize):
                        transform_original.transforms.insert(-1, corruption)
                    else:
                        transform_original.transforms.append(corruption)
                    self.test_data.transform = copy.deepcopy(TwoCropTransform(transform_original))
                    self.test_data.test_corruptted = True
    
    def reset_test(self):
        self.test_data.transform = copy.deepcopy(self.test_transform_orig)
                    
    def train(self, conf, device=CPU):
        if conf.round_id < conf.rounds - 5:
            self.train_feature(conf, device)
        else:
            self.train_cls(conf, self.model, device)

    def train_all(self, conf, device=CPU):
        """Execute client training.

        Args:
            conf (omegaconf.dictconfig.DictConfig): Client configurations.
            device (str): Hardware device for training, cpu or cuda devices.
        """
        start_time = time.time()
        self.model.train()
        self.model.to(device)
        loss_fn = torch.nn.CrossEntropyLoss()
        if self.train_loader is None:
            self.train_loader = self.load_loader(conf)

        if conf.optimizer.type == "Adam":
            optimizer = torch.optim.Adam(self.model.parameters(), lr=conf.optimizer.lr)
        else:
            # default using optimizer SGD
            optimizer = torch.optim.SGD(self.model.parameters(),
                                        lr=conf.optimizer.lr,
                                        momentum=conf.optimizer.momentum,
                                        weight_decay=conf.optimizer.weight_decay)
        self.train_loss = []
        for i in range(conf.local_epoch):
            batch_loss = []
            for images, labels in self.train_loader:
                original_img = images[0].to(self.device)
                # original_img = images.to(self.device)
                labels = labels.to(self.device)
                optimizer.zero_grad()
                _, log_probs = self.model(original_img)
                loss = loss_fn(log_probs, labels)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            current_epoch_loss = sum(batch_loss) / len(batch_loss)
            self.train_loss.append(float(current_epoch_loss))
            logger.debug("Client {}, local epoch: {}, loss: {}".format(self.cid, i, current_epoch_loss))
        self.train_time = time.time() - start_time
        logger.debug("Client {}, Train Time: {}".format(self.cid, self.train_time))

    def train_feature(self, conf, device=CPU):
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
            for images, labels in self.train_loader:
                images = torch.cat([images[0], images[1]], dim=0)
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                features, _ = self.model(images)
                bsz = labels.shape[0]
                f1, f2 = torch.split(features, [bsz, bsz], dim=0)
                features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
                loss = loss_fn(features, labels)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            current_epoch_loss = sum(batch_loss) / len(batch_loss)
            self.train_loss.append(float(current_epoch_loss))
            logger.debug("Client {}, local epoch: {}, loss: {}".format(self.cid, i, current_epoch_loss))
        self.train_time = time.time() - start_time
        logger.debug("Client {}, Train Time: {}".format(self.cid, self.train_time))

    def train_cls(self, conf, model, device=CPU):
        """Execute client training.

        Args:
            conf (omegaconf.dictconfig.DictConfig): Client configurations.
            device (str): Hardware device for training, cpu or cuda devices.
        """
        start_time = time.time()
        model.train()
        model.to(device)
        loss_fn = torch.nn.CrossEntropyLoss()
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
                original_img = images[0].to(self.device)
                labels = labels.to(self.device)
                optimizer.zero_grad()
                _, log_probs = model(original_img)
                loss = loss_fn(log_probs, labels)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            current_epoch_loss = sum(batch_loss) / len(batch_loss)
            self.train_loss.append(float(current_epoch_loss))
            logger.debug("Client {}, local epoch: {}, loss: {}".format(self.cid, i, current_epoch_loss))
        self.train_time = time.time() - start_time
        logger.debug("Client {}, Train Time: {}".format(self.cid, self.train_time))

        for name, param in model.named_parameters():
            param.requires_grad = True

    def load_loss_fn(self, conf):
        return SupConLoss(self.temperature, self.base_temperature)

    def load_optimizer(self, conf, model=None):
        """Load training optimizer. Implemented Adam and SGD."""
        model = self.model if model is None else model
        if conf.optimizer.type == "Adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=conf.optimizer.lr_sc)
        else:
            # default using optimizer SGD
            optimizer = torch.optim.SGD(model.parameters(),
                                        lr=conf.optimizer.lr_sc,
                                        momentum=conf.optimizer.momentum,
                                        weight_decay=conf.optimizer.weight_decay)
        return optimizer

    def test(self, conf, device=CPU):
        begin_test_time = time.time()
        self.local_model.eval()
        self.local_model.to(device)
        loss_fn = torch.nn.CrossEntropyLoss()
        self.test_loss = 0
        correct = 0
        with torch.no_grad():
            for batched_x, batched_y in self.test_loader:
                if isinstance(batched_x, tuple):
                    x = batched_x[0].to(device)
                else:
                    x = batched_x.to(device)
                y = batched_y.to(device)
                _, log_probs = self.local_model(x)
                loss = loss_fn(log_probs, y)
                _, y_pred = torch.max(log_probs, -1)
                correct += y_pred.eq(y.data.view_as(y_pred)).long().cpu().sum()
                self.test_loss += loss.item()
            test_size = self.test_data.size(self.cid)
            self.test_loss /= len(self.test_loader)
            self.test_accuracy = 100.0 * float(correct) / test_size

        logger.debug('Client {}, testing -- Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
            self.cid, self.test_loss, correct, test_size, self.test_accuracy))

        self.test_time = time.time() - begin_test_time
        self.test_metric = {"accuracy": self.test_accuracy, "loss": self.test_loss}
        self.local_model = self.local_model.cpu()
        self.model = self.model.cpu()

    def pre_test(self):
        """Preprocessing before testing."""

        # train a local classifier head for local test
        self.train_cls(self.conf, self.local_model, self.device)

        if self.test_loader is None:
            self.test_loader = self.test_data.loader(self.conf.test_batch_size, self.cid, shuffle=False)

        if self.conf.round_id == self.conf.rounds - 1:
            if self.conf.shift_type == 'd_shift':  # exchange test set for simulating domain shift
                num_clients = len(self.test_data.data)
                dataset_id = str(num_clients - 1 - int(self.cid[-7:]))
                self.test_loader = self.test_data.loader(self.conf.test_batch_size, dataset_id, shuffle=False)

            elif self.conf.shift_type == 'c_shift':
                self.set_test_corruption()
            
            # fine-tune the feature extractor during the test-time adaptation phase (last round)
            # self.test_time_adaptation(self.conf, self.device)
            self.tent(self.conf, self.device)

            if self.conf.shift_type == 'c_shift':
                self.reset_test()
                self.set_test_corruption()

    def test_time_adaptation(self, conf, device=CPU):
        start_time = time.time()
        self.local_model.train()
        self.local_model.to(device)
        loss_fn = self.load_loss_fn(conf)
        optimizer = self.load_optimizer(conf, self.local_model)
        # params, _ = collect_params(self.local_model)
        # optimizer = set_optimizer(conf, params)
        for i in range(conf.ft_epoch):
            for images, labels in self.test_loader:
                if isinstance(images, tuple):
                    images = torch.cat([images[0], images[1]], dim=0)
                else:
                    images = torch.cat([images, images], dim=0)
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                features, _ = self.local_model(images)
                bsz = labels.shape[0]
                f1, f2 = torch.split(features, [bsz, bsz], dim=0)
                features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
                loss = loss_fn(features)
                loss.backward()
                optimizer.step()
        self.adap_time = time.time() - start_time
        logger.debug("Client {}, Adaptation Time: {}".format(self.cid, self.adap_time))

    def tent(self, conf, device=CPU):
        self.local_model.train()
        self.local_model.to(device)
        params, _ = collect_params(self.local_model)
        optimizer = set_optimizer(conf, params)
        for i in range(conf.ft_epoch):
            for images, labels in self.test_loader:
                if isinstance(images, tuple):
                    images = images[0]
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                _, logit = self.local_model(images)
                loss = softmax_entropy(logit).mean(0)
                loss.backward()
                optimizer.step()

    def set_model(self, model):
        """Set the given model as the client model.
        This method should be overwritten for different training backend, the default is PyTorch.

        Args:
            model (options: nn.Module, tf.keras.Model, ...): Global model distributed from the server.
        """
        if self.model:
            self.model.load_state_dict(model.state_dict())
            self.local_model.load_state_dict(model.state_dict())
        else:
            self.model = copy.deepcopy(model)
            self.local_model = copy.deepcopy(self.model)


def collect_params(model):
    """Collect the affine scale + shift parameters from batch norms.

    Walk the model's modules and collect all batch normalization parameters.
    Return the parameters and their names.

    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:  # weight is scale, bias is shift
                    params.append(p)
                    names.append(f"{nm}.{np}")
    return params, names


def set_optimizer(conf, params=None):
        """Load training optimizer. Implemented Adam and SGD."""
        if conf.optimizer.type == "Adam":
            optimizer = torch.optim.Adam(params, lr=conf.optimizer.lr_sc)
        else:
            # default using optimizer SGD
            optimizer = torch.optim.SGD(params,
                                        lr=conf.optimizer.lr_sc,
                                        momentum=conf.optimizer.momentum,
                                        weight_decay=conf.optimizer.weight_decay)
        return optimizer
