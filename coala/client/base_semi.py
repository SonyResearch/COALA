import copy
import logging
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from coala.datasets.dataset_util import TransformDataset
from coala.client.base import BaseClient
from coala.distributed.distributed import CPU

logger = logging.getLogger(__name__)


class SemiFLClient(BaseClient):
    """Implementation of federated semi-supervised learning client.

    Args:
        cid (str): Client id.
        conf (omegaconf.dictconfig.DictConfig): Client configurations.
        s_train_data (:obj:`FederatedDataset`): Labeled Training dataset.
        u_train_data (:obj:`FederatedDataset`): Unlabeled Training dataset.
        test_data (:obj:`FederatedDataset`): Test dataset.
        device (str): Hardware device for training, cpu or cuda devices.
    """

    def __init__(self,
                 cid,
                 conf,
                 s_train_data,
                 u_train_data,
                 test_data,
                 device,
                 sleep_time=0,
                 is_remote=False,
                 local_port=23000,
                 server_addr="localhost:22999",
                 tracker_addr="localhost:12666"
                 ):
        super(SemiFLClient, self).__init__(cid, conf, s_train_data, test_data, device, sleep_time,
                                           is_remote, local_port, server_addr, tracker_addr)
        self.s_train_data = s_train_data
        self.u_train_data = u_train_data
        self.s_train_loader = None
        self.u_train_loader = None
        self.semi_scenario = "label_in_server" if s_train_data is None else "label_in_client"
        self.bs = conf.batch_size
        self.num_step = round(self.s_train_data.size(self.cid) / self.bs) if self.s_train_data is not None else None
        self.u_bs = self.u_train_data.size(self.cid) // self.num_step if self.num_step else conf.batch_size
        self.threshold = 0.95  # threshold for pseudo labeling
        self.lam = 1.0
        self.beta = torch.distributions.beta.Beta(torch.tensor(0.75), torch.tensor(0.75))

        self.fix_loader = None
        self.mix_loader = None
        self.selected_size = 0
        self._data_size = self.u_train_data.size(self.cid) if self.semi_scenario == "label_in_server"\
            else self.u_train_data.size(self.cid) + self.s_train_data.size(self.cid)

    def train(self, conf, device=CPU):
        """Execute client training.

        Args:
            conf (omegaconf.dictconfig.DictConfig): Client configurations.
            device (str): Hardware device for training, cpu or cuda devices.
        """
        if self.semi_scenario == 'label_in_server':
            # self.train_unsupervised(conf, device) # when dynamic pseudo-labeling is used during local training
            self.train_unsupervised_fix(conf, device)  # when fixed pseudo-labeling is used during local training
        else:
            self.train_semi_supervised(conf, device)

    def train_unsupervised(self, conf, device=CPU):
        """Execute client unsupervised training with pseudo labeling from real-time local model.

        Args:
            conf (omegaconf.dictconfig.DictConfig): Client configurations.
            device (str): Hardware device for training, cpu or cuda devices.
        """
        start_time = time.time()
        _, optimizer = self.pretrain_setup(conf, device)
        self.train_loss = []
        global_model = copy.deepcopy(self.model)
        for i in range(conf.local_epoch):
            batch_loss = []
            for (batched_u_w_x, batched_u_s_x), y in self.u_train_loader:
                weak_x, strong_x = batched_u_w_x.to(device), batched_u_s_x.to(device)
                inputs = torch.cat((weak_x, strong_x), dim=0)

                optimizer.zero_grad()
                out = self.model(inputs)
                logits_u_g = global_model(weak_x)
                logits_u_w, logits_u_s = out.chunk(2, dim=0)
                del out
                # unsupervised loss
                pseudo_label = torch.softmax(logits_u_g.detach() / 1.0, dim=-1)
                max_probs, targets_u = torch.max(pseudo_label, dim=-1)
                mask = max_probs.ge(self.threshold).float()
                loss = (F.cross_entropy(logits_u_s, targets_u, reduction='none') * mask).mean()

                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            current_epoch_loss = sum(batch_loss) / len(batch_loss)
            self.train_loss.append(float(current_epoch_loss))
            logger.debug("Client {}, local epoch: {}, loss: {}".format(self.cid, i, current_epoch_loss))
        self.train_time = time.time() - start_time
        logger.debug("Client {}, Train Time: {}".format(self.cid, self.train_time))

    def train_semi_supervised(self, conf, device=CPU):
        """Execute client semi-supervised training.

        Args:
            conf (omegaconf.dictconfig.DictConfig): Client configurations.
            device (str): Hardware device for training, cpu or cuda devices.
        """
        start_time = time.time()
        loss_fn, optimizer = self.pretrain_setup(conf, device)

        self.train_loss = []
        for i in range(conf.local_epoch):
            batch_loss = []
            for ((batched_u_w_x, batched_u_s_x), _), (batched_s_x, batched_s_y) in zip(self.u_train_loader,
                                                                                       self.s_train_loader):
                bs = len(batched_s_x)
                weak_x, strong_x, super_x, super_y = batched_u_w_x.to(device), batched_u_s_x.to(device), \
                    batched_s_x.to(device), batched_s_y.to(device),
                inputs = torch.cat((super_x, weak_x, strong_x), dim=0)

                optimizer.zero_grad()
                out = self.model(inputs)

                logits_s = out[:bs]
                logits_u_w, logits_u_s = out[bs:].chunk(2, dim=0)
                del out
                # unsupervised loss
                pseudo_label = torch.softmax(logits_u_w.detach() / 1.0, dim=-1)
                max_probs, targets_u = torch.max(pseudo_label, dim=-1)
                mask = max_probs.ge(self.threshold).float()
                un_loss = (F.cross_entropy(logits_u_s, targets_u,
                                           reduction='none') * mask).mean()
                # supervised loss
                su_loss = loss_fn(logits_s.contiguous(), super_y)
                loss = su_loss + self.lam * un_loss

                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            current_epoch_loss = sum(batch_loss) / len(batch_loss)
            self.train_loss.append(float(current_epoch_loss))
            logger.debug("Client {}, local epoch: {}, loss: {}".format(self.cid, i, current_epoch_loss))
        self.train_time = time.time() - start_time
        logger.debug("Client {}, Train Time: {}".format(self.cid, self.train_time))

    def train_unsupervised_fix(self, conf, device=CPU):
        """Execute client unsupervised training with fix-match and mix-up datasets.

        Args:
            conf (omegaconf.dictconfig.DictConfig): Client configurations.
            device (str): Hardware device for training, cpu or cuda devices.
        """
        start_time = time.time()
        loss_fn, optimizer = self.pretrain_setup(conf, device)

        if self.fix_loader is None:
            return

        self.train_loss = []
        for i in range(conf.local_epoch):
            batch_loss = []
            for _, (fix_input, mix_input) in enumerate(zip(self.fix_loader, self.mix_loader)):
                (fix_w_x, fix_s_x), fix_y = fix_input  # weak augmentation, strong augmentation for fixmatch set
                (mix_w_x, _), mix_y = mix_input  # weak augmentation for mixup set
                fix_w_x, fix_s_x = fix_w_x.to(device), fix_s_x.to(device)
                mix_w_x = mix_w_x.to(device)
                fix_y, mix_y = fix_y.to(device), mix_y.to(device)

                lam_mix = self.beta.sample()
                mix_x = (lam_mix * fix_w_x + (1 - lam_mix) * mix_w_x).detach()

                inputs = torch.cat((fix_s_x, mix_x), dim=0)
                optimizer.zero_grad()
                out = self.model(inputs)
                logits_fix, logits_mix = out.chunk(2, dim=0)

                # unsupervised loss
                loss_fix = loss_fn(logits_fix, fix_y)
                loss_mix = lam_mix * loss_fn(logits_mix, fix_y) + (1 - lam_mix) * loss_fn(logits_mix, mix_y)
                loss = loss_fix + self.lam * loss_mix
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)
                optimizer.step()
                batch_loss.append(loss.item())
            current_epoch_loss = sum(batch_loss) / len(batch_loss)
            self.train_loss.append(float(current_epoch_loss))
            logger.debug("Client {}, local epoch: {}, loss: {}".format(self.cid, i, current_epoch_loss))
        self.train_time = time.time() - start_time
        logger.debug("Client {}, Train Time: {}".format(self.cid, self.train_time))

    def pretrain_setup(self, conf, device):
        """Setup loss function and optimizer before training."""
        self.simulate_straggler()
        self.model.train()
        self.model.to(device)
        loss_fn = self.load_loss_fn(conf)
        optimizer = self.load_optimizer(conf)
        if self.u_train_loader is None:
            self.u_train_loader = self.unlabeled_data_loader(conf)
        if self.semi_scenario == 'label_in_client' and self.s_train_loader is None:
            self.s_train_loader = self.labeled_data_loader(conf)

        self.fix_loader, self.mix_loader = self.unlabeled_data_loader_fixmatch(conf, device)

        return loss_fn, optimizer

    def labeled_data_loader(self, conf):
        """Load the supervised training data loader.

        Args:
            conf (omegaconf.dictconfig.DictConfig): Client configurations.
        Returns:
            torch.utils.data.DataLoader: Data loader.
        """
        s_train_loader = self.s_train_data.loader(self.bs, self.cid, shuffle=True, seed=conf.seed)
        return s_train_loader

    def unlabeled_data_loader(self, conf):
        """Load the unsupervised training data loader.

        Args:
            conf (omegaconf.dictconfig.DictConfig): Client configurations.
        Returns:
            torch.utils.data.DataLoader: Data loader.
        """
        u_train_loader = self.u_train_data.loader(self.bs, self.cid, shuffle=True, seed=conf.seed)
        return u_train_loader

    def unlabeled_data_loader_fixmatch(self, conf, device=CPU):
        """Load the unsupervised training data loader.

        Args:
            conf (omegaconf.dictconfig.DictConfig): Client configurations.
        Returns:
            torch.utils.data.DataLoader: Data loader.
        """
        global_model = copy.deepcopy(self.model)
        u_train_loader = self.u_train_data.loader(self.u_bs, self.cid, shuffle=False, seed=conf.seed)
        u_train_data = self.u_train_data.data[self.cid]
        transform = self.u_train_data.transform
        fix_data = copy.deepcopy(u_train_data)
        mix_data = copy.deepcopy(u_train_data)
        soft_label = []

        global_model.to(device)
        global_model.eval()
        with torch.no_grad():
            for (x_w, x_s), _ in u_train_loader:
                x_w = x_w.to(device)
                logits_u = global_model(x_w)
                output = torch.softmax(logits_u.detach() / 1.0, dim=-1)
                soft_label.append(output)

        soft_label = torch.cat(soft_label, dim=0)
        max_probs, targets_u = torch.max(soft_label, dim=-1)
        mask = max_probs.ge(self.threshold).cpu()

        # fixmatch dataset
        fix_data['x'] = u_train_data['x'][mask]
        fix_data['y'] = targets_u[mask].cpu()

        fix_size = len(fix_data['x'])
        self._data_size = fix_size

        if fix_size == 0:
            return None, None

        # mixup dataset
        mix_idx = np.random.choice(range(len(u_train_data['x'])), fix_size, replace=True)
        mix_data['x'] = u_train_data['x'][mix_idx]
        mix_data['y'] = targets_u[mix_idx].cpu()

        fix_loader = load_new_data(fix_data, self.bs, shuffle=True, transform=transform)
        mix_loader = load_new_data(mix_data, self.bs, shuffle=True, transform=transform)

        return fix_loader, mix_loader


def load_new_data(data, batch_size, shuffle, transform):
    """Load the fix-match and mix-up data loader.

    Args:
        data (dict): data and label.
        batch_size (int): batch_size for unlabeled data.
        shuffle (bool): data shuffling during sampling.
        transform (torchvision.transforms.transforms.Compose, optional): Data transformation.
    Returns:
        torch.utils.data.DataLoader: Data loader.
    """
    data_x = np.array(data['x'])
    data_y = np.array(data['y'])
    dataset = TransformDataset(data_x, data_y, transform_x=transform)
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)

    return loader
