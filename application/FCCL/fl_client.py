import copy
import logging
import time
import torch
from coala.client import BaseClient
from coala.distributed.distributed import CPU

logger = logging.getLogger(__name__)


class ContinualClient(BaseClient):
    def __init__(self, cid, conf, train_data, test_data, device, **kwargs):
        super(ContinualClient, self).__init__(cid, conf, train_data, test_data, device, **kwargs)
        self.total_task = train_data.num_tasks()
        self.cur_task = 0
        self.known_classes = 0
        self.train_loader = None
        self.syn_data_loader = None
        self.old_model = None
        self._datasize = 0
        self.update_datasize()
    
    def update_datasize(self):
        self._datasize = self.train_data.size(self.cid, self.cur_task) if self.train_data else 0


    def train(self, conf, device=CPU):
        if self.cur_task != conf.task_index:
            self.train_loader = None
            self.cur_task = conf.task_index
            self.known_classes = self.train_data.get_known_classes(self.cur_task)
            self.update_datasize()
        if conf.task_index == 0:
            self.init_train(conf, device)
        elif self.syn_data_loader is not None:
            self.finetune_with_synthesis(conf, device)
        else:
            self.finetune_without_synthesis(conf, device)

    def load_loader(self, conf):
        """Load the training data loader.

        Args:
            conf (omegaconf.dictconfig.DictConfig): Client configurations.
        Returns:
            torch.utils.data.DataLoader: Data loader.
        """
        return self.train_data.loader(conf.batch_size, conf.task_index, self.cid, shuffle=True, seed=conf.seed)
    
    def set_model(self, model):
        """Set the given model as the client model.
        This method should be overwritten for different training backend, the default is PyTorch.

        Args:
            model (options: nn.Module, tf.keras.Model, ...): Global model distributed from the server.
        """
        self.model = copy.deepcopy(model)

    def receive(self, old_model, syn_data_loader, conf):
        self.conf = conf
        self.old_model = copy.deepcopy(old_model)
        self.syn_data_loader = copy.deepcopy(syn_data_loader)

    def init_train(self, conf, device=CPU):
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
                out = self.model(x)
                loss = loss_fn(out, y)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            current_epoch_loss = sum(batch_loss) / len(batch_loss)
            self.train_loss.append(float(current_epoch_loss))
            logger.debug("Client {}, local epoch: {}, loss: {}".format(self.cid, i, current_epoch_loss))
        self.train_time = time.time() - start_time
        logger.debug("Client {}, Train Time: {}".format(self.cid, self.train_time))

    def finetune_without_synthesis(self, conf, device=CPU):
        start_time = time.time()
        loss_fn, optimizer = self.pretrain_setup(conf, device)
        self.train_loss = []
        for i in range(conf.local_epoch):
            batch_loss = []
            for _, (images, labels) in enumerate(self.train_loader):
                x, y = images.to(device), labels.to(device)
                optimizer.zero_grad()
                fake_targets = y - self.known_classes
                out = self.model(x)
                loss = loss_fn(out[:, self.known_classes:], fake_targets)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            current_epoch_loss = sum(batch_loss) / len(batch_loss)
            self.train_loss.append(float(current_epoch_loss))
            logger.debug("Client {}, local epoch: {}, loss: {}".format(self.cid, i, current_epoch_loss))
        self.train_time = time.time() - start_time
        logger.debug("Client {}, Train Time: {}".format(self.cid, self.train_time))

    def finetune_with_synthesis(self, conf, device=CPU):
        start_time = time.time()
        loss_fn, optimizer = self.pretrain_setup(conf, device)
        teacher = copy.deepcopy(self.old_model).to(device)
        teacher.eval()
        self.train_loss = []
        for i in range(conf.local_epoch):
            batch_loss = []
            total_local = 0.0
            total_syn = 0.0
            iter_loader = enumerate(zip((self.train_loader), (self.syn_data_loader)))
            for _, ((images, labels), syn_input) in iter_loader:
                x, y, syn_input = images.to(device), labels.to(device), syn_input.to(device)
                optimizer.zero_grad()
                fake_targets = y - self.known_classes
                out = self.model(x)
                loss_ce = loss_fn(out[:, self.known_classes:], fake_targets)
                with torch.no_grad():
                    t_out = teacher(syn_input.detach())
                    total_syn += syn_input.shape[0]
                    total_local += images.shape[0]
                # for old task
                loss_kd = 0
                if conf.kd_alpha > 0:
                    s_out = self.model(syn_input)
                    loss_kd = self.kd_loss(s_out[:, :self.known_classes], t_out.detach(), 2)
                loss = loss_ce + conf.kd_alpha * loss_kd
                loss.backward()
                optimizer.step()
                batch_loss.append(loss_ce.item())
            current_epoch_loss = sum(batch_loss) / len(batch_loss)
            self.train_loss.append(float(current_epoch_loss))
            logger.debug("Client {}, local epoch: {}, loss: {}".format(self.cid, i, current_epoch_loss))
        self.train_time = time.time() - start_time
        logger.debug("Client {}, Train Time: {}".format(self.cid, self.train_time))

    def kd_loss(self, predict, soft, temp=1):
        predict = torch.log_softmax(predict / temp, dim=1)
        soft = torch.softmax(soft / temp, dim=1)
        return -1 * torch.mul(soft, predict).sum() / predict.shape[0]
