import logging
import os
import time
import copy
import numpy as np
import math
from omegaconf import OmegaConf
from coala.distributed.distributed import CPU
import torch.distributed as dist
import torch
from torchvision import transforms
from coala.server.base import BaseServer
from coala.tracking import metric
from coala.protocol import codec
from coala.utils.float import rounding

from synthesizer import GlobalSynthesizer, weight_init, Generator, UnlabeledImageDataset, Normalizer
from utils import KLDiv

logger = logging.getLogger(__name__)

data_normalize = dict(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761))
normalizer = Normalizer(**dict(data_normalize))

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(**dict(data_normalize)),
])


class ContinualServer(BaseServer):
    """Default implementation of federated continual learning server.

    Args:
        conf (omegaconf.dictconfig.DictConfig): Configurations of COALA.
        test_data (:obj:`FederatedDataset`): Test dataset for centralized testing in server, optional.
        val_data (:obj:`FederatedDataset`): Validation dataset for centralized validation in server, optional.
        is_remote (bool): A flag to indicate whether start remote training.
        local_port (int): The port of remote server service.
    """

    def __init__(self,
                 conf,
                 test_data=None,
                 val_data=None,
                 is_remote=False,
                 local_port=22999):
        super(ContinualServer, self).__init__(conf, test_data, val_data, is_remote, local_port)

        self.is_training = False
        self.bs = conf.server.batch_size
        self.save_dir = conf.data.syn_dir
        self.total_tasks = test_data.num_tasks()
        self.cur_task = -1
        self.known_classes = 0
        self.total_classes = 0
        self.test_loader = None
        self.syn_size = conf.data.synthetic_size
        self.syn_data_loader = None
        self.old_model = None
        self.synthesis = self.conf.data.synthesis
        syn_round = self.conf.synth.syn_round
        syn_bs = self.conf.synth.synthesis_batch_size
        if syn_bs * syn_round < self.syn_size:
            # self.conf.synth.syn_round = int(self.syn_size/syn_bs+1)
            self.syn_size = syn_bs * syn_round

        assert self.is_remote is False, "remote is currently not supported"

    def start(self, model, clients):
        """Start federated learning process, including training and testing.

        Args:
            model (nn.Module): The model to train.
            clients (list[:obj:`BaseClient`]|list[str]): Available clients.
                Clients are actually client grpc addresses when in remote training.
        """
        # Setup
        self._start_time = time.time()
        self._reset()
        self.set_model(model)
        self.set_clients(clients)

        for task_id in range(self.total_tasks):
            self._reset()
            self.cur_task += 1
            self.conf.task_id = str(task_id)
            self.conf.client.task_index = task_id
            cur_classes = self.test_data.get_task_size(task_id)
            self.total_classes = self.known_classes + cur_classes

            # update model output layer
            self.model.update_fc(self.total_classes)

            if self._should_track():
                self.tracker.create_task(self.conf.task_id, OmegaConf.to_container(self.conf))

            # Get initial testing accuracies
            if self.conf.server.test_all:
                if self._should_track():
                    self.tracker.set_round(self.current_round)
                self.test()
                self.save_tracker()

            while not self.should_terminate():
                self._round_time = time.time()

                self.current_round += 1
                self.print_("\n-------- round {} --------".format(self.current_round))

                # Train
                self.pre_train()
                self.train()
                self.post_train()

                # Test
                if self._do_every(self.conf.server.test_every, self.current_round, self.conf.server.rounds):
                    self.pre_test()
                    self.test()
                    self.post_test()

                # Save Model
                self.save_model()

                self.track(metric.ROUND_TIME, time.time() - self._round_time)
                self.save_tracker()

            for key, values in self.performance_metrics.items():
                self.print_("{}: {}".format(str(key).capitalize(), rounding(values, 4)))

            self.after_task()

        self.print_("Cumulative training time: {}".format(rounding(self.cumulative_times, 2)))

    def pre_train(self):
        if self.cur_task > 0 and self.synthesis:
            self.syn_data_loader = self.get_syn_data_loader()

    def after_task(self):
        self.known_classes = self.total_classes
        self.old_model = self.model.copy().freeze()
        if self.cur_task + 1 != self.total_tasks and self.synthesis:
            if self.is_primary_server():
                self.data_generation()
            if self.conf.is_distributed:
                dist.barrier()

    def test_in_server(self, device=CPU):
        mean_acc, mean_loss = self.compute_mean_acc(self.model, device)
        test_results = {metric.TEST_METRIC: {"accuracy": float(mean_acc), "loss": float(mean_loss)}}

        return test_results

    def distribution_to_train_locally(self):
        """Conduct training sequentially for selected clients in the group."""
        uploaded_models = {}
        uploaded_weights = {}
        uploaded_metrics = []
        for client in self.grouped_clients:
            # Update client config before training
            self.conf.client.task_id = self.conf.task_id
            self.conf.client.round_id = self.current_round

            if self.synthesis:
                client.receive(self.old_model, self.syn_data_loader, self.conf.client)

            uploaded_request = client.run_train(self.model, self.conf.client)
            uploaded_content = uploaded_request.content

            model = self.decompression(codec.unmarshal(uploaded_content.data))
            uploaded_models[client.cid] = model
            uploaded_weights[client.cid] = uploaded_content.data_size
            uploaded_metrics.append(metric.ClientMetric.from_proto(uploaded_content.metric))

        self.set_client_uploads_train(uploaded_models, uploaded_weights, uploaded_metrics)

    def kd_train(self, student, teacher, criterion, optimizer):
        student.to(self.conf.device)
        student.train()
        teacher.to(self.conf.device)
        teacher.eval()
        loader = self.get_all_syn_data()
        data_iter = iter(loader)
        iter_num = len(data_iter)
        for i in range(self.conf.synth.kd_steps):
            if i % iter_num == 0:
                data_iter = iter(loader)
            images = next(data_iter).to(self.conf.device)
            with torch.no_grad():
                t_out = teacher(images)
            s_out = student(images.detach())
            loss_s = criterion(s_out, t_out.detach())
            optimizer.zero_grad()
            loss_s.backward()
            optimizer.step()

    def get_syn_data_loader(self):
        dataset_size = 50000
        iters = math.ceil(
            dataset_size / (self.conf.data.num_of_clients * self.total_tasks * self.conf.client.batch_size))
        syn_bs = int(self.syn_size / iters)
        data_dir = os.path.join(self.save_dir, "task_{}".format(self.cur_task - 1))

        syn_dataset = UnlabeledImageDataset(data_dir, transform=train_transform, nums=self.syn_size)
        syn_data_loader = torch.utils.data.DataLoader(syn_dataset, batch_size=syn_bs,
                                                      shuffle=True, num_workers=4, pin_memory=True)
        return syn_data_loader

    def get_all_syn_data(self):
        data_dir = os.path.join(self.save_dir, "task_{}".format(self.cur_task))
        syn_dataset = UnlabeledImageDataset(data_dir, transform=train_transform)
        loader = torch.utils.data.DataLoader(syn_dataset, batch_size=self.conf.synth.sample_batch_size,
                                             shuffle=True, num_workers=4, pin_memory=True, sampler=None)
        return loader

    def data_generation(self):
        nz = 256
        img_size = 32
        img_shape = (3, 32, 32)

        generator = Generator(nz=nz, ngf=64, img_size=img_size, nc=3).to(self.conf.device)
        teacher = copy.deepcopy(self.model).to(self.conf.device)
        student = copy.deepcopy(teacher)
        student.apply(weight_init)
        tmp_dir = os.path.join(self.save_dir, "task_{}".format(self.cur_task))

        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)

        syn_hyp = self.conf.synth
        synthesizer = GlobalSynthesizer(teacher, student, generator,
                                        nz=nz, num_classes=self.total_classes, img_size=img_shape, init_dataset=None,
                                        save_dir=tmp_dir, transform=train_transform, normalizer=normalizer,
                                        synthesis_batch_size=syn_hyp.synthesis_batch_size,
                                        sample_batch_size=syn_hyp.sample_batch_size,
                                        iterations=syn_hyp.g_steps, warmup=syn_hyp.warmup,
                                        lr_g=syn_hyp.lr_g, lr_z=syn_hyp.lr_z,
                                        adv=syn_hyp.adv, bn=syn_hyp.bn, oh=syn_hyp.oh,
                                        reset_l0=syn_hyp.reset_l0, reset_bn=syn_hyp.reset_bn,
                                        bn_mmt=syn_hyp.bn_mmt, is_maml=syn_hyp.is_maml,
                                        device=self.conf.device)

        criterion = KLDiv(T=syn_hyp.T)
        optimizer = torch.optim.SGD(student.parameters(), lr=0.2, weight_decay=0.0001,
                                    momentum=0.9)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 200, eta_min=2e-4)

        for it in range(syn_hyp.syn_round):
            synthesizer.synthesize()  # generate synthetic data
            if it >= syn_hyp.warmup:
                self.kd_train(student, self.model, criterion, optimizer)  # kd_steps
                test_acc, _ = self.compute_mean_acc(student, self.conf.device)
                print("Task {}, Data Generation, Epoch {}/{} =>  Student test_acc: {:.2f}".format(
                    self.cur_task, it + 1, syn_hyp.syn_round, test_acc, ))
                scheduler.step()
        del teacher
        del student
        logger.info("For task {}, data generation completed! ".format(self.cur_task))

    def compute_mean_acc(self, model=None, device=CPU):
        model = self.model if model is None else model
        model.eval()
        model.to(device)
        loss_fn = torch.nn.CrossEntropyLoss().to(device)
        accuracies = []
        losses = []
        for task_id in range(self.cur_task + 1):
            test_loss = 0
            correct = 0
            test_loader = self.test_data.loader(self.conf.server.batch_size, task_id, seed=self.conf.seed)
            with torch.no_grad():
                for batched_x, batched_y in test_loader:
                    x = batched_x.to(device)
                    y = batched_y.to(device)
                    log_probs = model(x)
                    loss = loss_fn(log_probs, y)
                    _, y_pred = torch.max(log_probs, -1)
                    correct += y_pred.eq(y.data.view_as(y_pred)).long().cpu().sum()
                    test_loss += loss.item()
                test_data_size = self.test_data.size()
                test_loss /= len(test_loader)
                accuracy = 100.00 * correct / test_data_size
                accuracies.append(accuracy)
                losses.append(test_loss)
        self.model = self.model.cpu()

        mean_acc = sum(accuracies) / len(accuracies)
        mean_loss = sum(losses) / len(losses)

        return mean_acc, mean_loss
