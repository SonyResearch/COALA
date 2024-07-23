import argparse
import copy
import logging
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from coala.datasets.dataset_util import TransformDataset
from coala.client.base import BaseClient
from coala.communication import grpc_wrapper
from coala.distributed.distributed import CPU
from coala.pb import common_pb2 as common_pb
from coala.pb import server_service_pb2 as server_pb
from coala.protocol import codec
from coala.tracking import metric
from coala.tracking.client import init_tracking
from coala.tracking.evaluation import bit_to_megabyte
logger = logging.getLogger(__name__).setLevel(logging.DEBUG)


class BaseSFLClient(BaseClient):
    """Implementation of split federated learning client.
    Reference: SFLV1 in Thapa, Chandra, et al. "Splitfed: When federated learning meets split learning." 
    https://ojs.aaai.org/index.php/AAAI/article/view/20825
    
    Args:
        cid (str): Client id.
        conf (omegaconf.dictconfig.DictConfig): Client configurations.
        train_data (:obj:`FederatedDataset`): Labeled Training dataset.
        test_data (:obj:`FederatedDataset`): Test dataset.
        device (str): Hardware device for training, cpu or cuda devices.
    """

    def __init__(self,
                 cid,
                 conf,
                 train_data,
                 test_data,
                 device,
                 sleep_time=0,
                 is_remote=False,
                 local_port=23000,
                 server_addr="localhost:22999",
                 tracker_addr="localhost:12666"
                 ):
        super(BaseSFLClient, self).__init__(cid, conf, train_data, test_data, device, sleep_time,
                                           is_remote, local_port, server_addr, tracker_addr)
        
        self.label = None
        self.feature = None
        self.forward_time = None
        self.data_iterator = None
        self.bs = conf.batch_size
        self.lr_scheduler = None
        self.scheduler = conf.scheduler
        self.client_rounds = conf.rounds # default is 200, only affects scheduler

    def pretrain_setup(self, conf):
        """Setup loss function and optimizer before training."""
        self.simulate_straggler()
        
        if self.train_loader is None:
            self.train_loader = self.load_loader(conf)
        if self.data_iterator is None:
            self.data_iterator = iter(self.train_loader)

    def pre_train(self, model, conf):
        """Download up-to-date local client-side model from server.
        """
        self.pretrain_setup(conf)
        
        self.conf = conf
        if conf.track:
            self._tracker.set_client_context(conf.task_id, conf.round_id, self.cid)

        self._is_train = True
        
        self.set_model(model)

        # set up scheduler:
        self.scheduler_init()

        self.model.train()
        self.model.to(self.device)
        self.track(metric.TRAIN_DOWNLOAD_SIZE, self.calculate_model_size(model))
        
        self.decompression()
    
    def next_data_batch(self):
        """Get image/label from Client.
        """
        try:
            images, labels = next(self.data_iterator)
            if images.size(0) != self.bs:
                try:
                    next(self.data_iterator)
                except StopIteration:
                    pass
                self.data_iterator = iter(self.train_loader)
                images, labels = next(self.data_iterator)
        except StopIteration:
            self.data_iterator = iter(self.train_loader)
            images, labels = next(self.data_iterator)
        return images, labels

    def run_forward(self):
        """Client forward computation.
        """
        start_time = time.time()
        
        image, label = self.next_data_batch()
        self.label = label
        image = image.to(self.device)
        self.feature = self.model(image)# pass to online 
        
        self.forward_time = time.time() - start_time
        self.track('train_upload_feature_size', self.calculate_tensor_size(self.feature.detach()))
    

    def operate(self, model, conf, index, is_train=True):
        """A wrapper over operations (training/testing) on clients.

        Args:
            model (nn.Module): Model for operations.
            conf (omegaconf.dictconfig.DictConfig): Client configurations.
            index (int): Client index in the client list, for retrieving data. TODO: improvement.
            is_train (bool): The flag to indicate whether the operation is training, otherwise testing.
        """
        try:
            # Load the data index depending on server request
            self.cid = self.train_data.users[index]
        except IndexError:
            logger.error("Data index exceed the available data, abort training")
            return

        if self.conf.track and self._tracker is None:
            self._tracker = init_tracking(init_store=False)

        if is_train:
            self.run_backward(model) # model here refers to gradient
            # trigger upload request
            return self.upload("model")
        else:
            self.pre_train(model, conf) # model here refers to client-side model
            self.run_forward()
            # trigger upload request
            return self.upload("feature_label")

    def load_optimizer(self, conf):
        """Load training optimizer. Implemented Adam and SGD."""
        if conf.optimizer.type == "Adam":
            optimizer = torch.optim.Adam(self.model.parameters(), lr=conf.optimizer.lr)
        else:
            # default using optimizer SGD
            optimizer = torch.optim.SGD(self.model.parameters(),
                                        lr=conf.optimizer.lr,
                                        momentum=conf.optimizer.momentum,
                                        weight_decay=conf.optimizer.weight_decay if hasattr(conf.optimizer, "weight_decay") else 0.005)
        return optimizer

    def scheduler_init(self):
        if self.lr_scheduler is None:
            optimizer = self.load_optimizer(self.conf)
            if self.scheduler == "cos_anneal":
                self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.client_rounds)  # learning rate decay 
            elif self.scheduler == "multi_step":
                milestones = [int(0.3*self.client_rounds), int(0.6*self.client_rounds), int(0.8*self.client_rounds)]
                self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones)  # learning rate decay 
            else:
                raise NotImplementedError

    def scheduler_step(self):
        self.lr_scheduler.step()
        self.conf.optimizer.lr = float(self.lr_scheduler.get_last_lr()[0])
    
    def run_backward(self, external_grad):
        """Client backward computation.
        """
        external_grad = external_grad.to(self.device)
        self.track('train_download_gradient_size', self.calculate_tensor_size(external_grad))
        start_time = time.time()
        optimizer = self.load_optimizer(self.conf)
        optimizer.zero_grad()
        if self.feature is not None:
            self.feature.backward(gradient=external_grad)
            self.feature = None
        optimizer.step()
        
        self.track('client_train_step_time', time.time() - start_time + self.forward_time)

    def post_train(self, conf):
        """Update local client-side model to server.
        """
        if conf.local_test:
            self.test_local()
        self.compression()
        self.track(metric.TRAIN_UPLOAD_SIZE, self.calculate_model_size(self.model))
        return self.upload("model")
    
    def upload(self, content="model"):
        """Upload the messages from client to the server.

        Returns:
            :obj:`UploadRequest`: The upload request defined in protobuf to unify local and remote operations.
                Only applicable for local training as remote training upload through a gRPC request.
        """
        request = self.construct_upload_request(content)
        if not self.is_remote:
            self.post_upload()
            return request

        self.upload_remotely(request)
        self.post_upload()

    def construct_upload_request(self, content="model"):
        """Construct client upload request for feature, label or model updates.

        Returns:
            :obj:`UploadRequest`: The upload request defined in protobuf to unify local and remote operations.
        """
        data = codec.marshal(server_pb.Performance(metric=self.test_metric))
        typ = common_pb.DATA_TYPE_PERFORMANCE

        try:
            if self._is_train:
                data, typ = self.marshal_data(content)
                data_size = self._datasize
            else:
                data_size = 1 if not self.test_data else self.test_data.size(self.cid)
        except KeyError:
            # When the datasize cannot be obtained from dataset, default to use equal aggregate
            data_size = 1

        m = self._tracker.get_client_metric().to_proto() if self._tracker else common_pb.ClientMetric()
        return server_pb.UploadRequest(
            task_id=self.conf.task_id,
            round_id=self.conf.round_id,
            client_id=self.cid,
            content=server_pb.UploadContent(
                data=data,
                type=typ,
                data_size=data_size,
                metric=m,
            ),
        )
    
    def marshal_data(self, content):
        if content == "model":
            data = codec.marshal(copy.deepcopy(self.model))
            typ = common_pb.DATA_TYPE_PARAMS
        elif content == "feature_label":
            customize_dict = {"content": [copy.deepcopy(self.feature.detach().cpu()), copy.deepcopy(self.label)], 
                              "name": ["feature", "label"]}
            data = codec.marshal(customize_dict)
            typ = common_pb.DATA_TYPE_FEATURE # type_
        return data, typ

    def calculate_tensor_size(self, tensor, param_size=32):
        """Calculate size of any tensor.
        Should be overwritten for different training backend.

        Args:
            tensor (options: torch.Tensor): A tensor.
            param_size (int): The size of a parameter, default using float32.

        Returns:
            float: The model size in MB.
        """
        # sum(p.numel() for p in model.parameters() if p.requires_grad) for only trainable parameters
        params = tensor.numel()
        return bit_to_megabyte(params * param_size)

class MocoSFLClient(BaseSFLClient):
    """Implementation of MocoSFL client.

    Args:
        cid (str): Client id.
        conf (omegaconf.dictconfig.DictConfig): Client configurations.
        train_data (:obj:`FederatedDataset`): Unlabeled Training dataset.
        test_data (:obj:`FederatedDataset`): Test dataset.
        device (str): Hardware device for training, cpu or cuda devices.
    """

    def __init__(self,
                 cid,
                 conf,
                 train_data,
                 test_data,
                 device,
                 sleep_time=0,
                 is_remote=False,
                 local_port=23000,
                 server_addr="localhost:22999",
                 tracker_addr="localhost:12666"
                 ):
        super(MocoSFLClient, self).__init__(cid, conf, train_data, test_data, device, sleep_time,
                                           is_remote, local_port, server_addr, tracker_addr)
        
        self.hidden_pkey = None
        self.hidden_query = None

    def pre_train(self, model, conf):
        """Download up-to-date local client-side model from server.
        """
        self.pretrain_setup(conf)
        
        self.conf = conf
        if conf.track:
            self._tracker.set_client_context(conf.task_id, conf.round_id, self.cid)

        self._is_train = True
        
        self.set_model(model)

        # set up scheduler:
        self.scheduler_init()
        
        self.model.train()
        self.model.to(self.device)

        self.t_model = copy.deepcopy(self.model)
        for param_t in self.t_model.parameters():
            param_t.requires_grad = False  # not update by gradient
        
        self.track(metric.TRAIN_DOWNLOAD_SIZE, self.calculate_model_size(model))
        
        self.decompression()
    
    def next_data_batch(self):
        """Get image/label from Client.
        """
        try:
            query, pkey = next(self.data_iterator)
            if query.size(0) != self.bs:
                try:
                    next(self.data_iterator)
                except StopIteration:
                    pass
                self.data_iterator = iter(self.train_loader)
                query, pkey = next(self.data_iterator)
        except StopIteration:
            self.data_iterator = iter(self.train_loader)
            query, pkey = next(self.data_iterator)
        return query, pkey


    def operate(self, model, conf, index, is_train=True):
        """A wrapper over operations (training/testing) on clients.

        Args:
            model (nn.Module): Model for operations.
            conf (omegaconf.dictconfig.DictConfig): Client configurations.
            index (int): Client index in the client list, for retrieving data. TODO: improvement.
            is_train (bool): The flag to indicate whether the operation is training, otherwise testing.
        """
        try:
            # Load the data index depending on server request
            self.cid = self.train_data.users[index]
        except IndexError:
            logger.error("Data index exceed the available data, abort training")
            return

        if self.conf.track and self._tracker is None:
            self._tracker = init_tracking(init_store=False)

        if is_train:
            self.run_backward(model) # model here refers to gradient
            # trigger upload request
            return self.upload("model")
        else:
            self.pre_train(model, conf) # model here refers to client-side model
            self.run_forward()
            # trigger upload request
            return self.upload("query_pkey")

    def run_forward(self):
        """Client forward computation.
        """
        start_time = time.time()
        
        query, pkey = self.next_data_batch()
        query = query.to(self.device)
        pkey = pkey.to(self.device)

        self.hidden_query = self.model(query)# pass to online 
        self.update_moving_average()
        with torch.no_grad():
            self.hidden_pkey = self.t_model(pkey).detach() # pass to momentum 
        
        self.forward_time = time.time() - start_time
        self.track('train_upload_feature_size', 2 * self.calculate_tensor_size(self.hidden_query.detach()))
        
    def run_backward(self, external_grad):
        """Client backward computation.
        """
        external_grad = external_grad.to(self.device)
        self.track('train_download_gradient_size', self.calculate_tensor_size(external_grad))
        start_time = time.time()
        try:
            optimizer = self.load_optimizer(self.conf)
        except ValueError as error:
            logger.error(f"client {self.cid}'s {error}")
            self.track('client_train_step_time', time.time() - start_time + self.forward_time)
            exit()
            return 
        optimizer.zero_grad()
        if self.hidden_query is not None:
            self.hidden_query.backward(gradient=external_grad)
            self.hidden_query = None
        optimizer.step()

        '''MocoSFL adopts an aggressive LR scheduler for clients'''
        if self.scheduler == "cos_anneal":
            self.scheduler_step()
            
        self.track('client_train_step_time', time.time() - start_time + self.forward_time)

    def marshal_data(self, content):
        if content == "model":
            data = codec.marshal(copy.deepcopy(self.model))
            typ = common_pb.DATA_TYPE_PARAMS
        elif content == "query_pkey":
            customize_dict = {"content": [copy.deepcopy(self.hidden_query.detach().cpu()), copy.deepcopy(self.hidden_pkey.detach().cpu())], 
                              "name": ["query", "pkey"]}
            data = codec.marshal(customize_dict)
            typ = common_pb.DATA_TYPE_FEATURE # type_
        return data, typ
    
    @torch.no_grad()
    def update_moving_average(self, tau=0.99):
        for online, target in zip(self.model.parameters(), self.t_model.parameters()):
            target.data = tau * target.data + (1 - tau) * online.data