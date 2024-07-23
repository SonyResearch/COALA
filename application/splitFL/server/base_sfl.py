import logging
import concurrent.futures
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from omegaconf import OmegaConf
from coala.communication import grpc_wrapper
from coala.server.base import BaseServer, AGGREGATION_CONTENT_PARAMS
from coala.server.base import MODEL, DATA_SIZE
from coala.distributed.distributed import CPU
from coala.pb import client_service_pb2 as client_pb
from coala.pb import common_pb2 as common_pb
import time
from coala.protocol import codec
from coala.tracking import metric
from tools import accuracy, AverageMeter
import copy
from tqdm import tqdm
import numpy as np
logger = logging.getLogger(__name__)


class BaseSFLServer(BaseServer):
    """Implementation of split federated learning server.
    Reference: SFLV1 in Thapa, Chandra, et al. "Splitfed: When federated learning meets split learning." 
    https://ojs.aaai.org/index.php/AAAI/article/view/20825

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
        super(BaseSFLServer, self).__init__(conf, test_data, val_data, is_remote, local_port)
        
        self.conf.server.aggregation_content = AGGREGATION_CONTENT_PARAMS
        self.bs = conf.server.clients_per_round * conf.client.batch_size
        self.lr_scheduler = None

        self.scheduler = conf.server.scheduler

    def scheduler_init(self):
        if self.lr_scheduler is None:
            optimizer = self.load_optimizer(self.conf)
            if self.scheduler == "cos_anneal":
                self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.conf.server.rounds)  # learning rate decay 
            elif self.scheduler == "multi_step":
                milestones = [int(0.3*self.conf.server.rounds), int(0.6*self.conf.server.rounds), int(0.8*self.conf.server.rounds)]
                self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones)  # learning rate decay 
            else:
                self.lr_scheduler = None

    def scheduler_step(self):
        self.lr_scheduler.step()
        self.conf.server.optimizer.lr = float(self.lr_scheduler.get_last_lr()[0])

    def set_model(self, model, load_dict=False):
        """Update the universal model in the server.
        Overwrite this method for different training backend, default is for PyTorch.

        Args:
            model (nn.Module): New model.
            load_dict (bool): A flag to indicate whether load state dict or copy the model.
        """
        if load_dict:
            self.model.load_state_dict(model.state_dict())
        else:
            self.model = copy.deepcopy(model)
        # set up scheduler:
        self.scheduler_init()

    def pretrain_setup(self, conf, device):
        """Setup loss function and optimizer before training.
        """
        self.model.train()
        self.model.to(device)
        loss_fn = torch.nn.CrossEntropyLoss()

        try:
            optimizer = self.load_optimizer(conf)
        except ValueError as error:
            logger.error(f"server's {error}")
            exit()
        
        return loss_fn, optimizer
    
    def server_compute(self, loss_fn, stack_feature, stack_label):
        """Server perform loss computation and initialize backward propagation
        """
        stack_feature.requires_grad=True

        stack_feature.retain_grad()

        out = self.model.server[0](stack_feature)
        loss = loss_fn(out, stack_label)

        loss_val = loss.detach().cpu().numpy()
        loss.backward()

        prec1 = accuracy(out.data, stack_label)[0]
        accu_val = prec1.item()

        gradients = stack_feature.grad.detach().clone() # get gradient, the -1 is important, since updates are added to the weights in cpp.

        return loss_val, accu_val, gradients

    def aggregation(self):
        """Aggregate training updates of local copies of client-side model from clients.
        Server aggregates trained models from clients via federated averaging.
        """
        uploaded_content = self.get_client_uploads()
        models = list(uploaded_content[MODEL].values())
        weights = list(uploaded_content[DATA_SIZE].values())
        
        model = self.aggregate(models, weights)
        
        self.model.client_cloud_copy[0].load_state_dict(model.state_dict())
    

    def train(self):
        """Training process of federated learning."""
        self.print_("--- start training ---")

        self.selection(self.clients, self.conf.server.clients_per_round)
        self.grouping_for_distributed()
        self.compression()

        begin_train_time = time.time()

        self.distribution_to_train()
        # self.aggregation() # delete intentionally - we move aggregation in distribution_to_train()

        train_time = time.time() - begin_train_time
        self.print_("Server train time: {:.2f}s".format(train_time))
        self.track(metric.TRAIN_TIME, train_time)
        self.tracking_visualization({metric.TRAIN_TIME: train_time})

    def pack_customize_content_train(self, name1="content1", name2="content2"):
        value_idx_list = [0] # [0, 100, 200, ...]

        customize_content1 = self.client_uploads[name1]
        customize_content2 = self.client_uploads[name2]

        for value in customize_content1.values():
            value_idx_list.append(value.size(0) + value_idx_list[-1])
        
        # pack dict to a torch.array, use gradient_idx_list to record the original index
        stack_content1 = torch.cat(list(customize_content1.values()), dim = 0)
        stack_content1 = stack_content1.to(self.conf.device)
        
        stack_content2 = torch.cat(list(customize_content2.values()), dim = 0)
        stack_content2 = stack_content2.to(self.conf.device)
        
        return stack_content1, stack_content2, value_idx_list

    def distribution_to_train_locally(self):
        """Conduct training sequentially for selected clients in the group."""
        uploaded_models = {}
        uploaded_weights = {}
        uploaded_metrics = []
        
        uploaded_feature_list = [None for _ in range(len(self.grouped_clients))]
        uploaded_label_list = [None for _ in range(len(self.grouped_clients))]
        
        # get server optimizer, loss_fn
        loss_fn, optimizer = self.pretrain_setup(self.conf, self.conf.device)

        # Update client config before training
        self.conf.client.task_id = self.conf.task_id
        self.conf.client.round_id = self.current_round

        # Get number of steps
        num_step = 0
        for cid, client in enumerate(self.grouped_clients):
            client.pre_train(self.model.client_cloud_copy[0], self.conf.client)
            client_num_step = round(client.train_data.size(client.cid) / client.bs) if client.train_data is not None else 0
            num_step = client_num_step if client_num_step > num_step else num_step
        num_step *= self.conf.client.local_epoch

        training_loss = AverageMeter()
        training_accu = AverageMeter()
        
        # start current round
        for step in range(num_step):
            
            gradient_idx_list = [0] # [0, 100, 200, ...]
            
            for cid, client in enumerate(self.grouped_clients):
                
                # client_forward
                client.run_forward()
                
                # client upload feature and label
                uploaded_request = client.upload("feature_label")
                
                # server gather feature and label
                uploaded_content = uploaded_request.content
                feature_label = self.decompression(codec.unmarshal(uploaded_content.data))
                uploaded_feature_list[cid] = feature_label["content"][0]
                uploaded_label_list[cid] = feature_label["content"][1]
                
                gradient_idx_list.append(feature_label["content"][0].size(0) + gradient_idx_list[-1])
            
            # aggregate client feature and label
            stack_feature = torch.cat(uploaded_feature_list, dim = 0)
            stack_feature = stack_feature.to(self.conf.device)
            
            stack_label = torch.cat(uploaded_label_list, dim = 0)
            stack_label = stack_label.to(self.conf.device)

            # perform server computation
            optimizer.zero_grad()
            loss_val, accu_val, gradients = self.server_compute(loss_fn, stack_feature, stack_label)
            optimizer.step()

            training_loss.update(loss_val, stack_feature.size(0))
            training_accu.update(accu_val, stack_feature.size(0))
            
            # return gradient back to clients, and clients perform backward
            for cid, client in enumerate(self.grouped_clients):
                gradient = gradients[gradient_idx_list[cid]:gradient_idx_list[cid+1], :]
                client.run_backward(gradient)
        
        
            # Client-side Model Aggregation
            if step == num_step - 1 or (step % (num_step//self.conf.server.aggregation_freq) == (num_step//self.conf.server.aggregation_freq) - 1):
                
                self.aggregate_locally(uploaded_models, uploaded_weights, uploaded_metrics)

                # client get newest model
                if step != num_step - 1:
                    for client in self.grouped_clients:
                        client.pre_train(self.model.client_cloud_copy[0], self.conf.client)
        
        # print average training loss of this round
        self.print_(f"Train Loss: {training_loss.avg:.2f}, Train Accuracy: {training_accu.avg:.2f}%")
        # step scheduler
        self.scheduler_step()
        for client in self.grouped_clients:
            client.scheduler_step()


    def aggregate_locally(self, uploaded_models, uploaded_weights, uploaded_metrics):
        for client in self.grouped_clients:
            uploaded_request = client.post_train(self.conf.client)
            uploaded_content = uploaded_request.content

            model = self.decompression(codec.unmarshal(uploaded_content.data))
            uploaded_models[client.cid] = model
            uploaded_weights[client.cid] = uploaded_content.data_size
            uploaded_metrics.append(metric.ClientMetric.from_proto(uploaded_content.metric))
        
        self.set_client_uploads_train(uploaded_models, uploaded_weights, uploaded_metrics)

        self.aggregation()
    
    def _distribution_remotely(self, cid, request):
        """Distribute request to the assigned client to conduct operations.

        Args:
            cid (str): Client id.
            request (:obj:`OperateRequest`): gRPC request of specific operations.
        """
        resp = self.client_stubs[cid].Operate(request)
        # mute print

    def client_backward_remotely(self, gradient_dict):
        start_time = time.time()
        should_track = self.tracker is not None and self.conf.client.track
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for i, client in enumerate(self.grouped_clients):
                request = client_pb.OperateRequest(
                    type=client_pb.OP_TYPE_TRAIN,  # set OP_TYPE_TRAIN to trigger client's forward computation service
                    model=codec.marshal(gradient_dict[i]),
                    data_index=client.index,
                    config=client_pb.OperateConfig(
                        batch_size=self.conf.client.batch_size,
                        local_epoch=self.conf.client.local_epoch,
                        seed=self.conf.seed,
                        local_test=self.conf.client.local_test,
                        optimizer=client_pb.Optimizer(
                            type=self.conf.client.optimizer.type,
                            lr=self.conf.client.optimizer.lr,
                            momentum=self.conf.client.optimizer.momentum,
                        ),
                        task_id=self.conf.task_id,
                        round_id=self.current_round,
                        track=should_track,
                    ),
                )
                executor.submit(self._distribution_remotely, client.client_id, request)

            distribute_time = time.time() - start_time
            self.track("train_distribute_time_client_backward", distribute_time)
            # mute
            # logger.info("Distribute to clients, time: {}".format(distribute_time))
        with self._condition:
            self._condition.wait()

    def client_forward_remotely(self, model):
        start_time = time.time()
        should_track = self.tracker is not None and self.conf.client.track
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for client in self.grouped_clients:
                request = client_pb.OperateRequest(
                    type=client_pb.OP_TYPE_TEST, # set OP_TYPE_TEST to trigger client's forward computation service
                    model=codec.marshal(model),
                    data_index=client.index,
                    config=client_pb.OperateConfig(
                        batch_size=self.conf.client.batch_size,
                        local_epoch=self.conf.client.local_epoch,
                        seed=self.conf.seed,
                        local_test=self.conf.client.local_test,
                        optimizer=client_pb.Optimizer(
                            type=self.conf.client.optimizer.type,
                            lr=self.conf.client.optimizer.lr,
                            momentum=self.conf.client.optimizer.momentum,
                        ),
                        task_id=self.conf.task_id,
                        round_id=self.current_round,
                        track=should_track,
                    ),
                )
                executor.submit(self._distribution_remotely, client.client_id, request)

            distribute_time = time.time() - start_time
            self.track("train_distribute_time_client_forward", distribute_time)
            # mute
            # logger.info("Distribute to clients, time: {}".format(distribute_time))
        with self._condition:
            self._condition.wait()



    def distribution_to_train_remotely(self):
        """Distribute training requests to remote clients through multiple threads.
        The main thread waits for signal to proceed. The signal can be triggered via notification, as below example.

        Example to trigger signal:
            >>> with self.condition():
            >>>     self.notify_all()
        """
        # get server optimizer, loss_fn
        loss_fn, optimizer = self.pretrain_setup(self.conf, self.conf.device) #TODO: this can be questionable for remote
        
        # Update client config before training
        self.conf.client.task_id = self.conf.task_id
        self.conf.client.round_id = self.current_round

        # Get maximum number of steps among clients, or set to a fixed value
        num_step = 50

        training_loss = AverageMeter()
        training_accu = AverageMeter()
        
        # start current round
        for step in range(num_step):
            
            # break down to four steps: [client_forward_remotely, server_compute, client_backward_remotely, aggregation]
            self.client_forward_remotely(self.model.client_cloud_copy[0])
            stack_feature, stack_label, gradient_idx_list = self.pack_customize_content_train("feature", "label")

            # perform server computation
            optimizer.zero_grad()
            loss_val, accu_val, gradients = self.server_compute(loss_fn, stack_feature, stack_label)
            optimizer.step()

            training_loss.update(loss_val, stack_feature.size(0))
            training_accu.update(accu_val, stack_feature.size(0))

            # distribute gradients according to client_id_list and gradient_idx_list
            # return gradient back to clients, and clients perform backward
            gradient_dict = {}
            for i in range(len(self.grouped_clients)):
                gradient_dict[i] = gradients[gradient_idx_list[i]:gradient_idx_list[i+1], :]

            self.client_backward_remotely(gradient_dict)

            # Client-side Model Aggregation
            if step == num_step - 1 or (step % (num_step//self.conf.server.aggregation_freq) == (num_step//self.conf.server.aggregation_freq) - 1):
                self.aggregation()
        
        # print average training loss of this round
        self.print_(f"Train Loss: {training_loss.avg:.2f}, Train Accuracy: {training_accu.avg:.2f}%")
        
        # step scheduler
        self.scheduler_step()
        self.conf.client.optimizer.lr = self.conf.server.optimizer.lr


    
    def load_optimizer(self, conf):
        """Load training optimizer. Implemented Adam and SGD."""
        if conf.server.optimizer.type == "Adam":
            optimizer = torch.optim.Adam(self.model.server[0].parameters(), lr=conf.server.optimizer.lr)
        else:
            # default using optimizer SGD
            optimizer = torch.optim.SGD(self.model.server[0].parameters(),
                                        lr=conf.server.optimizer.lr,
                                        momentum=conf.server.optimizer.momentum,
                                        weight_decay=conf.server.optimizer.weight_decay)
        return optimizer
    



class MocoSFLServer(BaseSFLServer):
    """Implementation of MOCOSFL server.
    The code is adapted from https://colab.research.google.com/github/facebookresearch/moco/blob/colab-notebook/colab/moco_cifar10_demo.ipynb
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
        super(MocoSFLServer, self).__init__(conf, test_data, None, is_remote, local_port)
        
        self.t_model = None
        self.MLP = None
        self.t_MLP = None
        
        self.moco_version = conf.moco.version
        self.K = conf.moco.K
        self.T = conf.moco.T
        self.K_dim = conf.moco.K_dim
        self.symmetric = conf.moco.symmetric

        self.eval_data = val_data

        self.queue = torch.randn(self.K_dim, self.K).to(self.conf.device)
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.queue_ptr = torch.zeros(1, dtype=torch.long)

    def set_model(self, model, load_dict=False):
        """Update the universal model in the server.
        Overwrite this method for different training backend, default is for PyTorch.

        Args:
            model (nn.Module): New model.
            load_dict (bool): A flag to indicate whether load state dict or copy the model.
        """
        if load_dict:
            self.model.load_state_dict(model.state_dict())
        else:
            self.model = copy.deepcopy(model)
        
        # chop the original classifier
        output_dim = self.chop_classifier(self.model.view_identifier)

        # create a new MLP for self-supervised learning
        if self.moco_version == "mocov2": # This one uses a larger classifier, same as in Zhuang et al. Divergence-aware paper
            if output_dim is None:
                MLP_list = [nn.LazyLinear(self.K_dim),
                                nn.ReLU(True),
                                nn.Linear(self.K_dim, self.K_dim)]
            else:
                MLP_list = [nn.Linear(output_dim, self.K_dim),
                                nn.ReLU(True),
                                nn.Linear(self.K_dim, self.K_dim)]
        else:
            raise("Unknown version! Please specify the classifier.")
        self.MLP = nn.Sequential(*MLP_list)

        # create momentum backbone model
        self.t_model = copy.deepcopy(self.model.server[0])
        for param_t in self.t_model.parameters():
            param_t.requires_grad = False  # not update by gradient

        # create momentum MLP model
        self.t_MLP = copy.deepcopy(self.MLP)
        for param_t in self.t_MLP.parameters():
            param_t.requires_grad = False  # not update by gradient

        # merge momentum model forever (as we do not need to unmerge it any more)
        self.t_model = nn.Sequential(*(list(self.t_model.children()) + list(self.t_MLP.children())))
        
        # set up scheduler:
        self.scheduler_init()
    
    def chop_classifier(self, view_identifier = "ViewLayer"):
        ''' Chop classifier (MLP layers) to create backbone for self-supervised learning
        '''
        # chop it from the server model
        stop_add_flag = False
        new_server_list = []
        for module in list(self.model.server[0].children()):
            if view_identifier in str(module):
                new_server_list.append(module)
                stop_add_flag = True
            
            if stop_add_flag:
                break
            else:
                new_server_list.append(module)
        self.model.server[0] = nn.Sequential(*new_server_list)
        
        stop_add_flag = False
        
        # also chop it from the main model
        new_model_list = []
        for module in list(self.model.model.children()):
            if view_identifier in str(module):
                new_model_list.append(module)
                stop_add_flag = True
                
            
            if stop_add_flag:
                break
            else:
                new_model_list.append(module)
        self.model.model = nn.Sequential(*new_model_list)

        if not hasattr(self.model, "backbone_output_dim"):
            return None
        
        return self.model.backbone_output_dim

    def merge_backbone_MLP(self):
        # merge self.model.server[0] with self.MLP as the new self.model.server[0]
        self.original_model = nn.Sequential(*(list(self.model.server[0].children())))
        self.model.server[0] = nn.Sequential(*(list(self.model.server[0].children()) + list(self.MLP.children())))
    
    def unmerge_backbone_MLP(self):
        # disconnect with the MLP
        self.model.server[0] = self.original_model

    def pre_test(self):
        self.unmerge_backbone_MLP()
    
    def server_compute(self, loss_fn, stack_query, stack_pkey, update_momentum = True, enqueue = True, tau = 0.99):
        """Server perform loss computation and initialize backward propagation
        """
        stack_query.requires_grad=True

        stack_query.retain_grad()

        # update moving average
        if update_momentum:
            self.update_moving_average(tau)
        
        # symmetric contrastive loss
        if self.symmetric:
            loss12, accu, q1, k2 = self.contrastive_loss(loss_fn, stack_query, stack_pkey)
            loss21, accu, q2, k1 = self.contrastive_loss(loss_fn, stack_pkey, stack_query)
            loss = loss12 + loss21
            pkey_out = torch.cat([k1, k2], dim = 0)
        else:
            loss, accu, query_out, pkey_out = self.contrastive_loss(loss_fn, stack_query, stack_pkey)

        if enqueue:
            self._dequeue_and_enqueue(pkey_out)

        loss_val = loss.detach().cpu().numpy()
        accu_val = accu[0]
        loss.backward()

        gradients = stack_query.grad.detach().clone()

        return loss_val, accu_val, gradients

    def distribution_to_train_locally(self):
        """Conduct training sequentially for selected clients in the group."""
        uploaded_models = {}
        uploaded_weights = {}
        uploaded_metrics = []
        
        uploaded_query_list = [None for _ in range(len(self.grouped_clients))]
        uploaded_pkey_list = [None for _ in range(len(self.grouped_clients))]
        
        # get server optimizer, loss_fn
        self.t_model.train()
        self.t_model.to(self.conf.device)
        self.merge_backbone_MLP()
        loss_fn, optimizer = self.pretrain_setup(self.conf, self.conf.device)

        # Update client config before training
        self.conf.client.task_id = self.conf.task_id
        self.conf.client.round_id = self.current_round

        # Get number of steps
        num_step = 0
        num_steps = []
        for cid, client in enumerate(self.grouped_clients):
            client.pre_train(self.model.client_cloud_copy[0], self.conf.client)
            client_num_step = round(client.train_data.size(client.cid) / client.bs) if client.train_data is not None else 0
            # num_step = client_num_step if client_num_step > num_step else num_step
            num_steps.append(client_num_step)
        # num_step *= self.conf.client.local_epoch
        num_step = int(np.mean(num_steps)) * self.conf.client.local_epoch
        print(num_step)

        training_loss = AverageMeter()
        training_accu = AverageMeter()
        
        # start current round
        for step in range(num_step):
            
            gradient_idx_list = [0] # [0, 100, 200, ...]
            
            for cid, client in enumerate(self.grouped_clients):
                
                # client_forward
                client.run_forward()
                
                # client upload feature and label
                uploaded_request = client.upload("query_pkey")
                
                # server gather feature and label
                uploaded_content = uploaded_request.content
                query_pkey = self.decompression(codec.unmarshal(uploaded_content.data))
                uploaded_query_list[cid] = query_pkey["content"][0]
                uploaded_pkey_list[cid] = query_pkey["content"][1]
                
                gradient_idx_list.append(query_pkey["content"][0].size(0) + gradient_idx_list[-1])
            
            # aggregate client feature and label
            stack_query = torch.cat(uploaded_query_list, dim = 0)
            stack_query = stack_query.to(self.conf.device)
            
            stack_pkey = torch.cat(uploaded_pkey_list, dim = 0)
            stack_pkey = stack_pkey.to(self.conf.device)

            # perform server computation
            optimizer.zero_grad()
            loss_val, accu_val, gradients = self.server_compute(loss_fn, stack_query, stack_pkey)
            optimizer.step()

            training_loss.update(loss_val, stack_query.size(0))
            training_accu.update(accu_val, stack_query.size(0))
            
            # return gradient back to clients, and clients perform backward
            for cid, client in enumerate(self.grouped_clients):
                gradient = gradients[gradient_idx_list[cid]:gradient_idx_list[cid+1], :]
                client.run_backward(gradient)
        
        
            # Client-side Model Aggregation
            if step == num_step - 1 or (step % (num_step//self.conf.server.aggregation_freq) == (num_step//self.conf.server.aggregation_freq) - 1):
                
                self.aggregate_locally(uploaded_models, uploaded_weights, uploaded_metrics)

                # client get newest model
                if step != num_step - 1:
                    for client in self.grouped_clients:
                        client.pre_train(self.model.client_cloud_copy[0], self.conf.client)
        
        # print average training loss of this round
        self.print_(f"Train Loss: {training_loss.avg:.2f}, Train Accuracy: {training_accu.avg:.2f}%")
        # step scheduler
        self.scheduler_step()
        if self.conf.client.scheduler != "cos_anneal": # if client's schduler is cos_anneal, we skip this. instead, the LR scheduelring is done in client.run_backward()
            for client in self.grouped_clients:
                client.scheduler_step()


    def distribution_to_train_remotely(self):
        """Distribute training requests to remote clients through multiple threads.
        The main thread waits for signal to proceed. The signal can be triggered via notification, as below example.

        Example to trigger signal:
            >>> with self.condition():
            >>>     self.notify_all()
        """
        # get server optimizer, loss_fn
        self.t_model.train()
        self.t_model.to(self.conf.device)
        self.merge_backbone_MLP()

        loss_fn, optimizer = self.pretrain_setup(self.conf, self.conf.device) #TODO: this can be questionable for remote
        
        # Update client config before training
        self.conf.client.task_id = self.conf.task_id
        self.conf.client.round_id = self.current_round

        # Get maximum number of steps among clients, or set to a fixed value
        num_step = 50

        training_loss = AverageMeter()
        training_accu = AverageMeter()
        
        # start current round
        for step in range(num_step):
            
            # break down to four steps: [client_forward_remotely, server_compute, client_backward_remotely, aggregation]
            self.client_forward_remotely(self.model.client_cloud_copy[0])
            stack_query, stack_pkey, gradient_idx_list = self.pack_customize_content_train("query", "pkey")

            # perform server computation
            optimizer.zero_grad()
            loss_val, accu_val, gradients = self.server_compute(loss_fn, stack_query, stack_pkey)
            optimizer.step()

            training_loss.update(loss_val, stack_query.size(0))
            training_accu.update(accu_val, stack_query.size(0))

            # distribute gradients according to client_id_list and gradient_idx_list
            # return gradient back to clients, and clients perform backward
            gradient_dict = {}
            for i in range(len(self.grouped_clients)):
                gradient_dict[i] = gradients[gradient_idx_list[i]:gradient_idx_list[i+1], :]

            self.client_backward_remotely(gradient_dict)

            # Client-side Model Aggregation
            if step == num_step - 1 or (step % (num_step//self.conf.server.aggregation_freq) == (num_step//self.conf.server.aggregation_freq) - 1):
                self.aggregation()
        
        # print average training loss of this round
        self.print_(f"Train Loss: {training_loss.avg:.2f}, Train Accuracy: {training_accu.avg:.2f}%")
        
        # step scheduler
        self.scheduler_step()
        if self.conf.client.scheduler != "cos_anneal": # if client's schduler is cos_anneal, we skip this. instead, the LR scheduelring is done in client.run_backward()
            self.conf.client.optimizer.lr = self.conf.server.optimizer.lr

    def test(self):
        """Testing process of federated learning."""
        self.print_("--- start testing ---")
        if self.is_primary_server():
            test_begin_time = time.time()
            test_results = {metric.TEST_TIME: 0}
            test_results = self.test_in_server(self.conf.device)
            test_results[metric.TEST_TIME] = time.time() - test_begin_time
            self.track_test_results(test_results)
            self.tracking_visualization(test_results)

    def test_in_server(self, device=CPU, comprehensive_eval = False):
        """Conduct testing in the server.
        Overwrite this method for different training backend, default is PyTorch.
        
        Args:
            device (str): The hardware device to conduct testing, either cpu or cuda devices.

        Returns:
            dict: Test metrics, {"test_metric": dict, "test_time": value}.
        """
        self.num_class = self.eval_data.num_class
        self.model.eval()
        self.model.to(device)

        loss_fn = torch.nn.CrossEntropyLoss().to(device)
        train_loader = self.eval_data.loader(self.conf.server.batch_size, seed=self.conf.seed)
        test_loader = self.test_data.loader(self.conf.server.batch_size, seed=self.conf.seed)
        
        # knn evaluation
        knn_accuracy = self.knn_eval(train_loader, test_loader)
        logger.info('Server, knn_eval testing -- Accuracy: ({:.2f}%)'.format(knn_accuracy))
        test_results = {metric.TEST_METRIC: {"accuracy (knn)": knn_accuracy, "loss": knn_accuracy}}

        if self.current_round == self.conf.server.rounds - 1:
            comprehensive_eval = True

        if comprehensive_eval:
            output_dim = self.model.backbone_output_dim if hasattr(self.model, "backbone_output_dim") else None
            
            # linear probe evaluation
            linear_accuracy = self.linear_eval(train_loader, test_loader, loss_fn, output_dim)
            logger.info('Server, linear_accuracy testing -- Accuracy: ({:.2f}%)'.format(linear_accuracy))
            
            # semi-supervised evaluation
            
            self.eval_data.data['x'] = self.eval_data.data['x'][:self.eval_data.data['x'].shape[0]//10, :, :, :]# 5000, 32, 32, 3
            self.eval_data.data['y'] = self.eval_data.data['y'][:self.eval_data.data['y'].shape[0]//10] # 5000, 1
            train_loader_10percent = self.eval_data.loader(self.conf.server.batch_size, seed=self.conf.seed)
            print(f"loader length of train_loader_10percent is {len(train_loader_10percent)}")
            semi_10percent_accuracy = self.semisupervise_eval(train_loader_10percent, test_loader, loss_fn, output_dim)
            logger.info('Server, semi_supervised_10percent_accuracy testing -- Accuracy: ({:.2f}%)'.format(semi_10percent_accuracy))
            
            self.eval_data.data['x'] = self.eval_data.data['x'][:self.eval_data.data['x'].shape[0]//10, :, :, :]# 500, 32, 32, 3
            self.eval_data.data['y'] = self.eval_data.data['y'][:self.eval_data.data['y'].shape[0]//10] # 500, 1
            train_loader_1percent = self.eval_data.loader(self.conf.server.batch_size, seed=self.conf.seed)
            print(f"loader length of train_loader_1percent is {len(train_loader_1percent)}")
            semi_1percent_accuracy = self.semisupervise_eval(train_loader_1percent, test_loader, loss_fn, output_dim)
            logger.info('Server, semi_supervised_1percent_accuracy testing -- Accuracy: ({:.2f}%)'.format(semi_1percent_accuracy))
            
            test_results = {metric.TEST_METRIC: {"accuracy (knn)": knn_accuracy, "loss": knn_accuracy, "accuracy (linear)": linear_accuracy, "accuracy (semi-1%)": semi_1percent_accuracy, "accuracy (semi-10%)": semi_10percent_accuracy}}
        
        return test_results


    def track_test_results(self, results):
        """Track test results collected from clients.

        Args:
            results (dict): Test metrics, format in {"test_metric": dict, "test_time": value}
        """
        self.cumulative_times.append(time.time() - self._start_time)
        test_metrics = results[metric.TEST_METRIC]
        for key, value in test_metrics.items():
            if key in self.performance_metrics:
                self.performance_metrics[key].append(value)
            else:
                self.performance_metrics[key] = [value]

        for metric_name in results:
            self.track(metric_name, results[metric_name])

        test_metric_content = ''.join(
            [", Test {}: {:.2f}%".format(key, value) for key, value in test_metrics.items() if key != 'loss'])
        self.print_('Test time {:.2f}s'.format(results[metric.TEST_TIME]) + test_metric_content)

    def knn_eval(self, train_loader, test_loader): # Use linear evaluation
        # test using a knn monitor
        def test():
            classes = self.num_class
            total_top1, total_top5, total_num, feature_bank, feature_labels = 0.0, 0.0, 0, [], []
            with torch.no_grad():
                # generate feature bank
                for data, target in train_loader:
                    feature = self.model(data.to(self.conf.device))
                    feature = F.normalize(feature, dim=1)
                    feature_bank.append(feature)
                    feature_labels.append(target)
                # [D, N]
                feature_bank = torch.cat(feature_bank, dim=0).t().contiguous().to(self.conf.device)
                # [N]
                feature_labels = torch.cat(feature_labels, dim=0).contiguous().to(self.conf.device)
                # feature_labels = torch.tensor(memory_data_loader.dataset.targets, device=feature_bank.device)
                # loop test data to predict the label by weighted knn search
                for data, target in test_loader:
                    data, target = data.to(self.conf.device), target.to(self.conf.device)
                    feature = self.model(data)
                    feature = F.normalize(feature, dim=1)
                    
                    pred_labels = knn_predict(feature, feature_bank, feature_labels, classes, 200, 0.1)

                    total_num += data.size(0)
                    total_top1 += (pred_labels[:, 0] == target).float().sum().item()

            return total_top1 / total_num * 100

        # knn monitor as in InstDisc https://arxiv.org/abs/1805.01978
        # implementation follows http://github.com/zhirongw/lemniscate.pytorch and https://github.com/leftthomas/SimCLR
        def knn_predict(feature, feature_bank, feature_labels, classes, knn_k, knn_t):
            # compute cos similarity between each feature vector and feature bank ---> [B, N]
            sim_matrix = torch.mm(feature, feature_bank)
            # [B, K]
            sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
            # [B, K]
            sim_labels = torch.gather(feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices)
            sim_weight = (sim_weight / knn_t).exp()

            # counts for each class
            one_hot_label = torch.zeros(feature.size(0) * knn_k, classes, device=sim_labels.device)
            # [B*K, C]
            one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
            # weighted score ---> [B, C]
            pred_scores = torch.sum(one_hot_label.view(feature.size(0), -1, classes) * sim_weight.unsqueeze(dim=-1), dim=1)

            pred_labels = pred_scores.argsort(dim=-1, descending=True)
            return pred_labels
        test_acc_1 = test()
        return test_acc_1
    
    def linear_eval(self, train_loader, test_loader, loss_fn, output_dim, num_epochs = 100, lr = 3.0): # Use linear evaluation
        """
        Run Linear evaluation
        """
        if output_dim is None:
            classifier_list = [nn.LazyLinear(self.num_class)]
        else:
            classifier_list = [nn.Linear(output_dim, self.num_class)]
        linear_classifier = nn.Sequential(*classifier_list)
        # def init_weights(m):
        #     nn.init.xavier_uniform_(m.weight, gain=1.0)
        #     m.bias.data.zero_()
        
        # linear_classifier.apply(init_weights)

        # linear_optimizer = torch.optim.SGD(list(linear_classifier.parameters()), lr=lr, momentum=0.9, weight_decay=1e-4)
        linear_optimizer = torch.optim.Adam(list(linear_classifier.parameters()))
        linear_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(linear_optimizer, num_epochs//4)  # learning rate decay 

        linear_classifier.to(self.conf.device)
        best_avg_accu = 0.0
        # Train the linear layer
        for epoch in tqdm(range(num_epochs)):
            linear_classifier.train()
            for input, label in train_loader:
                linear_optimizer.zero_grad()
                input = input.to(self.conf.device)
                label = label.to(self.conf.device)
                with torch.no_grad():
                    output = self.model(input)
                output = linear_classifier(output.detach())
                loss = loss_fn(output, label)
                loss.backward()
                linear_optimizer.step()
                linear_scheduler.step()
            
            """
            Run validation
            """
            top1 = AverageMeter()
            
            linear_classifier.eval()

            for input, target in test_loader:
                input = input.to(self.conf.device)
                target = target.to(self.conf.device)
                with torch.no_grad():
                    output = self.model(input)
                    output = linear_classifier(output.detach())
                prec1 = accuracy(output.data, target)[0]
                top1.update(prec1.item(), input.size(0))
            
            avg_accu = top1.avg
            if avg_accu > best_avg_accu:
                best_avg_accu = avg_accu
        return best_avg_accu

    def semisupervise_eval(self, partial_train_loader, test_loader, loss_fn, output_dim, num_epochs = 100, lr = 3.0): # Use semi-supervised learning as evaluation
        """
        Run Semisupervised evaluation
        """
        if output_dim is None:
            classifier_list = [nn.LazyLinear(512),
                                nn.BatchNorm1d(512),
                                nn.ReLU(True),
                                nn.Linear(512, self.num_class)]
        else:
            classifier_list = [nn.Linear(output_dim, 512),
                                nn.BatchNorm1d(512),
                                nn.ReLU(True),
                                nn.Linear(512, self.num_class)]
        semi_classifier = nn.Sequential(*classifier_list)
        # def init_weights(m):
        #     nn.init.xavier_uniform_(m.weight, gain=1.0)
        #     m.bias.data.zero_()
        # semi_classifier.apply(init_weights)

        linear_optimizer = torch.optim.Adam(list(semi_classifier.parameters()), lr=1e-3) # as in divergence-aware
        milestones = [int(0.3*num_epochs), int(0.6*num_epochs), int(0.8*num_epochs)]
        linear_scheduler = torch.optim.lr_scheduler.MultiStepLR(linear_optimizer, milestones=milestones, gamma=0.1)  # learning rate decay 

        semi_classifier.to(self.conf.device)
        best_avg_accu = 0.0
        # Train the linear layer
        for epoch in tqdm(range(num_epochs)):
            semi_classifier.train()
            for input, label in partial_train_loader:
                linear_optimizer.zero_grad()
                input = input.to(self.conf.device)
                label = label.to(self.conf.device)
                with torch.no_grad():
                    output = self.model(input)
                    output = output.view(output.size(0), -1)
                output = semi_classifier(output.detach())
                loss = loss_fn(output, label)
                loss.backward()
                linear_optimizer.step()
                linear_scheduler.step()
            
            """
            Run validation
            """
            top1 = AverageMeter()
            
            semi_classifier.eval()

            for input, target in test_loader:
                input = input.to(self.conf.device)
                target = target.to(self.conf.device)
                with torch.no_grad():
                    output = self.model(input)
                    output = semi_classifier(output.detach())

                prec1 = accuracy(output.data, target)[0]
                top1.update(prec1.item(), input.size(0))
            
            avg_accu = top1.avg
            if avg_accu > best_avg_accu:
                best_avg_accu = avg_accu
        return best_avg_accu

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        
        # replace the keys at ptr (dequeue and enqueue)
        if (ptr + batch_size) <= self.K:
            self.queue[:, ptr:ptr + batch_size] = keys.T
        else:
            self.queue[:, ptr:] = keys.T[:, :self.K - ptr]
            self.queue[:, 0:(batch_size + ptr - self.K)] = keys.T[:, self.K - ptr:]
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def update_moving_average(self, tau = 0.99):
        for online, target in zip(self.model.server[0].parameters(), self.t_model.parameters()):
            target.data = tau * target.data + (1 - tau) * online.data
    
    @torch.no_grad()
    def _batch_shuffle_single_gpu(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        """
        # random shuffle index
        idx_shuffle = torch.randperm(x.shape[0]).to(self.conf.device)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        return x[idx_shuffle], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_single_gpu(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        """
        return x[idx_unshuffle]

    def contrastive_loss(self, loss_fn, query, pkey):
        query_out = self.model.server[0](query)

        query_out = nn.functional.normalize(query_out, dim = 1)

        with torch.no_grad():  # no gradient to keys

            pkey_, idx_unshuffle = self._batch_shuffle_single_gpu(pkey)

            pkey_out = self.t_model(pkey_)

            pkey_out = nn.functional.normalize(pkey_out, dim = 1).detach()

            pkey_out = self._batch_unshuffle_single_gpu(pkey_out, idx_unshuffle)

        l_pos = torch.einsum('nc,nc->n', [query_out, pkey_out]).unsqueeze(-1)
        
        l_neg = torch.einsum('nc,ck->nk', [query_out, self.queue.clone().detach()])


        logits = torch.cat([l_pos, l_neg], dim=1)

        logits /= self.T
        
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.conf.device)

        loss = loss_fn(logits, labels)

        accu = accuracy(logits, labels)

        return loss, accu, query_out, pkey_out