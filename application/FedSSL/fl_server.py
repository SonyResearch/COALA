import copy
import logging
import os

import torch
import torch.distributed as dist
from torchvision import datasets
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import ssl_model as model
import utils
from coala.datasets.data import CIFAR100
from coala.distributed import reduce_models
from coala.distributed.distributed import CPU
from coala.server import strategies
from coala.server.base import BaseServer, MODEL, DATA_SIZE
from coala.tracking import metric

logger = logging.getLogger(__name__)
TARGET = "target"


class FedSSLServer(BaseServer):
    def __init__(self, conf, test_data=None, val_data=None, is_remote=False, local_port=22999):
        super(FedSSLServer, self).__init__(conf, test_data, val_data, is_remote, local_port)
        self.train_loader = None
        self.test_loader = None

    def aggregation(self):
        if self.conf.client.auto_scaler == 'y' and self.conf.server.random_selection:
            self._retain_weight_scaler()

        uploaded_content = self.get_client_uploads()
        models = list(uploaded_content[MODEL].values())
        weights = list(uploaded_content[DATA_SIZE].values())

        # Aggregate networks gradually with different components.
        if self.conf.model in [model.SimSiam, model.BYOL, model.SimCLR]:
            online_encoders = [m.online_encoder for m in models]
            online_encoder = self._federated_averaging(online_encoders, weights)
            self.model.online_encoder.load_state_dict(online_encoder.state_dict())

        if self.conf.model in [model.SimSiam, model.BYOL]:
            predictors = [m.online_predictor for m in models]
            predictor = self._federated_averaging(predictors, weights)
            self.model.online_predictor.load_state_dict(predictor.state_dict())

        if self.conf.model in [model.BYOL]:
            target_encoders = [m.target_encoder for m in models]
            target_encoder = self._federated_averaging(target_encoders, weights)
            self.model.target_encoder = copy.deepcopy(target_encoder)

        if self.conf.model in [model.MoCo, model.MoCoV2]:
            encoder_qs = [m.encoder_q for m in models]
            encoder_q = self._federated_averaging(encoder_qs, weights)
            self.model.encoder_q.load_state_dict(encoder_q.state_dict())

            encoder_ks = [m.encoder_k for m in models]
            encoder_k = self._federated_averaging(encoder_ks, weights)
            self.model.encoder_k.load_state_dict(encoder_k.state_dict())

    def _retain_weight_scaler(self):
        self.client_id_to_index = {c.cid: i for i, c in enumerate(self.clients)}

        client_index = self.client_id_to_index[self.grouped_clients[0].cid]
        weight_scaler = self.grouped_clients[0].weight_scaler if self.grouped_clients[0].weight_scaler else 0
        scaler = torch.tensor((client_index, weight_scaler)).to(self.conf.device)
        scalers = [torch.zeros_like(scaler) for _ in self.selected_clients]
        dist.barrier()
        dist.all_gather(scalers, scaler)

        logger.info(f"Synced scaler {scalers}")
        for i, client in enumerate(self.clients):
            for scaler in scalers:
                scaler = scaler.cpu().numpy()
                if self.client_id_to_index[client.cid] == int(scaler[0]) and not client.weight_scaler:
                    self.clients[i].weight_scaler = scaler[1]

    def _federated_averaging(self, models, weights):
        fn_average = strategies.federated_averaging
        fn_sum = strategies.weighted_sum
        fn_reduce = reduce_models

        if self.conf.is_distributed:
            dist.barrier()
            model_, sample_sum = fn_sum(models, weights)
            fn_reduce(model_, torch.tensor(sample_sum).to(self.conf.device))
        else:
            model_ = fn_average(models, weights)
        return model_

    def test_in_server(self, device=CPU):
        testing_model = self._get_testing_model()
        testing_model.eval()
        testing_model.to(device)

        self._get_test_data()

        with torch.no_grad():
            accuracy = knn_monitor(testing_model, self.train_loader, self.test_loader, device=device)

        test_results = {metric.TEST_METRIC: {"accuracy": float(accuracy), "loss": 0}}
        
        return test_results

    def _get_test_data(self):
        transformation = self._load_transform()
        if self.train_loader is None or self.test_loader is None:
            if self.conf.data.dataset == CIFAR100:
                data_path = "./data/cifar100"
                train_dataset = datasets.CIFAR100(data_path, download=True, transform=transformation)
                test_dataset = datasets.CIFAR100(data_path, train=False, download=True, transform=transformation)
            else:
                data_path = "./data/cifar10"
                train_dataset = datasets.CIFAR10(data_path, download=True, transform=transformation)
                test_dataset = datasets.CIFAR10(data_path, train=False, download=True, transform=transformation)

            if self.train_loader is None:
                self.train_loader = DataLoader(train_dataset, batch_size=512)

            if self.test_loader is None:
                self.test_loader = DataLoader(test_dataset, batch_size=512)

    def _load_transform(self):
        transformation = utils.get_transformation(self.conf.model)
        return transformation().test_transform

    def _get_testing_model(self, net=False):
        if self.conf.model in [model.MoCo, model.MoCoV2]:
            testing_model = self.model.encoder_q
        elif self.conf.model in [model.SimSiam, model.SimCLR]:
            testing_model = self.model.online_encoder
        else:
            # BYOL
            if self.conf.client.aggregate_encoder == TARGET:
                self.print_("Use aggregated target encoder for testing")
                testing_model = self.model.target_encoder
            else:
                self.print_("Use aggregated online encoder for testing")
                testing_model = self.model.online_encoder
        return testing_model

    def save_model(self):
        if self._do_every(self.conf.server.save_model_every, self.current_round,
                          self.conf.server.rounds) and self.is_primary_server():
            save_path = self.conf.server.save_model_path
            if save_path == "":
                save_path = os.path.join(os.getcwd(), "saved_models", self.conf.task_id)
            os.makedirs(save_path, exist_ok=True)
            save_path = os.path.join(save_path,
                                     "{}_global_model_r_{}.pth".format(self.conf.task_id, self.current_round))

            torch.save(self._get_testing_model().cpu().state_dict(), save_path)
            self.print_("Encoder model saved at {}".format(save_path))

            if self.conf.server.save_predictor:
                if self.conf.model in [model.SimSiam, model.BYOL]:
                    save_path = save_path.replace("global_model", "predictor")
                    torch.save(self.model.online_predictor.cpu().state_dict(), save_path)
                    self.print_("Predictor model saved at {}".format(save_path))


def knn_monitor(net, memory_data_loader, test_data_loader, k=200, t=0.1, hide_progress=False, device=None):
    net.eval()
    classes = len(memory_data_loader.dataset.classes)
    total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
    with torch.no_grad():
        # generate feature bank
        for data, target in tqdm(memory_data_loader, desc='Feature extracting', leave=False, disable=hide_progress):
            if device is None:
                data = data.cuda(non_blocking=True)
            else:
                data = data.to(device, non_blocking=True)
            feature = net(data)
            feature = F.normalize(feature, dim=1)
            feature_bank.append(feature)
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        # [N]
        feature_labels = torch.tensor(memory_data_loader.dataset.targets, device=feature_bank.device)
        # loop test data to predict the label by weighted knn search
        test_bar = tqdm(test_data_loader, desc='kNN', disable=hide_progress)
        for data, target in test_bar:
            if device is None:
                data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            else:
                data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            feature = net(data)
            feature = F.normalize(feature, dim=1)

            pred_labels = knn_predict(feature, feature_bank, feature_labels, classes, k, t)

            total_num += data.size(0)
            total_top1 += (pred_labels[:, 0] == target).float().sum().item()
            test_bar.set_postfix({'Accuracy': total_top1 / total_num * 100})
        print("Accuracy: {}".format(total_top1 / total_num * 100))
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
