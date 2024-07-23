import logging
import os
import torch
import copy
from coala.server.base import BaseServer, MODEL, DATA_SIZE
from coala.distributed.distributed import broadcast_model_dict
import torch.distributed as dist

logger = logging.getLogger(__name__)


class ClusterServer(BaseServer):
    def __init__(self, conf, test_data=None, val_data=None, is_remote=False, local_port=22999):
        super(ClusterServer, self).__init__(conf, test_data, val_data, is_remote, local_port)

    def set_model(self, model, load_dict=False):
        """Update the universal model in the server.
        Overwrite this method for different training backend, default is for PyTorch.

        Args:
            model (nn.Module or dict (nn.Module): New model.
            load_dict (bool): A flag to indicate whether load state dict or copy the model.
        """
        num_clusters = self.conf.num_of_clusters

        if load_dict:
            for i in range(num_clusters):
                if i in model:
                    self.model[i].load_state_dict(copy.deepcopy(model[i].state_dict()))
        else:
            self.model = {}
            for i in range(num_clusters):
                seed = torch.random.seed()
                model_i = copy.deepcopy(model)
                model_i.model_id = i
                self.model[i] = model_i.to(self.conf.device)
            if self.conf.is_distributed and self.is_primary_server:
                dist.barrier()
                broadcast_model_dict(self.model, src=0)

    def aggregation(self):
        """Aggregate training updates from clients.
        Server aggregates trained models from clients via federated averaging.
        """
        uploaded_content = self.get_client_uploads()
        models = list(uploaded_content[MODEL].values())
        weights = list(uploaded_content[DATA_SIZE].values())
        model_clusters = {}
        for idx in range(self.conf.num_of_clusters):
            model_list = []
            weight_list = []
            for i in range(len(models)):
                if models[i].model_id == idx:
                    model_list.append(models[i])
                    weight_list.append(weights[i])
            if len(weight_list) == 0:
                model_list.append(copy.deepcopy(self.model[idx].to(self.conf.device)))
                weight_list.append(0)
            model_clusters[idx] = self.aggregate(model_list, weight_list)

        self.set_model(model_clusters, load_dict=True)

    def save_model(self):
        """Save the model in the server.
        Overwrite this method for different training backend, default is PyTorch.
        """
        if self._do_every(self.conf.server.save_model_every, self.current_round, self.conf.server.rounds) and \
                self.is_primary_server():
            save_path = self.conf.server.save_model_path
            if save_path == "":
                save_path = os.path.join(os.getcwd(), "saved_models")
            os.makedirs(save_path, exist_ok=True)
            self.print_("Model saved at {}".format(save_path))
            for key in self.model.keys():
                save_path_i = os.path.join(save_path,
                                           "{}_global_model_r_{}_c{}.pth".format(self.conf.task_id, self.current_round,
                                                                                 key))
                torch.save(self.model[key].cpu().state_dict(), save_path_i)
