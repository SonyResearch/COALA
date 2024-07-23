import logging
import time
import torch
import copy
from coala.client import BaseClient
from coala.distributed.distributed import CPU
from coala.tracking import metric
from coala.tracking.evaluation import bit_to_megabyte

logger = logging.getLogger(__name__)


class ClusterClient(BaseClient):
    def __init__(self, cid, conf, train_data, test_data, device, **kwargs):
        super(ClusterClient, self).__init__(cid, conf, train_data, test_data, device, **kwargs)
        self.model_id = 0

    def model_select(self, conf, models, device=CPU):
        """Execute client testing.

        Args:
            conf (omegaconf.dictconfig.DictConfig): Client configurations.
            models (models for test)
            device (str): Hardware device for training, cpu or cuda devices.
        """
        val_loss_list = {}
        if self.train_loader is None:
            self.train_loader = self.load_loader(conf)
        for id in sorted(models.keys()):
            model = models[id]
            model.eval()
            model.to(device)
            loss_fn = self.load_loss_fn(conf)
            val_loss = 0
            with torch.no_grad():
                for batched_x, batched_y in self.train_loader:
                    x = batched_x.to(device)
                    y = batched_y.to(device)
                    log_probs = model(x)
                    loss = loss_fn(log_probs, y)
                    val_loss += loss.item()
                val_loss /= len(self.train_loader)
                val_loss_list[id] = val_loss
            model.cpu()
        model_id = min(val_loss_list, key=val_loss_list.get)
        logger.debug(val_loss_list, model_id)

        return model_id

    def set_model(self, models):
        """Set the given model as the client model.
        This method should be overwritten for different training backend, the default is PyTorch.

        Args:
            models (options: nn.Module, tf.keras.Model, ...): Global model distributed from the server.
        """
        self.model_id = self.model_select(self.conf, copy.deepcopy(models), self.device)
        if self.model:
            self.model.load_state_dict(copy.deepcopy(models[self.model_id].state_dict()))
        else:
            self.model = copy.deepcopy(models[self.model_id])
        self.model.model_id = self.model_id

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

        self.model.load_state_dict(copy.deepcopy(model[self.model_id].state_dict()))

        self.pre_test()
        self.test(conf, self.device)
        self.post_test()

        self.track(metric.TEST_METRIC, {"accuracy": float(self.test_accuracy), "loss": float(self.test_loss)})
        self.track(metric.TEST_TIME, self.test_time)

        return self.upload()

    def calculate_model_size(self, model, param_size=32):
        """Calculate the model parameter sizes, including non-trainable parameters.
        Should be overwritten for different training backend.

        Args:
            model (options: nn.Module, tf.keras.Model, ...): A model.
            param_size (int): The size of a parameter, default using float32.

        Returns:
            float: The model size in MB.
        """
        # sum(p.numel() for p in model.parameters() if p.requires_grad) for only trainable parameters
        if isinstance(model, dict):
            params = sum(p.numel() for p in model[self.model_id].parameters())
        else:
            params = sum(p.numel() for p in model.parameters())

        return bit_to_megabyte(params * param_size)
