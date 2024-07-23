import coala
from coala.server import BaseServer
import wandb

config = {
    "task_id": "cifar10_wandb",
    "data": {"dataset": "cifar10"},
    "model": "simple_cnn",
    "test_mode": "test_in_server",
    "server": {"visualize": True}
}


class CustomizedServer(BaseServer):
    def __init__(self, conf, **kwargs):
        super(CustomizedServer, self).__init__(conf, **kwargs)

    def init_visualization(self):
        """
        init the external visualization tool, e.g., wandb, tensorboard
        """
        wandb.init(project=self.conf.task_id)
        wandb.config = {"split_type": self.conf.data.split_type, "num_of_clients": self.conf.data.num_of_clients}

    def tracking_visualization(self, results):
        """
        Args:
            results (dict): training and test metrics need tracking
        """
        wandb.log(results)


coala.register_server(CustomizedServer)
coala.init(config)
coala.run()
