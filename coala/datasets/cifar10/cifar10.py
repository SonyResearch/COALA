import logging
import os

import torchvision

from coala.datasets.simulation import data_simulation
from coala.datasets.utils.base_dataset import BaseDataset, CIFAR10
from coala.datasets.dataset import FederatedTensorDataset
from coala.datasets.utils.util import save_dict, load_dict
from coala.datasets.cifar10.preprocess import transform_train_cifar, transform_test_cifar

logger = logging.getLogger(__name__)


class Cifar10(BaseDataset):
    def __init__(self,
                 root,
                 fraction,
                 split_type,
                 user,
                 iid_user_fraction=0.1,
                 train_test_split=0.9,
                 minsample=10,
                 num_class=80,
                 num_of_client=100,
                 class_per_client=2,
                 setting_folder=None,
                 seed=-1,
                 weights=None,
                 alpha=0.5):
        super(Cifar10, self).__init__(root,
                                      CIFAR10,
                                      fraction,
                                      split_type,
                                      user,
                                      iid_user_fraction,
                                      train_test_split,
                                      minsample,
                                      num_class,
                                      num_of_client,
                                      class_per_client,
                                      setting_folder,
                                      seed)
        self.train_data, self.test_data = {}, {}
        self.split_type = split_type
        self.num_of_client = num_of_client
        self.weights = weights
        self.alpha = alpha
        self.min_size = minsample
        self.class_per_client = class_per_client

    def download_packaged_dataset_and_extract(self, filename):
        pass

    def download_raw_file_and_extract(self):
        train_set = torchvision.datasets.CIFAR10(root=self.base_folder, train=True, download=True)
        test_set = torchvision.datasets.CIFAR10(root=self.base_folder, train=False, download=True)

        self.train_data = {
            'x': train_set.data,
            'y': train_set.targets
        }

        self.test_data = {
            'x': test_set.data,
            'y': test_set.targets
        }

    def preprocess(self):
        train_data_path = os.path.join(self.data_folder, "train")
        test_data_path = os.path.join(self.data_folder, "test")
        if not os.path.exists(self.data_folder):
            os.makedirs(self.data_folder)
        if self.weights is None and os.path.exists(train_data_path):
            return
        logger.info("Start CIFAR10 data simulation")
        _, train_data = data_simulation(self.train_data['x'],
                                        self.train_data['y'],
                                        self.num_of_client,
                                        self.split_type,
                                        self.weights,
                                        self.alpha,
                                        self.min_size,
                                        self.class_per_client)
        logger.info("Complete CIFAR10 data simulation")
        save_dict(train_data, train_data_path)
        save_dict(self.test_data, test_data_path)

    def convert_data_to_json(self):
        pass


def construct_cifar10_datasets(root,
                               dataset_name,
                               num_of_clients,
                               split_type,
                               min_size,
                               class_per_client,
                               data_amount,
                               iid_fraction,
                               user,
                               train_test_split,
                               quantity_weights,
                               alpha):
    user_str = "sample"
    setting = BaseDataset.get_setting_folder(dataset_name, split_type, num_of_clients, min_size, class_per_client,
                                             data_amount, iid_fraction, user_str, train_test_split, alpha,
                                             quantity_weights)

    data_dir = os.path.join(root, dataset_name)
    if not data_dir:
        os.makedirs(data_dir)
    split_data_dir = os.path.join(data_dir, setting)

    if not os.path.exists(split_data_dir):
        dataset = Cifar10(root=data_dir,
                          fraction=data_amount,
                          split_type=split_type,
                          user=user,
                          iid_user_fraction=iid_fraction,
                          train_test_split=train_test_split,
                          minsample=min_size,
                          num_of_client=num_of_clients,
                          class_per_client=class_per_client,
                          setting_folder=setting,
                          alpha=alpha,
                          weights=quantity_weights)

        dataset.setup()

    train_data_dir = os.path.join(data_dir, setting, "train")
    test_data_dir = os.path.join(data_dir, setting, "test")

    train_data = load_dict(train_data_dir)
    test_data = load_dict(test_data_dir)

    transform_train = transform_train_cifar
    transform_test = transform_test_cifar

    train_data = FederatedTensorDataset(train_data,
                                        simulated=True,
                                        do_simulate=False,
                                        transform=transform_train,
                                        process_x=None,
                                        process_y=None)

    test_data = FederatedTensorDataset(test_data,
                                       simulated=False,
                                       do_simulate=False,
                                       num_of_clients=num_of_clients,
                                       simulation_method=split_type,
                                       alpha=alpha,
                                       transform=transform_test,
                                       process_x=None,
                                       process_y=None)

    return train_data, test_data
