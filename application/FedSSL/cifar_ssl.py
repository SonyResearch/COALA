import logging
import os

from coala.datasets.utils.base_dataset import BaseDataset
from coala.datasets.dataset import FederatedTensorDataset
from coala.datasets.utils.util import load_dict
from coala.datasets import Cifar10, Cifar100
from coala.datasets.cifar10.preprocess import transform_test_cifar
from utils import get_transformation
from coala.datasets.data import CIFAR100

logger = logging.getLogger(__name__)


def construct_cifar_ssl_datasets(root,
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
        data_load = Cifar100 if dataset_name == CIFAR100 else Cifar10
        dataset = data_load(root=data_dir,
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

    transform_unlabel = get_transformation("byol")()
    transform_test = transform_test_cifar

    train_data = FederatedTensorDataset(train_data,
                                        simulated=True,
                                        do_simulate=False,
                                        transform=transform_unlabel,
                                        process_x=None,
                                        process_y=None)

    test_data = FederatedTensorDataset(test_data,
                                       simulated=False,
                                       do_simulate=False,
                                       transform=transform_test,
                                       process_x=None,
                                       process_y=None)

    return train_data, test_data
