import logging
import os
import numpy as np
import torchvision
from torchvision.datasets.utils import download_url
from coala.datasets.utils.base_dataset import BaseDataset
from coala.datasets.dataset import FederatedTensorDataset
from coala.datasets.utils.util import load_dict
from coala.datasets.cifar10.cifar10 import Cifar10
from coala.datasets.cifar10.preprocess import transform_train_cifar, transform_test_cifar

logger = logging.getLogger(__name__)

CIFAR10_1 = "cifar10.1"


class Cifar10_1(Cifar10):
    def __init__(self,
                 root,
                 fraction,
                 split_type,
                 user=False,
                 iid_user_fraction=0.1,
                 train_test_split=0.9,
                 minsample=10,
                 num_class=10,
                 num_of_client=20,
                 class_per_client=2,
                 setting_folder=None,
                 seed=-1,
                 weights=None,
                 alpha=0.5):
        super(Cifar10_1, self).__init__(root,
                                        fraction,
                                        split_type,
                                        user,
                                        iid_user_fraction=iid_user_fraction,
                                        train_test_split=train_test_split,
                                        minsample=minsample,
                                        num_class=num_class,
                                        num_of_client=num_of_client,
                                        class_per_client=class_per_client,
                                        setting_folder=setting_folder,
                                        seed=seed,
                                        weights=weights,
                                        alpha=alpha)
        self.dataset_name = CIFAR10_1

    def download_raw_file_and_extract(self):
        train_set = torchvision.datasets.CIFAR10(root=self.base_folder, train=True, download=True)
        test_set = CIFAR10Val1(root=self.base_folder)

        self.train_data = {
            'x': train_set.data,
            'y': train_set.targets
        }

        self.test_data = {
            'x': test_set.data,
            'y': test_set.targets
        }


def construct_cifar10_1_datasets(root,
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
        dataset = Cifar10_1(root=data_dir,
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
                                       do_simulate=True,
                                       num_of_clients=num_of_clients,
                                       simulation_method=split_type,
                                       alpha=alpha,
                                       transform=transform_test,
                                       process_x=None,
                                       process_y=None)

    return train_data, test_data


class CIFAR10Val1(object):
    """Borrowed from https://github.com/modestyachts/CIFAR-10.1"""

    stats = {
        "v4": {
            "data": "https://github.com/modestyachts/CIFAR-10.1/raw/master/datasets/cifar10.1_v4_data.npy",
            "labels": "https://github.com/modestyachts/CIFAR-10.1/raw/master/datasets/cifar10.1_v4_labels.npy",
        },
        "v6": {
            "data": "https://github.com/modestyachts/CIFAR-10.1/raw/master/datasets/cifar10.1_v6_data.npy",
            "labels": "https://github.com/modestyachts/CIFAR-10.1/raw/master/datasets/cifar10.1_v6_labels.npy",
        },
    }

    def __init__(self, root, data_name=None, version=None, transform=None):
        version = "v6" if version is None else version
        assert version in ["v4", "v6"]

        self.data_name = data_name
        self.path_data = os.path.join(root, f"cifar10.1_{version}_data.npy")
        self.path_labels = os.path.join(root, f"cifar10.1_{version}_labels.npy")
        self._download(root, version)

        self.data = np.load(self.path_data)
        self.targets = np.load(self.path_labels).tolist()
        self.data_size = len(self.data)

        self.transform = transform

    def _download(self, root, version):
        if self._check_integrity():
            print("Files already downloaded and verified")
            return

        download_url(url=self.stats[version]["data"], root=root)
        download_url(url=self.stats[version]["labels"], root=root)

    def _check_integrity(self) -> bool:
        if os.path.exists(self.path_data) and os.path.exists(self.path_labels):
            return True
        else:
            return False

    def __getitem__(self, index):
        img_array = self.data[index]
        target = self.targets[index]

        if self.transform is not None:
            img = self.transform(img_array)

        return img, target

    def __len__(self):
        return self.data_size
