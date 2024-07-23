import importlib
import json
import logging
import numpy as np
import os
from sfl_datasets.cifar10_self import *
from sfl_datasets.cifar100_self import *

logger = logging.getLogger(__name__)

def construct_datasets_selfsupervised(root,
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
                            alpha,
                            augmentation_option):
    """Construct and load provided federated learning datasets.

        Args:
            root (str): The root directory where datasets stored.
            dataset_name (str): The name of the dataset. It currently supports: femnist, shakespeare, cifar10, and cifar100.
                Among them, femnist and shakespeare are adopted from LEAF benchmark.
            num_of_clients (int): The targeted number of clients to construct.
            split_type (str): The type of statistical simulation, options: iid, dir, and class.
                `iid` means independent and identically distributed data.
                `niid` means non-independent and identically distributed data for Femnist and Shakespeare.
                `dir` means using Dirichlet process to simulate non-iid data, for CIFAR-10 and CIFAR-100 datasets.
                `class` means partitioning the dataset by label classes, for datasets like CIFAR-10, CIFAR-100.
            min_size (int): The minimal number of samples in each client.
                It is applicable for LEAF datasets and dir simulation of CIFAR-10 and CIFAR-100.
            class_per_client (int): The number of classes in each client. Only applicable when the split_type is 'class'.
            data_amount (float): The fraction of data sampled for LEAF datasets.
                e.g., 10% means that only 10% of total dataset size are used.
            iid_fraction (float): The fraction of the number of clients used when the split_type is 'iid'.
            user (bool): A flag to indicate whether partition users of the dataset into train-test groups.
                Only applicable to LEAF datasets.
                True means partitioning users of the dataset into train-test groups.
                False means partitioning each users' samples into train-test groups.
            train_test_split (float): The fraction of data for training; the rest are for testing.
                e.g., 0.9 means 90% of data are used for training and 10% are used for testing.
            quantity_weights (list[float]): The targeted distribution of quantities to simulate data quantity heterogeneity.
                The values should sum up to 1. e.g., [0.1, 0.2, 0.7].
                The `num_of_clients` should be divisible by `len(weights)`.
                None means clients are simulated with the same data quantity.
            alpha (float): The parameter for Dirichlet distribution simulation, applicable only when split_type is `dir`.
            semi_scenario (str): scenario of semi-supervised FL
            num_labels_per_class (int):

        Returns:
            :obj:`FederatedDataset`: Training dataset.
            :obj:`FederatedDataset`: Testing dataset.
        """
    if dataset_name == 'cifar10':
        construct_fl_datasets = construct_cifar10_selfsupervised_datasets
    elif dataset_name == 'cifar100':
        construct_fl_datasets = construct_cifar100_selfsupervised_datasets
    else:
        raise NotImplementedError

    u_train_data, eval_data, test_data = construct_fl_datasets(root,
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
                                                                  alpha,
                                                                  augmentation_option)

    return u_train_data, eval_data, test_data