from coala.datasets.data import construct_datasets
from coala.datasets.dataset import (
    FederatedDataset,
    FederatedImageDataset,
    FederatedTensorDataset,
    FederatedTorchDataset,
    TEST_IN_SERVER,
    TEST_IN_CLIENT,
)
from coala.datasets.simulation import (
    data_simulation,
    iid,
    non_iid_dirichlet,
    non_iid_class,
    equal_division,
    quantity_hetero,
)
from coala.datasets.utils.base_dataset import BaseDataset
from coala.datasets.femnist import Femnist
from coala.datasets.shakespeare import Shakespeare
from coala.datasets.cifar10 import Cifar10
from coala.datasets.cifar100 import Cifar100

__all__ = ['FederatedDataset', 'FederatedImageDataset', 'FederatedTensorDataset', 'FederatedTorchDataset',
           'construct_datasets', 'data_simulation', 'iid', 'non_iid_dirichlet', 'non_iid_class',
           'equal_division', 'quantity_hetero', 'BaseDataset', 'Femnist', 'Shakespeare', 'Cifar10', 'Cifar100']
