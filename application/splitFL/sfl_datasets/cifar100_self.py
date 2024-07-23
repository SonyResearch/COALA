import logging
import os
from PIL import Image
import torchvision
import torchvision.transforms as transforms
from coala.datasets.simulation import data_simulation
from coala.datasets.utils.base_dataset import BaseDataset, CIFAR100
from coala.datasets.dataset import FederatedTensorDataset
from coala.datasets.utils.util import save_dict, load_dict
from coala.datasets.cifar100.preprocess import transform_train_cifar, transform_test_cifar
from tools import GaussianBlur
import numpy as np

logger = logging.getLogger(__name__)


class Cifar100SelfSupervised(BaseDataset):
    def __init__(self,
                 root,
                 fraction,
                 split_type,
                 user,
                 iid_user_fraction=0.1,
                 train_test_split=0.9,
                 minsample=10,
                 num_class=100,
                 num_of_client=100,
                 class_per_client=2,
                 setting_folder=None,
                 seed=-1,
                 weights=None,
                 alpha=0.5):
        super(Cifar100SelfSupervised, self).__init__(root,
                                      CIFAR100,
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

        class CIFAR100Pair(torchvision.datasets.CIFAR100):
            """CIFAR100 Dataset.
            """
            def __getitem__(self, index):
                img = self.data[index]
                img = Image.fromarray(img)
                if self.transform is not None:
                    im_1 = self.transform(img)
                    im_2 = self.transform(img)
                return im_1, im_2

        train_set = CIFAR100Pair(root=self.base_folder, train=True, download=True)
        test_set = torchvision.datasets.CIFAR100(root=self.base_folder, train=False, download=True)

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


def construct_cifar100_selfsupervised_datasets(root,
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
    user_str = "sample"
    setting = BaseDataset.get_setting_folder(dataset_name, split_type, num_of_clients, min_size, class_per_client,
                                             data_amount, iid_fraction, user_str, train_test_split, alpha,
                                             quantity_weights)

    data_dir = os.path.join(root, dataset_name)
    if not data_dir:
        os.makedirs(data_dir)
    split_data_dir = os.path.join(data_dir, setting)

    if not os.path.exists(split_data_dir):
        dataset = Cifar100SelfSupervised(root=data_dir,
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
    eval_data = load_dict(train_data_dir)

    if augmentation_option == "mocov1":
        transform_train = transforms.Compose([
            transforms.ToPILImage(mode='RGB'),
            transforms.RandomResizedCrop(32),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.49139968, 0.48215827, 0.44653124), (0.24703233, 0.24348505, 0.26158768)),])
    elif augmentation_option == "mocov2":
        transform_train = transforms.Compose([
            transforms.ToPILImage(mode='RGB'),
            transforms.RandomResizedCrop(32),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.49139968, 0.48215827, 0.44653124), (0.24703233, 0.24348505, 0.26158768)),])
    else:
        transform_train = transforms.Compose([
            transforms.ToPILImage(mode='RGB'),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.49139968, 0.48215827, 0.44653124), (0.24703233, 0.24348505, 0.26158768)),])

    transform_test = transform_test_cifar

    # current train_data has (data, label), we need to swap it to (data, data)
    for key, value in train_data.items():
        value['y'] = value['x']
    
    # combining all clients' training data to eval_data dict, which has key ['x'], ['y']
    new_eval_data = {'x': [], 'y': []}
    for key, value in eval_data.items():
        new_eval_data['x'].append(value['x'])
        new_eval_data['y'].append(value['y'])
    new_eval_data['x'] = np.concatenate(new_eval_data['x'], axis = 0)
    new_eval_data['y'] = np.concatenate(new_eval_data['y'], axis = 0)
    del eval_data

    train_data = FederatedTensorDataset(train_data,
                                        simulated=True,
                                        do_simulate=False,
                                        transform=transform_train,
                                        target_transform = transform_train,
                                        process_x=None,
                                        process_y=None)

    eval_data = FederatedTensorDataset(new_eval_data,
                                        simulated=False,
                                        do_simulate=False,
                                        transform=transform_train,
                                        process_x=None,
                                        process_y=None)
    
    eval_data.num_class = 100

    test_data = FederatedTensorDataset(test_data,
                                       simulated=False,
                                       do_simulate=False,
                                       transform=transform_test,
                                       process_x=None,
                                       process_y=None)

    return train_data, eval_data, test_data
