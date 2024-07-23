import logging
import os
import torchvision

from coala.datasets.simulation import data_simulation, split_labeled_and_unlabeled
from coala.datasets.utils.base_dataset import BaseDataset, CIFAR10
from coala.datasets.dataset import FederatedTensorDataset
from coala.datasets.utils.util import save_dict, load_dict
from coala.datasets.cifar10.preprocess import transform_train_cifar, transform_test_cifar, TransformFixMatch

logger = logging.getLogger(__name__)


class Cifar10Semi(BaseDataset):
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
                 alpha=0.5,
                 semi_scenario='label_in_server',
                 num_labels_per_class=5,
                 s_split_type="iid"):
        super(Cifar10Semi, self).__init__(root,
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
        self.semi_scenario = semi_scenario
        self.num_labels_per_class = num_labels_per_class
        self.s_split_type = s_split_type
        self.s_train = None
        self.u_train = None

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
        s_train_data_path = os.path.join(self.data_folder,
                                         "train_s_{}_{}".format(self.semi_scenario, self.num_labels_per_class))
        u_train_data_path = os.path.join(self.data_folder,
                                         "train_u_{}_{}".format(self.semi_scenario, self.num_labels_per_class))

        test_data_path = os.path.join(self.data_folder, "test")
        if not os.path.exists(self.data_folder):
            os.makedirs(self.data_folder)

        if os.path.exists(u_train_data_path) or os.path.exists(s_train_data_path):
            return

        # semi_scenario, num_labels_per_class, num_of_client
        # split the training data into labeled and unlabeled sets
        self.s_train, self.u_train = split_labeled_and_unlabeled(self.train_data['x'], self.train_data['y'],
                                                                 self.semi_scenario, self.num_labels_per_class,
                                                                 self.num_of_client)

        logger.info("Start CIFAR10 data simulation")
        if self.semi_scenario != 'label_in_server':
            ssl_s_split_type = self.s_split_type
            _, s_train_data = data_simulation(self.s_train['x'],
                                              self.s_train['y'].tolist(),
                                              self.num_of_client,
                                              ssl_s_split_type,
                                              self.weights,
                                              self.alpha,
                                              self.min_size,
                                              self.class_per_client)
        else:
            s_train_data = {
                'x': self.s_train['x'],
                'y': self.s_train['y'].tolist()
            }

        _, u_train_data = data_simulation(self.u_train['x'],
                                          self.u_train['y'].tolist(),
                                          self.num_of_client,
                                          self.split_type,
                                          self.weights,
                                          self.alpha,
                                          self.min_size,
                                          self.class_per_client)

        logger.info("Complete CIFAR10 data simulation")

        save_dict(s_train_data, s_train_data_path)
        save_dict(u_train_data, u_train_data_path)

        if not os.path.exists(test_data_path):
            save_dict(self.test_data, test_data_path)

    def convert_data_to_json(self):
        pass


def construct_cifar10_semi_datasets(root,
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
                                    semi_scenario,
                                    num_labels_per_class):
    user_str = "sample"
    setting = BaseDataset.get_setting_folder(dataset_name, split_type, num_of_clients, min_size, class_per_client,
                                             data_amount, iid_fraction, user_str, train_test_split, alpha,
                                             quantity_weights)

    data_dir = os.path.join(root, dataset_name)
    if not data_dir:
        os.makedirs(data_dir)

    s_train_data_dir = os.path.join(data_dir, setting, "train_s_{}_{}".format(semi_scenario, num_labels_per_class))
    u_train_data_dir = os.path.join(data_dir, setting, "train_u_{}_{}".format(semi_scenario, num_labels_per_class))
    test_data_dir = os.path.join(data_dir, setting, "test")

    if not os.path.exists(s_train_data_dir):
        dataset = Cifar10Semi(root=data_dir,
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
                              semi_scenario=semi_scenario,
                              weights=quantity_weights,
                              num_labels_per_class=num_labels_per_class)

        dataset.setup()

    s_train_data = load_dict(s_train_data_dir)
    u_train_data = load_dict(u_train_data_dir)
    test_data = load_dict(test_data_dir)

    transform_train = transform_train_cifar
    transform_test = transform_test_cifar
    transform_u = TransformFixMatch()

    simulated = False if semi_scenario == 'label_in_server' else True
    train_labeled_data = FederatedTensorDataset(s_train_data,
                                                simulated=simulated,
                                                do_simulate=False,
                                                transform=transform_train,
                                                process_x=None,
                                                process_y=None)

    train_unlabeled_data = FederatedTensorDataset(u_train_data,
                                                  simulated=True,
                                                  do_simulate=False,
                                                  transform=transform_u,
                                                  process_x=None,
                                                  process_y=None)

    test_data = FederatedTensorDataset(test_data,
                                       simulated=False,
                                       do_simulate=False,
                                       transform=transform_test,
                                       process_x=None,
                                       process_y=None)

    return train_labeled_data, train_unlabeled_data, test_data
