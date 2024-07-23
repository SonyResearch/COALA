import logging
import os
import numpy as np
from coala.datasets.simulation import data_simulation
from coala.datasets.utils.base_dataset import BaseDataset
from coala.datasets.dataset import FederatedContinualDataset
from coala.datasets.utils.util import save_dict, load_dict
from coala.datasets.cifar10.cifar10 import Cifar10
from coala.datasets.cifar10.preprocess import transform_train_cifar, transform_test_cifar, map_new_class_index

logger = logging.getLogger(__name__)


class Cifar10Continual(Cifar10):
    """
       class-continual data stream split of CIFAR-10 for federated continual learning
    """

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
                 num_tasks=2,
                 init_cls=0,
                 increment=0,
                 shuffle=False):
        super(Cifar10Continual, self).__init__(root,
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
        self.train_data, self.test_data = {}, {}
        self.split_type = split_type
        self.num_of_client = num_of_client
        self.weights = weights
        self.alpha = alpha
        self.min_size = minsample
        self.class_per_client = class_per_client

        self.shuffle = shuffle
        self.class_order = []
        self.init_cls = init_cls
        self.increment = increment
        self.num_tasks = num_tasks
        self.cls_batches = []

        self.known_classes = 0
        self.total_classes = 0

    def get_task_size(self, task):
        return self.cls_batches[task]

    def get_known_size(self, task):
        return sum([self.cls_batches[i] for i in range(task)])

    def get_total_size(self, task):
        return sum([self.cls_batches[i] for i in range(task + 1)])

    def get_total_class_num(self):
        return len(self.class_order)

    def class_split(self):
        # data class ordering
        order = [i for i in range(len(np.unique(self.train_data['y'])))]
        if self.shuffle:
            np.random.seed(self.seed)
            order = np.random.permutation(len(order)).tolist()
        self.class_order = order

        # re-labeling according to the class order
        self.train_data['y'] = map_new_class_index(self.train_data['y'], self.class_order)
        self.test_data['y'] = map_new_class_index(self.test_data['y'], self.class_order)

        # set the sequences of class increments
        assert self.init_cls <= len(self.class_order), "No enough classes."

        if self.init_cls == 0:
            self.init_cls = int(len(self.class_order) / self.num_tasks)
        if self.increment == 0:
            self.increment = int(len(self.class_order) / self.num_tasks)

        self.cls_batches = [self.init_cls]
        while sum(self.cls_batches) + self.increment <= len(self.class_order):
            self.cls_batches.append(self.increment)
        rest = len(self.class_order) - sum(self.cls_batches)
        if rest > 0:
            self.cls_batches[-1] = self.cls_batches[-1] + rest

        assert len(self.cls_batches) == self.num_tasks, "Number of tasks is not consistent"

    def preprocess(self):
        train_data_path = os.path.join(self.data_folder, "train")
        test_data_path = os.path.join(self.data_folder, "test")
        if self.weights is None and os.path.exists(train_data_path):
            return
        if not os.path.exists(train_data_path):
            os.makedirs(train_data_path)
            os.makedirs(test_data_path)

        logger.info("Start CIFAR10 class-continual data stream simulation")

        for task_id in range(len(self.cls_batches)):
            train_data_batch_path = os.path.join(train_data_path, f"batch_{task_id}")
            test_data_batch_path = os.path.join(test_data_path, f"batch_{task_id}")

            cur_class_size = self.get_task_size(task_id)
            known_classes = self.get_known_size(task_id)
            total_classes = self.get_total_size(task_id)

            train_batch_x, train_batch_y = self.select_by_label(self.train_data['x'], self.train_data['y'],
                                                                known_classes, total_classes)
            test_batch_x, test_batch_y = self.select_by_label(self.test_data['x'], self.test_data['y'],
                                                              known_classes, total_classes)

            _, train_data = data_simulation(train_batch_x,
                                            train_batch_y,
                                            self.num_of_client,
                                            self.split_type,
                                            self.weights,
                                            self.alpha,
                                            self.min_size,
                                            min(self.class_per_client, cur_class_size)
                                            )
            test_data = {
                'x': test_batch_x,
                'y': test_batch_y
            }
            save_dict(train_data, train_data_batch_path)
            save_dict(test_data, test_data_batch_path)

        save_dict(self.cls_batches, os.path.join(self.data_folder, "class_batches"))

        logger.info("Complete CIFAR10 incremental data simulation")

    def select_by_label(self, x, y, low_range, high_range):
        indices = np.where(np.logical_and(y >= low_range, y < high_range))[0]
        return x[indices], y[indices]

    def setup(self):
        self.download_raw_file_and_extract()
        self.class_split()
        self.preprocess()
        self.convert_data_to_json()


def construct_cifar10_continual_datasets(root,
                                         dataset_name,
                                         num_of_clients,
                                         split_type,
                                         min_size=10,
                                         class_per_client=1,
                                         data_amount=0.05,
                                         iid_fraction=0.1,
                                         user=False,
                                         train_test_split=0.9,
                                         quantity_weights=None,
                                         alpha=0.5,
                                         num_tasks=5,
                                         init_cls=0,
                                         increment=0,
                                         shuffle=False
                                         ):
    user_str = "sample"
    setting = BaseDataset.get_setting_folder(dataset_name, split_type, num_of_clients, min_size, num_tasks,
                                             data_amount, iid_fraction, user_str, train_test_split, alpha,
                                             quantity_weights)

    data_dir = os.path.join(root, dataset_name)
    if not data_dir:
        os.makedirs(data_dir)
    split_data_dir = os.path.join(data_dir, setting)

    if not os.path.exists(split_data_dir):
        dataset = Cifar10Continual(root=data_dir,
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
                                   weights=quantity_weights,
                                   num_tasks=num_tasks,
                                   init_cls=init_cls,
                                   increment=increment,
                                   shuffle=shuffle
                                   )

        dataset.setup()

    train_data_dir = os.path.join(data_dir, setting, "train")
    test_data_dir = os.path.join(data_dir, setting, "test")

    train_data_list = []
    test_data_list = []

    for task_id in range(num_tasks):
        train_data_batch_dir = os.path.join(train_data_dir, f"batch_{task_id}")
        test_data_batch_dir = os.path.join(test_data_dir, f"batch_{task_id}")
        train_data = load_dict(train_data_batch_dir)
        test_data = load_dict(test_data_batch_dir)
        train_data_list.append(train_data)
        test_data_list.append(test_data)

    class_batches = load_dict(os.path.join(data_dir, setting, "class_batches"))

    transform_train = transform_train_cifar
    transform_test = transform_test_cifar

    train_data = FederatedContinualDataset(train_data_list,
                                           simulated=True,
                                           do_simulate=False,
                                           transform=transform_train,
                                           process_x=None,
                                           process_y=None,
                                           cls_batches=class_batches)

    test_data = FederatedContinualDataset(test_data_list,
                                          simulated=False,
                                          do_simulate=False,
                                          transform=transform_test,
                                          process_x=None,
                                          process_y=None,
                                          cls_batches=class_batches)

    return train_data, test_data
