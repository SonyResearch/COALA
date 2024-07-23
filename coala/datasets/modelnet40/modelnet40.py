import os
import h5py
import logging
import numpy as np
from glob import glob
from coala.datasets.utils.base_dataset import BaseDataset
from coala.datasets.utils.download import download_url, extract_archive
from coala.datasets.simulation import data_simulation
from coala.datasets.dataset import FederatedTorchDataset
from coala.datasets.utils.util import save_dict, load_dict
from .dataset import ModelNetDataset

logger = logging.getLogger(__name__)


class ModelNet40(BaseDataset):
    """ModelNet40 dataset implementation. It stores the raw image paths locally.

    Attributes:
        base_folder (str): The base folder path of the dataset folder.
        raw_data_folder (str): The folder to store the raw datasets of BDD100K.
    """

    def __init__(self,
                 root,
                 fraction=1.0,
                 split_type='iid',
                 user=False,
                 iid_user_fraction=1.0,
                 train_test_split=0.9,
                 minsample=10,
                 num_class=10,
                 num_of_client=4,
                 class_per_client=10,
                 setting_folder=None,
                 seed=-1,
                 weights=None,
                 alpha=0.5
                 ):
        super(ModelNet40, self).__init__(root,
                                         "modelnet40",
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
        self.weights = weights
        self.alpha = alpha
        self.min_size = minsample
        self.raw_data_folder = os.path.join(self.base_folder, "modelnet40_hdf5_2048")
        self.raw_data_url = "https://cloud.tsinghua.edu.cn/f/b3d9fe3e2a514def8097/?dl=1"
        self.path_h5py_all = []
        self.train_data = {}
        self.test_data = {}

    def download_raw_file_and_extract(self):
        if os.path.exists(self.raw_data_folder):
            logger.info("raw image files exist")
        else:
            file_path = download_url(self.raw_data_url, self.base_folder, "modelnet40_hdf5_2048.zip")
            extract_archive(file_path, remove_finished=True)
            logger.info("raw images are downloaded")

    def get_path(self, type):
        self.path_h5py_all = []
        path_h5py = os.path.join(self.raw_data_folder, f'{type}*.h5')
        paths = glob(path_h5py)
        paths_sort = [os.path.join(self.raw_data_folder, type + str(i) + '.h5') for i in range(len(paths))]
        self.path_h5py_all += paths_sort

    def load_h5py(self, path):
        all_data = []
        all_label = []
        for h5_name in path:
            f = h5py.File(h5_name, 'r+')
            data = f['data'][:].astype('float32')
            label = f['label'][:].astype('int64')
            f.close()
            all_data.extend(data)
            all_label.extend(label)
        all_data = np.array(all_data)
        all_label = np.array(all_label)
        return all_data, all_label

    def data_load_h5(self):
        self.get_path("train")
        train_x, train_y = self.load_h5py(self.path_h5py_all)
        self.train_data = {
            'x': train_x,
            'y': train_y
        }

        self.get_path("test")
        test_x, test_y = self.load_h5py(self.path_h5py_all)
        self.test_data = {
            'x': test_x,
            'y': test_y
        }

    def preprocess(self):
        train_data_path = os.path.join(self.data_folder, "train")
        test_data_path = os.path.join(self.data_folder, "test")
        if not os.path.exists(self.data_folder):
            os.makedirs(self.data_folder)
        if os.path.exists(train_data_path):
            return
        logger.info("Start ModelNet40 data simulation")
        _, train_data = data_simulation(self.train_data['x'],
                                        self.train_data['y'],
                                        self.num_of_client,
                                        self.split_type,
                                        self.weights,
                                        self.alpha,
                                        self.min_size,
                                        self.class_per_client)
        logger.info("Complete ModelNet40 data simulation")
        save_dict(train_data, train_data_path)
        save_dict(self.test_data, test_data_path)

    def setup(self):
        self.download_raw_file_and_extract()
        self.data_load_h5()
        self.preprocess()


def construct_modelnet40_datasets(root,
                                  dataset_name,
                                  num_of_clients,
                                  split_type='iid',
                                  min_size=10,
                                  class_per_client=1,
                                  data_amount=0.05,
                                  iid_fraction=0.1,
                                  user=False,
                                  train_test_split=0.9,
                                  quantity_weights=None,
                                  alpha=0.5):
    data_dir = os.path.join(root, dataset_name)
    user_str = "sample"
    setting = BaseDataset.get_setting_folder(dataset_name, split_type, num_of_clients, min_size, class_per_client,
                                             data_amount, iid_fraction, user_str, train_test_split, alpha,
                                             quantity_weights)
    if not data_dir:
        os.makedirs(data_dir)
    split_data_dir = os.path.join(data_dir, setting)

    if not os.path.exists(split_data_dir):
        dataset = ModelNet40(root=data_dir,
                             fraction=data_amount,
                             split_type=split_type,
                             user=user,
                             iid_user_fraction=iid_fraction,
                             train_test_split=train_test_split,
                             minsample=min_size,
                             num_of_client=num_of_clients,
                             class_per_client=class_per_client,
                             setting_folder=setting)
        try:
            dataset.download_raw_file_and_extract()
            logger.info(f"Downloaded raw dataset {dataset_name}")
        except Exception as e:
            logger.info(f"Please download raw dataset {dataset_name} manually: {e.args}")

        dataset.setup()
        logger.info(f"data splitting accomplished")

    train_data_dir = os.path.join(data_dir, setting, "train")
    test_data_dir = os.path.join(data_dir, setting, "test")

    train_data = load_dict(train_data_dir)
    test_data = load_dict(test_data_dir)

    train_sets = prepare_data(train_data, is_train=True)
    test_set = prepare_data(test_data, is_train=False)
    clients = [str(i) for i in range(num_of_clients)]

    train_data = FederatedTorchDataset(train_sets, clients, is_loaded=False)
    test_data = FederatedTorchDataset(test_set, clients, is_loaded=False)

    return train_data, test_data


def prepare_data(data_files, is_train=True, transform=None):
    if is_train:
        data_sets = {}
        for cid, data in data_files.items():
            data_set = ModelNetDataset(data['x'], data['y'], transform)
            data_sets[cid] = data_set
    else:
        data_sets = ModelNetDataset(data_files['x'], data_files['y'], transform)

    return data_sets
