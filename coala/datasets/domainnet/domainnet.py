import os
import numpy as np
import logging

from coala.datasets.utils.base_dataset import BaseDataset
from coala.datasets.utils.download import download_url, extract_archive
from coala.datasets.dataset import FederatedTensorDataset
from coala.datasets.dataset_util import ColorImageDataset
from coala.datasets.domainnet.preprocess import transform_train_domainnet, transform_test_domainnet, in_domain_split

logger = logging.getLogger(__name__)


class DomainNet(BaseDataset):
    """DomainNet dataset implementation. It stores the raw and pre-split image paths locally, respectively.

    Attributes:
        base_folder (str): The base folder path of the dataset folder.
        pre_split_url (str): The url to get the pre-split path-label of DomainNet.
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
                 alpha=0.5,
                 pre_split=True):
        super(DomainNet, self).__init__(root,
                                        "domainnet",
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
        self.domain_list = ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"]
        self.num_domain = len(self.domain_list)
        self.pre_split = pre_split
        self.pre_split_url = "https://cloud.tsinghua.edu.cn/f/f5949b3a560c4c86a223/?dl=1"
        self.packaged_file = "domainnet_dataset.zip"

    def download_raw_file_and_extract(self):

        raw_data_folder = self.base_folder
        if not os.path.exists(raw_data_folder):
            os.makedirs(raw_data_folder)

        url_list = ["http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/clipart.zip",
                    "http://csr.bu.edu/ftp/visda/2019/multi-source/infograph.zip",
                    "http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/painting.zip",
                    "http://csr.bu.edu/ftp/visda/2019/multi-source/quickdraw.zip",
                    "http://csr.bu.edu/ftp/visda/2019/multi-source/real.zip",
                    "http://csr.bu.edu/ftp/visda/2019/multi-source/sketch.zip"]

        for dom, url in zip(self.domain_list, url_list):
            if os.path.exists(os.path.join(self.base_folder, dom)):
                continue
            raw_data_url = url
            file_path = download_url(raw_data_url, raw_data_folder, f"{dom}.zip")
            extract_archive(file_path, remove_finished=True)
        logger.info("raw file is downloaded")

    def download_packaged_dataset_and_extract(self, filename=None):

        packaged_data_folder = os.path.join(self.base_folder, "pre_split")
        if not os.path.exists(packaged_data_folder):
            os.makedirs(packaged_data_folder)
        elif os.listdir(packaged_data_folder):
            logger.info("packaged file exists")
            return
        file_path = download_url(self.pre_split_url, packaged_data_folder, self.packaged_file)
        extract_archive(file_path, remove_finished=True)

    def data_splitting(self, setting):

        # raw-data splitting into distributed sets
        if self.pre_split:
            raw_data_folder = os.path.join(self.base_folder, "pre_split")
            split_data_folder = os.path.join(self.base_folder, setting)
            num_client_per_domain_avg = int(self.num_of_client / self.num_domain)
            num_client_per_domain = {}
            for dom in self.domain_list:
                num = num_client_per_domain_avg
                if dom == self.domain_list[-1]:
                    num = self.num_of_client - num_client_per_domain_avg*(self.num_domain-1)
                num_client_per_domain[dom] = num
                
            class_to_idx = {'bird': 0, 'feather': 1, 'headphones': 2, 'ice_cream': 3, 'teapot': 4,
                            'tiger': 5, 'whale': 6, 'windmill': 7, 'wine_glass': 8, 'zebra': 9}
            for dom in self.domain_list:
                logger.info('Partitioning {}...'.format(dom))
                dom_split_folder = os.path.join(split_data_folder, dom)

                # split training set
                paths, labels = np.load(os.path.join(raw_data_folder, f"{dom}_train.pkl"), allow_pickle=True)
                path_new = ['/'.join(path.split('/')[1:]) for path in paths]
                data_x = [os.path.join(self.base_folder, path) for path in path_new]
                data_y = [class_to_idx[i] for i in labels]
                data_set = {'x': data_x, 'y': data_y}
                in_domain_split(data_set, dom_split_folder, num_client_per_domain[dom], train=True)

                # split test set
                paths, labels = np.load(os.path.join(raw_data_folder, f"{dom}_test.pkl"), allow_pickle=True)
                path_new = ['/'.join(path.split('/')[1:]) for path in paths]
                data_x = [os.path.join(self.base_folder, path) for path in path_new]
                data_y = [class_to_idx[i] for i in labels]
                data_set = {'x': data_x, 'y': data_y}
                in_domain_split(data_set, dom_split_folder, num_client_per_domain[dom], train=False)
        else:
            raise NotImplementedError


def read_domainnet_partitions(data_dir):
    """Load datasets from data directories.

    Args:
        data_dir (str): The directory of data, e.g., ..data/dataset_name/setting

    Returns:
        dict: A dictionary of training data, e.g., {"id1": {"x": data_path, "y": label}, "id2": {"x": data_path, "y": label}}.
        dict: A dictionary of testing data.
    """
    domains= ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"]
    train_data = {}
    test_data = {}
    client_id = 0
    for dom in domains:
        train_path = os.path.join(data_dir, dom, 'train')
        test_path = os.path.join(data_dir, dom, 'test')
        files = os.listdir(train_path)
        for i in range(len(files)):
            # load training set
            train_x, train_y = np.load(os.path.join(train_path, 'train_part{}.pkl'.format(i)), allow_pickle=True)
            train_data[str(client_id)] = {'x': train_x, 'y': train_y}
            # load test set
            test_x, test_y = np.load(os.path.join(test_path, 'test_part{}.pkl'.format(i)), allow_pickle=True)
            test_data[str(client_id)] = {'x': test_x, 'y': test_y}
            # update the client id (total num of clients)
            client_id += 1

    return train_data, test_data


def construct_domainnet_datasets(root,
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
                                 pre_split=True):
    user_str = "sample"
    setting = BaseDataset.get_setting_folder(dataset_name, split_type, num_of_clients, min_size, class_per_client,
                                             data_amount, iid_fraction, user_str, train_test_split, alpha,
                                             quantity_weights)

    data_dir = os.path.join(root, dataset_name)
    if not data_dir:
        os.makedirs(data_dir)
    split_data_dir = os.path.join(data_dir, setting)

    if not os.path.exists(split_data_dir):
        dataset = DomainNet(root=data_dir,
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
                            pre_split=pre_split)

        try:
            dataset.download_raw_file_and_extract()
            logger.info(f"Downloaded raw dataset {dataset_name}")
        except Exception as e:
            logger.info(f"Failed to download raw dataset: {e.args}")

        try:
            dataset.download_packaged_dataset_and_extract()
            filename = dataset.packaged_file
            logger.info(f"Downloaded packaged dataset {dataset_name}: {filename}")
        except Exception as e:
            logger.info(f"Failed to download packaged dataset: {e.args}")

        dataset.data_splitting(setting)
        logger.info(f"data splitting accomplished")

    train_data, test_data = read_domainnet_partitions(split_data_dir)

    assert len(train_data) == num_of_clients, "the num of clients should match the pre-split files"

    transform_train = {str(c): transform_train_domainnet for c in range(num_of_clients)}
    transform_test = {str(c): transform_test_domainnet for c in range(num_of_clients)}

    train_data = FederatedTensorDataset(train_data,
                                        simulated=True,
                                        do_simulate=False,
                                        transform=transform_train,
                                        process_x=None,
                                        process_y=None,
                                        data_wrapping=ColorImageDataset)

    test_data = FederatedTensorDataset(test_data,
                                       simulated=True,
                                       do_simulate=False,
                                       transform=transform_test,
                                       process_x=None,
                                       process_y=None,
                                       data_wrapping=ColorImageDataset)
    return train_data, test_data
