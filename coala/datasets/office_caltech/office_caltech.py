import os
import numpy as np
import logging

from torchvision.datasets.folder import make_dataset
from coala.datasets.utils.base_dataset import BaseDataset
from coala.datasets.utils.download import download_url, extract_archive
from coala.datasets.dataset import FederatedTensorDataset
from coala.datasets.dataset_util import ColorImageDataset
from coala.datasets.office_caltech.preprocess import IMG_EXTENSIONS, transform_train_office, transform_test_office, \
    in_domain_split

logger = logging.getLogger(__name__)


class OfficeCaltech(BaseDataset):
    """Office_Caltech dataset implementation. It stores the raw and pre-split image paths locally, respectively.

    Attributes:
        base_folder (str): The base folder path of the dataset folder.
        raw_data_url (str): The url to get the raw datasets of Office-Caltech.
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
                 seed=0,
                 weights=None,
                 alpha=0.5):
        super(OfficeCaltech, self).__init__(root,
                                            "office_caltech",
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
        self.domain_list = ["amazon", "caltech", "dslr", "webcam"]
        self.num_domain = len(self.domain_list)
        self.raw_data_url = "https://cloud.tsinghua.edu.cn/f/5aca3f6c81b5448f85bc/?dl=1"
        self.raw_file = "office_caltech_10.zip"

    def download_raw_file_and_extract(self):

        raw_data_folder = self.base_folder
        if not os.path.exists(raw_data_folder):
            os.makedirs(raw_data_folder)
        elif os.listdir(raw_data_folder):
            logger.info("raw file exists")
            return
        file_path = download_url(self.raw_data_url, raw_data_folder, self.raw_file)
        extract_archive(file_path, remove_finished=True)
        logger.info("raw file is downloaded")

    def data_splitting(self, setting):

        # raw-data splitting into distributed sets
        raw_data_folder = os.path.join(self.base_folder)
        split_data_folder = os.path.join(self.base_folder, setting)

        num_client_per_domain = int(self.num_of_client / self.num_domain)
        class_to_idx = {'back_pack': 0, 'bike': 1, 'calculator': 2, 'headphones': 3, 'keyboard': 4,
                        'laptop_computer': 5, 'monitor': 6, 'mouse': 7, 'mug': 8, 'projector': 9}

        for dom in self.domain_list:
            logger.info('Partitioning {}...'.format(dom))
            dom_data_folder = os.path.join(raw_data_folder, dom)
            dom_split_folder = os.path.join(split_data_folder, dom)
            samples = make_dataset(dom_data_folder, class_to_idx, extensions=IMG_EXTENSIONS)
            if len(samples) == 0:
                msg = "Found 0 files in sub-folders of: {}\n".format(dom_data_folder)
                raise RuntimeError(msg)

            # random shuffle before allocation
            data_x = [i[0] for i in samples]
            data_y = [i[1] for i in samples]
            np.random.seed(self.seed)
            rng_state = np.random.get_state()
            np.random.shuffle(data_x)
            np.random.set_state(rng_state)
            np.random.shuffle(data_y)

            dom_data_set = {'x': data_x, 'y': data_y}

            in_domain_split(dom_data_set, dom_split_folder, num_client_per_domain, self.train_test_split)


def read_office_caltech_partitions(data_dir):
    """Load datasets from data directories.

    Args:
        data_dir (str): The directory of data, e.g., ..data/dataset_name/setting

    Returns:
        dict: A dictionary of training data, e.g., {"id1": {"x": data_path, "y": label}, "id2": {"x": data_path, "y": label}}.
        dict: A dictionary of testing data.
    """
    domains = ["amazon", "caltech", "dslr", "webcam"]
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


def construct_office_caltech_datasets(root,
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
        dataset = OfficeCaltech(root=data_dir,
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
        try:
            dataset.download_raw_file_and_extract()
            filename = dataset.raw_file
            logger.info(f"Downloaded raw dataset {dataset_name}: {filename}")
        except Exception as e:
            logger.info(f"Failed to download raw dataset: {e.args}")

        dataset.data_splitting(setting)
        logger.info(f"data splitting accomplished")

    train_data, test_data = read_office_caltech_partitions(split_data_dir)

    assert len(train_data) == num_of_clients, "the num of clients should match the pre-split files"

    transform_train = {str(c): transform_train_office for c in range(num_of_clients)}
    transform_test = {str(c): transform_test_office for c in range(num_of_clients)}

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
