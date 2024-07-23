import os
import numpy as np
import logging
from torchvision.datasets.folder import make_dataset
from coala.datasets.simulation import data_simulation
from coala.datasets.utils.base_dataset import BaseDataset
from coala.datasets.utils.download import download_url, extract_archive
from coala.datasets.dataset import FederatedMultiDomainDataset
from coala.datasets.dataset_util import ImageDataset
from coala.datasets.casia_webface.preprocess import IMG_EXTENSIONS, transform_train_face, transform_test_face
from coala.datasets.utils.util import save_dict, load_dict
from .utils import read_pairs, get_paths

logger = logging.getLogger(__name__)


class CASIAWebFace(BaseDataset):
    """CASIAWebFace dataset implementation. It stores the raw and pre-split image paths locally, respectively.

    Attributes:
        base_folder (str): The base folder path of the dataset folder.
        raw_data_url (str): The url to get the raw datasets of CASIAWebFace.
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
        super(CASIAWebFace, self).__init__(root,
                                           "casia_webface",
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
        self.raw_data_url = "https://cloud.tsinghua.edu.cn/f/5aca3f6c81b5448f85bc/?dl=1"
        self.raw_file = "CASIA-WebFace.zip"
        self.test_folder = "./data/face_test/RFW"  # can be changed
        self.test_sets = ['Indian', 'African', 'Caucasian', 'Asian']

        if not os.path.exists(self.test_folder):
            raise NotImplementedError("Test set is not available")

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

    def data_splitting(self):
        train_data_path = os.path.join(self.data_folder, "train")
        test_data_path = os.path.join(self.data_folder, "test")
        if not os.path.exists(self.data_folder):
            os.makedirs(self.data_folder)
        if os.path.exists(train_data_path):
            return

        # raw-data splitting into distributed sets
        raw_data_folder = os.path.join(self.base_folder, "CASIA-WebFace")
        samples = make_dataset(raw_data_folder, extensions=IMG_EXTENSIONS)
        if len(samples) == 0:
            msg = "Found 0 files in folder of: {}\n".format(raw_data_folder)
            raise RuntimeError(msg)

        # random shuffle before allocation
        data_x = [i[0] for i in samples]
        data_y = [i[1] for i in samples]
        np.random.seed(self.seed)
        rng_state = np.random.get_state()
        np.random.shuffle(data_x)
        np.random.set_state(rng_state)
        np.random.shuffle(data_y)

        self.train_data = {'x': data_x, 'y': data_y}

        _, train_data = data_simulation(self.train_data['x'],
                                        self.train_data['y'],
                                        self.num_of_client,
                                        self.split_type,
                                        self.weights,
                                        self.alpha,
                                        self.min_size,
                                        self.class_per_client)
        logger.info("Complete WebFace data simulation")

        save_dict(train_data, train_data_path)

        # get test set
        for set in self.test_sets:
            img_path = os.path.join(self.test_folder, "data", set)
            pair_path = os.path.join(self.test_folder, "txts", set, f"{set}_pairs.txt")
            pairs = read_pairs(pair_path)
            path_list, is_same_list = get_paths(img_path, pairs)
            self.test_data[set] = {'x': path_list, 'y': is_same_list}
        save_dict(self.test_data, test_data_path)

    def setup(self):
        self.download_raw_file_and_extract()
        self.data_splitting()


def construct_casia_webface_datasets(root,
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
    user_str = "sample"
    setting = BaseDataset.get_setting_folder(dataset_name, split_type, num_of_clients, min_size, class_per_client,
                                             data_amount, iid_fraction, user_str, train_test_split, alpha,
                                             quantity_weights)

    data_dir = os.path.join(root, dataset_name)
    if not data_dir:
        os.makedirs(data_dir)
    split_data_dir = os.path.join(data_dir, setting)

    if not os.path.exists(split_data_dir):
        dataset = CASIAWebFace(root=data_dir,
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

        dataset.setup()
        logger.info(f"data splitting accomplished")

    train_data_dir = os.path.join(data_dir, setting, "train")
    test_data_dir = os.path.join(data_dir, setting, "test")

    train_data = load_dict(train_data_dir)
    test_data = load_dict(test_data_dir)

    assert len(train_data) == num_of_clients, "the num of clients should match the pre-split files"

    train_data = FederatedMultiDomainDataset(train_data,
                                             simulated=True,
                                             do_simulate=False,
                                             transform=transform_train_face,
                                             process_x=None,
                                             process_y=None,
                                             data_wrapping=ImageDataset)

    test_data = FederatedMultiDomainDataset(test_data,
                                            simulated=True,
                                            do_simulate=False,
                                            transform=transform_test_face,
                                            process_x=None,
                                            process_y=None,
                                            data_wrapping=ImageDataset)
    return train_data, test_data
