import os
import json
import collections
import torchvision.transforms as transforms
from coala.datasets.utils.base_dataset import BaseDataset
from coala.datasets.utils.download import download_url, extract_archive
from coala.datasets.dataset import FederatedTorchDataset
from coala.datasets.bdd100k.data_split import *
from .dataset import LandmarksDataset, read_csv

logger = logging.getLogger(__name__)


class Landmarks(BaseDataset):
    """Landmarks dataset implementation. It stores the raw image paths locally.
       The gld23k dataset contains 203 classes, 233 clients and 23080 images.
       The gld160k dataset contains 2,028 classes, 1262 clients and 164,172 images.
       the user_id is string of numbers, e.g., '0'~'232'.

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
                 seed=-1):
        super(Landmarks, self).__init__(root,
                                        "landmarks",
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
        self.raw_data_folder = os.path.join(self.base_folder, "images")
        self.raw_label_folder = os.path.join(self.base_folder, "data_user")

        self.raw_data_url = "https://fedcv.s3-us-west-1.amazonaws.com/landmark/images.zip"
        self.user_csv_url = "https://fedcv.s3-us-west-1.amazonaws.com/landmark/data_user_dict.zip"

    def download_raw_file_and_extract(self):
        if os.path.exists(self.raw_data_folder):
            logger.info("raw image files exist")
        else:
            file_path = download_url(self.raw_data_url, self.base_folder, "images.zip")
            extract_archive(file_path, remove_finished=True)
            logger.info("raw images are downloaded")

        if os.path.exists(self.raw_label_folder):
            logger.info("raw data_user files exist")
        else:
            dict_path = download_url(self.user_csv_url, self.base_folder, "data_user_dict.zip")
            extract_archive(dict_path, remove_finished=True)
            logger.info("user_data dicts are downloaded")

    def setup(self):
        pass


def read_landmarks_partitions(data_dir, split='gld23k'):
    """Load datasets from data directories.

    Args:
        data_dir (str): The directory of data, e.g., ..data/dataset_name/setting
        split (str): The split of landmarks dataset, including 23k and 160k

    Returns:
        dict: A dictionary of training data, e.g., {"id1": {"x": data_path, "y": label_path}}.
        dict: A dictionary of testing data.
    """
    train_csv = os.path.join(data_dir, "data_user_dict/gld23k_user_dict_train.csv")
    test_csv = os.path.join(data_dir, "data_user_dict/gld23k_user_dict_train.csv")
    if split == 'gld160k':
        train_csv = os.path.join(data_dir, "data_user_dict/gld160k_user_dict_train.csv")
        test_csv = os.path.join(data_dir, "data_user_dict/gld160k_user_dict_train.csv")

    train_mapping = read_csv(train_csv)
    mapping_per_user = collections.defaultdict(list)
    test_mapping = read_csv(test_csv)
    train_data = {}
    for row in train_mapping:
        user_id = row["user_id"]
        mapping_per_user[user_id].append(row)
    for user_id, samples in mapping_per_user.items():
        train_data[str(user_id)] = samples
    test_data = test_mapping

    return train_data, test_data


def construct_landmarks_datasets(root,
                                 dataset_name,
                                 num_of_clients,
                                 split_type='iid',
                                 min_size=10,
                                 class_per_client=1,
                                 data_amount=0.05,
                                 iid_fraction=0.1,
                                 user=False,
                                 train_test_split=0.9,
                                 ):
    setting = "gld23k" if dataset_name in ["gld23k", "landmarks"] else "gld160k"
    data_dir = os.path.join(root, "landmarks")
    image_dir = os.path.join(data_dir, "images")
    if not data_dir:
        os.makedirs(data_dir)
    data_user_dir = os.path.join(data_dir, "data_user_dict")

    if not os.path.exists(data_user_dir):
        dataset = Landmarks(root=data_dir,
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

    train_data, test_data = read_landmarks_partitions(data_dir, split=setting)

    assert len(train_data) == num_of_clients, "the num of clients should match the split datasets"

    train_sets = prepare_data(image_dir, train_data, is_train=True, transform=transform_train)
    test_set = prepare_data(image_dir, test_data, is_train=False, transform=transform_test)

    clients = [str(i) for i in range(num_of_clients)]

    train_data = FederatedTorchDataset(train_sets, clients, is_loaded=False)
    test_data = FederatedTorchDataset(test_set, clients, is_loaded=False)

    return train_data, test_data


def prepare_data(root, train_files, is_train=True, transform=None):
    if is_train:
        clients = [str(i) for i in range(len(train_files))]
        data_sets = {}
        for cid in clients:
            data_set = LandmarksDataset(root, train_files[cid], is_train, transform)
            data_sets[cid] = data_set
    else:
        data_sets = LandmarksDataset(root, train_files, is_train, transform)

    return data_sets


transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5]),
])

transform_test = transforms.Compose([
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5]),
])
