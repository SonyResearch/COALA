import logging
import os
import json

from coala.datasets.femnist.preprocess.data_to_json import data_to_json
from coala.datasets.femnist.preprocess.get_file_dirs import get_file_dir
from coala.datasets.femnist.preprocess.get_hashes import get_hash
from coala.datasets.femnist.preprocess.group_by_writer import group_by_writer
from coala.datasets.femnist.preprocess.match_hashes import match_hash
from coala.datasets.utils.base_dataset import BaseDataset
from coala.datasets.utils.download import download_url, extract_archive
from coala.datasets.dataset import FederatedTensorDataset
from coala.datasets.femnist.data_process import process_x, process_y

logger = logging.getLogger(__name__)


class Femnist(BaseDataset):
    """FEMNIST dataset implementation. It gets FEMNIST dataset according to configurations.
     It stores the processed datasets locally.

    Attributes:
        base_folder (str): The base folder path of the dataset folder.
        class_url (str): The url to get the by_class split FEMNIST.
        write_url (str): The url to get the by_write split FEMNIST.
    """

    def __init__(self,
                 root,
                 fraction,
                 split_type,
                 user,
                 iid_user_fraction=0.1,
                 train_test_split=0.9,
                 minsample=10,
                 num_class=62,
                 num_of_client=100,
                 class_per_client=2,
                 setting_folder=None,
                 seed=-1,
                 **kwargs):
        super(Femnist, self).__init__(root,
                                      "femnist",
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
        self.class_url = "https://s3.amazonaws.com/nist-srd/SD19/by_class.zip"
        self.write_url = "https://s3.amazonaws.com/nist-srd/SD19/by_write.zip"
        self.packaged_data_files = {
            "femnist_niid_100_10_1_0.05_0.1_sample_0.9.zip": "https://dl.dropboxusercontent.com/s/oyhegd3c0pxa0tl/femnist_niid_100_10_1_0.05_0.1_sample_0.9.zip",
            "femnist_iid_100_10_1_0.05_0.1_sample_0.9.zip": "https://dl.dropboxusercontent.com/s/jcg0xrz5qrri4tv/femnist_iid_100_10_1_0.05_0.1_sample_0.9.zip"
        }
        # Google Drive ids
        # self.packaged_data_files = {
        #     "femnist_niid_100_10_1_0.05_0.1_sample_0.9.zip": "11vAxASl-af41iHpFqW2jixs1jOUZDXMS",
        #     "femnist_iid_100_10_1_0.05_0.1_sample_0.9.zip": "1U9Sn2ACbidwhhihdJdZPfK2YddPMr33k"
        # }

    def download_packaged_dataset_and_extract(self, filename):
        file_path = download_url(self.packaged_data_files[filename], self.base_folder)
        extract_archive(file_path, remove_finished=True)

    def download_raw_file_and_extract(self):
        raw_data_folder = os.path.join(self.base_folder, "raw_data")
        if not os.path.exists(raw_data_folder):
            os.makedirs(raw_data_folder)
        elif os.listdir(raw_data_folder):
            logger.info("raw file exists")
            return
        class_path = download_url(self.class_url, raw_data_folder)
        write_path = download_url(self.write_url, raw_data_folder)
        extract_archive(class_path, remove_finished=True)
        extract_archive(write_path, remove_finished=True)
        logger.info("raw file is downloaded")

    def preprocess(self):
        intermediate_folder = os.path.join(self.base_folder, "intermediate")
        if not os.path.exists(intermediate_folder):
            os.makedirs(intermediate_folder)
        if not os.path.exists(intermediate_folder + "/class_file_dirs.pkl"):
            logger.info("extracting file directories of images")
            get_file_dir(self.base_folder)
            logger.info("finished extracting file directories of images")
        if not os.path.exists(intermediate_folder + "/class_file_hashes.pkl"):
            logger.info("calculating image hashes")
            get_hash(self.base_folder)
            logger.info("finished calculating image hashes")
        if not os.path.exists(intermediate_folder + "/write_with_class.pkl"):
            logger.info("assigning class labels to write images")
            match_hash(self.base_folder)
            logger.info("finished assigning class labels to write images")
        if not os.path.exists(intermediate_folder + "/images_by_writer.pkl"):
            logger.info("grouping images by writer")
            group_by_writer(self.base_folder)
            logger.info("finished grouping images by writer")

    def convert_data_to_json(self):
        all_data_folder = os.path.join(self.base_folder, "all_data")
        if not os.path.exists(all_data_folder):
            os.makedirs(all_data_folder)
        if not os.listdir(all_data_folder):
            logger.info("converting data to .json format")
            data_to_json(self.base_folder)
            logger.info("finished converting data to .json format")


def construct_femnist_datasets(root,
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
    user_str = "user" if user else "sample"
    setting = BaseDataset.get_setting_folder(dataset_name, split_type, num_of_clients, min_size, class_per_client,
                                             data_amount, iid_fraction, user_str, train_test_split, alpha,
                                             quantity_weights)
    dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    dataset_file = os.path.join(dir_path, "data_process", "{}.py".format(dataset_name))
    if not os.path.exists(dataset_file):
        logger.error("Please specify a valid process file path for process_x and process_y functions.")

    data_dir = os.path.join(root, dataset_name)
    if not data_dir:
        os.makedirs(data_dir)
    train_data_dir = os.path.join(data_dir, setting, "train")
    test_data_dir = os.path.join(data_dir, setting, "test")

    if not os.path.exists(train_data_dir) or not os.path.exists(test_data_dir):
        dataset = Femnist(root=data_dir,
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
            filename = f"{setting}.zip"
            dataset.download_packaged_dataset_and_extract(filename)
            logger.info(f"Downloaded packaged dataset {dataset_name}: {filename}")
        except Exception as e:
            logger.info(f"Failed to download packaged dataset: {e.args}")

        if not os.path.exists(train_data_dir):
            dataset.setup()
            dataset.sampling()

    train_clients, train_groups, train_data = read_json_dir(train_data_dir)
    test_clients, test_groups, test_data = read_json_dir(test_data_dir)

    test_simulated = True

    train_data = FederatedTensorDataset(train_data,
                                        simulated=True,
                                        do_simulate=False,
                                        process_x=process_x,
                                        process_y=process_y,
                                        transform=None)
    test_data = FederatedTensorDataset(test_data,
                                       simulated=test_simulated,
                                       do_simulate=False,
                                       process_x=process_x,
                                       process_y=process_y,
                                       transform=None)

    return train_data, test_data


def read_json_dir(data_dir):
    clients = []
    groups = []
    data = {}

    files = os.listdir(data_dir)
    files = [f for f in files if f.endswith('.json')]
    for f in files:
        file_path = os.path.join(data_dir, f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        clients.extend(cdata['users'])
        if 'hierarchies' in cdata:
            groups.extend(cdata['hierarchies'])
        data.update(cdata['user_data'])

    clients = list(sorted(data.keys()))
    return clients, groups, data
