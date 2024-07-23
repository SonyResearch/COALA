import logging
from coala.datasets.utils.base_dataset import BaseDataset
from coala.datasets.utils.download import download_url, extract_archive
from coala.datasets.dataset import FederatedTensorDataset
from coala.datasets.dataset_util import ArrayImageDataset
from coala.datasets.digits_five.preprocess import *

logger = logging.getLogger(__name__)


class DigitsFive(BaseDataset):
    """Digits-Five dataset implementation. It stores the raw and pre-split datasets locally, respectively.

    Attributes:
        pre_split (Boolean): whether to use the pre-split dataset or not, default: True
        base_folder (str): The base folder path of the dataset folder.
        pre_split_url (str): The url to get the pre-split package of Digits-Five.
        raw_data_url (str): The url to get the raw sub-datasets of Digits-Five.
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
                 num_of_client=50,
                 class_per_client=10,
                 setting_folder=None,
                 seed=0,
                 weights=None,
                 alpha=0.5,
                 pre_split=True):
        super(DigitsFive, self).__init__(root,
                                         "digits_five",
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
        self.domain_list = ["MNIST", "SVHN", "USPS", "SynthDigits", "MNIST_M"]
        self.num_domain = len(self.domain_list)
        self.pre_split = pre_split

        self.pre_split_url = "https://www.dropbox.com/s/gxvis4tv47nwvw9/digit_dataset.zip?dl=1"
        self.raw_data_url = "https://www.dropbox.com/s/mxwxb0j1kyy63ca/raw_data.zip?dl=1"
        self.packaged_file = "digits_split.zip"
        self.raw_file = "raw_data.zip"
        if self.pre_split:
            assert self.num_of_client == 50, "pre-split dataset contains 10 clients for each of 5 domains"

    def download_packaged_dataset_and_extract(self, filename=None):

        packaged_data_folder = os.path.join(self.base_folder, "pre_split")
        if not os.path.exists(packaged_data_folder):
            os.makedirs(packaged_data_folder)
        elif os.listdir(packaged_data_folder):
            logger.info("packaged file exists")
            return
        file_path = download_url(self.pre_split_url, packaged_data_folder, self.packaged_file)
        extract_archive(file_path, remove_finished=True)

        # further split the test set and rename the training path
        domains = ['MNIST', 'SVHN', 'USPS', 'SynthDigits', 'MNIST_M']
        for dom in domains:
            train_path = os.path.join(packaged_data_folder, dom, "partitions")
            train_save = os.path.join(packaged_data_folder, dom, "train")
            os.rename(train_path, train_save)
            test_file = os.path.join(packaged_data_folder, dom)
            test_save = os.path.join(packaged_data_folder, dom)
            num_client_per_domain = 10
            in_domain_split(test_file, test_save, num_parts=num_client_per_domain, filename='test')

    def download_raw_file_and_extract(self):

        raw_data_folder = os.path.join(self.base_folder, "raw")
        if not os.path.exists(raw_data_folder):
            os.makedirs(raw_data_folder)
        elif os.listdir(raw_data_folder):
            logger.info("raw file exists")
            return

        file_path = download_url(self.raw_data_url, raw_data_folder, self.raw_file)
        extract_archive(file_path, remove_finished=True)
        logger.info("raw file is downloaded")

    def pre_process(self):
        logger.info("Digits_Five Raw Data Processing...")
        if not os.path.exists(os.path.join(self.base_folder, "raw", "MNIST/train.pkl")):
            process_mnist()

        if not os.path.exists(os.path.join(self.base_folder, "raw", "SVHN/train.pkl")):
            process_svhn()

        if not os.path.exists(os.path.join(self.base_folder, "raw", "USPS/train.pkl")):
            process_usps()

        if not os.path.exists(os.path.join(self.base_folder, "raw", "SynthDigits/train.pkl")):
            process_synth()

        if not os.path.exists(os.path.join(self.base_folder, "raw", "MNIST_M/train.pkl")):
            process_mnistm()

    def data_splitting(self, setting):
        # raw data pre-processing
        self.pre_process()
        logger.info("Raw data pre-processing accomplished")

        # raw-data splitting into distributed sets
        raw_data_folder = os.path.join(self.base_folder, "raw")
        processed_data_folder = os.path.join(self.base_folder, setting)
        raw_paths = [
            '{}/MNIST'.format(raw_data_folder),
            '{}/SVHN'.format(raw_data_folder),
            '{}/USPS'.format(raw_data_folder),
            '{}/SynthDigits'.format(raw_data_folder),
            '{}/MNIST_M'.format(raw_data_folder),
        ]
        save_paths = [
            '{}/MNIST'.format(processed_data_folder),
            '{}/SVHN'.format(processed_data_folder),
            '{}/USPS'.format(processed_data_folder),
            '{}/SynthDigits'.format(processed_data_folder),
            '{}/MNIST_M'.format(processed_data_folder),
        ]

        num_client_per_domain = int(self.num_of_client / len(raw_paths))
        for raw_dir, save_dir in zip(raw_paths, save_paths):
            logger.info('Splitting {}...'.format(os.path.basename(raw_dir)))
            in_domain_split(raw_dir, save_dir, num_parts=num_client_per_domain, filename='train')
            in_domain_split(raw_dir, save_dir, num_parts=num_client_per_domain, filename='test')

    def convert_data_to_json(self):
        pass


def read_digits_five_partitions(data_dir):
    """Load datasets from data directories.

    Args:
        data_dir (str): The directory of data, e.g., ..data/dataset_name/setting

    Returns:
        dict: A dictionary of training data, e.g., {"id1": {"x": data, "y": label}, "id2": {"x": data, "y": label}}.
        dict: A dictionary of testing data.
        function: A function to preprocess training data.
        function: A function to preprocess testing data.
        dict: A collection of per-client torchvision.transforms.transforms.Compose: Training data transformation.
        dict: A collection of per-client torchvision.transforms.transforms.Compose: Testing data transformation.
    """
    domains = ['MNIST', 'SVHN', 'USPS', 'SynthDigits', 'MNIST_M']

    transform_train_dict = transform_train_digits
    transform_test_dict = transform_test_digits

    train_data = {}
    test_data = {}
    transform_train = {}
    transform_test = {}
    client_id = 0
    for dom in domains:
        train_path = os.path.join(data_dir, dom, 'train')
        test_path = os.path.join(data_dir, dom, 'test')
        files = os.listdir(train_path)
        for i in range(len(files)):
            # load training set
            train_x, train_y = np.load(os.path.join(train_path, 'train_part{}.pkl'.format(i)), allow_pickle=True)
            train_data[str(client_id)] = {'x': train_x, 'y': train_y}
            transform_train[str(client_id)] = transform_train_dict[dom]
            # load test set
            test_x, test_y = np.load(os.path.join(test_path, 'test_part{}.pkl'.format(i)), allow_pickle=True)
            test_data[str(client_id)] = {'x': test_x, 'y': test_y}
            transform_test[str(client_id)] = transform_test_dict[dom]
            # update the client id (total num of clients)
            client_id += 1

    return train_data, test_data, transform_train, transform_test


def construct_digits_five_datasets(root,
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
    if num_of_clients != 50:
        pre_split = False
    if pre_split:
        setting = "pre_split"

    data_dir = os.path.join(root, dataset_name)
    if not data_dir:
        os.makedirs(data_dir)
    split_data_dir = os.path.join(data_dir, setting)

    if not os.path.exists(split_data_dir):
        dataset = DigitsFive(root=data_dir,
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
            filename = dataset.raw_file
            logger.info(f"Downloaded raw dataset {dataset_name}: {filename}")
        except Exception as e:
            logger.info(f"Failed to download raw dataset: {e.args}")

        dataset.data_splitting(setting)
        logger.info(f"data splitting accomplished")

        if pre_split:
            try:
                dataset.download_packaged_dataset_and_extract()
                filename = dataset.packaged_file
                logger.info(f"Downloaded packaged dataset {dataset_name}: {filename}")
            except Exception as e:
                logger.info(f"Failed to download packaged dataset: {e.args}")

    train_data, test_data, transform_train, transform_test = read_digits_five_partitions(split_data_dir)

    assert len(train_data) == num_of_clients, "the num of clients should match the pre-split files"

    train_data = FederatedTensorDataset(train_data,
                                        simulated=True,
                                        do_simulate=False,
                                        transform=transform_train,
                                        process_x=None,
                                        process_y=None,
                                        data_wrapping=ArrayImageDataset)

    test_data = FederatedTensorDataset(test_data,
                                       simulated=True,
                                       do_simulate=False,
                                       transform=transform_test,
                                       process_x=None,
                                       process_y=None,
                                       data_wrapping=ArrayImageDataset)

    return train_data, test_data
