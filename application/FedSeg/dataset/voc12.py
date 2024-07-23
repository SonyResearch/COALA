import os
from coala.datasets.utils.base_dataset import BaseDataset
from coala.datasets.dataset import FederatedTensorDataset
from coala.datasets.dataset_util import ImageSegDataset
from coala.datasets.bdd100k.data_split import *
from .transform import train_transform, test_transform
from .utils import load_files_paths
from PIL import Image

logger = logging.getLogger(__name__)

classes = {
    0: 'background',
    1: 'aeroplane',
    2: 'bicycle',
    3: 'bird',
    4: 'boat',
    5: 'bottle',
    6: 'bus',
    7: 'car',
    8: 'cat',
    9: 'chair',
    10: 'cow',
    11: 'diningtable',
    12: 'dog',
    13: 'horse',
    14: 'motorbike',
    15: 'person',
    16: 'pottedplant',
    17: 'sheep',
    18: 'sofa',
    19: 'train',
    20: 'tvmonitor'
}


class PascalVoC2012_Segmentation(BaseDataset):
    """PascalVoC2012 dataset implementation. It stores the raw image paths locally.

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
                 num_class=20,
                 num_of_client=4,
                 class_per_client=10,
                 setting_folder=None,
                 seed=-1,
                 weights=None,
                 alpha=0.5):
        super(PascalVoC2012_Segmentation, self).__init__(root,
                                                         "voc12",
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
        self.raw_data_folder = os.path.join(self.base_folder, "JPEGImages")
        self.raw_mask_folder = os.path.join(self.base_folder, "SegmentationClass")
        self.train_file = os.path.join(self.base_folder, "ImageSets/Segmentation/train.txt")
        self.val_file = os.path.join(self.base_folder, "ImageSets/Segmentation/val.txt")
        self.categories = ['__background__', 'airplane', 'bicycle', 'bird', 'boat', 'bottle',
                           'bus', 'car', 'cat', 'chair', 'cow', 'dining table', 'dog', 'horse',
                           'motorcycle', 'person', 'potted-plant', 'sheep', 'sofa', 'television',
                           'train']
        self.classes = [i for i in range(len(self.categories))]

    def download_raw_file_and_extract(self):
        if os.path.exists(self.raw_data_folder) and os.path.exists(self.raw_mask_folder):
            logger.info("raw file exists")
            return
        else:
            raise FileNotFoundError

    def data_splitting(self, data_file, save_dir):
        if os.path.exists(save_dir):
            return
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        train_val = save_dir.split('/')[-1]

        images = []
        masks = []
        # split training set
        with open(data_file, 'r') as f:
            image_names = f.read().strip().splitlines()
            for image_name in image_names:
                images.append(os.path.join(self.raw_data_folder, image_name + '.jpg'))
                masks.append(os.path.join(self.raw_mask_folder, image_name + '.png'))

        counter = 0
        image_idx = []
        image_class = []
        # label collecting, only consider the first category
        for img_mask in masks:
            mask = Image.open(img_mask)
            labels = np.unique(np.array(mask, dtype=np.uint8))
            if len(labels) > 1 and labels[1] < 255:
                image_class.append(labels[1])
            else:
                image_class.append(labels[0])
            image_idx.append(counter)
            counter += 1

        # split image id/name based on the distributions of classes
        image_idx = np.array(image_idx)
        image_class = np.array(image_class)
        if self.split_type == 'iid':
            federated_idx = iid(image_idx, image_class, self.num_of_client)
        elif self.split_type == 'dir':
            federated_idx = non_iid_dirichlet(image_idx, image_class, self.num_of_client, self.alpha,
                                              self.min_size)
        else:
            raise NotImplementedError

        for i in range(len(federated_idx)):
            data_x = [images[idx] for idx in federated_idx[i]]
            data_y = [masks[idx] for idx in federated_idx[i]]
            save_dir_i = os.path.join(save_dir, f"client{i}")
            os.makedirs(save_dir_i)
            with open(os.path.join(save_dir_i, f'voc12_seg_image_{train_val}_{i}.txt'), 'w') as f:
                f.write('\n'.join(data_x))
            with open(os.path.join(save_dir_i, f'voc12_seg_label_{train_val}_{i}.txt'), 'w') as f:
                f.write('\n'.join(data_y))

        if train_val == 'val':
            with open(os.path.join(save_dir, f'voc12_seg_image_{train_val}.txt'), 'w') as f:
                f.write('\n'.join(images))
            with open(os.path.join(save_dir, f'voc12_seg_label_{train_val}.txt'), 'w') as f:
                f.write('\n'.join(masks))

    def setup(self):
        split_data_folder = os.path.join(self.base_folder, self.setting_folder)
        # dataset split based on object classes
        train_idx_path = os.path.join(split_data_folder, 'train')
        self.data_splitting(self.train_file, train_idx_path)
        test_idx_path = os.path.join(split_data_folder, 'val')
        self.data_splitting(self.val_file, test_idx_path)


def read_voc12_partitions(data_dir):
    """Load datasets from data directories.

    Args:
        data_dir (str): The directory of data, e.g., ..data/dataset_name/setting

    Returns:
        dict: A dictionary of training data, e.g., {"id1": {"x": data_path, "y": label_path}}.
        dict: A dictionary of testing data.
    """
    train_data_dir = os.path.join(data_dir, 'train')
    test_data_dir = os.path.join(data_dir, 'val')
    train_data = {}
    test_data = {}
    clients = os.listdir(train_data_dir)
    for i in range(len(clients)):
        train_x_path = os.path.join(train_data_dir, f'client{i}', f'voc12_seg_image_train_{i}.txt')
        train_y_path = os.path.join(train_data_dir, f'client{i}', f'voc12_seg_label_train_{i}.txt')
        train_x = load_files_paths(train_x_path)
        train_y = load_files_paths(train_y_path)
        train_data[str(i)] = {'x': train_x, 'y': train_y}

        val_x_path = os.path.join(test_data_dir, f'client{i}', f'voc12_seg_image_val_{i}.txt')
        val_y_path = os.path.join(test_data_dir, f'client{i}', f'voc12_seg_label_val_{i}.txt')
        val_x = load_files_paths(val_x_path)
        val_y = load_files_paths(val_y_path)
        test_data[str(i)] = {'x': val_x, 'y': val_y}

    val_x_path = os.path.join(test_data_dir, 'voc12_seg_image_val.txt')
    val_y_path = os.path.join(test_data_dir, 'voc12_seg_label_val.txt')
    val_x = load_files_paths(val_x_path)
    val_y = load_files_paths(val_y_path)
    global_val_data = {'x': val_x, 'y': val_y}

    return train_data, test_data, global_val_data


def construct_voc12_datasets(root,
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
                             ):
    user_str = "sample"
    setting = BaseDataset.get_setting_folder(dataset_name, split_type, num_of_clients, min_size, class_per_client,
                                             data_amount, iid_fraction, user_str, train_test_split, alpha,
                                             quantity_weights)

    data_dir = os.path.join(root, dataset_name)
    if not data_dir:
        os.makedirs(data_dir)
    split_data_dir = os.path.join(data_dir, setting)

    if not os.path.exists(split_data_dir):
        dataset = PascalVoC2012_Segmentation(root=data_dir,
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
            logger.info(f"Downloaded raw dataset {dataset_name}")
        except Exception as e:
            logger.info(f"Please download raw dataset {dataset_name} manually: {e.args}")

        dataset.setup()
        logger.info(f"data splitting accomplished")

    train_sets, val_sets, global_val = read_voc12_partitions(split_data_dir)

    assert len(train_sets) == num_of_clients, "the num of clients should match the split datasets"

    train_data = FederatedTensorDataset(train_sets,
                                        simulated=True,
                                        do_simulate=False,
                                        process_x=None,
                                        process_y=None,
                                        transform=train_transform,
                                        data_wrapping=ImageSegDataset)

    val_data = FederatedTensorDataset(val_sets,
                                      simulated=True,
                                      do_simulate=False,
                                      process_x=None,
                                      process_y=None,
                                      transform=test_transform,
                                      data_wrapping=ImageSegDataset)

    return train_data, val_data
