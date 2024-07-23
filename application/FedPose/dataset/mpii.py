import os
import json
import torchvision.transforms as transforms
from coala.datasets.utils.base_dataset import BaseDataset
from coala.datasets.utils.download import download_url, extract_archive
from coala.datasets.dataset import FederatedTorchDataset
from coala.datasets.bdd100k.data_split import *
from .JointsDataset import JointsDataset


logger = logging.getLogger(__name__)


class MPII(BaseDataset):
    """MPII dataset implementation. It stores the raw image paths locally.
       MPII has 22246 training images

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
                 alpha=0.5):
        super(MPII, self).__init__(root,
                                   "mpii",
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
        self.raw_data_folder = os.path.join(self.base_folder, "images")
        self.raw_label_folder = os.path.join(self.base_folder, "annot")

        self.raw_data_url = "https://datasets.d2.mpi-inf.mpg.de/andriluka14cvpr/mpii_human_pose_v1.tar.gz"
        self.annot_json_url = "https://cloud.tsinghua.edu.cn/f/b82b756013a244998305/?dl=1"

    def download_raw_file_and_extract(self):
        if os.path.exists(self.raw_data_folder) and os.path.exists(self.raw_label_folder):
            logger.info("raw file exists")
            return
        else:
            file_path = download_url(self.raw_data_url, self.base_folder, "images.tar.gz")
            extract_archive(file_path, remove_finished=True)
            logger.info("raw images is downloaded")

            annot_path = download_url(self.annot_json_url, self.base_folder, "annot.zip")
            extract_archive(annot_path, remove_finished=True)
            logger.info("annotation is downloaded")

    def data_splitting(self, data_dir, save_dir):
        if os.path.exists(save_dir):
            return
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        train_val = save_dir.split('/')[-1]

        # split training set
        with open(data_dir) as anno_file:
            data_anno = json.load(anno_file)

        if train_val == 'val':
            with open(os.path.join(save_dir, f'mpii_annot_{train_val}.json'), 'w') as f:
                json.dump(data_anno, f)
            return

        image_idx = [i for i in range(len(data_anno))]
        image_names = []
        for a in data_anno:
            image_names.append(a['image'])
        # split image id/name
        image_idx = np.array(image_idx)
        if self.split_type == 'iid':
            federated_idx = iid(image_idx, None, self.num_of_client)
        else:
            raise NotImplementedError

        for i in range(len(federated_idx)):
            user_anno = [data_anno[idx] for idx in federated_idx[i]]
            save_dir_i = os.path.join(save_dir, f"client{i}")
            os.makedirs(save_dir_i)
            with open(os.path.join(save_dir_i, f'mpii_annot_{train_val}_{i}.json'), 'w') as f:
                json.dump(user_anno, f)

    def setup(self):
        split_data_folder = os.path.join(self.base_folder, self.setting_folder)
        label_dir = self.raw_label_folder

        # dataset split based on images attributes
        train_dir = os.path.join(label_dir, 'train.json')
        train_idx_path = os.path.join(split_data_folder, 'train')
        self.data_splitting(train_dir, train_idx_path)

        test_dir = os.path.join(label_dir, 'valid.json')
        test_idx_path = os.path.join(split_data_folder, 'val')
        self.data_splitting(test_dir, test_idx_path)


def read_mpii_partitions(data_dir):
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
    clients = os.listdir(train_data_dir)
    for i in range(len(clients)):
        train_annot = os.path.join(train_data_dir, f'client{i}', f'mpii_annot_train_{i}.json')
        train_data[str(i)] = train_annot
    test_data = os.path.join(test_data_dir, f'mpii_annot_val.json')

    return train_data, test_data


def construct_mpii_datasets(conf,
                            root,
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
        dataset = MPII(root=data_dir,
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

    train_anno_path, val_anno_path = read_mpii_partitions(split_data_dir)

    assert len(train_anno_path) == num_of_clients, "the num of clients should match the split datasets"

    train_loaders = prepare_data(conf, data_dir, train_anno_path, is_train=True, transform=transform_train)
    val_loaders = prepare_data(conf, data_dir, val_anno_path, is_train=False, transform=transform_test)

    clients = [str(i) for i in range(num_of_clients)]

    train_data = FederatedTorchDataset(train_loaders, clients, is_loaded=False)
    val_data = FederatedTorchDataset(val_loaders, clients, is_loaded=False)

    return train_data, val_data


def prepare_data(conf, root, anno_paths, is_train=True, transform=None):
    if is_train:
        clients = [str(i) for i in range(len(anno_paths))]
        data_sets = {}
        for cid in clients:
            data_set = MPIIDataset(conf, root, anno_paths[cid], is_train, transform)
            data_sets[cid] = data_set
    else:
        data_sets = MPIIDataset(conf, root, anno_paths, is_train, transform)

    return data_sets

class MPIIDataset(JointsDataset):
    def __init__(self, cfg, root, data_path, is_train, transform=None):
        super().__init__(cfg, root, data_path, is_train, transform)

        self.num_joints = 16
        self.flip_pairs = [[0, 5], [1, 4], [2, 3], [10, 15], [11, 14], [12, 13]]
        self.parent_ids = [1, 2, 6, 6, 3, 4, 6, 6, 7, 8, 11, 12, 7, 7, 13, 14]

        self.db = self._get_db()

        if is_train and cfg.data.select_data:
            self.db = self.select_data(self.db)

        logger.info('=> load {} samples'.format(len(self.db)))

    def _get_db(self):
        # create train/val split
        # file_name = os.path.join(self.root,
        #                          'annot',
        #                          self.image_set + '.json')

        file_name = self.data_path
        with open(file_name) as anno_file:
            anno = json.load(anno_file)

        gt_db = []
        for a in anno:
            image_name = a['image']

            c = np.array(a['center'], dtype=np.float)
            s = np.array([a['scale'], a['scale']], dtype=np.float)

            # Adjust center/scale slightly to avoid cropping limbs
            if c[0] != -1:
                c[1] = c[1] + 15 * s[1]
                s = s * 1.25

            # MPII uses matlab format, index is based 1,
            # we should first convert to 0-based index
            c = c - 1

            joints_3d = np.zeros((self.num_joints, 3), dtype=np.float)
            joints_3d_vis = np.zeros((self.num_joints, 3), dtype=np.float)
            if self.image_set != 'test':
                joints = np.array(a['joints'])
                joints[:, 0:2] = joints[:, 0:2] - 1
                joints_vis = np.array(a['joints_vis'])
                assert len(joints) == self.num_joints, \
                    'joint num diff: {} vs {}'.format(len(joints),
                                                      self.num_joints)

                joints_3d[:, 0:2] = joints[:, 0:2]
                joints_3d_vis[:, 0] = joints_vis[:]
                joints_3d_vis[:, 1] = joints_vis[:]

            image_dir = 'images.zip@' if self.data_format == 'zip' else 'images'
            gt_db.append({
                'image': os.path.join(self.root, image_dir, image_name),
                'center': c,
                'scale': s,
                'joints_3d': joints_3d,
                'joints_3d_vis': joints_3d_vis,
                'filename': '',
                'imgnum': 0,
            })

        return gt_db

transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225]),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225]),
])