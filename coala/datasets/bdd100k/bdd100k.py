import pickle as pkl
from coala.datasets.utils.base_dataset import BaseDataset
from coala.datasets.dataset import FederatedTorchDataset
from coala.datasets.bdd100k.data_stats import *
from coala.datasets.bdd100k.data_split import *
from coala.datasets.bdd100k.label_convert import *
from coala.datasets.bdd100k.dataload import create_dataloader
from coala.datasets.bdd100k.bdd100k_utils import colorstr

logger = logging.getLogger(__name__)


class BDD100K(BaseDataset):
    """BDD100K dataset implementation. It stores the raw image paths locally.

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
        super(BDD100K, self).__init__(root,
                                      "bdd100k",
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
        self.weather = ["rainy", "snowy", "clear", "overcast", "partly cloudy", "foggy"]
        self.scene = ["tunnel", "residential", "parking lot", "city street", "gas stations", "highway"]
        self.timeofday = ["daytime", "night", "dawn/dusk"]
        self.raw_data_folder = os.path.join(self.base_folder, "images/100k")
        self.raw_label_folder = os.path.join(self.base_folder, "labels")

    def download_raw_file_and_extract(self):
        if os.path.exists(self.raw_data_folder) and os.path.exists(self.raw_label_folder):
            logger.info("raw file exists")
            return
        else:
            raise FileNotFoundError

    def data_splitting(self, data_dir, save_dir):
        if os.path.exists(save_dir):
            return
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # split training set
        with open(data_dir) as f:
            train_labels = json.load(f)

        weather_to_id = {"rainy": 0, "snowy": 1, "clear": 2, "overcast": 3, "partly cloudy": 4, "foggy": 5}
        scene_to_id = {"tunnel": 0, "residential": 1, "parking lot": 2, "city street": 3, "gas stations": 4,
                       "highway": 5}
        time_to_id = {"daytime": 0, "night": 1, "dawn/dusk": 2}

        counter = 0
        image_idx = []
        image_names = []
        weathers = []
        scenes = []
        timeofdays = []
        # attributes collecting
        for img in train_labels:
            weather = img['attributes']['weather']
            scene = img['attributes']['scene']
            timeofday = img['attributes']['timeofday']
            if weather in self.weather and scene in self.scene and timeofday in self.timeofday:
                image_idx.append(counter)
                image_names.append(img['name'])
                weathers.append(weather_to_id[weather])
                scenes.append(scene_to_id[scene])
                timeofdays.append(time_to_id[timeofday])
                counter += 1
        # split image id/name based on the distributions of three attributes
        image_idx = np.array(image_idx)
        image_type = np.vstack([weathers, scenes, timeofdays]).transpose()  # shape: (datasize, 3)
        if self.split_type == 'iid':
            federated_idx = iid(image_idx, image_type[:, 0], self.num_of_client)
        elif self.split_type == 'dir':
            federated_idx = non_iid_dirichlet(image_idx, image_type[:, 0], self.num_of_client, self.alpha,
                                              self.min_size)
        else:
            federated_idx = non_iid_dirichlet_hierarchical(image_idx, image_type, self.num_of_client, self.alpha,
                                                           self.alpha, self.alpha, self.min_size)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        train_val = save_dir.split('/')[-1]
        if train_val == "train":
            # save federated_idx and image_type as pkl
            split_data_folder = os.path.join(self.base_folder, self.setting_folder)
            with open(os.path.join(split_data_folder, 'federated_idx_attributes.pkl'), 'wb') as f:
                pkl.dump((federated_idx, image_type), f, pkl.HIGHEST_PROTOCOL)
            # save image distribution statistics
            data_distribution_display(image_type, federated_idx, show=False, save_path=split_data_folder)
            pairwise_divergence(image_type, federated_idx, show=False, save_path=split_data_folder)
            datasize_distribution(federated_idx, show=False, save_path=split_data_folder)

        for i in range(len(federated_idx)):
            user_img = [image_names[idx] for idx in federated_idx[i]]
            data_x = [os.path.join(self.raw_data_folder, train_val, x) for x in user_img]
            data_y = [os.path.join(self.raw_label_folder, train_val, x.replace(os.path.splitext(x)[-1], '.txt')) for x
                      in user_img]
            save_dir_i = os.path.join(save_dir, f"client{i}")
            os.makedirs(save_dir_i)
            with open(os.path.join(save_dir_i, f'bdd100k_image_{train_val}_{i}.txt'), 'w') as f:
                f.write('\n'.join(data_x))
            with open(os.path.join(save_dir_i, f'bdd100k_label_{train_val}_{i}.txt'), 'w') as f:
                f.write('\n'.join(data_y))

    def format_convert(self):
        label_dir = os.path.join(self.base_folder, "labels")
        coco_label_dir = os.path.join(self.base_folder, "coco_labels")
        yolo_label_dir = os.path.join(self.base_folder, "labels/train")
        if not os.path.exists(coco_label_dir):
            bdd_to_coco_convert(label_dir, coco_label_dir)
        if not os.path.exists(yolo_label_dir):
            coco_to_yolo_convert(label_dir, coco_label_dir)

    def setup(self):
        split_data_folder = os.path.join(self.base_folder, self.setting_folder)
        label_dir = self.raw_label_folder

        # dataset split based on images attributes
        train_dir = os.path.join(label_dir, 'bdd100k_labels_images_train.json')
        train_idx_path = os.path.join(split_data_folder, 'train')
        self.data_splitting(train_dir, train_idx_path)

        test_dir = os.path.join(label_dir, 'bdd100k_labels_images_val.json')
        test_idx_path = os.path.join(split_data_folder, 'val')
        self.data_splitting(test_dir, test_idx_path)

        # convert original label format
        self.format_convert()


def read_bdd100k_partitions(data_dir):
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
        train_x = os.path.join(train_data_dir, f'client{i}', f'bdd100k_image_train_{i}.txt')
        train_y = os.path.join(train_data_dir, f'client{i}', f'bdd100k_label_train_{i}.txt')
        train_data[str(i)] = {'x': train_x, 'y': train_y}

        val_x = os.path.join(test_data_dir, f'client{i}', f'bdd100k_image_val_{i}.txt')
        val_y = os.path.join(test_data_dir, f'client{i}', f'bdd100k_label_val_{i}.txt')
        test_data[str(i)] = {'x': val_x, 'y': val_y}

    return train_data, test_data


def construct_bdd100k_datasets(root,
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
                               batch_size=32,
                               test_batch_size=64,
                               imagesize=640,
                               grid_size=32,
                               hyp_kws=None,
                               cache="ram"
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
        dataset = BDD100K(root=data_dir,
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

    train_path, val_path = read_bdd100k_partitions(split_data_dir)

    assert len(train_path) == num_of_clients, "the num of clients should match the split datasets"

    train_loaders = prepare_data(train_path, hyp_kws, imagesize, grid_size, batch_size=batch_size, workers=1,
                                 cache=cache, nc=13, train=True)
    val_loaders = prepare_data(val_path, hyp_kws, imagesize, grid_size, batch_size=test_batch_size, workers=1,
                               cache=cache, nc=13, train=False)

    clients = [str(i) for i in range(num_of_clients)]

    train_data = FederatedTorchDataset(train_loaders, clients, is_loaded=True)
    val_data = FederatedTorchDataset(val_loaders, clients, is_loaded=True)

    return train_data, val_data


def prepare_data(data_sets, hyp, imgsz, gs, batch_size=64, workers=1, cache="ram", nc=13, train=True):
    clients = [str(i) for i in range(len(data_sets))]
    data_loaders = {}
    for cid in clients:
        data_loader = create_dataloader(data_sets[cid],
                                        imgsz,
                                        batch_size,
                                        gs,
                                        hyp=hyp,
                                        augment=True if train else False,
                                        cache=cache,
                                        rect=False if train else True,
                                        workers=workers,
                                        prefix=colorstr('train: ') if train else colorstr('val: '),
                                        shuffle=True if train else False
                                        )
        labels = np.concatenate(data_loader.dataset.labels, 0)
        mlc = int(labels[:, 0].max())  # max label class
        assert mlc < nc, f'Label class {mlc} exceeds nc={nc}. Possible class labels are 0-{nc - 1}'
        data_loaders[cid] = data_loader

    return data_loaders
