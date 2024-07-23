import os
import pickle as pkl
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')

transform_train_office = transforms.Compose([
    transforms.Resize([256, 256]),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation((-30, 30)),
    transforms.ToTensor(),
])

transform_test_office = transforms.Compose([
    transforms.Resize([256, 256]),
    transforms.ToTensor(),
])


def in_domain_split(data_set, save_path, num_parts=1, train_ratio=0.8, split_type='iid'):
    """
    split each single dataset into multiple partitions for client scaling training
    """
    images = data_set['x']
    labels = data_set['y']

    part_len = len(images) / num_parts  # data size for each domain is 958, 1123, 157, 295, choose the smallest value

    save_path_train = os.path.join(save_path, 'train')
    save_path_test = os.path.join(save_path, 'test')
    if not os.path.exists(save_path_train):
        os.makedirs(save_path_train)
    if not os.path.exists(save_path_test):
        os.makedirs(save_path_test)

    for num in range(num_parts):
        images_part = images[int(part_len * num):int(part_len * (num + 1))]
        labels_part = labels[int(part_len * num):int(part_len * (num + 1))]

        x_train, x_test, y_train, y_test = train_test_split(images_part, labels_part, train_size=train_ratio)

        with open(os.path.join(save_path_train, f'train_part{num}.pkl'), 'wb') as f:
            pkl.dump((x_train, y_train), f, pkl.HIGHEST_PROTOCOL)

        with open(os.path.join(save_path_test, f'test_part{num}.pkl'), 'wb') as f:
            pkl.dump((x_test, y_test), f, pkl.HIGHEST_PROTOCOL)
