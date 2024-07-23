"""
This file is used to pre-process all data in Digit-5 dataset.
i.e., splitted data into train&test set  in a stratified way.
The function to process data into 10 partitions is also provided.
"""

import sys, os

base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)

import torch
import pickle as pkl
import scipy.io as scio
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from collections import Counter
import torchvision.transforms as transforms

raw_base_folder = "./data/digits_five/raw"


transform_mnist = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_svhn = transforms.Compose([
    transforms.Resize([28, 28]),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_usps = transforms.Compose([
    transforms.Resize([28, 28]),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_synth = transforms.Compose([
    transforms.Resize([28, 28]),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_mnistm = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_train_digits = {
    "MNIST": transform_mnist,
    "SVHN": transform_svhn,
    "USPS": transform_usps,
    "SynthDigits": transform_synth,
    "MNIST_M": transform_mnistm,
}

transform_test_digits = {
    "MNIST": transform_mnist,
    "SVHN": transform_svhn,
    "USPS": transform_usps,
    "SynthDigits": transform_synth,
    "MNIST_M": transform_mnistm,
}



def stratified_split(X, y):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)

    for train_index, test_index in sss.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        print('Train:', Counter(y_train))
        print('Test:', Counter(y_test))

    return (X_train, y_train), (X_test, y_test)


def process_mnist():
    """
    train:
    (56000, 28, 28)
    (56000,)
    test:
    (14000, 28, 28)
    (14000,)
    """
    mnist_train = os.path.join(raw_base_folder, "MNIST/training.pt")
    mnist_test = os.path.join(raw_base_folder, "MNIST/test.pt")
    train = torch.load(mnist_train)
    test = torch.load(mnist_test)

    train_img = train[0].numpy()
    train_tar = train[1].numpy()

    test_img = test[0].numpy()
    test_tar = test[1].numpy()

    all_img = np.concatenate([train_img, test_img])
    all_tar = np.concatenate([train_tar, test_tar])

    train_stratified, test_stratified = stratified_split(all_img, all_tar)
    print('# After spliting:')
    print('Train imgs:\t', train_stratified[0].shape)
    print('Train labels:\t', train_stratified[1].shape)
    print('Test imgs:\t', test_stratified[0].shape)
    print('Test labels:\t', test_stratified[1].shape)

    with open(os.path.join(raw_base_folder, "MNIST/train.pkl"), 'wb') as f:
        pkl.dump(train_stratified, f, pkl.HIGHEST_PROTOCOL)

    with open(os.path.join(raw_base_folder, "MNIST/test.pkl"), 'wb') as f:
        pkl.dump(test_stratified, f, pkl.HIGHEST_PROTOCOL)


def process_svhn():
    """
    train:
    (79431, 32, 32, 3)
    (79431,)
    test:
    (19858, 32, 32, 3)
    (19858,)
    """
    train = scio.loadmat('{}/SVHN/train_32x32.mat'.format(raw_base_folder))
    test = scio.loadmat('{}/SVHN/test_32x32.mat'.format(raw_base_folder))

    train_img = train['X']
    train_tar = train['y'].astype(np.int64).squeeze()

    test_img = test['X']
    test_tar = test['y'].astype(np.int64).squeeze()

    train_img = np.transpose(train_img, (3, 0, 1, 2))
    test_img = np.transpose(test_img, (3, 0, 1, 2))

    np.place(train_tar, train_tar == 10, 0)
    np.place(test_tar, test_tar == 10, 0)

    all_img = np.concatenate([train_img, test_img])
    all_tar = np.concatenate([train_tar, test_tar])

    train_stratified, test_stratified = stratified_split(all_img, all_tar)
    print('# After spliting:')
    print('Train imgs:\t', train_stratified[0].shape)
    print('Train labels:\t', train_stratified[1].shape)
    print('Test imgs:\t', test_stratified[0].shape)
    print('Test labels:\t', test_stratified[1].shape)

    with open('{}/SVHN/train.pkl'.format(raw_base_folder), 'wb') as f:
        pkl.dump(train_stratified, f, pkl.HIGHEST_PROTOCOL)

    with open('{}/SVHN/test.pkl'.format(raw_base_folder), 'wb') as f:
        pkl.dump(test_stratified, f, pkl.HIGHEST_PROTOCOL)


def process_usps():
    """
    train:
    (7438, 16, 16)
    (7438,)
    test:
    (1860, 16, 16)
    (1860,)
    :return:
    """
    import bz2
    train_path = '{}/USPS/usps.bz2'.format(raw_base_folder)
    with bz2.open(train_path) as fp:
        raw_data = [l.decode().split() for l in fp.readlines()]
    imgs = [[x.split(':')[-1] for x in data[1:]] for data in raw_data]
    imgs = np.asarray(imgs, dtype=np.float32).reshape((-1, 16, 16))
    imgs = ((imgs + 1) / 2 * 255).astype(dtype=np.uint8)
    targets = [int(d[0]) - 1 for d in raw_data]

    train_img = imgs
    train_tar = np.array(targets)

    test_path = '{}/USPS/usps.t.bz2'.format(raw_base_folder)
    with bz2.open(test_path) as fp:
        raw_data = [l.decode().split() for l in fp.readlines()]
    imgs = [[x.split(':')[-1] for x in data[1:]] for data in raw_data]
    imgs = np.asarray(imgs, dtype=np.float32).reshape((-1, 16, 16))
    imgs = ((imgs + 1) / 2 * 255).astype(dtype=np.uint8)
    targets = [int(d[0]) - 1 for d in raw_data]

    test_img = imgs
    test_tar = np.array(targets)

    all_img = np.concatenate([train_img, test_img])
    all_tar = np.concatenate([train_tar, test_tar])

    train_stratified, test_stratified = stratified_split(all_img, all_tar)
    print('# After spliting:')
    print('Train imgs:\t', train_stratified[0].shape)
    print('Train labels:\t', train_stratified[1].shape)
    print('Test imgs:\t', test_stratified[0].shape)
    print('Test labels:\t', test_stratified[1].shape)

    with open('{}/USPS/train.pkl'.format(raw_base_folder), 'wb') as f:
        pkl.dump(train_stratified, f, pkl.HIGHEST_PROTOCOL)

    with open('{}/USPS/test.pkl'.format(raw_base_folder), 'wb') as f:
        pkl.dump(test_stratified, f, pkl.HIGHEST_PROTOCOL)


def process_synth():
    """
    (391162, 32, 32, 3)
    (391162,)
    (97791, 32, 32, 3)
    (97791,)
    """
    train = scio.loadmat('{}/SynthDigits/synth_train_32x32.mat'.format(raw_base_folder))
    test = scio.loadmat('{}/SynthDigits/synth_test_32x32.mat'.format(raw_base_folder))

    train_img = train['X']
    train_tar = train['y'].astype(np.int64).squeeze()

    test_img = test['X']
    test_tar = test['y'].astype(np.int64).squeeze()

    train_img = np.transpose(train_img, (3, 0, 1, 2))
    test_img = np.transpose(test_img, (3, 0, 1, 2))

    all_img = np.concatenate([train_img, test_img])
    all_tar = np.concatenate([train_tar, test_tar])

    train_stratified, test_stratified = stratified_split(all_img, all_tar)
    print('# After spliting:')
    print('Train imgs:\t', train_stratified[0].shape)
    print('Train labels:\t', train_stratified[1].shape)
    print('Test imgs:\t', test_stratified[0].shape)
    print('Test labels:\t', test_stratified[1].shape)

    with open('{}/SynthDigits/train.pkl'.format(raw_base_folder), 'wb') as f:
        pkl.dump(train_stratified, f, pkl.HIGHEST_PROTOCOL)

    with open('{}/SynthDigits/test.pkl'.format(raw_base_folder), 'wb') as f:
        pkl.dump(test_stratified, f, pkl.HIGHEST_PROTOCOL)


def process_mnistm():
    """
    (56000, 28, 28, 3)
    (56000,)
    (14000, 28, 28, 3)
    (14000,)
    :return:
    """
    data = np.load('{}/MNIST_M/mnistm_data.pkl'.format(raw_base_folder), allow_pickle=True)
    train_img = data['train']
    train_tar = data['train_label']
    valid_img = data['valid']
    valid_tar = data['valid_label']
    test_img = data['test']
    test_tar = data['test_label']

    all_img = np.concatenate([train_img, valid_img, test_img])
    all_tar = np.concatenate([train_tar, valid_tar, test_tar])

    train_stratified, test_stratified = stratified_split(all_img, all_tar)
    print('# After spliting:')
    print('Train imgs:\t', train_stratified[0].shape)
    print('Train labels:\t', train_stratified[1].shape)
    print('Test imgs:\t', test_stratified[0].shape)
    print('Test labels:\t', test_stratified[1].shape)

    with open('{}/MNIST_M/train.pkl'.format(raw_base_folder), 'wb') as f:
        pkl.dump(train_stratified, f, pkl.HIGHEST_PROTOCOL)

    with open('{}/MNIST_M/test.pkl'.format(raw_base_folder), 'wb') as f:
        pkl.dump(test_stratified, f, pkl.HIGHEST_PROTOCOL)


def in_domain_split(raw_path, save_path, num_parts=10, filename='train', split_type='iid'):
    """
    split each single dataset into multiple partitions for client scaling training
    each part remain the same size according to the smallest datasize (i.e. 743 for num_client=10)
    filename: sub-list of ["train", "test"]
    """
    images, labels = np.load(os.path.join(raw_path, f"{filename}.pkl"), allow_pickle=True)
    part_len = 7438 / num_parts if filename == "train" else min(len(images), 10000) / num_parts

    save_path_file = os.path.join(save_path, filename)
    if not os.path.exists(save_path_file):
        os.makedirs(save_path_file)

    for num in range(num_parts):
        images_part = images[int(part_len * num):int(part_len * (num + 1)), :, :]
        labels_part = labels[int(part_len * num):int(part_len * (num + 1))]

        with open(os.path.join(save_path_file, f'{filename}_part{num}.pkl'), 'wb') as f:
            pkl.dump((images_part, labels_part), f, pkl.HIGHEST_PROTOCOL)
