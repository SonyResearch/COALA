import os
import pickle as pkl
import torchvision.transforms as transforms

transform_train_domainnet = transforms.Compose([
    transforms.Resize([256, 256]),
    # transforms.Resize([224, 224]),  # for parameter efficient finetuning of ViT
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation((-30, 30)),
    transforms.ToTensor(),
])

transform_test_domainnet = transforms.Compose([
    transforms.Resize([256, 256]),
    # transforms.Resize([224, 224]),  # for parameter efficient finetuning of ViT
    transforms.ToTensor(),
])


def in_domain_split(data_set, save_path, num_parts=10, train=True, split_type='iid'):
    """
    split each single dataset into multiple partitions for client scaling training
    each part remain the same size according to the smallest datasize
    filename: sub-list of ["train", "test"]
    """
    if train:
        filename = "train"
    else:
        filename = "test"

    images = data_set['x']
    labels = data_set['y']

    # training data size for each domain is 2103, 2626, 2472, 4000, 4864, 2213
    part_len = 1000 / num_parts if train else len(images) / num_parts

    save_path_file = os.path.join(save_path, filename)
    if not os.path.exists(save_path_file):
        os.makedirs(save_path_file)

    for num in range(num_parts):
        images_part = images[int(part_len * num):int(part_len * (num + 1))]
        labels_part = labels[int(part_len * num):int(part_len * (num + 1))]

        with open(os.path.join(save_path_file, f'{filename}_part{num}.pkl'), 'wb') as f:
            pkl.dump((images_part, labels_part), f, pkl.HIGHEST_PROTOCOL)
