import torch
import torch.utils.data as data

LABELS = ['airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car',
          'chair', 'cone', 'cup', 'curtain', 'desk', 'door', 'dresser', 'flower_pot',
          'glass_box', 'guitar', 'keyboard', 'lamp', 'laptop', 'mantel', 'monitor',
          'night_stand', 'person', 'piano', 'plant', 'radio', 'range_hood', 'sink',
          'sofa', 'stairs', 'stool', 'table', 'tent', 'toilet', 'tv_stand', 'vase',
          'wardrobe', 'xbox']


class ModelNetDataset(data.Dataset):
    def __init__(
            self,
            data,
            labels,
            num_points=2048,
            transform=None,
            target_transform=None,
    ):
        self.data = data
        self.labels = labels
        self.num_points = num_points
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        point_set = self.data[idx][:self.num_points]
        label = self.labels[idx]

        if self.transform:
            point_set = self.transform(point_set)
        if self.target_transform:
            label = self.target_transform(label)

        # convert numpy array to pytorch Tensor
        point_set = torch.from_numpy(point_set)
        label = torch.from_numpy(label).long()
        label = label.squeeze(0)

        return point_set, label
