from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class ImageDataset(Dataset):
    def __init__(self, images, labels, transform_x=None, transform_y=None):
        self.images = images
        self.labels = labels
        self.transform_x = transform_x
        self.transform_y = transform_y

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        data, label = self.images[index], self.labels[index]
        if self.transform_x is not None:
            data = self.transform_x(Image.open(data))
        else:
            data = Image.open(data)
        if self.transform_y is not None:
            label = self.transform_y(label)
        return data, label


class ColorImageDataset(Dataset):
    def __init__(self, images, labels, transform_x=None, transform_y=None):
        self.images = images
        self.labels = labels
        self.transform_x = transform_x
        self.transform_y = transform_y

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        data, label = self.images[index], self.labels[index]
        image = Image.open(data).convert('RGB')

        if self.transform_x is not None:
            image = self.transform_x(image)

        if self.transform_y is not None:
            label = self.transform_y(label)
        return image, label


class ArrayImageDataset(Dataset):
    def __init__(self, images, labels, transform_x=None, transform_y=None):
        self.images = images
        self.labels = labels
        self.transform_x = transform_x
        self.transform_y = transform_y

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        data, label = self.images[idx], self.labels[idx]

        # Depending on the channel number of images, the attr:mode varies
        # The shape of array data is (height, width) for grayscale images and (height, width, channel) for color images
        mode = 'L' if len(data.shape) == 2 else 'RGB'
        if self.transform_x is not None:
            data = self.transform_x(Image.fromarray(data, mode))
        else:
            data = Image.fromarray(data, mode)
        if self.transform_y is not None:
            label = self.transform_y(label)
        return data, label


class TransformDataset(Dataset):
    def __init__(self, images, labels, transform_x=None, transform_y=None):
        self.data = images
        self.targets = labels
        self.transform_x = transform_x
        self.transform_y = transform_y

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        target = self.targets[idx]

        if self.transform_x:
            sample = self.transform_x(sample)
        if self.transform_y:
            target = self.transform_y(target)

        return sample, target


class ImageSegDataset(Dataset):
    def __init__(self, images, labels, transform=None, *args, **kwargs):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        data = Image.open(self.images[index]).convert('RGB')
        label = Image.open(self.labels[index])
        if self.transform is not None:
            data, label = self.transform(data, label)

        return data, label.long()
