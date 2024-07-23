import numpy as np
import torch
import torchvision
from torchvision import transforms
from coala.datasets.data_process.randaug import RandAugmentMC


class Cutout(object):
    """Cutout data augmentation is adopted from https://github.com/uoguelph-mlrg/Cutout"""

    def __init__(self, length=16):
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).

        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


transform_train_cifar = transforms.Compose([
    torchvision.transforms.ToPILImage(mode='RGB'),
    transforms.RandomCrop(32, padding=4),
    # The following two lines are for parameter efficient finetuning of ViT
    # transforms.Resize([224, 224]),
    # transforms.RandomCrop(224, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.49139968, 0.48215827, 0.44653124), (0.24703233, 0.24348505, 0.26158768)),
])

transform_train_cifar.transforms.append(Cutout())

transform_test_cifar = transforms.Compose([
    # The following two lines are for parameter efficient finetuning of ViT
    # torchvision.transforms.ToPILImage(mode='RGB'),
    # transforms.Resize([224, 224]),
    transforms.ToTensor(),
    transforms.Normalize((0.49139968, 0.48215827, 0.44653124), (0.24703233, 0.24348505, 0.26158768)),
])



class TransformFixMatch(object):
    def __init__(self):
        self.weak = transforms.Compose([
            torchvision.transforms.ToPILImage(mode='RGB'),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),])
        self.strong = transforms.Compose([
            torchvision.transforms.ToPILImage(mode='RGB'),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            RandAugmentMC(n=2, m=10)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.49139968, 0.48215827, 0.44653124), (0.24703233, 0.24348505, 0.26158768))])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)


def map_new_class_index(y, order):
    return np.array(list(map(lambda x: order.index(x), y)))
