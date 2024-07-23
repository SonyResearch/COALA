import copy
import torch
from torch import nn
from .base_model import resnet18, resnet34, resnet50
from .linear import SimpleLinear


def get_backbone(conf, pretrained=False):
    name = conf.model.lower()
    if name == "resnet18":
        return resnet18(pretrained=pretrained, conf=conf)
    elif name == "resnet34":
        return resnet34(pretrained=pretrained, conf=conf)
    elif name == "resnet50":
        return resnet50(pretrained=pretrained, conf=conf)
    else:
        raise NotImplementedError("Unknown type {}".format(name))


class BaseNet(nn.Module):
    def __init__(self, conf, pretrained=False):
        super(BaseNet, self).__init__()

        self.backbone = get_backbone(conf, pretrained)
        self.fc = None

    @property
    def feature_dim(self):
        return self.backbone.out_dim

    def extract_vector(self, x):
        return self.backbone(x)

    def forward(self, x):
        x = self.backbone(x)
        out = self.fc(x)

        return out

    def update_fc(self, nb_classes):
        pass

    def generate_fc(self, in_dim, out_dim):
        pass

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

        return self


class IncrementalNet(BaseNet):
    def __init__(self, conf, pretrained=False):
        super().__init__(conf, pretrained)

    def update_fc(self, num_classes):
        fc = self.generate_fc(self.feature_dim, num_classes)
        if self.fc is not None:
            num_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:num_output] = weight
            fc.bias.data[:num_output] = bias
        self.fc = fc

    def weight_align(self, increment):
        weights = self.fc.weight.data
        newnorm = torch.norm(weights[-increment:, :], p=2, dim=1)
        oldnorm = torch.norm(weights[:-increment, :], p=2, dim=1)
        meannew = torch.mean(newnorm)
        meanold = torch.mean(oldnorm)
        gamma = meanold / meannew
        self.fc.weight.data[-increment:, :] *= gamma

    def generate_fc(self, in_dim, out_dim):
        fc = SimpleLinear(in_dim, out_dim)

        return fc

    def forward(self, x):
        x = self.backbone(x)
        out = self.fc(x)

        return out
