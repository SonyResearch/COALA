import torch
import torch.nn as nn
import torch.nn.functional as F
from models.split_models import BaseSFLModel, ViewLayer

NUM_CHANNEL_GROUP = 4

class WSConv2d(nn.Conv2d): # This module is taken from https://github.com/joe-siyuan-qiao/WeightStandardization

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(WSConv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias)

    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2,
                                  keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        return F.conv2d(x, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = WSConv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.GroupNorm(num_groups = max(planes//NUM_CHANNEL_GROUP, 1), num_channels = planes)
        self.conv2 = WSConv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.GroupNorm(num_groups = max(planes//NUM_CHANNEL_GROUP, 1), num_channels = planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                WSConv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(num_groups = max(self.expansion * planes//NUM_CHANNEL_GROUP, 1), num_channels = self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = WSConv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.GroupNorm(num_groups = max(planes//NUM_CHANNEL_GROUP, 1), num_channels = planes)
        self.conv2 = WSConv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.GroupNorm(num_groups = max(planes//NUM_CHANNEL_GROUP, 1), num_channels = planes)
        self.conv3 = WSConv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.GroupNorm(num_groups = max(self.expansion * planes//NUM_CHANNEL_GROUP, 1), num_channels = self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                WSConv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(num_groups = max(self.expansion * planes//NUM_CHANNEL_GROUP, 1), num_channels = self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Model(BaseSFLModel):
    """ResNet18 model
    Note two main differences from official pytorch version:
    1. conv1 kernel size: pytorch version uses kernel_size=7
    2. average pooling: pytorch version uses AdaptiveAvgPool
    """

    def __init__(self, block=Bottleneck, num_blocks=[3, 4, 6, 3], num_classes=10):
        super(Model, self).__init__()
        self.in_planes = 64

        super(Model, self).__init__()
        
        model_list = [WSConv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
                      nn.GroupNorm(num_groups = max(64//NUM_CHANNEL_GROUP, 1), num_channels = 64),
                      nn.ReLU()]
                      
        self._make_layer(model_list, block, 64, num_blocks[0], stride=1)
        self._make_layer(model_list, block, 128, num_blocks[1], stride=2)
        self._make_layer(model_list, block, 256, num_blocks[2], stride=2)
        self._make_layer(model_list, block, 512, num_blocks[3], stride=2)
        
        
        model_list.extend([nn.AdaptiveAvgPool2d((1,1)),
                      ViewLayer(),
                      nn.Linear(512 * block.expansion, num_classes)])
        
        self.model = nn.Sequential(*model_list)

        self.backbone_output_dim = 512 * block.expansion

        self.split()
        
        self.initialize_weights()

    def _make_layer(self, model_list, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        for stride in strides:
            model_list.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

