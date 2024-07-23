"""
Reference:
https://github.com/hshustc/CVPR19_Incremental_Learning/blob/master/cifar100-class-incremental/modified_linear.py
"""

import torch
from torch import nn
from torch.nn import functional as F


class SimpleLinear(nn.Module):
    """
    Reference:
    https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/linear.py
    """

    def __init__(self, in_features, out_features, bias=True):
        super(SimpleLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, nonlinearity='linear')
        nn.init.constant_(self.bias, 0)

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)

    def update(self):
        pass
