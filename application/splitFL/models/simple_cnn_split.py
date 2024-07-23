import torch
import torch.nn.functional as F
from torch import nn
from models.split_models import BaseSFLModel, ViewLayer

class Model(BaseSFLModel):
    def __init__(self, channels=32, num_classes=10, num_clients = 1):
        super(Model, self).__init__()
        self.num_channels = channels
        model_list = [nn.Conv2d(3, self.num_channels, 3, stride=1),
                      nn.MaxPool2d(kernel_size=2, stride=2),
                      nn.ReLU(),
                      nn.Conv2d(self.num_channels, self.num_channels * 2, 3, stride=1),
                      nn.MaxPool2d(kernel_size=2, stride=2),
                      nn.ReLU(),
                      nn.Conv2d(self.num_channels * 2, self.num_channels * 2, 3, stride=1),
                      ViewLayer(),
                      nn.Linear(4 * 4 * self.num_channels * 2, self.num_channels * 2),
                      nn.ReLU(),
                      nn.Linear(self.num_channels * 2, num_classes)]
        
        self.model = nn.Sequential(*model_list)

        self.backbone_output_dim = 4 * 4 * self.num_channels * 2

        self.split()

        self.initialize_weights()