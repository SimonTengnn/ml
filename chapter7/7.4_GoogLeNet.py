import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

class Inception(nn.Module):
    def __init__(self, in_channels, channel_path1, channel_path2, channel_path3, channel_path4, **kwargs):
        super(Inception).__init__(**kwargs)
        self.path1_1 = nn.Conv2d(in_channels, channel_path1, kernel_size=1)
        self.path2_1 = nn.Conv2d(in_channels, channel_path2[0], kernel_size=1)
        self.path2_2 = nn.Conv2d(channel_path2[0], channel_path2[1], kernel_size=3, padding=1)
        self.path3_1 = nn.Conv2d(in_channels, channel_path3[0], kernel_size=1)
        self.path3_2 = nn.Conv2d(channel_path3[0], channel_path3[1], kernel_size=5, padding=2)
        self.path4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1) # 池化层不改变通道数，参数和通道数无关
        self.path4_2 = nn.Conv2d(in_channels, channel_path4, kernel_size=1)

    def forward(self, x):
        path1 = F.relu(self.path1_1(x))
        path2 = F.relu(self.path2_2(F.relu(self.path2_1(x))))
        path3 = F.relu(self.path3_2(F.relu(self.path3_1(x))))
        path4 = F.relu(self.path4_2(F.relu(self.path4_1(x))))
        return torch.cat((path1, path2, path3, path4), dim=1)