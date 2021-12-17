import torch
import torch.nn as nn
from model.embeddings import InvertedResidual
import math

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        # if bilinear, use the normal convolutions to reduce the number of channels
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = InvertedResidual(out_channels, out_channels, (1, 1, 1), 1)
        self.bn = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        x = self.up(x)
        x = self.bn(self.conv(x))
        return x


class UpNet(nn.Module):
    def __init__(self, start_channels, image_size=128):
        super(UpNet, self).__init__()
        self.n_layers = int(math.log2(image_size))
        cur_channel = start_channels
        layers = []
        for i in range(self.n_layers - 1):
            layers.append(Up(cur_channel, cur_channel // 2))
            cur_channel = cur_channel // 2
        layers.append(Up(cur_channel, 1))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
