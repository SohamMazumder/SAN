"""
Compare the SAM block and Residual block.
"""

import torch
import torch.nn as nn
import time

from model.san2 import SAM


class Bottleneck(nn.Module):
    def __init__(self, sa_type, in_planes, rel_planes, mid_planes, out_planes, share_planes=8, kernel_size=3, stride=1):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.sam = SAM(sa_type, in_planes, rel_planes, mid_planes, share_planes, kernel_size, stride)
        self.bn2 = nn.BatchNorm2d(mid_planes)
        self.conv = nn.Conv2d(mid_planes, out_planes, kernel_size=1)
        self.relu = nn.LeakyReLU()
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.relu((self.sam(x)))
        out = self.conv(out)
        out = self.relu(out)
        out += identity
        return out


class ResidualBlock(torch.nn.Module):
    def __init__(self, channels, kernel_size=3, activation='lrelu'):
        """
        Residual block.
        :param channels: number of channels.
        :param kernel_size: kernel size.
        :param activation: activation function.
        """
        super().__init__()
        self.channels = channels

        self.conv1 = nn.Conv2d(channels, channels, kernel_size, padding='same')
        self.conv2 = nn.Conv2d(channels, channels, kernel_size, padding='same')
        self.proj = nn.Conv2d(channels, channels, 1)
        self.activation = nn.LeakyReLU()


    def forward(self, x):
        skip = x

        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.activation(x)
        x = self.proj(x)
        x = (x + skip) * (2 ** -0.5)

        return x


if __name__ == '__main__':
    x = torch.rand((1, 64, 64, 64)).cuda()

    bottleneck = Bottleneck(sa_type=0, in_planes=64, rel_planes=64//16, mid_planes=64//4, out_planes=64).cuda()
    num_params = sum(p.numel() for p in bottleneck.parameters())
    print(num_params)
    out = bottleneck(x)
    start = time.perf_counter_ns()
    for _ in range(100):
        out = bottleneck(x)
    print(f'Time: {(time.perf_counter_ns() - start) / 1e6 / 100:.2f} ms')
    print(out.shape)

    residual_block = ResidualBlock(channels=64).cuda()
    num_params = sum(p.numel() for p in residual_block.parameters())
    print(num_params)
    out = residual_block(x)
    start = time.perf_counter_ns()
    for _ in range(100):
        out = residual_block(x)
    print(f'Time: {(time.perf_counter_ns() - start) / 1e6 / 100:.2f} ms')
    print(out.shape)
