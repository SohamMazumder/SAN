"""
SAN without custom CUDA kernels
"""

import torch
import torch.nn as nn


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Aggregation(nn.Module):

    def __init__(self, kernel_size, stride, padding, dilation, pad_mode):
        super(Aggregation, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.pad_mode = pad_mode

    def forward(self, x, weight):
        return self._aggregation(x, weight)

    def _aggregation(self, x, weight):
        assert x.shape[0] == weight.shape[0] and (x.shape[1] % weight.shape[1] == 0) and self.pad_mode in [0, 1]
        if x.is_cuda:
            if self.pad_mode == 0:
                out = self._aggregation_zeropad(x, weight)
            elif self.pad_mode == 1:
                out = self._aggregation_refpad(x, weight)
        else:
            raise NotImplementedError
        return out

    def _aggregation_zeropad(self, x, w):
        batch_size, input_channels, input_height, input_width = x.size()
        _, weight_channels, weight_height, weight_width = w.size()
        out_height = int(
            (input_height + 2 * self.padding - (self.dilation * (self.kernel_size - 1) + 1)) / self.stride + 1)
        out_width = int(
            (input_width + 2 * self.padding - (self.dilation * (self.kernel_size - 1) + 1)) / self.stride + 1)

        unfold_j = torch.nn.Unfold(kernel_size=self.kernel_size,
                                   dilation=self.dilation,
                                   padding=self.padding,
                                   stride=self.stride)
        x2 = unfold_j(x).view(batch_size,
                              input_channels // weight_channels,
                              weight_channels,
                              pow(self.kernel_size, 2),
                              out_height * out_width)
        y2 = (w.unsqueeze(1) * x2).sum(-2).view(batch_size, input_channels, out_height, out_width)
        return y2

    def _aggregation_refpad(self, x, w):
        batch_size, input_channels, input_height, input_width = x.size()
        _, weight_channels, weight_height, weight_width = w.size()
        out_height = int(
            (input_height + 2 * self.padding - (self.dilation * (self.kernel_size - 1) + 1)) / self.stride + 1)
        out_width = int(
            (input_width + 2 * self.padding - (self.dilation * (self.kernel_size - 1) + 1)) / self.stride + 1)

        unfold_j = torch.nn.Unfold(kernel_size=self.kernel_size,
                                   dilation=self.dilation,
                                   padding=0,
                                   stride=self.stride)
        pad = torch.nn.ReflectionPad2d(self.padding)
        x2 = unfold_j(pad(x)).view(batch_size,
                                   input_channels // weight_channels,
                                   weight_channels,
                                   pow(self.kernel_size, 2),
                                   out_height * out_width)
        y2 = (w.unsqueeze(1) * x2).sum(-2).view(batch_size, input_channels, out_height, out_width)
        return y2


class SAM(nn.Module):
    def __init__(self, sa_type, in_planes, rel_planes, out_planes, share_planes, kernel_size=3, stride=1, dilation=1):
        super(SAM, self).__init__()
        self.sa_type, self.kernel_size, self.stride = sa_type, kernel_size, stride
        self.conv1 = nn.Conv2d(in_planes, rel_planes, kernel_size=1)
        self.conv2 = nn.Conv2d(in_planes, rel_planes, kernel_size=1)
        self.conv3 = nn.Conv2d(in_planes, out_planes, kernel_size=1)

        #patchwise
        self.conv_w = nn.Sequential(nn.BatchNorm2d(rel_planes * (pow(kernel_size, 2) + 1)), nn.ReLU(inplace=True),
                                    nn.Conv2d(rel_planes * (pow(kernel_size, 2) + 1), out_planes // share_planes, kernel_size=1, bias=False),
                                    nn.BatchNorm2d(out_planes // share_planes), nn.ReLU(inplace=True),
                                    nn.Conv2d(out_planes // share_planes, pow(kernel_size, 2) * out_planes // share_planes, kernel_size=1))
        self.unfold_i = nn.Unfold(kernel_size=1, dilation=dilation, padding=0, stride=stride)
        self.unfold_j = nn.Unfold(kernel_size=kernel_size, dilation=dilation, padding=0, stride=stride)
        self.pad = nn.ReflectionPad2d(kernel_size // 2)

        self.aggregation = Aggregation(kernel_size, stride, (dilation * (kernel_size - 1) + 1) // 2, dilation, pad_mode=1)

    def forward(self, x):
        x1, x2, x3 = self.conv1(x), self.conv2(x), self.conv3(x)
        # patchwise
        if self.stride != 1:
            x1 = self.unfold_i(x1)
        x1 = x1.view(x.shape[0], -1, 1, x.shape[2]*x.shape[3])
        x2 = self.unfold_j(self.pad(x2)).view(x.shape[0], -1, 1, x1.shape[-1])
        w = self.conv_w(torch.cat([x1, x2], 1)).view(x.shape[0], -1, pow(self.kernel_size, 2), x1.shape[-1])

        x = self.aggregation(x3, w)
        return x


class Bottleneck(nn.Module):
    def __init__(self, sa_type, in_planes, rel_planes, mid_planes, out_planes, share_planes=8, kernel_size=7, stride=1):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.sam = SAM(sa_type, in_planes, rel_planes, mid_planes, share_planes, kernel_size, stride)
        self.bn2 = nn.BatchNorm2d(mid_planes)
        self.conv = nn.Conv2d(mid_planes, out_planes, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(x))
        out = self.relu(self.bn2(self.sam(out)))
        out = self.conv(out)
        out += identity
        return out


class SAN(nn.Module):
    def __init__(self, sa_type, block, layers, kernels, num_classes):
        super(SAN, self).__init__()
        c = 64
        self.conv_in, self.bn_in = conv1x1(3, c), nn.BatchNorm2d(c)
        self.conv0, self.bn0 = conv1x1(c, c), nn.BatchNorm2d(c)
        self.layer0 = self._make_layer(sa_type, block, c, layers[0], kernels[0])

        c *= 4
        self.conv1, self.bn1 = conv1x1(c // 4, c), nn.BatchNorm2d(c)
        self.layer1 = self._make_layer(sa_type, block, c, layers[1], kernels[1])

        c *= 2
        self.conv2, self.bn2 = conv1x1(c // 2, c), nn.BatchNorm2d(c)
        self.layer2 = self._make_layer(sa_type, block, c, layers[2], kernels[2])

        c *= 2
        self.conv3, self.bn3 = conv1x1(c // 2, c), nn.BatchNorm2d(c)
        self.layer3 = self._make_layer(sa_type, block, c, layers[3], kernels[3])

        c *= 2
        self.conv4, self.bn4 = conv1x1(c // 2, c), nn.BatchNorm2d(c)
        self.layer4 = self._make_layer(sa_type, block, c, layers[4], kernels[4])

        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(c, num_classes)

    def _make_layer(self, sa_type, block, planes, blocks, kernel_size=7, stride=1):
        layers = []
        for _ in range(0, blocks):
            layers.append(block(sa_type, planes, planes // 16, planes // 4, planes, 8, kernel_size, stride))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn_in(self.conv_in(x)))
        x = self.relu(self.bn0(self.layer0(self.conv0(self.pool(x)))))
        x = self.relu(self.bn1(self.layer1(self.conv1(self.pool(x)))))
        x = self.relu(self.bn2(self.layer2(self.conv2(self.pool(x)))))
        x = self.relu(self.bn3(self.layer3(self.conv3(self.pool(x)))))
        x = self.relu(self.bn4(self.layer4(self.conv4(self.pool(x)))))

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def san(sa_type, layers, kernels, num_classes):
    model = SAN(sa_type, Bottleneck, layers, kernels, num_classes)
    return model


if __name__ == '__main__':
    import time

    net = san(sa_type=1, layers=(2, 1, 2, 4, 1), kernels=[3, 7, 7, 7, 7], num_classes=1000).cuda().eval()
    print(sum(p.numel() for p in net.parameters() if p.requires_grad))
    y = net(torch.randn(4, 3, 224, 224).cuda())
    start = time.perf_counter_ns()
    for _ in range(10):
        y = net(torch.randn(4, 3, 224, 224).cuda())
    print(f'SAN: {(time.perf_counter_ns() - start) / 1e6 / 10:.4f}ms')
    print(y.size())

    from torchvision.models import resnet18

    res = resnet18(pretrained=False).eval().cuda()
    print(sum(p.numel() for p in res.parameters() if p.requires_grad))
    y = res(torch.randn(4, 3, 224, 224).cuda())
    start = time.perf_counter_ns()
    for _ in range(10):
        y = res(torch.randn(4, 3, 224, 224).cuda())
    print(f'ResNet18: {(time.perf_counter_ns() - start) / 1e6 / 10:.4f}ms')
    print(y.size())
