#! /usr/bin/python3
# -*- coding:utf-8 -*-
from typing import List
import torch
from torch import nn
from torch.nn import functional as F


class Conv2dBNHardswish(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=None, groups=1, bias=False):
        if padding is None:
            padding = (kernel_size - 1) // 2
        super(Conv2dBNHardswish, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=bias)
        self.bn   = nn.BatchNorm2d(out_planes)
        self.act  = nn.Hardswish()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


# to run script.pt on cuda in C++, you need to rewrite Focus
class Focus(nn.Module):
    # Focus wh information into c-space
    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding=None, groups=1):
        super(Focus, self).__init__()
        self.conv = Conv2dBNHardswish(in_planes * 4, out_planes, kernel_size, stride, padding, groups)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        #this is a workaround, see https://github.com/DeepVAC/yolov5/issues/5
        xcpu = x.cpu()
        xnew = torch.cat([xcpu[..., ::2, ::2], xcpu[..., 1::2, ::2], xcpu[..., ::2, 1::2], xcpu[..., 1::2, 1::2]], 1)
        return self.conv(xnew.to(x.device))


class BottleneckStd(nn.Module):
    def __init__(self, in_planes, out_planes, groups=1, shortcut=True, expansion=0.5):
        super(BottleneckStd, self).__init__()
        hidden_planes = int(out_planes * expansion)
        self.conv1 = Conv2dBNHardswish(in_planes, hidden_planes, 1, 1,)
        self.conv2 = Conv2dBNHardswish(hidden_planes, out_planes, 3, 1, groups=groups)
        self.add = shortcut and in_planes == out_planes

    def forward(self, x):
        return x + self.conv2(self.conv1(x)) if self.add else self.conv2(self.conv1(x))


class BottleneckCSP(nn.Module):
    def __init__(self, in_planes, out_planes, bottle_std_num=1, shortcut=True, groups=1, expansion=0.5):
        super(BottleneckCSP, self).__init__()
        hidden_planes = int(out_planes * expansion)
        self.conv1 = Conv2dBNHardswish(in_planes, hidden_planes, 1, 1,)
        self.conv2 = nn.Conv2d(in_planes, hidden_planes, 1, 1, bias=False)
        self.conv3 = nn.Conv2d(hidden_planes, hidden_planes, 1, 1, bias=False)
        self.conv4 = Conv2dBNHardswish(2 * hidden_planes, out_planes, 1, 1,)
        self.bn = nn.BatchNorm2d(2 * hidden_planes)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.std_bottleneck_list = nn.Sequential(*[BottleneckStd(hidden_planes, hidden_planes, groups=groups, shortcut=shortcut, expansion=1.0) for _ in range(bottle_std_num)])

    def forward(self, x):
        y1 = self.conv3(self.std_bottleneck_list(self.conv1(x)))
        y2 = self.conv2(x)
        return self.conv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))


class Concat(nn.Module):
    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.d = dimension

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        return torch.cat(x, self.d)


class SPP(nn.Module):
    def __init__(self, in_planes, out_planes, pool_kernel_size=(5, 9, 13)):
        super(SPP, self).__init__()
        hidden_planes = in_planes // 2
        self.conv1 = Conv2dBNHardswish(in_planes, hidden_planes, 1, 1)
        self.conv2 = Conv2dBNHardswish(hidden_planes * (len(pool_kernel_size) + 1), out_planes, 1, 1)
        self.pool_list = nn.ModuleList([nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2) for k in pool_kernel_size])

    def forward(self, x):
        x = self.conv1(x)
        return self.conv2(torch.cat([x] + [pool(x) for pool in self.pool_list], 1))


class SiLU(nn.Module):
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)


class Hardswish(nn.Module):
    @staticmethod
    def forward(x):
        return x * F.hardsigmoid(x)  # for torchscript and CoreML
        return x * F.hardtanh(x + 3, 0., 6.) / 6.  # for torchscript, CoreML and ONNX


class Mish(nn.Module):
    @staticmethod
    def forward(x):
        return x * F.softplus(x).tanh()


def setCoreml(model):
    for m in model.modules():
        if isinstance(m, Conv2dBNHardswish) and isinstance(m.act, torch.nn.Hardswish):
            m.act = Hardswish()
