#! /usr/bin/python3
# -*- coding:utf-8 -*-
from typing import List
import torch
from torch import nn


class Conv2dBnAct(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding=None, groups=1, act=True):
        super(Conv2dBnAct, self).__init__()
        if padding is None:
            padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups, bias=False)
        self.bn   = nn.BatchNorm2d(out_planes)
        self.act  = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class BottleneckStd(nn.Module):
    def __init__(self, in_planes, out_planes, groups=1, shortcut=True, expansion=0.5):
        super(BottleneckStd, self).__init__()
        hidden_planes = int(out_planes * expansion)
        self.conv1 = Conv2dBnAct(in_planes, hidden_planes, 1, 1)
        self.conv2 = Conv2dBnAct(hidden_planes, out_planes, 3, 1, groups=groups)
        self.add = shortcut and in_planes == out_planes

    def forward(self, x):
        return x + self.conv2(self.conv1(x)) if self.add else self.conv2(self.conv1(x))


class BottleneckC3(nn.Module):
    def __init__(self, in_planes, out_planes, bottle_std_num=1, shortcut=True, groups=1, expansion=0.5):
        super(BottleneckC3, self).__init__()
        hidden_planes = int(out_planes * expansion)
        self.conv1 = Conv2dBnAct(in_planes, hidden_planes, 1, 1)
        self.conv2 = Conv2dBnAct(in_planes, hidden_planes, 1, 1)
        self.conv3 = Conv2dBnAct(2 * hidden_planes, out_planes, 1, 1)
        self.std_bottleneck_list = nn.Sequential(*[BottleneckStd(hidden_planes, hidden_planes, groups=groups, shortcut=shortcut, expansion=1.0) for _ in range(bottle_std_num)])

    def forward(self, x):
        y1 = self.std_bottleneck_list(self.conv1(x))
        y2 = self.conv2(x)
        return self.conv3(torch.cat((y1, y2), dim=1))


class SPP(nn.Module):
    def __init__(self, in_planes, out_planes, pool_kernel_size=(5, 9, 13)):
        super(SPP, self).__init__()
        hidden_planes = in_planes // 2  # hidden channels
        self.conv1 = Conv2dBnAct(in_planes, hidden_planes, 1, 1)
        self.conv2 = Conv2dBnAct(hidden_planes * (len(pool_kernel_size) + 1), out_planes, 1, 1)
        self.pool_list = nn.ModuleList([nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2) for k in pool_kernel_size])

    def forward(self, x):
        x = self.conv1(x)
        return self.conv2(torch.cat([x] + [pool(x) for pool in self.pool_list], 1))


class Concat(nn.Module):
    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.d = dimension

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        return torch.cat(x, self.d)
