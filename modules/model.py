#! /usr/bin/python3
# -*- coding:utf-8 -*-
import torch
from torch import nn, Tensor
from typing import List, Tuple
from .utils import *


class Detect(nn.Module):
    def __init__(self, class_num=80, anchors=(), in_planes_list=()):
        super(Detect, self).__init__()
        self.class_num = class_num
        self.output_num_per_anchor = class_num + 5
        self.detect_layer_num = len(anchors)
        self.anchor_num = len(anchors[0]) // 2
        self.grid = [torch.zeros(1)] * self.detect_layer_num
        anchor_t = torch.tensor(anchors).float().view(self.detect_layer_num, -1, 2)
        self.register_buffer('anchors', anchor_t)  # shape(detect_layer_num, anchor_num,2)
        self.register_buffer('anchor_grid', anchor_t.clone().view(self.detect_layer_num, 1, -1, 1, 1, 2))  # shape(detect_layer_num,1, anchor_num,1,1,2)
        self.conv_list = nn.ModuleList(nn.Conv2d(x, self.output_num_per_anchor * self.anchor_num, 1) for x in in_planes_list)

        self.output: List[Tensor] = [torch.empty(0),] * 3

    def forward(self, x: List[Tensor]) -> Tensor:
        inference_result: List[Tensor] = []
        for i, layer in enumerate(self.conv_list):
            x[i] = layer(x[i])
            bs, _, ny, nx = x[i].shape
            #x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.anchor_num, self.output_num_per_anchor, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            # if not self.is_training:
            if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                self.grid[i] = self.makeGrid(nx, ny).to(x[i].device)

            y = x[i].sigmoid()
            y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i].to(x[i].device)) * self.strides[i]  # xy
            y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
            inference_result.append(y.view(bs, -1, self.output_num_per_anchor))

        self.output = x
        return torch.cat(inference_result, 1)

    @staticmethod
    def makeGrid(nx: int=20, ny: int=20) -> Tensor:
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view(1, 1, ny, nx, 2).float()


class Yolov5S(nn.Module):
    '''
        yolov5s
    '''
    def __init__(self, class_num=80, strides=[8, 16, 32]):
        super(Yolov5S, self).__init__()
        self.class_num = class_num
        self.upsample = nn.Upsample(scale_factor=2., mode="nearest")
        self.cat = Concat(1)
        self.initBlock1()
        self.initBlock2()
        self.initBlock3()
        self.initBlock4()
        self.initBlock5()
        self.initDetect()
        self.initBNParam()
        self.detect.strides = torch.Tensor(strides)
        self.initParamGroup()

    def buildBlock(self, cfgs):
        layers = []
        #init the 4 layers
        for m, args in cfgs:
            layers.append(m(*args))
        return nn.Sequential(*layers)

    def forward(self, x):
        x1 = self.block1(x)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        x = self.upsample(x3)

        #cat from point2
        x = self.cat([x, x2])
        x4 = self.block4(x)
        x = self.upsample(x4)

        #cat from point1
        x = self.cat([x, x1])
        c1 = self.csp1(x)
        x = self.conv1(c1)

        #cat from point4
        x = self.cat([x,x4])
        c2 = self.csp2(x)
        x = self.conv2(c2)

        #cat from point3
        x = self.cat([x,x3])
        c3 = self.csp3(x)

        return self.detect([c1, c2, c3])

    def initBlock1(self):
        cfgs = [
            [Focus, [3, 32, 3, 1] ],
            [Conv2dBNHardswish, [32, 64, 3, 2] ],
            [BottleneckCSP, [64, 64, 1, True] ],
            [Conv2dBNHardswish, [64, 128, 3, 2] ],
            [BottleneckCSP, [128, 128, 3, True] ]
        ]
        self.block1 = self.buildBlock(cfgs)

    def initBlock2(self):
        cfgs = [
            [Conv2dBNHardswish, [128, 256, 3, 2] ],
            [BottleneckCSP, [256, 256, 3, True] ]
        ]
        self.block2 = self.buildBlock(cfgs)

    def initBlock3(self):
        cfgs = [
            [Conv2dBNHardswish, [256, 512, 3, 2] ],
            [SPP, [512, 512, [5, 9, 13]] ],
            [BottleneckCSP, [512, 512, 1, False] ],
            [Conv2dBNHardswish, [512, 256, 1, 1] ]
        ]
        self.block3 = self.buildBlock(cfgs)

    def initBlock4(self):
        cfgs = [
            [BottleneckCSP, [512, 256, 1, False] ],
            [Conv2dBNHardswish, [256, 128, 1, 1] ],
        ]
        self.block4 = self.buildBlock(cfgs)

    def initBlock5(self):
        self.csp1 = self.buildBlock([[BottleneckCSP, [256, 128, 1, False]],])
        self.conv1 = self.buildBlock([[Conv2dBNHardswish, [128, 128, 3, 2]],])
        self.csp2 = self.buildBlock([[BottleneckCSP, [256, 256, 1, False]],])
        self.conv2 = self.buildBlock([[Conv2dBNHardswish, [256, 256, 3, 2]],])
        self.csp3 = self.buildBlock([[BottleneckCSP, [512, 512, 1, False]],])

    def initDetect(self):
        #initial anchors
        self.detect = Detect(self.class_num, [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]], [128, 256, 512])

    def initBNParam(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eps = 0.001
                m.momentum = 0.01

    def initParamGroup(self):
        self.pg0 = []
        self.pg1 = []
        self.pg2 = []
        for k, v in self.named_modules():
            if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
                self.pg2.append(v.bias)  # biases
            if isinstance(v, nn.BatchNorm2d):
                self.pg0.append(v.weight)  # no decay
            elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
                self.pg1.append(v.weight)  # apply decay


class Yolov5L(Yolov5S):
    '''
        yolov5-L
    '''
    def __init__(self, class_num=80, strides=[8, 16, 32]):
        super(Yolov5L, self).__init__(class_num, strides)

    def initBlock1(self):
        cfgs = [
            [Focus, [3, 64, 3, 1] ],
            [Conv2dBNHardswish, [64, 128, 3, 2] ],
            [BottleneckCSP, [128, 128, 3, True] ],
            [Conv2dBNHardswish, [128, 256, 3, 2] ],
            [BottleneckCSP, [256, 256, 9, True] ]
        ]
        self.block1 = self.buildBlock(cfgs)

    def initBlock2(self):
        cfgs = [
            [Conv2dBNHardswish, [256, 512, 3, 2] ],
            [BottleneckCSP, [512, 512, 9, True] ]
        ]
        self.block2 = self.buildBlock(cfgs)

    def initBlock3(self):
        cfgs = [
            [Conv2dBNHardswish, [512, 1024, 3, 2] ],
            [SPP, [1024, 1024, [5, 9, 13]] ],
            [BottleneckCSP, [1024, 1024, 3, False] ],
            [Conv2dBNHardswish, [1024, 512, 1, 1] ]
        ]
        self.block3 = self.buildBlock(cfgs)

    def initBlock4(self):
        cfgs = [
            [BottleneckCSP, [1024, 512, 3, False] ],
            [Conv2dBNHardswish, [512, 256, 1, 1] ],
        ]
        self.block4 = self.buildBlock(cfgs)

    def initBlock5(self):
        self.csp1 = self.buildBlock([[BottleneckCSP, [512, 256, 3, False]],])
        self.conv1 = self.buildBlock([[Conv2dBNHardswish, [256, 256, 3, 2]],])
        self.csp2 = self.buildBlock([[BottleneckCSP, [512, 512, 3, False]],])
        self.conv2 = self.buildBlock([[Conv2dBNHardswish, [512, 512, 3, 2]],])
        self.csp3 = self.buildBlock([[BottleneckCSP, [1024, 1024, 3, False]],])

    def initDetect(self):
        #initial anchors
        self.detect = Detect(self.class_num, [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]], [256, 512, 1024])
