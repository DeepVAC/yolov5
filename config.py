#! /usr/bin/python3
# -*- coding:utf-8 -*-
'''
@Author     : __MHGL__
@Data       : 2021/01/05
@Desciption : deepvac yolov5 config, include config, aug, train, val, test, loss
'''


import sys
sys.path.append("../deepvac/")
import math

from deepvac.syszux_config import *


# # # config # # #
config.disable_git = True

# train
config.amp = False
config.save_num = 1
config.epoch_num = 150
# print(">>> config.py +23 config.epoch changed")
# config.epoch_num = 50
config.nominal_batch_factor = 4
# print(">>> config.py +26 config.nominal_batch_factor changed")
# config.nominal_batch_factor = 1

# optimizer
config.lr0 = 0.01
config.lrf = 0.2
config.lr_step = lambda x: ((1 + math.cos(x * math.pi / config.epoch_num)) / 2) * (1 - config.lrf) + config.lrf
config.momentum = 0.937
config.nesterov = True
config.weight_decay = 5e-4

# warmup
config.warmup_epochs = 3.0
config.warmup_bias_lr = 0.1
config.warmup_momentum = 0.8

# model
config.device = 'cuda'
# print(">>> config.py +44 config.device changed")
# config.device = "cpu"
config.img_size = 416
config.class_num = 4
config.strides = [8, 16, 32]
config.model_path = "output/yolov5l.pth"

# ema model
config.ema = True
config.updates = 0
config.decay = 0.9999

# # # loss # # #
config.obj = 1.0
config.box = 0.05
config.cls = 0.5 * config.class_num / 80

# # # train # # #
config.train = AttrDict()
config.train.shuffle = True
config.train.hflip = 0.5
# print(">>> config.py +65 config.train.hflip changed")
# config.train.hflip = 1
config.train.batch_size = 16
# print(">>> config.py +67 config.train.batch_size changed")
# config.train.batch_size = 1
config.train.num_workers = 8
config.train.augment = True
config.train.pin_memory = True
config.train.img_size = config.img_size
config.train.border = [-config.img_size / 2] * 2
config.train.img_folder = "/gemfield/hostpv/PornYoloDataset/train"
# print(">>> config.py +76 config.train.img_folder changed")
# config.train.img_folder = "/gemfield/hostpv/PornTestData/images"
config.train.annotation = "/gemfield/hostpv/PornYoloDataset/train.json"
# print(">>> config.py +79 config.train.annotation changed")
# config.train.annotation = "/gemfield/hostpv/PornTestData/train.json"

# # # val # # #
config.val = AttrDict()
config.val.hflip = 0.5
config.val.augment = True
config.val.shuffle = False
config.val.batch_size = 20
config.val.img_size = config.img_size
config.val.border = [-config.img_size / 2] * 2
config.val.img_folder = "/gemfield/hostpv/PornYoloDataset/val"
config.val.annotation = "/gemfield/hostpv/PornYoloDataset/val.json"

# # # test # # #
config.test = AttrDict()
config.test.plot = False
config.test.iou_thres = 0.25
config.test.conf_thres = 0.45
config.test.idx_to_cls = ["Female-Breast", "Female-Gential", "Male-Gential", "Buttock"]
