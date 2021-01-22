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
config.disable_git = False
# train
config.amp = False
config.save_num = 1
config.epoch_num = 150
config.best_fitness = 0.0
config.nominal_batch_factor = 4
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
config.img_size = 416
config.num_classes = 4
config.strides = [8, 16, 32]
config.model_type = "yolov5l"
config.model_file = "modules/yolo.json"
config.model_path = "output/yolov5l.pkl"
# config.script_model_dir = f"output/{config.model_type}.torchscript.pt"
# if you want to export model to torchscript, make model.is_training = False first;
# ema model
config.ema = True
config.updates = 0
config.decay = 0.9999

# # # loss # # #
config.loss = AttrDict()
config.loss.gr = 1.0
config.loss.box = 0.05
config.loss.cls = 0.5 * config.num_classes / 80
config.loss.cls_pw = 1.0
config.loss.obj = 1.0
config.loss.obj_pw = 1.0
config.loss.iou_t = 0.2
config.loss.anchor_t = 4
config.loss.fl_gamma = 0.0

# # # train # # #
config.train = AttrDict()
config.train.shuffle = True
config.train.hflip = 1
config.train.batch_size = 16
config.train.num_workers = 8
config.train.augment = True
config.train.pin_memory = True
config.train.img_size = config.img_size
config.train.border = [-config.img_size / 2] * 2
config.train.img_folder = "/home/liyang/ai05/PornTestData/images"
config.train.annotation = "/home/liyang/ai05/PornTestData/train.json"

# # # val # # #
config.val = AttrDict()
config.val.shuffle = False
config.val.batch_size = 20
config.val.augment = config.aug
config.val.img_size = config.img_size
config.val.root = "/gemfield/hostpv/PornYoloDataset/val"
config.val.annotation = "/gemfield/hostpv/PornYoloDataset/val.json"

# # # test # # #
config.test = AttrDict()
config.test.plot = False
config.test.iou_thres = 0.25
config.test.conf_thres = 0.45
config.test.idx_to_cls = ["Female-Breast", "Female-Gential", "Male-Gential", "Buttock"]
