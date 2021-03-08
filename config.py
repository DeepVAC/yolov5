#! /usr/bin/python3
# -*- coding:utf-8 -*-
import math

from deepvac import config


# # # config # # #
config.disable_git = True

# # # training # # #
config.amp = False
config.save_num = 1
config.epoch_num = 150
config.nominal_batch_factor = 4

# # # optimizer # # #
config.lr0 = 0.01
config.lrf = 0.2
config.lr_step = lambda x: ((1 + math.cos(x * math.pi / config.epoch_num)) / 2) * (1 - config.lrf) + config.lrf
config.momentum = 0.937
config.nesterov = True
config.weight_decay = 5e-4

# # # warmup # # #
config.warmup_epochs = 3.0
config.warmup_bias_lr = 0.1
config.warmup_momentum = 0.8

# # # model # # #
config.device = 'cuda'
config.img_size = 416
config.class_num = 80
config.strides = [8, 16, 32]
# load model path
config.model_path = "<pretrained-model-path>"
# load script and quantize model path
# config.jit_model_path = "<pretrained-model-path>"

# ema model
config.ema = True
# define ema_decay with other func
# config.ema_decay = lambda x: 0.9999 * (1 - math.exp(-x / 2000))

# # # output # # #
config.output_dir = "output"
# now yolov5-1 support torch.jit.script, not support torch.jit.trace, that will cause fix input_size in c++ program
# config.trace_model_dir = "output/trace.pt"
config.script_model_dir = "output/script.pt"
# bug to fix: now yolov5-1 supoort static quantize, but got error while test
# config.static_quantize_dir = "output/script.pt.sq"

# # # loss # # #
config.obj = 1.0
config.box = 0.05
config.cls = 0.5 * config.class_num / 80

# # # train # # #
# augment
config.train.hflip = 0.5
config.train.augment = True
# dataloader
config.train.shuffle = True
config.train.batch_size = 16  # for single GeForce RTX 2080 Ti
config.train.num_workers = 0
config.train.pin_memory = True
# input
config.train.img_size = config.img_size
config.train.border = [-config.img_size / 2] * 2
config.train.img_folder = "data/coco/images/train2017"
config.train.annotation = "data/coco/instances_train2017.json"

# # # val # # #
# augment
config.val.hflip = 0.5
config.val.augment = True
# dataloader
config.val.batch_size = 20
# input
config.val.img_size = config.img_size
config.val.border = [-config.img_size / 2] * 2
config.val.img_folder = "data/coco/images/val2017"
config.val.annotation = "data/coco/instances_val2017.json"

# # # test # # #
config.test.plot = False
config.test.plot_dir = "output/detect"

config.test.augment = False
config.test.radios = [1, 0.83, 0.67]
# None: no flip  1: cflip  2: wflip  3: hhlip
config.test.flip_p = [None, 3, None]

config.test.iou_thres = 0.45
config.test.conf_thres = 0.25
config.test.input_dir = "data/coco/images/test2017"
config.test.idx_to_cls = ["cls{}".format(i) for i in range(config.class_num)]
