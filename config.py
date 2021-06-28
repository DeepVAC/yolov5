#! /usr/bin/python3
# -*- coding:utf-8 -*-

# import first:  system libs
import math

# import second: third party libs
import torch
from torchvision import transforms

# import third:  libs in your program
from deepvac import AttrDict, new
from deepvac.aug.yolo_aug import *
from aug.aug import Yolov5TrainComposer, Yolov5ValComposer
from data.datasets import Yolov5MosaicDataset, Yolov5Dataset
from modules import Yolov5S, Yolov5L, Yolov5Loss


################################################################################
### TRAIN
################################################################################
config = new("Yolov5Train")
### ---------------------------------- common ----------------------------------
config.core.Yolov5Train.class_num = 80
config.core.Yolov5Train.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config.core.Yolov5Train.output_dir = "output"
config.core.Yolov5Train.log_every = 10
config.core.Yolov5Train.disable_git = True
config.core.Yolov5Train.model_reinterpret_cast = True
config.core.Yolov5Train.cast_state_dict_strict = True

### ---------------------------------- training --------------------------------
config.core.Yolov5Train.ema = True
# define ema_decay with other func
# config.ema_decay = lambda x: 0.9999 * (1 - math.exp(-x / 2000))
config.core.Yolov5Train.amp = False
config.core.Yolov5Train.epoch_num = 150
config.core.Yolov5Train.save_num = 1
config.core.Yolov5Train.shuffle = True
config.core.Yolov5Train.batch_size = 16
config.core.Yolov5Train.num_workers = 8
config.core.Yolov5Train.pin_memory = True
config.core.Yolov5Train.nominal_batch_factor = 4
config.core.Yolov5Train.model_path = "output/trained.pth"
# load script and quantize model path
# config.core.Yolov5Train.jit_model_path = "<pretrained-model-path>"

### ---------------------------------- tensorboard -----------------------------
# config.core.Yolov5Train.tensorboard_port = "6007"
# config.core.Yolov5Train.tensorboard_ip = None

### ------------------------------ script and quantize -------------------------
# config.cast.TraceCast = AttrDict()
# config.cast.TraceCast.model_dir = "./script.pt"
# config.cast.TraceCast.static_quantize_dir = "./script.sq"
# config.cast.TraceCast.dynamic_quantize_dir = "./quantize.sq"

### ---------------------------------- dataset ---------------------------------
config.core.Yolov5Train.img_size = 416
config.core.Yolov5Train.border = [-config.core.Yolov5Train.img_size / 2] * 2

config.aug.HSVAug = AttrDict()
config.aug.HSVAug.hgain = 0.015
config.aug.HSVAug.sgain = 0.7
config.aug.HSVAug.vgain = 0.4

config.aug.YoloPerspectiveAug = AttrDict()
config.aug.YoloPerspectiveAug.scale = 0.5
config.aug.YoloPerspectiveAug.shear = 0.0
config.aug.YoloPerspectiveAug.degrees = 0.0
config.aug.YoloPerspectiveAug.translate = 0.1
config.aug.YoloPerspectiveAug.perspective = 0.0
config.aug.YoloPerspectiveAug.border = config.core.Yolov5Train.border

config.datasets.Yolov5MosaicDataset = AttrDict()
config.datasets.Yolov5MosaicDataset.composer = Yolov5TrainComposer(config)
train_sample_path = "data/coco/images/train2017"
train_target_path = "data/coco/instances_train2017.json"
config.core.Yolov5Train.train_dataset = Yolov5MosaicDataset(config, train_sample_path, train_target_path, config.core.Yolov5Train.img_size, config.core.Yolov5Train.border)
config.core.Yolov5Train.train_loader = torch.utils.data.DataLoader(config.core.Yolov5Train.train_dataset,
        batch_size=config.core.Yolov5Train.batch_size,
        shuffle=config.core.Yolov5Train.shuffle,
        num_workers=config.core.Yolov5Train.num_workers,
        pin_memory=config.core.Yolov5Train.pin_memory,
        collate_fn=Yolov5MosaicDataset.collate_fn)

config.datasets.Yolov5Dataset = AttrDict()
config.datasets.Yolov5Dataset.composer = Yolov5ValComposer(config)
val_sample_path = "data/coco/images/val2017"
val_target_path = "data/coco/instances_val2017.json"
config.core.Yolov5Train.val_dataset = Yolov5Dataset(config, val_sample_path, val_target_path, config.core.Yolov5Train.img_size)
config.core.Yolov5Train.val_loader = torch.utils.data.DataLoader(config.core.Yolov5Train.val_dataset,
        batch_size=config.core.Yolov5Train.batch_size,
        num_workers=config.core.Yolov5Train.num_workers,
        collate_fn=Yolov5Dataset.collate_fn)

### ---------------------------------- model -----------------------------------
config.core.Yolov5Train.strides = [8, 16, 32]
# support model include (Yolov5S, Yolov5L) now
config.core.Yolov5Train.net = Yolov5S(config.core.Yolov5Train.class_num, config.core.Yolov5Train.strides)

### ---------------------------------- optimizer -------------------------------
config.core.Yolov5Train.lr = 0.01
config.core.Yolov5Train.momentum = 0.937
config.core.Yolov5Train.nesterov = True
config.core.Yolov5Train.optimizer = torch.optim.SGD(config.core.Yolov5Train.net.pg0, lr=config.core.Yolov5Train.lr, momentum=config.core.Yolov5Train.momentum, nesterov=config.core.Yolov5Train.nesterov)
config.core.Yolov5Train.optimizer.add_param_group({'params': config.core.Yolov5Train.net.pg1, 'weight_decay': 5e-4})
config.core.Yolov5Train.optimizer.add_param_group({'params': config.core.Yolov5Train.net.pg2})

del config.core.Yolov5Train.net.pg0
del config.core.Yolov5Train.net.pg1
del config.core.Yolov5Train.net.pg2

### ---------------------------------- scheduler -------------------------------
config.core.Yolov5Train.lr_lambda = lambda x: ((1 + math.cos(x * math.pi / config.core.Yolov5Train.epoch_num)) / 2) * (1 - 0.2) + 0.2
config.core.Yolov5Train.scheduler = torch.optim.lr_scheduler.LambdaLR(config.core.Yolov5Train.optimizer, lr_lambda=config.core.Yolov5Train.lr_lambda)
config.core.Yolov5Train.warmup_bias_lr = 0.1
config.core.Yolov5Train.warmup_momentum = 0.8
config.core.Yolov5Train.warmup_iter = max(3 * len(config.core.Yolov5Train.train_loader), 1e3)

### ---------------------------------- loss ------------------------------------
cls_scale = 0.5 * config.core.Yolov5Train.class_num / 80
box_scale = 0.05
obj_scale = 1.0
config.core.Yolov5Train.criterion = Yolov5Loss(config, config.core.Yolov5Train.net.detect, cls_scale, box_scale, obj_scale, config.core.Yolov5Train.strides, config.core.Yolov5Train.device)

################################################################################
### TEST
################################################################################
config.core.Yolov5Test = AttrDict()
### ---------------------------------- test ------------------------------------
config.core.Yolov5Test.device = "cpu"
config.core.Yolov5Test.class_num = 80
config.core.Yolov5Test.img_size = 640
config.core.Yolov5Test.strides = [8, 16, 32]
config.core.Yolov5Test.model_reinterpret_cast = True
config.core.Yolov5Test.cast_state_dict_strict = True
config.core.Yolov5Test.net = Yolov5S(config.core.Yolov5Test.class_num, config.core.Yolov5Test.strides)

config.core.Yolov5Test.model_path = "output/trained.pth"
config.core.Yolov5Test.test_sample_path = "your test sample path"
config.core.Yolov5Test.half = False
config.core.Yolov5Test.show_output_dir = "output/show"
config.core.Yolov5Test.iou_thres = 0.45
config.core.Yolov5Test.conf_thres = 0.25
config.core.Yolov5Test.idx2cat = ["cls{}".format(i) for i in range(config.core.Yolov5Test.class_num)]

################################################################################
### CAST
config.core.Yolov5Train.cast2cpu = True
config.core.Yolov5Test.cast2cpu = True
################################################################################
import coremltools

config.cast.CoremlCast = AttrDict()
config.cast.TraceCast = AttrDict()

config.cast.TraceCast.model_dir = "output/trace.pt"
config.cast.CoremlCast.model_dir = "output/coreml.mlmodel"
config.cast.CoremlCast.input_type = None
config.cast.CoremlCast.scale = 1.0 / 255.0
config.cast.CoremlCast.color_layout = 'BGR'
config.cast.CoremlCast.blue_bias = 0
config.cast.CoremlCast.green_bias = 0
config.cast.CoremlCast.red_bias = 0
config.cast.CoremlCast.minimum_deployment_target = coremltools.target.iOS13
config.cast.CoremlCast.classfier_config = ["cls{}".format(i) for i in range(config.core.Yolov5Test.class_num)]
