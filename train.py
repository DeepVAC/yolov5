#! /usr/bin/python3
# -*- coding:utf-8 -*-
import os
import sys
sys.path.append("../deepvac")
import math
import torch
import numpy as np

from copy import deepcopy
from deepvac.syszux_log import LOG
from deepvac.syszux_yolo import Yolov5L
from deepvac.syszux_loss import Yolov5Loss
from data.dataset import Yolov5MosaicDataset
from deepvac.syszux_deepvac import DeepvacTrain


class ModelEMA:
    def __init__(self, model, deepvac_config):
        self.ema = deepcopy(model.module if self.is_parallel(model) else model).eval()
        self.ema = self.ema.to(deepvac_config.device)
        self.updates = deepvac_config.updates
        self.decay = lambda x: deepvac_config.decay * (1 - math.exp(-x / 2000))
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)
            msd = model.module.state_dict() if self.is_parallel(model) else model.state_dict()
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1. - d) * msd[k].detach()

    def is_parallel(self, model):
        return type(model) in (torch.nn.parallel.DataParallel, torch.nn.parallel.DistributedDataParallel)

    def update_attr(self, model, include=(), exclude=('process_group', 'reducer')):
        copy_attr(self.ema, model, include, exclude)


class DeepvacYolov5Train(DeepvacTrain):
    def __init__(self, config):
        super(DeepvacYolov5Train, self).__init__(config)
        self.warmup_iter = max(round(self.conf.warmup_epochs * len(self.train_loader)), 1e3)

    def initNetWithCode(self):
        self.net = Yolov5L(self.conf)
        self.conf.model = self.net
        self.net.detect.is_training = True
        if self.conf.ema:
            self.ema = ModelEMA(self.net, self.conf)

    def initTrainLoader(self):
        self.train_dataset = Yolov5MosaicDataset(self.conf.train)
        self.train_loader = torch.utils.data.DataLoader(dataset=self.train_dataset,
                                                        batch_size=self.conf.train.batch_size,
                                                        shuffle=self.conf.train.shuffle,
                                                        num_workers=self.conf.train.num_workers,
                                                        pin_memory=self.conf.train.pin_memory,
                                                        collate_fn=Yolov5MosaicDataset.collate_fn)

    def initValLoader(self):
        self.val_dataset = Yolov5MosaicDataset(self.conf.val)
        self.val_loader = torch.utils.data.DataLoader(dataset=self.val_dataset,
                                                        batch_size=self.conf.val.batch_size,
                                                        collate_fn=Yolov5MosaicDataset.collate_fn)

    def initSgdOptimizer(self):
        pg0, pg1, pg2 = [], [], []
        for k, v in self.net.named_parameters():
            v.requires_grad = True
            if '.bias' in k:
                pg2.append(v)
            elif (k.split('.')[-2] in ['1', "bn"]) and ("detect" not in k):
                pg0.append(v)
            else:
                pg1.append(v)

        self.optimizer = torch.optim.SGD(pg0, lr=self.conf.lr0, momentum=self.conf.momentum, nesterov=self.conf.nesterov)
        self.optimizer.add_param_group({'params': pg1, 'weight_decay': self.conf.weight_decay})
        self.optimizer.add_param_group({'params': pg2})
        del pg0, pg1, pg2

    def initCriterion(self):
        self.criterion = Yolov5Loss(self.conf)

    def preIter(self):
        if self.is_val:
            return
        ni = self.iter
        if ni <= self.warmup_iter:
            xi = [0, self.warmup_iter]
            for j, x in enumerate(self.optimizer.param_groups):
                x['lr'] = np.interp(ni, xi, [self.conf.warmup_bias_lr if j == 2 else 0.0, x["initial_lr"] * self.conf.lr_step(self.epoch)])
                if 'momentum' in x:
                    x['momentum'] = np.interp(ni, xi, [self.conf.warmup_momentum, self.conf.momentum])

    def earlyIter(self):
        super(DeepvacYolov5Train, self).earlyIter()
        self.sample = self.sample.float() / 255.

    def doLoss(self):
        self.loss, loss_items = self.criterion(self.output, self.target)
        self.addScalar('{}/boxLoss'.format(self.phase), loss_items[0], self.epoch)
        self.addScalar('{}/objLoss'.format(self.phase), loss_items[1], self.epoch)
        self.addScalar('{}/clsLoss'.format(self.phase), loss_items[2], self.epoch)
        self.accuracy = 0

    def doBackward(self):
        if self.conf.amp:
            self.scaler.scale(self.loss).backward()
        else:
            self.loss.backward()

    def doOptimize(self):
        super(DeepvacYolov5Train, self).doOptimize()
        if self.conf.ema:
            self.ema.update(self.net)

    def process(self):
        self.iter = 0
        epoch_start = self.epoch
        if self.conf.ema:
            self.ema.updates = epoch_start * len(self.train_loader) // self.conf.nominal_batch_factor
        self.optimizer.zero_grad()
        for epoch in range(epoch_start, self.conf.epoch_num):
            self.epoch = epoch
            LOG.logI('Epoch {} started...'.format(self.epoch))
            self.processTrain()
            if self.epoch % 10 == 0:
                self.processVal()
                self.processAccept()


if __name__ == '__main__':
    from config import config

    det = DeepvacYolov5Train(config)
    det()

