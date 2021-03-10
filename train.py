#! /usr/bin/python3
# -*- coding:utf-8 -*-
import torch
import numpy as np

from utils import *
from modules.model import Yolov5L
from data.dataset import Yolov5MosaicDataset
from deepvac import LOG, Yolov5Loss, DeepvacTrain, ClassifierReport


class DeepvacYolov5Train(DeepvacTrain):
    def __init__(self, config):
        super(DeepvacYolov5Train, self).__init__(config)
        if self.conf.report:
            self.report = ClassifierReport(ds_name=self.conf.ds_name, cls_num=self.conf.class_num)

    def auditConfig(self):
        super(DeepvacYolov5Train, self).auditConfig()
        self.warmup_iter = max(round(self.conf.warmup_epochs * len(self.train_loader)), 1e3)

    def initNetWithCode(self):
        self.net = Yolov5L(self.conf.class_num, self.conf.strides)
        self.conf.model = self.net

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
        if ni > self.warmup_iter:
            return
        xi = [0, self.warmup_iter]
        for j, x in enumerate(self.optimizer.param_groups):
            x['lr'] = np.interp(ni, xi, [self.conf.warmup_bias_lr if j == 2 else 0.0, x["initial_lr"] * self.conf.lr_step(self.epoch)])
            if 'momentum' in x:
                x['momentum'] = np.interp(ni, xi, [self.conf.warmup_momentum, self.conf.momentum])

    def doLoss(self):
        self.loss, loss_items = self.criterion(self.net.detect.output, self.target)
        self.addScalar('{}/boxLoss'.format(self.phase), loss_items[0], self.epoch)
        self.addScalar('{}/objLoss'.format(self.phase), loss_items[1], self.epoch)
        self.addScalar('{}/clsLoss'.format(self.phase), loss_items[2], self.epoch)

        if self.is_train or (not self.conf.report):
            return
        preds = [postProcess(i, self.conf.test.conf_thres, self.conf.test.iou_thres) for i in self.output]
        res = metrics(self.sample, preds, self.target, 0.7)
        if not res:
            return
        for gt, pred in res:
            self.report.add(gt, pred)

    def postEpoch(self):
        self.accuracy = 0
        if self.is_train or (not self.conf.report):
            return
        self.report()
        self.accuracy = self.report.accuracy
        self.report.reset()


if __name__ == '__main__':
    from config import config

    det = DeepvacYolov5Train(config)
    det()

