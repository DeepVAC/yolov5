#! /usr/bin/python3
# -*- coding:utf-8 -*-
import numpy as np
import torch
from deepvac import LOG, DeepvacTrain


class Yolov5Train(DeepvacTrain):
    def preIter(self):
        if not self.config.is_train:
            return
        ni = self.config.iter
        if ni > self.config.warmup_iter:
            return
        xi = [0, self.config.warmup_iter]
        for j, x in enumerate(self.config.optimizer.param_groups):
            x['lr'] = np.interp(ni, xi, [self.config.warmup_bias_lr if j == 2 else 0.0, x["initial_lr"] * self.config.lr_lambda(self.config.epoch)])
            if 'momentum' in x:
                x['momentum'] = np.interp(ni, xi, [self.config.warmup_momentum, self.config.momentum])

    def doLoss(self):
        self.config.loss, loss_items = self.config.criterion(self.config.net.detect.output, self.config.target)
        self.addScalar('{}/boxLoss'.format(self.config.phase), loss_items[0], self.config.epoch)
        self.addScalar('{}/objLoss'.format(self.config.phase), loss_items[1], self.config.epoch)
        self.addScalar('{}/clsLoss'.format(self.config.phase), loss_items[2], self.config.epoch)
        self.config.accuracy = 0


if __name__ == '__main__':
    from config import config

    det = Yolov5Train(config)
    det()

