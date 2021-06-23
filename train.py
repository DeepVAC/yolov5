#! /usr/bin/python3
# -*- coding:utf-8 -*-
import numpy as np
import torch
from deepvac import LOG, DeepvacTrain
from utils import postProcess, metrics


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

        if self.config.is_train or (not self.config.reporter):
            return
        preds = [postProcess(i) for i in self.config.output]
        res = metrics(self.config.sample, self.config.output, self.config.target, 0.7)
        if not res:
            return
        for gt, pd in res:
            print(">>> gt: ", gt)
            print(">>> pd: ", pd)
            self.config.reporter.add(gt, pd)

    def postEpoch(self):
        self.config.accuracy = 0
        if self.config.is_train or (not self.config.reporter):
            return
        self.config.reporter()
        self.config.accuracy = self.config.reporter.accuracy
        self.config.reporter.reset()


if __name__ == '__main__':
    from config import config

    det = Yolov5Train(config)
    det()

