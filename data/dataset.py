#! /usr/bin/python3
# -*- coding:utf-8 -*-
import os
import cv2
import sys
import torch
import random
sys.path.append("../deepvac")
import numpy as np

from PIL import Image
from config import config
from deepvac.syszux_loader import CocoCVDataset
from deepvac.syszux_executor import YoloAugExecutor


class Yolov5MosaicDataset(CocoCVDataset):
    def __init__(self, deepvac_config):
        super(Yolov5MosaicDataset, self).__init__(deepvac_config)
        self.border = deepvac_config.border
        self.img_size = deepvac_config.img_size
        self.aug_executor = YoloAugExecutor(deepvac_config) if deepvac_config.augment else None

    def _getSample(self, index):
        label4 = []
        yc, xc = [int(random.uniform(-x, 2 * self.img_size + x)) for x in self.border]
        indices = [index] + [random.randint(0, len(self.ids)-1) for _ in range(3)]
        for i, index in enumerate(indices):
            # build img
            img = self.loadImgs(index)
            h0, w0, _ = img.shape
            r = self.img_size / max(h0, w0)
            if r != 1:
                interp = cv2.INTER_AREA if r < 1 and not self.aug_executor else cv2.INTER_LINEAR
                img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)
            h, w, _ = img.shape
            hr, wr = h / h0, w / w0
            if i == 0:
                img4 = np.full((self.img_size * 2, self.img_size * 2, img.shape[2]), 114, dtype=np.uint8)
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
            elif i == 1:
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, self.img_size * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(self.img_size * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, self.img_size * 2), min(self.img_size * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]
            padw = x1a - x1b
            padh = y1a - y1b
            # build label
            cls, det, _ = self.loadAnns(index)
            label = np.hstack((cls.reshape(-1, 1), det))
            label = label if label.sum() else np.empty((0, 5))
            if label.size:
                label[:, 3] = wr * (label[:, 1] + label[:, 3]) + padw
                label[:, 4] = hr * (label[:, 2] + label[:, 4]) + padh
                label[:, 1] = wr * label[:, 1] + padw
                label[:, 2] = hr * label[:, 2] + padh
            label4.append(label)
        label4 = np.concatenate(label4, 0)
        if label4.size:
            np.clip(label4[:, 1:], 0, 2 * self.img_size, out=label4[:, 1:])
        return [img4, label4], None

    def _buildSample(self, img, target):
         img, label = img
         # img
         img = img[:, :, ::-1].transpose(2, 0, 1)
         img = np.ascontiguousarray(img)
         img = torch.from_numpy(img)
         # label
         label_out = torch.zeros(len(label), 6)
         label_out[:, 1:] = torch.from_numpy(label)
         return img, label_out

    @staticmethod
    def collate_fn(batch):
        img, label = zip(*batch)
        for i, l in enumerate(label):
            l[:, 0] = i
        return torch.stack(img, 0), torch.cat(label, 0)

