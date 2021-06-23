#! /usr/bin/python3
# -*- coding:utf-8 -*-
import cv2
import torch
import random
import numpy as np
from deepvac.datasets import OsWalkBaseDataset, CocoCVDataset


class Yolov5MosaicDataset(CocoCVDataset):
    def __init__(self, deepvac_config, sample_path, target_path, img_size, border, cat2idx=None):
        super(Yolov5MosaicDataset, self).__init__(deepvac_config, sample_path, target_path, cat2idx)
        self.img_size = img_size
        self.border = border

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
                interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
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
            label = np.hstack((cls.reshape(-1, 1), det)) if cls.sum() else np.empty((0, 5))
            if label.size:
                label[:, 3] = wr * (label[:, 1] + label[:, 3]) + padw
                label[:, 4] = hr * (label[:, 2] + label[:, 4]) + padh
                label[:, 1] = wr * label[:, 1] + padw
                label[:, 2] = hr * label[:, 2] + padh
            label4.append(label)
        label4 = np.concatenate(label4, 0)
        if label4.size:
            np.clip(label4[:, 1:], 0, 2 * self.img_size, out=label4[:, 1:])

        # mixup or not
        if self.config.mixup:
            [img_mix, label_mix], _ = self._getSample(random.randint(0, self.__len__() - 1))
            r = np.random.beta(32.0, 32.0)
            img4 = (img4 * r + img_mix * (1 - r)).astype(np.uint8)
            label4 = np.concatenate((label4, label_mix), 0)
        # for transform or composer
        return [img4, label4], None

    def _buildSample(self, img, target):
         img, label = img
         # img
         img = img[:, :, ::-1].transpose((2, 0, 1))
         img = np.ascontiguousarray(img)
         img = torch.from_numpy(img)
         img = img.to(dtype=torch.get_default_dtype()).div(255)
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

    @staticmethod
    def collate_fn4(batch):
        img, label = zip(*batch)
        n = len(img) // 4  # batch
        img4, label4 = [], []

        ho = torch.tensor([[0., 0, 0, 1, 0, 0]])
        wo = torch.tensor([[0., 0, 1, 0, 0, 0]])
        s = torch.tensor([[1, 1, .5, .5, .5, .5]])

        for i in range(n):
            i *= 4
            if random.random() < 0.5:
                im = F.interpolate(img[i].unsqueeze(0).float(), scale_factor=2., mode='bilinear', align_corners=False)[0].type(img[i].type())
                # ratio auto fit to scale
                l = label[i]
            else:
                im = torch.cat((torch.cat((img[i], img[i + 1]), 1), torch.cat((img[i + 2], img[i + 3]), 1)), 2)
                l = torch.cat((label[i], label[i + 1] + ho, label[i + 2] + wo, label[i + 3] + ho + wo), 0) * s
            img4.append(im)
            label4.append(l)

        for i, l in enumerate(label4):
            l[:, 0] = i
        return torch.stack(img4, 0), torch.cat(label4, 0), path4, shapes4


class Yolov5Dataset(CocoCVDataset):
    def __init__(self, deepvac_config, sample_path, target_path, img_size, cat2idx=None):
        super(Yolov5Dataset, self).__init__(deepvac_config, sample_path, target_path, cat2idx)
        self.img_size = img_size

    def _getSample(self, index):
        # img
        img = self.loadImgs(index)
        # h0, w0, _ = img.shape
        # r = np.array(self.img_size) / max(h0, w0)
        # if r != 1:
        #     img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR)
        img, (wr, hr), (padw, padh) = letterbox(img, self.img_size, auto=False, scaleup=True)
        # label
        cls, det, _ = self.loadAnns(index)
        label = np.hstack((cls.reshape(-1, 1), det)) if cls.sum() else np.empty((0, 5))
        if label.size:
            # xywh2xyxy
            label[:, 3] = wr * (label[:, 1] + label[:, 3]) + padw
            label[:, 4] = hr * (label[:, 2] + label[:, 4]) + padh
            label[:, 1] = wr * label[:, 1] + padw
            label[:, 2] = hr * label[:, 2] + padh
            np.clip(label[:, 1:], 0, 2 * self.img_size, out=label[:, 1:])
        # for transform or composer
        return [img, label], None

    def _buildSample(self, img, target):
        img, label = img
        # img
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img)
        img = img.to(dtype=torch.get_default_dtype()).div(255)
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


class Yolov5TestDataset(OsWalkBaseDataset):
    def __init__(self, deepvac_config, sample_path, img_size):
        super(Yolov5TestDataset, self).__init__(deepvac_config, sample_path)
        self.img_size = img_size

    def __getitem__(self, index):
        filepath = self.files[index]
        img = cv2.imread(filepath, 1)

        img, r, pads = letterbox(img, self.img_size)

        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img)
        # img = img.to(dtype=torch.get_default_dtype()).div(255)
        img = img.float().div(255)
        return img, r, pads, filepath


def letterbox(img, target_shape, color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    h0, w0 = img.shape[:2]

    if isinstance(target_shape, (tuple, list)):
        h, w = target_shape
    elif isinstance(target_shape, int):
        h = w = target_shape
    else:
        raise TypeError("target_shape type must be [int, tuple, list]")

    r = min(h / h0, w / w0)
    if not scaleup:
        r = min(r, 1.0)
    ratio = r, r
    new_unpad = int(round(w0 * r)), int(round(h0 * r))
    dw, dh = w - new_unpad[0], h - new_unpad[1]
    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    elif scaleFill:
        dw, dh = 0.0, 0.0
        new_unpad = (w, h)
        ratio = w / w0, h / h0
    dw /= 2
    dh /= 2
    if (w0, h0) != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, ratio, (dw, dh)
