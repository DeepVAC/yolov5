#! /usr/bin/python3
# -*- coding:utf-8 -*-
import cv2
from deepvac.aug import AugFactory, Composer, CvAugBase2


class HistEqualizeAug(CvAugBase2):
    def auditConfig(self):
        self.config.clahe = self.addUserConfig('clahe', self.config.clahe, True)

    def forward(self, img):
        cv_img, label = img
        yuv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2YUV)

        if self.config.clahe:
            cv_ops = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            yuv_img[:, :, 0] = cv_ops.apply(yuv_img[:, :, 0])
        else:
            yuv_img[:, :, 0] = cv2.equalizeHist(yuv_img[:, :, 0])

        img = cv2.cvtColor(yuv_img, cv2.COLOR_YUV2BGR)
        return img, label


class ReplicateAug(CvAugBase2):
    def forward(self, img):
        img, label = img

        h, w = img.shape[:2]
        boxes = label[:, 1:].astype(int)
        x1, y1, x2, y2 = boxes.T
        s = ((x2 - x1) + (y2 - y1)) / 2
        for i in s.argsort()[:round(s.size * 0.5)]:
            x1b, y1b, x2b, y2b = boxes[i]
            bh, bw = y2b - y1b, x2b - x1b
            yc, xc = int(random.uniform(0, h - bh)), int(random.uniform(0, w - bw))
            x1a, y1a, x2a, y2a = [xc, yc, xc + bw, yc + bh]
            img[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]
            label = np.append(label, [[label[i, 0], x1a, y1a, x2a, y2a]], axis=0)
        return img, label


class Yolov5AugFactory(AugFactory):
    def initProducts(self):
        super(Yolov5AugFactory, self).initProducts()
        aug_name = 'HistEqualizeAug'
        self.addProduct(aug_name, eval(aug_name))
        aug_name = 'RelicateAug'
        self.addProduct(aug_name, eval(aug_name))


class Yolov5TrainComposer(Composer):
    def __init__(self, deepvac_config):
        super(Yolov5TrainComposer, self).__init__(deepvac_config)
        ac = AugFactory("YoloPerspectiveAug => HSVAug@0.5 => YoloNormalizeAug => YoloHFlipAug@0.5", deepvac_config)
        self.addAugFactory("ac", ac)


class Yolov5ValComposer(Composer):
    def __init__(self, deepvac_config):
        super(Yolov5ValComposer, self).__init__(deepvac_config)
        ac = AugFactory("YoloNormalizeAug", deepvac_config)
        self.addAugFactory("ac", ac)
