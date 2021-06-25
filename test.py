#! /usr/bin/python3
# -*- coding:utf-8 -*-
import os
import sys
import cv2
import numpy as np
import torch
from torchvision import ops
from deepvac import LOG, Deepvac
from deepvac.utils import pallete20
from data.datasets import Yolov5TestDataset
from modules.utils import setCoreml


class Yolov5Test(Deepvac):
    def __init__(self, deepvac_config):
        super(Yolov5Test, self).__init__(deepvac_config)
        os.makedirs(self.config.show_output_dir, exist_ok=True)
        if self.config.test_label_path is not None:
            ...
        if self.config.half:
            assert self.config.device.type == "cuda", "half forward must on cuda device"

    def test(self):
        self.config.non_det_num = 0
        super(Yolov5Test, self).test()
        LOG.logW("\nnon - detect: {} ".format(self.config.non_det_num) + ' * ' * 70)

    def preIter(self):
        self.config.ratio, self.config.pad, self.config.filepath = self.config.target
        assert len(self.config.filepath) == 1, 'config.core.Yolov5Test.test_batch_size must be set to 1 in current test mode.'

        self.config.ratio = torch.cat(self.config.ratio * 2).to(self.config.device)
        self.config.pad = torch.cat(self.config.pad * 2).to(self.config.device)
        self.config.filepath = self.config.filepath[0]

        if self.config.half:
            self.config.sample.half()

    def postIter(self):
        preds = self.config.output[self.config.output[..., 4] > self.config.conf_thres]  # filter with classifier confidence
        if not preds.size(0):
            LOG.logW("file: {0} >>> non - detect".format(self.config.filepath))
            self.config.non_det_num += 1
            return
        # Compute conf
        preds[:, 5:] *= preds[:, 4:5]  # conf = obj_conf * cls_conf
        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        preds[:, 0] = preds[:, 0] - preds[:, 2] / 2.0
        preds[:, 1] = preds[:, 1] - preds[:, 3] / 2.0
        preds[:, 2] += preds[:, 0]
        preds[:, 3] += preds[:, 1]
        # Detections matrix nx6 (xyxy, conf, cls)
        conf, idx = preds[:, 5:].max(dim=1, keepdim=True)
        preds = torch.cat((preds[:, :4], conf, idx), dim=1)[conf.view(-1) > self.config.conf_thres]  # filter with bbox confidence
        if not preds.size(0):
            LOG.logW("file: {0} >>> non - detect".format(self.config.filepath))
            self.config.non_det_num += 1
            return
        # nms on per class
        max_side = 4096
        class_offset = preds[:, 5:6] * max_side
        boxes, scores = preds[:, :4] + class_offset, preds[:, 4]
        idxs = ops.nms(boxes, scores, self.config.iou_thres)
        preds = torch.stack([preds[i] for i in idxs], dim=0)
        if not preds.size(0):
            LOG.logW("file: {0} >>> non - detect".format(self.config.filepath))
            self.config.non_det_num += 1
            return
        # coords scale
        classes = preds[:, -1].long().tolist()
        scores = [i.item() for i in preds[:, -2]]
        coords = preds[:, :4]
        coords -= self.config.pad
        coords /= self.config.ratio
        coords = coords.long().tolist()

        LOG.logI("file: {0} >>> class: {1} >>> score: {2} >>> coord: {3}".format(self.config.filepath, classes, scores, coords))
        self.plotRectangle(self.config.filepath, (classes, scores, coords), self.config.show_output_dir)

    def export3rd(self, output_file=None):
        if self.deepvac_config.cast and self.deepvac_config.cast.CoremlCast:
            setCoreml(self.config.net)
        super(Yolov5Test, self).export3rd(output_file)

    def plotRectangle(self, filepath, preds, save_dir):
        file_name = filepath.split('/')[-1]
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        new_filepath = os.path.join(save_dir, file_name)

        cv_img = cv2.imread(filepath)

        if not preds:
            return

        for cls, score, coord in zip(*preds):
            if (max(coord) > max(cv_img.shape)) or min(coord) < 0:
                continue
            # color = pallete20[cls]
            color = (0, 245, 233)
            text = "{0}-{1:.2f}".format(self.config.idx2cat[cls], score)
            cv2.rectangle(cv_img, (coord[0], coord[1]), (coord[2], coord[3]), color, 2)
            cv2.putText(cv_img, text, tuple(coord[:2]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 3)
            cv2.imwrite(new_filepath, cv_img)


if __name__ == "__main__":
    from config import config

    def check_args(idx, argv):
        return (len(argv) > idx) and (os.path.exists(argv[idx]))

    if check_args(1, sys.argv):
        config.core.Yolov5Test.model_path = sys.argv[1]
    if check_args(2, sys.argv):
        config.core.Yolov5Test.test_sample_path = sys.argv[2]
    if check_args(3, sys.argv):
        config.core.Yolov5Test.test_label_path = sys.argv[3]

    if (config.core.Yolov5Test.model_path is None) or (config.core.Yolov5Test.test_sample_path is None):
        helper = '''model_path or test_sample_path not found, please check:
                config.core.Yolov5Test.model_path or sys.argv[1] to init model path
                config.core.Yolov5Test.test_sample_path or sys.argv[2] to init test sample path
                config.core.Yolov5Test.test_label_path or sys.argv[3] to init test sample path (not required)
                for example:
                python3 test.py <trained-model-path> <test sample path> [test label path(not required)]'''
        print(helper)
        sys.exit(1)

    config.core.Yolov5Test.test_dataset = Yolov5TestDataset(config, config.core.Yolov5Test.test_sample_path, config.core.Yolov5Test.img_size)
    config.core.Yolov5Test.test_loader = torch.utils.data.DataLoader(config.core.Yolov5Test.test_dataset, batch_size=1)
    Yolov5Test(config)()

