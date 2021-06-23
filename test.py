#! /usr/bin/python3
# -*- coding:utf-8 -*-
import os
import sys
import cv2
import torch
from deepvac import LOG, Deepvac
from deepvac.utils import pallete20
from utils import postProcess, plotRectangle
from data.datasets import Yolov5TestDataset


pallete80 = pallete20 * 4

class Yolov5Test(Deepvac):
    def __init__(self, deepvac_config):
        super(Yolov5Test, self).__init__(deepvac_config)
        os.makedirs(self.config.show_output_dir, exist_ok=True)
        if self.config.test_label_path is not None:
            ...
        if self.config.half:
            assert self.config.device.type == "cuda", "half forward must on cuda device"
            self.config.net.half()

    def test(self):
        self.config.non_det_num = 0
        super(Yolov5Test, self).test()
        LOG.logW(' * ' * 25 + "non - detect: {} ".format(self.config.non_det_num) + ' * ' * 25)

    def preIter(self):
        self.config.ratio, self.config.pad, self.config.filepath = self.config.target
        assert len(self.config.filepath) == 1, 'config.core.Yolov5Test.test_batch_size must be set to 1 in current test mode.'

        self.config.ratio = torch.cat(self.config.ratio * 2).to(self.config.device)
        self.config.pad = torch.cat(self.config.pad * 2).to(self.config.device)
        self.config.filepath = self.config.filepath[0]

        if self.config.half:
            self.config.sample = self.config.sample.half()

    def postIter(self):
        preds = postProcess(self.config.output, self.config.conf_thres, self.config.iou_thres)
        if preds is None:
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
        plotRectangle(self.config.filepath, (classes, scores, coords), self.config.show_output_dir, self.config.idx2cat, pallete80)


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
