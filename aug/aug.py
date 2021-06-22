#! /usr/bin/python3
# -*- coding:utf-8 -*-
from deepvac.aug import AugFactory, Composer


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
