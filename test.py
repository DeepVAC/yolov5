import torch
import numpy as np

from utils import *
from modules.model import Yolov5L
from deepvac import LOG, Deepvac, OsWalkDataset


class Yolov5TestDataset(OsWalkDataset):
    def __init__(self, deepvac_config):
        self.conf = deepvac_config
        super(Yolov5TestDataset, self).__init__(deepvac_config.test)

    def __getitem__(self, index):
        image = self.files[index]
        img = cv2.imread(image, 1)
        h0, w0, _ = img.shape

        img = letterBox(img, self.conf.img_size)
        h, w, _ = img.shape

        r = min(h / h0, w / w0)
        pads = (w - w0 * r) / 2, (h - h0 * r) / 2

        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img)
        img = img.float() / 255.
        return image, img, r, pads


class Yolov5Test(Deepvac):
    def __init__(self, deepvac_config):
        super(Yolov5Test, self).__init__(deepvac_config)
        self.initTestLoader()

    def initNetWithCode(self):
        self.net = Yolov5L(self.conf.class_num, self.conf.strides)

    def initTestLoader(self):
        self.test_dataset = Yolov5TestDataset(self.conf)
        # to plot, set batch_size=1
        # to save cuda memory, set pin_memory=False
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=1, pin_memory=False)

    def process(self):
        # found half bug, if turn on, will detect None
        # half = self.device.type == "cuda"
        half = False
        # half net parameters
        if half:
            self.net = self.net.half()

        for image, img, r, pad in self.test_loader:
            r = r.to(self.device)
            img = img.to(self.device)
            pad = [i.to(self.device) for i in pad]
            # half img inputs
            if half:
                img = img.half()
            # predict augment
            if self.conf.test.augment:
                output = self.augProcess(img)
            else:
                output = self.net(img)
            # post process
            preds = postProcess(output, self.conf.test.conf_thres, self.conf.test.iou_thres)
            if preds.size(0):
                coords = restoreCoords(img, preds[:, :4], ratio_pad=(r, pad)).long().tolist()
                scores = [i.item() for i in preds[:, -2]]
                classes = [self.conf.test.idx_to_cls[i] for i in preds[:, -1].long()]

                print("file: {0} >>> class: {1} >>> score: {2} >>> coord: {3}".format(image, classes, scores, coords))

            if self.conf.test.plot:
                preds = [classes, scores, coords] if preds.size(0) else []
                if self.conf.test.plot_dir is None:
                    self.conf.test.plot_dir = "output/detect"
                plotRectangle(image[0], preds, self.conf.test.plot_dir)

    def augProcess(self, img):
        h, w = img.shape[-2:]

        y = []
        for r, p in zip(self.conf.test.radios, self.conf.test.flip_p):
            x = scaleImg(img.flip(p) if p else img, r, gs=max(self.conf.strides))
            yi = self.net(x)
            yi[..., :4] /= r
            if p == 2:
                yi[..., 1] = h - yi[..., 1]
            elif p == 3:
                yi[..., 0] = w - yi[..., 0]
            y.append(yi)
        return torch.cat(y, 1)


if __name__ == "__main__":
    import os

    from config import config
    '''
        you must assign:

        config.test.plot(optional)
        config.test.plot_dir(optional)

        config.class_num
        config.model_path | config.jit_model_path
        config.test.input_dir
        config.test.idx_to_cls
        first
    '''

    det = Yolov5Test(config)
    det()
