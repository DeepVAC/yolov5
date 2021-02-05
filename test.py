import cv2
import torch
import numpy as np

from torchvision import ops
from deepvac.syszux_yolo import Yolov5L
from deepvac import LOG, Deepvac, OsWalkDataset


class Yolov5TestDataset(OsWalkDataset):
    def __init__(self, deepvac_config):
        self.conf = deepvac_config
        super(Yolov5TestDataset, self).__init__(deepvac_config.test)

    def __getitem__(self, index):
        image = self.files[index]
        img = cv2.imread(image, 1)
        h0, w0, _ = img.shape

        img = self.letterBox(img, self.conf.img_size)
        h, w, _ = img.shape

        r = min(h / h0, w / w0)
        pads = (w - w0 * r) / 2, (h - h0 * r) / 2

        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img)
        img = img.float() / 255.
        return image, img, r, pads

    @staticmethod
    def letterBox(img, img_size, color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
        shape = img.shape[:2]
        if isinstance(img_size, int):
            new_shape = (img_size, img_size)

        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:
            r = min(r, 1.0)
        ratio = r, r

        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        if auto:
            dw, dh = np.mod(dw, 32), np.mod(dh, 32)
        elif scaleFill:
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]
        dw /= 2
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return img


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
        none = 0
        for image, img, r, pads in self.test_loader:
            output = self.net(img.to(self.device))
            preds = self.postProcess(output, self.conf.test.conf_thres, self.conf.test.iou_thres)
            if preds.size(0):
                coords = preds[:, :4]
                coords[:, [0, 2]] -= pads[0].to(self.device)
                coords[:, [1, 3]] -= pads[1].to(self.device)
                coords /= r.to(self.device)
                coords = coords.long().tolist()

                scores = [i.item() for i in preds[:, -2]]

                classes = [self.conf.test.idx_to_cls[i] for i in preds[:, -1].long()]

                print("file: {0} >>> class: {1} >>> score: {2} >>> coord: {3}".format(image, classes, scores, coords))
            else:
                none += 1

            if self.conf.test.plot:
                preds = [classes, scores, coords] if preds.size(0) else []
                if self.conf.test.plot_dir is None:
                    self.conf.test.plot_dir = "output/detect"
                self.plotRectangle(image[0], preds, self.conf.test.plot_dir)
        print(">>> none: ", none)

    @staticmethod
    def postProcess(output, conf_thres, iou_thres):
        '''
            preds: [x1, y1, w, h, confi, one-hot * num_classes]
            return: [x1, y1, x2, y2, confi, cls)
        '''
        preds = output[output[..., 4] > conf_thres]  # filter with classifier confidence
        if not preds.shape[0]:
            return torch.zeros((0, 6))
        # Compute conf
        preds[:, 5:] *= preds[:, 4:5]  # conf = obj_conf * cls_conf
        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        preds[:, 0] = preds[:, 0] - preds[:, 2] / 2.0
        preds[:, 1] = preds[:, 1] - preds[:, 3] / 2.0
        preds[:, 2] += preds[:, 0]
        preds[:, 3] += preds[:, 1]
        # Detections matrix nx6 (xyxy, conf, cls)
        conf, idx = preds[:, 5:].max(dim=1, keepdim=True)
        preds = torch.cat((preds[:, :4], conf, idx), dim=1)[conf.view(-1) > conf_thres]  # filter with bbox confidence
        if not preds.size(0):
            return torch.zeros((0, 6))
        # nms on per class
        max_side = 4096
        class_offset = preds[:, 5:6] * max_side
        boxes, scores = preds[:, :4] + class_offset, preds[:, 4]
        idxs = ops.nms(boxes, scores, iou_thres)
        preds = torch.stack([preds[i] for i in idxs], dim=0)
        return preds

    @staticmethod
    def plotRectangle(image, preds, save_dir):
        file_name = image.split('/')[-1]
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        new_image = os.path.join(save_dir, file_name)

        image = cv2.imread(image)

        if not preds:
            cv2.imwrite(new_image, image)
            return

        for cls, score, coord in zip(*preds):
            if (max(coord) > max(image.shape)) or min(coord) < 0:
                continue
            cv2.rectangle(image, (coord[0], coord[1]), (coord[2], coord[3]), (0, 0, 255), 2)
            cv2.putText(image, "{0}-{1:.2f}".format(cls, score), tuple(coord[:2]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 3)
            cv2.imwrite(new_image, image)


if __name__ == "__main__":
    import os
    import sys

    from config import config
    '''
        you must assign:

        config.test.plot(optional)
        config.test.plot_dir(optional)

        config.class_num
        config.model_path
        config.test.input_dir
        config.test.idx_to_cls
        first
    '''
    config.test.input_dir = sys.argv[1]
    config.model_path = sys.argv[2]
    config.class_num = 4

    det = Yolov5Test(config)
    det()
