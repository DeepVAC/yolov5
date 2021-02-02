import cv2
import torch
import numpy as np

from torchvision import ops
from deepvac.syszux_yolo import Yolov5L
from deepvac import LOG, Deepvac, OsWalkerLoader


class Yolov5Detection(Deepvac):
    def __init__(self, conf):
        conf.disable_git = True
        super(Yolov5Detection, self).__init__(conf)

    def initNetWithCode(self):
        self.net = Yolov5L(self.conf.class_num, self.conf.strides)
        self.net.detect.is_training = False

    def _letter_box(self, img, img_size, color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
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
        return img, ratio, (dw, dh)

    def _image_process(self, img):
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img)
        img = img.float()
        img /= 255.0
        img = img.unsqueeze(0)
        return img.to(self.device)

    def _post_process(self, preds):
        '''
            preds: [x1, y1, w, h, confi, one-hot * num_classes]
            return: [x1, y1, x2, y2, confi, cls)
        '''
        pred = preds[preds[..., 4] > self.conf.test.conf_thres]  # filter with classifier confidence
        if not pred.shape[0]:
            return torch.zeros((0, 6))
        # Compute conf
        pred[:, 5:] *= pred[:, 4:5]  # conf = obj_conf * cls_conf
        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        pred[:, 0] = pred[:, 0] - pred[:, 2] / 2.0
        pred[:, 1] = pred[:, 1] - pred[:, 3] / 2.0
        pred[:, 2] += pred[:, 0]
        pred[:, 3] += pred[:, 1]
        # Detections matrix nx6 (xyxy, conf, cls)
        conf, idx = pred[:, 5:].max(dim=1, keepdim=True)
        pred = torch.cat((pred[:, :4], conf, idx), dim=1)[conf.view(-1) > self.conf.test.conf_thres]  # filter with bbox confidence
        if not pred.shape[0]:
            return torch.zeros((0, 6))
        # nms on per class
        max_side = 4096
        class_offset = pred[:, 5:6] * max_side
        boxes, scores = pred[:, :4] + class_offset, pred[:, 4]
        idxs = ops.nms(boxes, scores, self.conf.test.iou_thres)
        pred = torch.stack([pred[i] for i in idxs], dim=0)
        return pred

    def _plot_rectangle(self, img, pred, file_path):
        file_name = file_path.split('/')[-1]
        save_dir = self.conf.test.plot_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        n, c, h, w = img.shape
        image = cv2.imread(file_path)
        h0, w0, c0 = image.shape

        if not len(pred):
            cv2.imwrite(os.path.join(save_dir, file_name), image)
            return

        for det in pred:
            coord = det[:4]
            score = det[4].item()
            cls = [self.conf.test.idx_to_cls[det[5].long()]]
            gain = min(h / h0, w / w0)
            pad = (w - w0 * gain) / 2, (h - h0 * gain) / 2
            coord[[0, 2]] -= pad[0]
            coord[[1, 3]] -= pad[1]
            coord /= gain
            coord = [int(x.item()) for x in coord]
            if (max(coord) > max(h0, w0)) or min(coord) < 0:
                continue
            cv2.rectangle(image, (coord[0], coord[1]), (coord[2], coord[3]), (0, 0, 255), 2)
            cv2.putText(image, "{0}-{1:.2f}".format(cls, score), tuple(coord[:2]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 3)
            cv2.imwrite(os.path.join(save_dir, file_name), image)

    def setInput(self, file_path):
        self.file_path = file_path
        image = cv2.imread(file_path, 1)
        img = self._letter_box(image, self.conf.img_size)[0]
        self.sample = self._image_process(img)

    def getOutput(self):
        pred = self._post_process(self.output)
        self.pred = pred
        if not pred.size(0):
            return None, 0
        if self.conf.test.plot:
            self._plot_rectangle(self.sample, self.pred, self.file_path)
        scores = pred[:, -2]
        classes = [self.conf.test.idx_to_cls[i] for i in pred[:, -1].long()]
        return classes, scores

    def process(self):
        self.output = self.net(self.sample)[0]


if __name__ == "__main__":
    import os

    from config import config
    '''
        you must assign:
        config.class_num
        config.model_path
        config.test.img_folder
        config.test.idx_to_cls
        first
    '''

    det = Yolov5Detection(config)
    test_dataset = OsWalkerLoader(config.test)
    for fp in test_dataset():
        print("img_file: ", fp)
        res = det(fp)
        print("result: ", res)
