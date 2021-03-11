# -*- coding:utf-8 -*-
import cv2
import math
import torch
import numpy as np

from torchvision import ops
from torch.nn import functional as F


def restoreCoords(img, coords, img0=None, ratio_pad=None):

    """
    Rescale box coords to origin image size
    Argument:
        img:       Tensor[N, C, H, W]
        coords:    Tensor[N, 6] = (x1, y1, x2, y2, conf, cls)
        img0:      CVMat[H0, W0, C]
        radio_pad: List = [radio, (h_pad, w_pad)]
    Returns:
        coords:    Tensor[N, 6]
    """

    assert img0 or ratio_pad, "you must speify one of (img0, radio_pad)"

    if ratio_pad is None:
        h, w = img.shape[2:]
        h0, w0 = img0.shape[:2]
        gain = min(h / h0, w / w0)
        pad = (w - w0 * gain) / 2, (h - h0 * gain) / 2
    else:
        gain, pad = ratio_pad

    coords[:, [0, 2]] -= pad[0]
    coords[:, [1, 3]] -= pad[1]
    coords[:, :4] /= gain
    coords[:, :4].clamp_(0)
    return coords


def xywh2xyxy(x):

    """
    Argument:
        x: [cx, cy, w, h]
    Returns:
        y: [x1, y1, x2, y2]
    """

    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


def boxIou(box1, box2):

    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 Tensor[N, 4]
        box2 Tensor[M, 4]
    Returns:
        iou  Tensor[N, M]
    """

    def box_area(box):
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)


def metrics(samples, preds, targets, ious_thres):

    """
    get classifer result per sample [Pred, Gt]
    Arguments:
        samples    Tensor[N, C, H, W]
        preds      Tensor[N, 6] = [x1, y1, x2, y2, cls, conf]
        targets    Tensor[M, 6] = [id, cls, x1, y1, w, h]
        radio_pad: List = [radio, (h_pad, w_pad)]
    Returns:
        res:       List = [[pred, gt], ... ...]
    """

    h, w = samples.shape[2:]
    targets[:, [2, 4]] *= w
    targets[:, [3, 5]] *= h

    res = []
    for i, pred in enumerate(preds):
        if not pred.size(0):
            continue

        labels = targets[targets[:, 0] == i, 1:]
        nl = len(labels)
        if not nl:
            continue

        tbox = xywh2xyxy(labels[:, 1:])

        ious, idx = boxIou(tbox, pred[:, :4]).max(1)
        tcls = (ious > ious_thres).nonzero(as_tuple=False).view(-1)
        pcls = idx[tcls]
        res.extend(zip(tcls.tolist(), pcls.tolist()))
    return res


def letterBox(img, img_size, color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):

    """
    resize and pad cv img to input sample size
    Arguments:
        img:       cvMat
        img_size:  int or tuple = (w, h)
        color:     pad color
        auto:      bool if True, pad with color
        scaleFill: bool if True, no pad and auto=False
        scaleup:   bool if True, can resize to large size
    Returns:
        img:       Resized cvMat
    """

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


def scaleImg(img, ratio=1.0, same_shape=False, gs=32):

    """
    scale img by ratio
    Arguments:
        img:       Tensor[N, C, H, W]
        ratio:     float
        same_shape:bool if False, w and h can divide by gs
        gs:        int stride
    Returns:
        img:       Tensor[N, C, H, W]
    """

    if ratio == 1.0:
        return img
    else:
        h, w = img.shape[2:]
        s = (int(h * ratio), int(w * ratio))
        img = F.interpolate(img, size=s, mode='bilinear', align_corners=False)
        if not same_shape:
            h, w = [math.ceil(x * ratio / gs) * gs for x in (h, w)]
    return F.pad(img, [0, w - s[1], 0, h - s[0]], value=0.447)


def postProcess(output, conf_thres, iou_thres):

    """
    postprocess to net output
    Arguments:
        output:    Tensor[N, C, (cls+4)*3]
        conf_thres:float
        iou_thres: float
    Returns:
        preds:     Tensor[N, C', (cls+4)*3]
    """

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


def plotRectangle(image, preds, save_dir):

    """
    draw retangle on image
    Arguments:
        image:     str image file path
        preds:     [classes, scores, coords]
        save_dir:  output image file save dir
    """

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
