#! /usr/bin/python3
# -*- coding:utf-8 -*-
import os
import cv2
import numpy as np
import torch
from torchvision import ops
from torch.nn import functional as F


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
        if not tcls.numel():
            continue

        idx  = idx[tcls]
        print(pred.shape)
        pcls = pred[:, 5:].argmax(1)[idx]
        print(">>> tcls: ", tcls)
        print(">>> pcls: ", pcls)
        res.extend(zip(tcls.tolist(), pcls.tolist()))
    return res


def postProcess(output, conf_thres=0.25, iou_thres=0.45):

    """
    postprocess to net output
    Arguments:
        output:    Tensor[N, C, (cls+4)*3]
        conf_thres:float
        iou_thres: float
    Returns:
        preds:     Tensor[N, C', (cls+4)*3]
    """

    preds = output[output[..., 4] > conf_thres]
    if not preds.size(0):
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
    preds = torch.cat((preds[:, :4], conf, idx), dim=1)[conf.view(-1) > conf_thres]
    if not preds.size(0):
        return
    # nms on per class
    max_side = 4096
    class_offset = preds[:, 5:6] * max_side
    boxes, scores = preds[:, :4] + class_offset, preds[:, 4]
    idxs = ops.nms(boxes, scores, iou_thres)
    preds = torch.stack([preds[i] for i in idxs], dim=0)
    if not preds.size(0):
        return
    return preds


def plotRectangle(filepath, preds, savedir, idx2cat, pallete=None):
    """
    draw pred rectangle on origin img
    Arguments:
        filepath: origin img file path
        preds: (cls_ids, scores, coords)
        savedir: result img save dir
        idx2cat: map idx to class name
        pallete: provide different color for cls
    """
    if not preds[0]:
        return

    assert os.path.exists(savedir)
    if pallete:
        assert len(pallete) >= max(preds[0]), "pallete length must >= max class num"

    filename = filepath.split('/')[-1]
    new_filepath = os.path.join(savedir, filename)

    cv_img = cv2.imread(filepath)

    for cls, score, coord in zip(*preds):
        if (max(coord) > max(cv_img.shape)) or min(coord) < 0:
            continue
        color = pallete[cls] if pallete else (255, 255, 255)
        text = "{0}-{1:.2f}".format(idx2cat[cls], score)
        cv2.rectangle(cv_img, (coord[0], coord[1]), (coord[2], coord[3]), color, 2)
        cv2.putText(cv_img, text, (coord[0] - 5, coord[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.imwrite(new_filepath, cv_img)
