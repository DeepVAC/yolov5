#! /usr/bin/python3
# -*- coding:utf-8 -*-
import math
import torch
import torch.nn as nn
from deepvac.loss import LossBase


class Yolov5Loss(LossBase):
    def __init__(self, deepvac_config, det, cls_scale, box_scale, obj_scale, strides=[8, 16, 32], device="cpu"):
        super(Yolov5Loss, self).__init__(deepvac_config)
        self.cls, self.box, self.obj = cls_scale, box_scale, obj_scale
        self.strides = torch.Tensor(strides)
        self.device = device
        # define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([self.cls_pw], device=self.device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([self.obj_pw], device=self.device))
        if self.fl_gamma > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, self.fl_gamma), FocalLoss(BCEobj, self.fl_gamma)
        self.BCEcls, self.BCEobj = BCEcls, BCEobj
        # model info
        det = det.to(self.device)
        self.ssi = (self.strides == 16).nonzero(as_tuple=False).item()
        for k in ("anchor_num", "class_num", "detect_layer_num", "anchors"):
            setattr(self, k, getattr(det, k))

    def auditConfig(self):
        self.gr = 1.0
        self.anchor_t = 4
        self.cls_pw = 1.0
        self.obj_pw = 1.0
        self.fl_gamma = 0.0
        self.autobalance = False
        self.balance = [4.0, 1.0, 0.4]
        self.cp, self.cn = self.smoothBCE(eps=0.0)

    def build_target(self, pred, target):
        target_num = target.size(0)
        tcls, tbox, indices, anch = [], [], [], []
        # normalized to gridspace gain
        gain = torch.ones(7, device=self.device)
        # targets: (N, 6) -> (3, N, 7), at last append anchor index
        target = [torch.cat([target, i.repeat(target_num).view(target_num, 1)], dim=1) for i in torch.arange(self.anchor_num, device=self.device)]
        target = torch.stack(target, dim=0).to(self.device)

        off = torch.tensor([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]], device=self.device) * 0.5
        for i in range(self.detect_layer_num):
            anchors = self.anchors[i]
            # pred: [(n, c, h1, w1, 9), (n, c, h2, w2, 9), (n, c, h3, w3, 9)] gain: [1, 1, w, h, w, h, 1]
            gain[2:6] = torch.tensor(pred[i].shape)[[3, 2, 3, 2]]
            # target: [[[id, cls, cx, cy, w, h, anchor_id], ...n...]]
            # normalize by w, h -> origin size
            t = target * gain
            if target_num:
                # anchors: (3, 2) -> (3, 1, 2)
                wh_radio = t[:, :, 4:6] / anchors.unsqueeze(dim=1)
                # index: (3, N) t: (3, N, 7) -> (n, 7)
                index = torch.max(wh_radio, 1/wh_radio).max(dim=2)[0] < self.anchor_t
                t = t[index]
                # cxcy: [[cx, xy], ...n...]
                cxcy = t[:, 2:4]
                # inverse_cxcy: [[w-cx, h-cy], ...n...]
                inverse_cxcy = gain[[2, 3]] - cxcy
                # cx_index: x_index  cy_index: y_index
                cx_index, cy_index = ((cxcy.fmod(1.) < 0.5) & (cxcy > 1.)).T
                # inverse_cx_index: x_index  inverse_cy_index: y_index
                inverse_cx_index, inverse_cy_index = ((inverse_cxcy % 1. < 0.5) & (inverse_cxcy > 1.)).T
                # cx_index: (n) -> (5, n)
                cx_index = torch.stack((torch.ones_like(cx_index), cx_index, cy_index, inverse_cx_index, inverse_cy_index))
                # t: (n, 7) -> (5, n, 7) -> (n', 7)
                t = t.unsqueeze(dim=0).repeat((5, 1, 1))[cx_index]
                offsets = (torch.zeros_like(cxcy.unsqueeze(dim=0)) + off.unsqueeze(dim=1))[cx_index]
                # offsets = (torch.zeros_like(cxcy)[None] + off[:, None])[cx_index]
            else:
                t = target[0]
                offsets = 0

            img_id, cls = t[:, :2].long().T
            cxcy = t[:, 2:4]
            wh = t[:, 4:6]
            cxcy_index = (cxcy - offsets).long()
            cx_index, cy_index = cxcy_index.T

            anchor_index = t[:, 6].long()
            indices.append((img_id, anchor_index, cy_index.clamp_(0, gain[3] - 1), cx_index.clamp_(0, gain[2] - 1)))
            tbox.append(torch.cat((cxcy - cxcy_index, wh), 1))
            anch.append(anchors[anchor_index])
            tcls.append(cls)
        return tcls, tbox, indices, anch

    def compute_loss(self, p, tcls, tbox, indices, anchors, balance):
        img_id, anchor_index, cy_index, cx_index = indices
        tobj = torch.zeros_like(p[..., 0], device=self.device)
        target_num = img_id.size(0)
        if not target_num:
            lobj = self.BCEobj(p[..., 4], tobj) * balance
            balance = balance * 0.9999 + 0.0001 / lobj.detach().item() if self.autobalance else balance
            return 0, 0, lobj
        # p: [px, py, pw, ph, conf, cls] ...
        ps = p[img_id, anchor_index, cy_index, cx_index]
        # Regression
        pcxcy = ps[:, :2].sigmoid() * 2. - 0.5
        pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors
        pbox = torch.cat((pcxcy, pwh), 1)
        iou = self.bbox_iou(pbox.T, tbox, x1y1x2y2=False, CIou=True)
        lbox = (1.0 - iou).mean()
        # Objectness
        tobj[img_id, anchor_index, cy_index, cx_index] = (1.0 - self.gr) + self.gr * iou.detach().clamp(0).type(tobj.dtype)
        lobj = self.BCEobj(p[..., 4], tobj) * balance
        balance = balance * 0.9999 + 0.0001 / lobj.detach().item() if self.autobalance else balance
        # Classification
        if self.class_num <= 1:
            return lbox, 0, lobj
        t = torch.full_like(ps[:, 5:], self.cn, device=self.device)
        t[range(target_num), tcls] = self.cp
        lcls = self.BCEcls(ps[:, 5:], t)
        return lbox, lcls, lobj

    def __call__(self, pred, target):
        lcls, lbox, lobj = torch.zeros(1, device=self.device), torch.zeros(1, device=self.device), torch.zeros(1, device=self.device)
        tcls, tbox, indices, anchors = self.build_target(pred, target)
        # pred: [(n, c, h1, w1, 9), (n, c, h2, w2, 9), (n, c, h3, w3, 9)]
        for i, p in enumerate(pred):
            libox, licls, liobj = self.compute_loss(p, tcls[i], tbox[i], indices[i], anchors[i], self.balance[i])
            lbox += libox
            lcls += licls
            lobj += liobj
        self.balance = [x / self.balance[self.ssi] for x in self.balance] if self.autobalance else self.balance
        lbox *= self.box
        lobj *= self.obj
        lcls *= self.cls
        bs = pred[0].size(0)
        loss = lbox + lobj + lcls
        return loss * bs, torch.cat((lbox, lobj, lcls, loss)).detach()

    @staticmethod
    def bbox_iou(box1, box2, x1y1x2y2=True, GIou=False, DIou=False, CIou=False, eps=1e-9):
        box2 = box2.T
        if x1y1x2y2:
            b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
            b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
        else:
            b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
            b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
            b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
            b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2
        inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
                (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)
        coords = (b1_x1, b1_x2, b1_y1, b1_y2, b2_x1, b2_x2, b2_y1, b2_y2)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
        whs = (w1, h1, w2, h2)
        union = w1 * h1 + w2 * h2 - inter + eps
        iou = inter / union

        if not GIou and not DIou and not CIou:
            return iou
        elif GIou:
            return Yolov5Loss.compute_GIou(iou, coords, union)[-1]
        elif DIou:
            return Yolov5Loss.compute_DIou(iou, coords, union)[-1]
        elif CIou:
            return Yolov5Loss.compute_CIou(iou, coords, whs, union)

    @staticmethod
    def compute_GIou(iou, coords, union, eps=1e-9):
        b1_x1, b1_x2, b1_y1, b1_y2, b2_x1, b2_x2, b2_y1, b2_y2 = coords
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)
        c_area = cw * ch + eps
        giou = iou - (c_area - union) / c_area
        return cw, ch, giou

    @staticmethod
    def compute_DIou(iou, coords, union, eps=1e-9):
        cw, ch, _ = Yolov5Loss.compute_GIou(iou, coords, union)
        c2 = cw ** 2 + ch ** 2 + eps
        b1_x1, b1_x2, b1_y1, b1_y2, b2_x1, b2_x2, b2_y1, b2_y2 = coords
        rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4
        diou = iou - rho2 / c2
        return c2, rho2, diou

    @staticmethod
    def compute_CIou(iou, coords, whs, union, eps=1e-9):
        cw, ch, _ = Yolov5Loss.compute_GIou(iou, coords, union)
        c2, rho2, _ = Yolov5Loss.compute_DIou(iou, coords, union)
        w1, h1, w2, h2 = whs
        v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
        with torch.no_grad():
            alpha = v / ((1 + eps) - iou + v)
        ciou = iou - (rho2 / c2 + v * alpha)
        return ciou

    @staticmethod
    def smoothBCE(eps=0.1):
        return 1.0 - 0.5 * eps, 0.5 * eps
