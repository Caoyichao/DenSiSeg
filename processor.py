# -*- coding: utf-8 -*
import os
import numpy as np
import torch
import torch.nn as nn

from image import transform_preds
from utils import _gather_feat, _tranpose_and_gather_feat

def _nms(heat, kernel=11):
    pad = (kernel - 1) // 2

    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep


def _topk(scores, K=40):
    batch, cat, height, width = scores.size()

    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds / width).int().float()
    topk_xs = (topk_inds % width).int().float()

    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_clses = (topk_ind / K).int()
    topk_inds = _gather_feat(
        topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs


def ctdet_decode(heat, wh, reg=None, cat_spec_wh=False, K=100):
    batch, cat, height, width = heat.size()

    # heat = torch.sigmoid(heat)
    # perform nms on heatmaps
    heat = _nms(heat)

    scores, inds, clses, ys, xs = _topk(heat, K=K)
    if reg is not None:
        reg = _tranpose_and_gather_feat(reg, inds)
        reg = reg.view(batch, K, 2)
        xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
        ys = ys.view(batch, K, 1) + reg[:, :, 1:2]
    else:
        xs = xs.view(batch, K, 1) + 0.5
        ys = ys.view(batch, K, 1) + 0.5
    wh = _tranpose_and_gather_feat(wh, inds)
    if cat_spec_wh:
        wh = wh.view(batch, K, cat, 2)
        clses_ind = clses.view(batch, K, 1, 1).expand(batch, K, 1, 2).long()
        wh = wh.gather(2, clses_ind).view(batch, K, 2)
    else:
        wh = wh.view(batch, K, 2)
    clses = clses.view(batch, K, 1).float()
    scores = scores.view(batch, K, 1)
    bboxes = torch.cat([xs - wh[..., 0:1] / 2,
                        ys - wh[..., 1:2] / 2,
                        xs + wh[..., 0:1] / 2,
                        ys + wh[..., 1:2] / 2], dim=2)
    detections = torch.cat([bboxes, scores, clses], dim=2)

    return detections


def ctdet_post_process(dets, c, s, h, w, num_classes):
  # dets: batch x max_dets x dim
  # return 1-based class det dict
  ret = []
  for i in range(dets.shape[0]):
    top_preds = {}
    dets[i, :, :2] = transform_preds(
          dets[i, :, 0:2], c[i], s[i], (w, h))
    dets[i, :, 2:4] = transform_preds(
          dets[i, :, 2:4], c[i], s[i], (w, h))
    classes = dets[i, :, -1]
    for j in range(num_classes):
      inds = (classes == j)
      top_preds[j + 1] = np.concatenate([
        dets[i, inds, :4].astype(np.float32),
        dets[i, inds, 4:5].astype(np.float32)], axis=1).tolist()
    ret.append(top_preds)
  return ret
