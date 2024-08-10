# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import numpy as np
import torch
from utiles import utils_fish_landmark_detection
def py_cpu_nms_box(dets, thresh):
    """Pure Python NMS baseline."""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    # print('*'*10)
    # print(f'max score {scores[0]}')

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep



def py_cpu_nms_landmarks(landmarks, scores, thresh, top_k):
    """Pure Python NMS baseline."""
    v, order_idx = scores.sort(0,descending = True)
    #print('*'*10)

    order_idx = order_idx[:top_k]
    # if v.size(0) != 0:
        # print(f'max score is {v[0]}')
        # print(f'last top_k score is {v[order_idx[-1]]}')
    keep = []
    while order_idx.size(0) > 0:
        # 取出分类评分最高的索引样本
        cur_index = int(order_idx[0])
        cur_max_score = scores[cur_index]
        keep.append(cur_index)
        order_idx = order_idx[1:]
        if order_idx.size(0) == 0:
            break
        # 取出分类评分最高的landmarks样本
        cur_optimal_landmarks = landmarks[cur_index].unsqueeze(0)
        #取出其他landmarks样本
        other_landmarks = torch.index_select(landmarks,0, order_idx)
        #相似性计算
        match_score = utils_fish_landmark_detection.fish_key_pts_match_score(cur_optimal_landmarks, other_landmarks)
        #取出相似程度大于阈值的样本
        order_idx = order_idx[match_score.gt(thresh).squeeze()]


    return keep
