import torch
import numpy as np

from utiles import utils_fish_landmark_detection
from utiles.py_cpu_nms import py_cpu_nms_box


class FishLandmarkRefinementWithBBoxnms:

    def __init__(self, param):
        # 类别数量
        # 输入图形长宽
        self.source_img_width = param.img_width
        self.source_img_height = param.img_height
        self.pytorch_device = torch.device("cpu" if param.CPU else "cuda")
        # [列，行，列，行]
        self.scale_bbox = torch.Tensor(
            [self.source_img_width, self.source_img_height, self.source_img_width, self.source_img_height]).float()
        self.scale_bbox = self.scale_bbox.to(self.pytorch_device)
        self.scale_key_pts = torch.Tensor([self.source_img_width, self.source_img_height,
                                           self.source_img_width, self.source_img_height,
                                           self.source_img_width, self.source_img_height,
                                           self.source_img_width, self.source_img_height
                                           ]).float()
        self.scale_key_pts = self.scale_key_pts.to(self.pytorch_device)
        # 极大值抑制过程用的阈值
        self.nms_thresh = param.nms_thresh_for_test
        if self.nms_thresh <= 0:
            raise ValueError('nms_threshold must be non negative.')
        # 分类概率阈值
        self.conf_thresh = param.conf_thresh_for_test
        # ??
        self.variance = param.variance
    #似乎只能一个一个来,暂时不能输入一个batch
    def forward(self, loc_data, conf_data, landm_data, prior_data):
        boxes = utils_fish_landmark_detection.decode_bbox(loc_data.data, prior_data, self.variance)
        boxes = boxes * self.scale_bbox
        boxes = boxes.cpu().numpy()
        scores = conf_data.squeeze(0).data.cpu().numpy()[:, 1]
        landms = utils_fish_landmark_detection.decode_landm_with_bbox(landm_data.data.squeeze(0), prior_data, self.variance)
        landms = landms * self.scale_key_pts
        landms = landms.cpu().numpy()

        # ignore low scores
        inds = np.where(scores > self.conf_thresh)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1]
        # order = scores.argsort()[::-1][:args.top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms_box(dets, self.nms_thresh)
        # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        dets = dets[keep, :]
        landms = landms[keep]

        # keep top-K faster NMS
        # dets = dets[:args.keep_top_k, :]
        # landms = landms[:args.keep_top_k, :]

        dets = np.concatenate((dets, landms), axis=1)

        return dets
