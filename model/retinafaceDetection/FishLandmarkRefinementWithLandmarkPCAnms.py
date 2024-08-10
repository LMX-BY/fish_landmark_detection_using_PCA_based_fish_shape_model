import torch
import numpy as np
from utiles import utils_fish_landmark_detection
from utiles.py_cpu_nms import py_cpu_nms_landmarks


class FishLandmarkRefinementWithLandmarkPCAnms:

    def __init__(self, train_param, network_param, pca_param_dict):
        # 类别数量
        # 输入图形长宽
        self.source_img_width = train_param.img_width
        self.source_img_height = train_param.img_height
        self.device = torch.device("cpu" if train_param.CPU else "cuda")
        # [列，行，列，行]
        self.scale_bbox = torch.Tensor(
            [self.source_img_width, self.source_img_height, self.source_img_width, self.source_img_height]).float()
        self.scale_bbox = self.scale_bbox.to(self.device)
        self.scale_landmarks = torch.Tensor([self.source_img_width, self.source_img_height,
                                             self.source_img_width, self.source_img_height,
                                             self.source_img_width, self.source_img_height,
                                             self.source_img_width, self.source_img_height
                                             ]).float()
        self.scale_landmarks = self.scale_landmarks.to(self.device)
        # 极大值抑制过程用的阈值
        self.nms_thresh = train_param.nms_thresh_for_test
        if self.nms_thresh < 0:
            raise ValueError('nms_threshold must be non negative.')
        # 分类概率阈值
        self.conf_thresh = train_param.conf_thresh_for_test
        # ??
        self.variance = train_param.variance
        self.top_k = 4000
        self.pca_feature_size = network_param.pca_feature_size
        self.pca_mean = pca_param_dict['mean_feature'].to(self.device).to(torch.float32)
        self.pca_features = pca_param_dict['svd_feature'].to(self.device).to(torch.float32)[0:self.pca_feature_size,:]
        self.img_size_normalized_pca_features = self.pca_features / self.scale_landmarks
        self.singular_values = pca_param_dict['normalized_singular_values'].to(self.device).to(torch.float32)[0:self.pca_feature_size]
        self.pca_features.requires_grad = False
        self.singular_values.requires_grad = False
        self.pca_mean.requires_grad = False

    def forward(self, loc_data, conf_data, landm_data, prior_data):
        prior_landmarks = prior_data['prior_landmarks']
        # decoded_landmarks_p = utils_fish_landmark_detection.decode_landmarks_with_prior_landmarks(landm_data.data,
        #                                                                                           prior_landmarks,
        #                                                                                        self.variance)
        decoded_landmarks_p = utils_fish_landmark_detection.decode_landmarks_with_pca_param(landm_data.data, prior_landmarks, self.img_size_normalized_pca_features)
        index_tensor_for_prior = torch.range(0,conf_data.size(0)).int().unsqueeze(1).cuda()
        scores = conf_data.squeeze(0)[:, 1]
        # landms = utils_fish_landmark_detection.decode_landmarks_with_prior_landmarks(landm_data.data.squeeze(0), prior_data, self.variance)
        # landms = landms * self.scale_key_pts
        # landms = landms.cpu().np()

        # ignore low scores
        inds = torch.where(scores > self.conf_thresh)[0]
        decoded_landmarks_p = decoded_landmarks_p[inds]
        prior_landmarks_copy = prior_landmarks[inds].detach().cpu().clone()
        index_tensor_for_prior = index_tensor_for_prior[inds]
        # landms = landms[inds]
        scores = scores[inds]

        # # keep top-K before NMS
        # order = scores.argsort()[::-1]
        # # order = scores.argsort()[::-1][:args.top_k]
        # decoded_landmarks_p = decoded_landmarks_p[order]
        # # landms = landms[order]
        # scores = scores[order]

        # do NMS
        keep = py_cpu_nms_landmarks(decoded_landmarks_p, scores, self.nms_thresh, self.top_k)
        # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        decoded_landmarks_p = decoded_landmarks_p[keep, :]
        decoded_landmarks_p = decoded_landmarks_p * self.scale_landmarks
        scores = scores[keep].unsqueeze(1)
        prior_landmarks_copy = prior_landmarks_copy[keep, :]
        index_tensor_for_prior = index_tensor_for_prior[keep, :]
        # landms = landms[keep]

        # keep top-K faster NMS
        # dets = dets[:args.keep_top_k, :]
        # landms = landms[:args.keep_top_k, :]

        dets = torch.cat([decoded_landmarks_p, scores], dim=1)

        return dets, prior_landmarks_copy, index_tensor_for_prior
