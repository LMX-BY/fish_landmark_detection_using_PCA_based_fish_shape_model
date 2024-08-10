import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utiles.box_utils import match, log_sum_exp
from utiles import utils_fish_landmark_detection
from model import focalloss

GPU = True
landmark_coordinate_size = 8


class RetinaFaceLandmarkPointsErrorLoss(nn.Module):

    def __init__(self, param):
        super(RetinaFaceLandmarkPointsErrorLoss, self).__init__()
        self.num_classes = param.num_of_classe
        self.positive_sample_score_threshold = param.pos_sample_score_threshold
        self.neg_pos_ratio = param.neg_pos_ratio
        self.variance = param.variance
        self.landmark_dim = param.landmark_size * 2
        self.useful_key_pys_num = 4
        if param.CPU:
            self.gpu = False
        else:
            self.gpu = True
        self.worst_score = 1

    def forward(self, predictions, priors, targets):

        loc_data, conf_data, landm_data = predictions
        priors_landmarks = priors['prior_landmarks']
        # !!先验要替换为keypts
        # priors = priors
        assert loc_data.size(1) == priors_landmarks.size(0)
        batch_size = loc_data.size(0)
        num_priors = (priors_landmarks.size(0))

        # match priors (default boxes) and ground truth boxes
        # 双向匹配问题
        loc_t = torch.Tensor(batch_size, num_priors, 4)
        landm_t = torch.Tensor(batch_size, num_priors, landmark_coordinate_size)
        conf_t = torch.LongTensor(batch_size, num_priors)
        for idx in range(batch_size):
            truths = targets[idx][:, :4].data
            gt_landms = targets[idx][:, 4:4 + self.landmark_dim].data
            labels = targets[idx][:, -1].data
            priors_landmarks_data = priors_landmarks.data  # ??.data
            utils_fish_landmark_detection.match_landmarks_cdist(self.positive_sample_score_threshold, gt_landms,
                                                                priors_landmarks_data, self.variance, labels, conf_t,
                                                                landm_t, idx, self.worst_score)
        if self.gpu:
            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()
            landm_t = landm_t.cuda()

        zeros = torch.tensor(0).cuda()
        # landm Loss (Smooth L1)
        # Shape: [batch,num_priors,10]
        pos1 = conf_t > zeros
        num_pos_landm = pos1.long().sum(1, keepdim=True)
        N1 = max(num_pos_landm.data.sum().float(), 1)
        pos_idx1 = pos1.unsqueeze(pos1.dim()).expand_as(landm_data)
        # landm_p 预测值
        landm_p = landm_data[pos_idx1].view(-1, landmark_coordinate_size)
        # landm_t：由gt编码后的先验
        landm_t = landm_t[pos_idx1].view(-1, landmark_coordinate_size)
        # pos_prior_landmark_decoded_list = []
        # for a_batch_pos_idx1 in pos_idx1:
        #     pos_prior_landmark_decoded = priors_landmarks[a_batch_pos_idx1].view(-1, landmark_coordinate_size)
        #     pos_prior_landmark_decoded_list.append(pos_prior_landmark_decoded)
        landm_p = landm_p[:, :self.useful_key_pys_num * 2]
        landm_t = landm_t[:, :self.useful_key_pys_num * 2]
        loss_landm = F.smooth_l1_loss(landm_p, landm_t, reduction='sum')

        pos = conf_t != zeros
        conf_t[pos] = 1

        # Localization Loss (Smooth L1)
        # Shape: [batch,num_priors,4]
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        loc_p = loc_data[pos_idx].view(-1, 4)
        loc_t = loc_t[pos_idx].view(-1, 4)
        loss_l = F.smooth_l1_loss(loc_p, loc_t, reduction='sum')

        # Compute max conf across batch for hard negative mining
        batch_conf = conf_data.view(-1, self.num_classes)
        batch_conf_t = conf_t.view(-1, 1).squeeze()
        focal_loss = 0  # focalloss.FocalLoss(gamma=7)(batch_conf, batch_conf_t)
        # soft_max_loss = nn.CrossEntropyLoss()(batch_conf, batch_conf_t)
        # test_1 = log_sum_exp(batch_conf)
        # test_2 = batch_conf.gather(1, conf_t.view(-1, 1))
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))
        # neg_score = batch_conf[0] - batch_conf[1]

        # Hard Negative Mining
        loss_c[pos.view(-1, 1)] = 0  # filter out pos boxes for now
        loss_c = loss_c.view(batch_size, -1)
        loss_c_sorted, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        num_pos = pos.long().sum(1, keepdim=True)
        num_neg = torch.clamp(self.neg_pos_ratio * num_pos, max=pos.size(1) - 1)
        neg = idx_rank < num_neg.expand_as(idx_rank)

        # Confidence Loss Including Positive and Negative Examples
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        conf_p = conf_data[(pos_idx + neg_idx).gt(0)].view(-1, self.num_classes)
        targets_weighted = conf_t[(pos + neg).gt(0)]
        loss_c = F.cross_entropy(conf_p, targets_weighted, reduction='sum')

        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        N = max(num_pos.data.sum().float(), 1)
        loss_l /= N
        loss_c /= N
        loss_landm /= N1

        pos_sample_index = []
        neg_sample_index = []
        for index_batch in range(batch_size):
            pos_sample_index.append(conf_t[index_batch].nonzero())
            neg_sample_index.append(neg[index_batch].nonzero())
        return loss_l, focal_loss, loss_c, loss_landm, pos_sample_index, neg_sample_index
