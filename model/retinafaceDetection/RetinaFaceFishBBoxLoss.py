import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utiles.box_utils import match, log_sum_exp
from utiles import utils_fish_landmark_detection

GPU = True
landmark_coordinate_size = 8


class RetinaFaceFishBBoxLoss(nn.Module):
    """SSD Weighted Loss Function
    Compute Targets:
        1) Produce Confidence Target Indices by matching  ground truth boxes
           with (default) 'priorboxes' that have jaccard index > threshold parameter
           (default threshold: 0.5).
        2) Produce localization target by 'encoding' variance into offsets of ground
           truth boxes and their matched  'priorboxes'.
        3) Hard negative mining to filter the excessive number of negative examples
           that comes with using a large number of default bounding boxes.
           (default negative:positive ratio 3:1)
    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by α which is set to 1 by cross val.
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """

    def __init__(self, params):
        super(RetinaFaceFishBBoxLoss, self).__init__()
        self.num_classes = params.num_of_class
        self.threshold = params.overlap_threshold
        self.negpos_ratio = params.neg_pos_ratio
        self.variance = params.variance
        self.landmark_size = params.landmark_size

        self.useful_key_pys_num = 4

    def forward(self, predictions, priors, targets):

        loc_data, conf_data, landm_data = predictions
        # !!先验要替换为keypts
        priors = priors
        assert loc_data.size(1) == priors.size(0)
        batch_size = loc_data.size(0)
        num_priors = (priors.size(0))

        # match priors (default boxes) and ground truth boxes
        loc_t = torch.Tensor(batch_size, num_priors, 4)
        landm_t = torch.Tensor(batch_size, num_priors, self.landmark_size * 2)
        conf_t = torch.LongTensor(batch_size, num_priors)
        for idx in range(batch_size):
            truths = targets[idx][:, :4].data
            landms = targets[idx][:, 4:12].data
            labels = targets[idx][:, -1].data
            defaults = priors.data
            utils_fish_landmark_detection.match_jaccard(self.threshold, truths, defaults, self.variance, labels, landms,
                                                        loc_t, conf_t, landm_t, idx)
        if GPU:
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
        landm_p = landm_data[pos_idx1].view(-1, landmark_coordinate_size)
        landm_t = landm_t[pos_idx1].view(-1, landmark_coordinate_size)
        landm_p = landm_p[:, :self.useful_key_pys_num * 2]
        landm_t = landm_t[:, :self.useful_key_pys_num * 2]
        loss_landm = F.smooth_l1_loss(landm_p, landm_t, reduction='sum')
        #正樣本可能包含有框无点的情况，因此标记是-1，考虑框的loss需要计算该部分正样本
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
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))

        # Hard Negative Mining
        loss_c[pos.view(-1, 1)] = 0  # filter out pos boxes for now
        loss_c = loss_c.view(batch_size, -1)
        loss_c_sorted, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        num_pos = pos.long().sum(1, keepdim=True)
        num_neg = torch.clamp(self.negpos_ratio * num_pos, max=pos.size(1) - 1)
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
        return loss_l, loss_c, loss_landm, pos_sample_index
