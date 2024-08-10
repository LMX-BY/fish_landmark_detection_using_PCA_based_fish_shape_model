import pathlib
import numpy as np
import torch
import json
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.init as init
import os
import cv2
import base64
import io
import PIL.ExifTags
import PIL.Image
from shapely.geometry import Polygon
import time

# landmark_coordinate_size = 8
ir_fish_img_means = torch.Tensor([0.3919, 0.3855, 0.3855])
ir_fish_img_stds = torch.Tensor([0.1832, 0.1796, 0.1793])





def collate_double(batch) -> tuple:
    """
    collate function for the ObjectDetectionDataSet.
    Only used by the dataloader.
    """

    # for a_data in batch:
    #     x = a_data['data'].unsqueeze(0)
    #     y = a_data['label'].unsqueeze(0)

    x = torch.cat([sample["data"].unsqueeze(0) for sample in batch])
    y = [sample["label"] for sample in batch]
    x_name = [sample["data_file_name"] for sample in batch]
    y_name = [sample["label_file_name"] for sample in batch]
    return x, torch.tensor(y), x_name, y_name


def img_to_json_suffix(img_file_name: pathlib.Path, label_file_path: pathlib.Path) -> pathlib.Path:
    return label_file_path / (img_file_name.stem + '.json')


def json_to_bmp_suffix(label_file_name: pathlib.Path, img_file_path: pathlib.Path) -> pathlib.Path:
    return img_file_path / (label_file_name.stem + '.bmp')


def rotation_matrix_between_two_2d_vector(source_vector, target_vector):
    norm_source_vector = source_vector / np.linalg.norm(source_vector)
    norm_target_vector = target_vector / np.linalg.norm(target_vector)
    source_vector_to_O = np.array(
        [[norm_source_vector[0], -norm_source_vector[1]], [norm_source_vector[1], norm_source_vector[0]]])
    target_vector_to_O = np.array(
        [[norm_target_vector[0], -norm_target_vector[1]], [norm_target_vector[1], norm_target_vector[0]]])
    return np.matmul(source_vector_to_O.transpose(), target_vector_to_O)


# 'test rotation_matrix_between_two_2d_vector'
# source_vector = np.array([1,0])
# target_vector = np.array([1,1])
# print(rotation_matrix_between_two_2d_vector(source_vector,target_vector))

def angle_between_2d_vector_x_to_y(source_vector, target_vector):
    rotation_matrix = rotation_matrix_between_two_2d_vector(source_vector, target_vector)
    ccos = rotation_matrix[0][0]
    ssin = rotation_matrix[1][0]
    assert ~np.isnan(ccos)
    assert ~np.isnan(ssin)
    return np.arctan2(ssin, ccos)


# 输入弧度，输出旋转矩阵
def rotation_matrix(orientation):
    cos_o = np.cos(orientation)
    sin_o = np.sin(orientation)
    rm = np.array([[cos_o, -sin_o], [sin_o, cos_o]])
    return rm


# test angle_between_2d_vector_x_to_y
# cur_angle = 0
# start_vector = np.array([1,0])
# delta_angle = np.pi / 6
# p_size = 5
# rotated_vectors = []
# rotated_vectors.append(start_vector)
# for index in range(p_size):
#     cur_angle = cur_angle + delta_angle
#     ccos = np.cos(cur_angle)
#     csin = np.sin(cur_angle)
#     rotation_M = np.array([[ccos,-csin],[csin,ccos]])
#     rotated_vector = rotation_M@start_vector
#     rotated_vectors.append(rotated_vector)
#
# angle1 = np.rad2deg(angle_between_2d_vector_x_to_y(rotated_vectors[0],rotated_vectors[1]))
# angle2 = np.rad2deg(angle_between_2d_vector_x_to_y(rotated_vectors[0],rotated_vectors[2]))
# angle3 = np.rad2deg(angle_between_2d_vector_x_to_y(rotated_vectors[0],rotated_vectors[3]))
# print(f'angle1 {angle1}')
# print(f'angle2 {angle2}')
# print(f'angle3 {angle3}')
# 判断向量集是否顺时针还是逆时针
def is_set_direction_consistency(pts_set: np.array) -> bool:
    vector_set_size = pts_set.shape[0]
    assert vector_set_size % 2 == 0

    before_point = pts_set[1]
    before_vector = pts_set[1] - pts_set[0]
    before_angle = 0
    isxy = True

    for index in range(2, vector_set_size, 2):
        cur_point = pts_set[index]
        cur_vector = cur_point - before_point
        angle = angle_between_2d_vector_x_to_y(cur_vector, before_vector)
        if angle * before_angle < 0:
            return False, None
        if angle <= 0:
            isxy = True
        else:
            isxy = False
        before_vector = cur_vector
        before_point = cur_point
        before_angle = angle
    return True, isxy


# test 'is_set_direction_consistency'
# points_xy_consistency = np.array([[0,0],[1,0],[1,1],[0,1]])
# points_yx_consistency = np.array([[0,0],[0,1],[1,1],[1,0]])
# points_inconsistency = np.array([[0,0],[1,1],[0,1],[1,0]])
# a11,a12 = is_set_direction_consistency(points_xy_consistency)
# a21,a22 = is_set_direction_consistency(points_yx_consistency)
# a31,a32 = is_set_direction_consistency(points_inconsistency)
# print(f'a11:{a11},a12:{a12}')
# print(f'a21:{a21},a22:{a22}')
# print(f'a31:{a31},a32:{a32}')
def key_pts_projection(key_pts, width_scale_factor, height_scale_factor, mode='f2i'):
    assert mode in ['f2i', 'i2f']

    batch_size = key_pts.size(dim=0)
    proj_key_pts = key_pts.clone().reshape(batch_size, -1, 8)
    # invalid_bbox_mask = (proj_key_pts == -1)  # indicating padded bboxes

    if mode == 'f2i':
        # activation map to pixel image
        proj_key_pts[:, :, [0, 2, 4, 6]] *= width_scale_factor
        proj_key_pts[:, :, [1, 3, 5, 7]] *= height_scale_factor
    else:
        # pixel image to activation map
        proj_key_pts[:, :, [0, 2, 4, 6]] /= width_scale_factor
        proj_key_pts[:, :, [1, 3, 5, 7]] /= height_scale_factor

    # proj_key_pts.masked_fill_(invalid_bbox_mask, -1)  # fill padded bboxes back with -1
    proj_key_pts.resize_as_(key_pts)

    return proj_key_pts


def point_4_loc_allocator_1(batch_key_pts):
    for key_pts in batch_key_pts:
        pt2_x = key_pts[2].clone()
        pt2_y = key_pts[3].clone()
        pt3_x = key_pts[4].clone()
        pt3_y = key_pts[5].clone()
        # pt4_x = key_pts[6]
        # pt4_y = key_pts[7]
        key_pts[2] = pt3_x
        key_pts[3] = pt3_y
        key_pts[4] = pt2_x
        key_pts[5] = pt2_y
    return batch_key_pts


def generate_orientated_anchor_key_points(orientation_size: int, anc_key_pts_single_direction):
    anc_key_pts_size_in_a_grid = anc_key_pts_single_direction.shape[0] * orientation_size
    orientated_anchor_key_points = torch.zeros(anc_key_pts_size_in_a_grid, 8)
    rotation_M_anc_key_pts = torch.zeros(orientation_size, 2, 2)
    delta_angle = 2 * np.pi / orientation_size
    cur_angle = 0

    for index in range(orientation_size):
        cos_theta = np.cos(cur_angle)
        sin_theta = np.sin(cur_angle)
        rotation_M_anc_key_pts[index] = torch.tensor([[cos_theta, -sin_theta], [sin_theta, cos_theta]])
        cur_angle = cur_angle + delta_angle

    grid_anc_key_points_index = 0
    for a_anc_key_pts_set in anc_key_pts_single_direction:
        pt1 = a_anc_key_pts_set[0:2]
        pt2 = a_anc_key_pts_set[2:4]
        pt3 = a_anc_key_pts_set[4:6]
        pt4 = a_anc_key_pts_set[6:8]
        for a_rotation_m in rotation_M_anc_key_pts:
            rpt1 = torch.matmul(a_rotation_m, pt1)
            rpt2 = torch.matmul(a_rotation_m, pt2)
            rpt3 = torch.matmul(a_rotation_m, pt3)
            rpt4 = torch.matmul(a_rotation_m, pt4)
            a_anc_key_pts = torch.tensor([rpt1[0], rpt1[1],
                                          rpt2[0], rpt2[1],
                                          rpt3[0], rpt3[1],
                                          rpt4[0], rpt4[1]])
            orientated_anchor_key_points[grid_anc_key_points_index] = a_anc_key_pts
            grid_anc_key_points_index = grid_anc_key_points_index + 1

    return orientated_anchor_key_points


# key_pts_set:numOfItem*8(4个点的坐标)
def fish_key_pts_match_score(key_pts_set1, key_pts_set2):
    return torch.cdist(key_pts_set1, key_pts_set2)


def detection_collate(batch):
    labels = []
    imgs = []
    for sample in batch:
        imgs.append(sample['img'])
        labels.append(sample['label'])
    return torch.stack(imgs, 0), labels


def xavier(param):
    init.xavier_uniform(param)


def kaiming(param):
    init.kaiming_normal_(param, mode='fan_in', nonlinearity='relu')


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()


def encode_bbox(matched, priors, variances):
    """Encode the variances from the priorbox layers into the ground truth boxes
    we have matched (based on jaccard overlap) with the prior boxes.
    Args:
        matched: (tensor) Coords of ground truth for each prior in point-form
            Shape: [num_priors, 4].
        priors: (tensor) Prior boxes in center-offset form
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        encoded boxes (tensor), Shape: [num_priors, 4]
    """

    # dist b/t match center and prior's center
    g_cxcy = (matched[:, :2] + matched[:, 2:]) / 2 - priors[:, :2]
    # encode variance
    g_cxcy /= (variances[0] * priors[:, 2:])
    # match wh / prior wh
    g_wh = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:]
    g_wh = torch.log(g_wh) / variances[1]
    # return target for smooth_l1_loss
    return torch.cat([g_cxcy, g_wh], 1)  # [num_priors,4]


def decode_bbox(loc, priors, variances):
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        loc (tensor): location predictions for loc layers,
            Shape: [num_priors,4]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded bounding box predictions
    """
    # print(f'loc size: {loc.dim()}')
    unsqueezed_loc = loc
    unsqueezed_priors = priors
    if loc.dim() == 1:
        unsqueezed_loc = loc.unsqueeze(0)
        unsqueezed_priors = priors.unsqueeze(0)

    boxes = torch.cat((
        unsqueezed_priors[:, :2] + unsqueezed_loc[:, :2] * variances[0] * unsqueezed_priors[:, 2:],
        unsqueezed_priors[:, 2:] * torch.exp(unsqueezed_loc[:, 2:] * variances[1])), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    # except:
    #     pass

    return boxes


def encode_landm_with_bbox(matched, priors, variances):
    """Encode the variances from the priorbox layers into the ground truth boxes
    we have matched (based on jaccard overlap) with the prior boxes.
    Args:
        matched: (tensor) Coords of ground truth for each prior in point-form
            Shape: [num_priors, 10].
        priors: (tensor) Prior boxes in center-offset form
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        encoded landm (tensor), Shape: [num_priors, 10]
    """

    # dist b/t match center and prior's center
    #key_pts_size = int(landmark_coordinate_size / 2)
    key_pts_size = int(matched.shape[1] / 2)
    matched = torch.reshape(matched, (matched.size(0), key_pts_size, 2))
    priors_cx = priors[:, 0].unsqueeze(1).expand(matched.size(0), key_pts_size).unsqueeze(2)
    priors_cy = priors[:, 1].unsqueeze(1).expand(matched.size(0), key_pts_size).unsqueeze(2)
    priors_w = priors[:, 2].unsqueeze(1).expand(matched.size(0), key_pts_size).unsqueeze(2)
    priors_h = priors[:, 3].unsqueeze(1).expand(matched.size(0), key_pts_size).unsqueeze(2)
    priors = torch.cat([priors_cx, priors_cy, priors_w, priors_h], dim=2)
    test = matched[:, :, :2]
    g_cxcy = matched[:, :, :2] - priors[:, :, :2]
    # encode variance
    # g_cxcy /= (variances[0] * priors[:, :, 2:])
    g_cxcy /= priors[:, :, 2:]
    # g_cxcy /= priors[:, :, 2:]
    g_cxcy = g_cxcy.reshape(g_cxcy.size(0), -1)
    # return target for smooth_l1_loss
    return g_cxcy


def decode_landm_with_bbox(pre, priors, variances):
    """Decode landm from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        pre (tensor): landm predictions for loc layers,
            Shape: [num_priors,10]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded landm predictions
    """
    # landms = torch.cat((priors[:, :2] + pre[:, :2] * variances[0] * priors[:, 2:],
    #                     priors[:, :2] + pre[:, 2:4] * variances[0] * priors[:, 2:],
    #                     priors[:, :2] + pre[:, 4:6] * variances[0] * priors[:, 2:],
    #                     priors[:, :2] + pre[:, 6:8] * variances[0] * priors[:, 2:]
    #                     ), dim=1)
    # print(f'prior shapes {priors.size()}')
    shaped_priors = priors
    shaped_pre = pre
    if priors.dim() == 1:
        shaped_priors = shaped_priors.unsqueeze(0)
        shaped_pre = shaped_pre.unsqueeze(0)

    landms = torch.cat((shaped_priors[:, :2] + shaped_pre[:, :2] * shaped_priors[:, 2:],
                        shaped_priors[:, :2] + shaped_pre[:, 2:4] * shaped_priors[:, 2:],
                        shaped_priors[:, :2] + shaped_pre[:, 4:6] * shaped_priors[:, 2:],
                        shaped_priors[:, :2] + shaped_pre[:, 6:8] * shaped_priors[:, 2:]
                        ), dim=1)
    return landms


def encode_landmarks_with_prior_landmarks(matched, priors, variances):
    return matched - priors


def decode_landmarks_with_prior_landmarks(loc, prior, variances):
    return loc + prior


def encode_landmarks_with_pca_param(loc, prior_mean, pca_features):
    loc_sub_mean = loc - prior_mean
    pca_features_transposes = torch.transpose(pca_features, 1,0)
    encoded_landmarks = torch.mm(loc_sub_mean, pca_features_transposes)
    return encoded_landmarks


def decode_landmarks_with_pca_param(loc, prior_mean, pca_features):
    try:
        decoded_landmarks = torch.mm(loc, pca_features) + prior_mean
    except:
        pass
    return decoded_landmarks


def log_sum_exp(x):
    """Utility function for computing log_sum_exp while determining
    This will be used to determine unaveraged confidence loss across
    all examples in a batch.
    Args:
        x (Variable(tensor)): conf_preds from conf layers
    """
    x_max = x.data.max()
    return torch.log(torch.sum(torch.exp(x - x_max), 1, keepdim=True)) + x_max


def point_form(boxes):
    """ Convert prior_boxes to (xmin, ymin, xmax, ymax)
    representation for comparison to point form ground truth data.
    Args:
        boxes: (tensor) center-size default boxes from priorbox layers.
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    # print(f'boxes dim: {boxes.dim()}')
    if boxes.dim() == 2:
        return torch.cat((boxes[:, :2] - boxes[:, 2:] / 2,  # xmin, ymin
                          boxes[:, :2] + boxes[:, 2:] / 2), 1)  # xmax, ymax
    if boxes.dim() == 1:
        unsqueezed_boxes = boxes.unsqueeze(0)
        return torch.cat((unsqueezed_boxes[:, :2] - unsqueezed_boxes[:, 2:] / 2,  # xmin, ymin
                          unsqueezed_boxes[:, :2] + unsqueezed_boxes[:, 2:] / 2), 1)  # xmax, ymax


def intersect(box_a, box_b):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [A,4].
      box_b: (tensor) bounding boxes, Shape: [B,4].
    Return:
      (tensor) intersection area, Shape: [A,B].
    """
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]


def jaccard(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2] - box_a[:, 0]) *
              (box_a[:, 3] - box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2] - box_b[:, 0]) *
              (box_b[:, 3] - box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


def polygon_iou(polygon_a, polygon_b):
    start_time = time.time()
    lamdmark_size = int(polygon_a.size(1) / 2)
    a_size = polygon_a.size(0)
    b_size = polygon_b.size(0)
    correlation_matrix = torch.zeros([a_size, b_size])
    for index_a in range(a_size):
        for index_b in range(b_size):
            p_a = Polygon(polygon_a[index_a].cpu().numpy().reshape(lamdmark_size, 2))
            p_b = Polygon(polygon_b[index_b].cpu().numpy().reshape(lamdmark_size, 2))
            intersection = p_a.intersection(p_b).area
            union = p_a.union(p_b).area
            iou = intersection / union
            correlation_matrix[index_a][index_b] = iou

    end_time = time.time()
    print(f'polygon_iou time: {(end_time - start_time) * 1000}')
    return correlation_matrix


def match(threshold, truths, priors, variances, labels, loc_t, conf_t, idx):
    """Match each prior box with the ground truth box of the highest jaccard
    overlap, encode the bounding boxes, then return the matched indices
    corresponding to both confidence and location preds.
    Args:
        threshold: (float) The overlap threshold used when mathing boxes.
        truths: (tensor) Ground truth boxes, Shape: [num_obj, num_priors].
        priors: (tensor) Prior boxes from priorbox layers, Shape: [n_priors,4].
        variances: (tensor) Variances corresponding to each prior coord,
            Shape: [num_priors, 4].
        labels: (tensor) All the class labels for the image, Shape: [num_obj].
        loc_t: (tensor) Tensor to be filled w/ endcoded location targets.
        conf_t: (tensor) Tensor to be filled w/ matched indices for conf preds.
        idx: (int) current batch index
    Return:
        The matched indices corresponding to 1)location and 2)confidence preds.
    """
    # jaccard index
    # numOfGroundTruthBox*numOfPriorsBox
    match_score = fish_key_pts_match_score(
        truths,
        priors
    )

    # (Bipartite Matching)
    # [1,num_objects] best prior for each ground truth
    # numOfGroundTruthBox*1
    # 和每个ground truth重合最大的prior box
    # ！！返回了和先验关键点集匹配最好的索引
    best_prior_overlap, best_prior_idx = match_score.min(1, keepdim=True)
    # [1,num_priors] best ground truth for each prior
    # 1*numOfPriorsBox：和每个piorbox重合最大的ground truth
    best_truth_overlap, best_truth_idx = match_score.min(0, keepdim=True)
    best_truth_idx.squeeze_(0)
    best_truth_overlap.squeeze_(0)
    best_prior_idx.squeeze_(1)
    best_prior_overlap.squeeze_(1)
    # ？？似乎是把和groundTruth匹配上的标记为0，标记出
    best_truth_overlap.index_fill_(0, best_prior_idx, 0)  # ensure best prior
    # TODO refactor: index  best_prior_idx with long tensor
    # ensure every gt matches with its prior of max overlap
    # ？？
    for j in range(best_prior_idx.size(0)):
        best_truth_idx[best_prior_idx[j]] = j
    # ??匹配成功的prior box，没有匹配上的都标识的是0，但是0特指了一个有效的ground truth box？？
    matches = truths[best_truth_idx]  # Shape: [num_priors,4]
    # 对类别进行+1，把0留出来作为背景
    conf = labels[best_truth_idx]  # Shape: [num_priors]
    # 将重叠程度小于阈值的标记为背景
    # ！！先验关键点样本中匹配的最优解，及，匹配分数优于特定阈值的解
    conf[best_truth_overlap > threshold] = 0  # label as background
    ########test 正样本选中的个数
    # test_pos_sample_count = conf.sum()
    # print(f'\n pos_sample_count is {test_pos_sample_count}')
    ########test 正样本选中的个数
    #######test 获取正样本索引
    pos_sample_index = conf.nonzero().squeeze()
    # pos_sample_index = pos_sample_index.squeeze()
    #######test 获取正样本索引
    # 先验关键点实在归一化特征地图坐标系,encode
    # gt也是在归一化特征地图坐标系
    loc = encode(matches, priors, variances)
    loc_t[idx] = loc  # [num_priors,4] encoded offsets to learn
    conf_t[idx] = conf  # [num_priors] top class label for each prior
    return best_prior_idx, pos_sample_index


def match_jaccard(threshold, truths, priors, variances, labels, landms, loc_t, conf_t, landm_t, idx):
    """Match each prior box with the ground truth box of the highest jaccard
    overlap, encode the bounding boxes, then return the matched indices
    corresponding to both confidence and location preds.
    Args:
        threshold: (float) The overlap threshold used when mathing boxes.
        truths: (tensor) Ground truth boxes, Shape: [num_obj, 4].
        priors: (tensor) Prior boxes from priorbox layers, Shape: [n_priors,4].
        variances: (tensor) Variances corresponding to each prior coord,
            Shape: [num_priors, 4].
        labels: (tensor) All the class labels for the image, Shape: [num_obj].
        landms: (tensor) Ground truth landms, Shape [num_obj, 10].
        loc_t: (tensor) Tensor to be filled w/ endcoded location targets.
        conf_t: (tensor) Tensor to be filled w/ matched indices for conf preds.
        landm_t: (tensor) Tensor to be filled w/ endcoded landm targets.
        idx: (int) current batch index
    Return:
        The matched indices corresponding to 1)location 2)confidence 3)landm preds.
    """
    # jaccard index
    # ！！获得匹配分数矩阵，不同评分方式输出形式相同(统一步骤)
    overlaps = jaccard(
        truths,
        point_form(priors)
    )
    # (Bipartite Matching)
    # [1,num_objects] best prior for each ground truth
    # 每个gt，选最好的先验框(统一步骤)
    best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True)

    # ignore hard gt
    # 针对每一个gt，就算最好的匹配先验框都不是太好，就直接返回，跳过该张图片，如果有一些不好的匹配，就过滤掉
    valid_gt_idx = best_prior_overlap[:, 0] >= 0.2
    best_prior_idx_filter = best_prior_idx[valid_gt_idx, :]
    if best_prior_idx_filter.shape[0] <= 0:
        loc_t[idx] = 0
        conf_t[idx] = 0
        return

    # [1,num_priors] best ground truth for each prior
    # 针对每一个先验框，选最高的gt（统一步骤）
    best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)
    best_truth_idx.squeeze_(0)
    best_truth_overlap.squeeze_(0)
    best_prior_idx.squeeze_(1)
    best_prior_idx_filter.squeeze_(1)
    best_prior_overlap.squeeze_(1)
    best_truth_overlap.index_fill_(0, best_prior_idx_filter, 2)  # ensure best prior
    # TODO refactor: index  best_prior_idx with long tensor
    # ensure every gt matches with its prior of max overlap
    for j in range(best_prior_idx.size(0)):  # 判别此anchor是预测哪一个boxes
        best_truth_idx[best_prior_idx[j]] = j

    matches = truths[best_truth_idx]  # Shape: [num_priors,4] 此处为每一个anchor对应的bbox取出来
    # (Bipartite Matching)
    conf = labels[best_truth_idx]  # Shape: [num_priors]      此处为每一个anchor对应的label取出来

    conf[best_truth_overlap < threshold] = 0  # label as background   overlap<0.35的全部作为负样本

    loc = encode_bbox(matches, priors, variances)

    matches_landm = landms[best_truth_idx]

    landm = encode_landm_with_bbox(matches_landm, priors, variances)

    loc_t[idx] = loc  # [num_priors,4] encoded offsets to learn

    conf_t[idx] = conf  # [num_priors] top class label for each prior

    landm_t[idx] = landm


# 根据gt和先验，计算每个cell的输出，包括分类输出，landmark偏差
def match_landmarks_cdist(threshold, truths, priors, variances, labels, conf_t, landm_t, idx, worst_score):
    """Match each prior box with the ground truth box of the highest jaccard
    overlap, encode the bounding boxes, then return the matched indices
    corresponding to both confidence and location preds.
    Args:
        threshold: (float) The overlap threshold used when mathing boxes.
        truths: (tensor) Ground truth boxes, Shape: [num_obj, 4].
        priors: (tensor) Prior boxes from priorbox layers, Shape: [n_priors,4].
        variances: (tensor) Variances corresponding to each prior coord,
            Shape: [num_priors, 4].
        labels: (tensor) All the class labels for the image, Shape: [num_obj].
        landms: (tensor) Ground truth landms, Shape [num_obj, 10].
        conf_t: (tensor) Tensor to be filled w/ matched indices for conf preds.
        landm_t: (tensor) Tensor to be filled w/ endcoded landm targets.
        idx: (int) current batch index
    Return:
        The matched indices corresponding to 1)location 2)confidence 3)landm preds.
    """
    # jaccard index
    # ！！获得匹配分数矩阵，不同评分方式输出形式相同(统一步骤)
    overlaps = fish_key_pts_match_score(
        truths,
        priors
    )
    # (Bipartite Matching)
    # [1,num_objects] best prior for each ground truth
    # 每个gt，选最好的先验框(统一步骤)
    best_prior_overlap, best_prior_idx = overlaps.min(1, keepdim=True)

    # ignore hard gt
    # 针对每一个gt，就算最好的匹配先验框都不是太好，就直接返回，跳过该张图片，如果有一些不好的匹配，就过滤掉
    valid_gt_idx = best_prior_overlap[:, 0] <= worst_score
    best_prior_idx_filter = best_prior_idx[valid_gt_idx, :]
    if best_prior_idx_filter.shape[0] <= 0:
        landm_t[idx] = 0
        conf_t[idx] = 0
        return

    # [1,num_priors] best ground truth for each prior
    # 针对每一个先验框，选最高的gt（统一步骤）
    best_truth_overlap, best_truth_idx = overlaps.min(0, keepdim=True)
    best_truth_idx.squeeze_(0)
    best_truth_overlap.squeeze_(0)
    best_prior_idx.squeeze_(1)
    best_prior_idx_filter.squeeze_(1)
    best_prior_overlap.squeeze_(1)
    best_truth_overlap.index_fill_(0, best_prior_idx_filter, 0)  # ensure best prior
    # TODO refactor: index  best_prior_idx with long tensor
    # ensure every gt matches with its prior of max overlap
    for j in range(best_prior_idx.size(0)):  # 判别此anchor是预测哪一个boxes
        best_truth_idx[best_prior_idx[j]] = j

    matches = truths[best_truth_idx]  # Shape: [num_priors,4] 此处为每一个anchor对应的bbox取出来
    # (Bipartite Matching)
    conf = labels[best_truth_idx]  # Shape: [num_priors]      此处为每一个anchor对应的label取出来

    conf[best_truth_overlap > threshold] = 0  # label as background   overlap<0.35的全部作为负样本

    # test 查看正样本的个数
    test_pos_sample_count = conf.sum()

    matches_landm = truths[best_truth_idx]

    landm = encode_landmarks_with_prior_landmarks(matches_landm, priors, variances)

    conf_t[idx] = conf  # [num_priors] top class label for each prior

    landm_t[idx] = landm


def match_landmarks_cdist_for_svd(threshold, truths, priors, variances, labels, conf_t, landm_svd_weight_t, pca_featurs, idx,
                                  worst_score):
    """Match each prior box with the ground truth box of the highest jaccard
    overlap, encode the bounding boxes, then return the matched indices
    corresponding to both confidence and location preds.
    Args:
        threshold: (float) The overlap threshold used when mathing boxes.
        truths: (tensor) Ground truth boxes, Shape: [num_obj, 4].
        priors: (tensor) Prior boxes from priorbox layers, Shape: [n_priors,4].
        variances: (tensor) Variances corresponding to each prior coord,
            Shape: [num_priors, 4].
        labels: (tensor) All the class labels for the image, Shape: [num_obj].
        landms: (tensor) Ground truth landms, Shape [num_obj, 10].
        conf_t: (tensor) Tensor to be filled w/ matched indices for conf preds.
        landm_t: (tensor) Tensor to be filled w/ endcoded landm targets.
        idx: (int) current batch index
    Return:
        The matched indices corresponding to 1)location 2)confidence 3)landm preds.
    """
    # jaccard index
    # ！！获得匹配分数矩阵，不同评分方式输出形式相同(统一步骤)
    overlaps = fish_key_pts_match_score(
        truths,
        priors
    )
    # (Bipartite Matching)
    # [1,num_objects] best prior for each ground truth
    # 每个gt，选最好的先验框(统一步骤)
    best_prior_overlap, best_prior_idx = overlaps.min(1, keepdim=True)

    # ignore hard gt
    # 针对每一个gt，就算最好的匹配先验框都不是太好，就直接返回，跳过该张图片，如果有一些不好的匹配，就过滤掉
    valid_gt_idx = best_prior_overlap[:, 0] <= worst_score
    best_prior_idx_filter = best_prior_idx[valid_gt_idx, :]
    if best_prior_idx_filter.shape[0] <= 0:
        landm_svd_weight_t[idx] = 0
        conf_t[idx] = 0
        return

    # [1,num_priors] best ground truth for each prior
    # 针对每一个先验框，选最高的gt（统一步骤）
    best_truth_overlap, best_truth_idx = overlaps.min(0, keepdim=True)
    best_truth_idx.squeeze_(0)
    best_truth_overlap.squeeze_(0)
    best_prior_idx.squeeze_(1)
    best_prior_idx_filter.squeeze_(1)
    best_prior_overlap.squeeze_(1)
    best_truth_overlap.index_fill_(0, best_prior_idx_filter, 0)  # ensure best prior
    # TODO refactor: index  best_prior_idx with long tensor
    # ensure every gt matches with its prior of max overlap
    for j in range(best_prior_idx.size(0)):  # 判别此anchor是预测哪一个boxes
        best_truth_idx[best_prior_idx[j]] = j

    matches = truths[best_truth_idx]  # Shape: [num_priors,4] 此处为每一个anchor对应的bbox取出来
    # (Bipartite Matching)
    conf = labels[best_truth_idx]  # Shape: [num_priors]      此处为每一个anchor对应的label取出来

    conf[best_truth_overlap > threshold] = 0  # label as background   overlap<0.35的全部作为负样本
    conf[conf < 0] = 0
    # test 查看正样本的个数
    # 如果bbox中没有
    test_pos_sample_count = conf.sum()

    matches_landm = truths[best_truth_idx]

    matches_landm_svd_weight = encode_landmarks_with_pca_param(matches_landm, priors, pca_featurs)

    conf_t[idx] = conf  # [num_priors] top class label for each prior

    landm_svd_weight_t[idx] = matches_landm_svd_weight


def nms(loc_data, scores, overlap=0.5, top_k=200):
    """Apply non-maximum suppression at test time to avoid detecting too many
    overlapping bounding boxes for a given object.
    Args:
        boxes: (tensor) The location preds for the img, Shape: [num_priors,8].
        scores: (tensor) The class predscores for the img, Shape:[num_priors].
        overlap: (float) The overlap thresh for suppressing unnecessary boxes.
        top_k: (int) The Maximum number of box preds to consider.
    Return:
        The indices of the kept boxes with respect to num_priors.
    """
    # 创建和score形状、设备一样的tensor
    keep = scores.new(scores.size(0)).zero_().long()
    if loc_data.numel() == 0:
        return keep
    # x1 = boxes[:, 0]
    # y1 = boxes[:, 1]
    # x2 = boxes[:, 2]
    # y2 = boxes[:, 3]
    # area = torch.mul(x2 - x1, y2 - y1)
    # 对评分进行排序
    v, idx = scores.sort(0)  # sort in ascending order
    # I = I[v >= 0.01]
    idx = idx[-top_k:]  # indices of the top-k largest vals
    # xx1 = boxes.new()
    # yy1 = boxes.new()
    # xx2 = boxes.new()
    # yy2 = boxes.new()
    # w = boxes.new()
    # h = boxes.new()

    # keep = torch.Tensor()
    count = 0
    while idx.numel() > 0:
        i = idx[-1]  # index of current largest val
        # keep.append(i)
        keep[count] = i
        count += 1
        if idx.size(0) == 1:
            break
        # view上移除一个最大值，内存上没有移除
        idx = idx[:-1]  # remove kept element from view
        cur_max_loc = loc_data[i]
        cur_max_loc = cur_max_loc.unsqueeze(0)
        selected_loc_data = torch.index_select(loc_data, 0, idx)
        match_score = fish_key_pts_match_score(cur_max_loc, selected_loc_data)
        idx = idx[match_score.le(overlap).squeeze()]
    return keep, count


def save_IR_fish_detected_results(detected_results, img, img_files_name, save_folder_path):
    # save_name = save_folder_path + '/' + img_files_name + ".txt"
    # dirname = os.path.dirname(save_name)
    # if not os.path.isdir(dirname):
    #     os.makedirs(dirname)
    # with open(save_name, "w") as fd:
    #     bboxs = detected_results
    #     file_name = os.path.basename(save_name)[:-4] + "\n"
    #     bboxs_num = str(len(bboxs)) + "\n"
    #     fd.write(file_name)
    #     fd.write(bboxs_num)
    #     for box in bboxs:
    #         x = int(box[0])
    #         y = int(box[1])
    #         w = int(box[2]) - int(box[0])
    #         h = int(box[3]) - int(box[1])
    #         confidence = str(box[4])
    #         line = str(x) + " " + str(y) + " " + str(w) + " " + str(h) + " " + confidence + " \n"
    #         fd.write(line)

    save_img_name = save_folder_path + '/' + img_files_name + '.bmp'
    cv2.imwrite(save_img_name, img)


def detected_results_13_to_MeanAveragePrecision_preds_dict(detected_results_13):
    bbox_num = detected_results_13.shape[0]
    boxes = torch.zeros([bbox_num, 4])
    scores = torch.zeros(bbox_num)
    labels = torch.zeros(bbox_num).int()
    for index, a_detected_results_13 in enumerate(detected_results_13):
        boxes[index] = torch.from_numpy(a_detected_results_13[0:4])
        scores[index] = float(a_detected_results_13[4])
        labels[index] = 1
    # return list[boxes, scores, labels]
    return [dict(
        boxes=boxes,
        scores=scores,
        labels=labels
    )]


def detected_results_gt_to_MeanAveragePrecision_target_dict(gt_data, img_height, img_width):
    bbox_num = gt_data.shape[0]
    boxes = torch.zeros([bbox_num, 4])
    labels = torch.zeros(bbox_num).int()
    for index, a_gt_data in enumerate(gt_data):
        boxes[index] = a_gt_data[0:4].detach().clone()
        boxes[index][0:4:2] = boxes[index][0:4:2] * img_width
        boxes[index][1:5:2] = boxes[index][1:5:2] * img_height
        labels[index] = 1

    # return list[boxes, labels]
    return [dict(
        boxes=boxes,
        labels=labels
    )]


def json_img_data_to_opencvImage(image_data):
    imageDecoded = base64.b64decode(image_data)
    f = io.BytesIO()
    f.write(imageDecoded)
    PIL_imge = PIL.Image.open(f)
    open_cv_image = cv2.cvtColor(np.array(PIL_imge), cv2.COLOR_RGB2BGR)
    return open_cv_image


def gen_anc_centers(out_size):
    out_h, out_w = out_size

    anc_pts_x = torch.arange(0, out_w) + 0.5
    anc_pts_y = torch.arange(0, out_h) + 0.5

    return anc_pts_x, anc_pts_y


'N指间隔数，个数为N+1'


def line_split_evenly(point1, point2, N: int) -> list:
    assert N > 1
    '需补充point和point2的类型检测'
    delta_vector = point2 - point1
    distance = np.linalg.norm(delta_vector)
    normolized_delat_vector = delta_vector / distance
    delta_distance = distance / (N - 1)
    results_points = np.zeros((N, 2), dtype=np.float32)
    next_point = point1
    for count in range(N):
        results_points[count] = next_point
        next_point = next_point + normolized_delat_vector * delta_distance
    # results_points[N] = point2
    return results_points


# def test_line_split_evenly():
#     point1 = np.array([0., 0.])
#     point2 = np.array([10., 10.])
#     split_points = line_split_evenly(point1, point2, 10)
#     plt.scatter(split_points[:, 0], split_points[:, 1], marker='.', color='r')
#     # plt.scatter(point1[0],point1[1],marker='.',c='g')
#     # plt.scatter(point2[0],point2[1],marker='.',c='g')
#     plt.show()


'bounding box array:[TopLeftX,TopLeftY,BottomRightX,BottomRightY]'


def is_point_in_boundingBox(point: np.array, boundingBox: np.array) -> bool:
    '需补充point和point2，boundingBox的类型检测'
    point_x, point_y = point[0], point[1]
    # top_left_x,top_left_y,bottom_right_x,bottom_right_y = boundingBox[0],boundingBox[1],boundingBox[2],boundingBox[3]
    top_left_x, top_left_y, bottom_right_x, bottom_right_y = boundingBox
    assert top_left_x < bottom_right_x
    assert top_left_y < bottom_right_y
    if point_x < top_left_x or point_x > bottom_right_x:
        return False
    if point_y < top_left_y or point_y > bottom_right_y:
        return False
    return True


'bounding point to [TopLeftX,TopLeftY,BottomRightX,BottomRightY]'
'输入形式[[],[]]'


def two_bounding_point_to_TL_BR_pattern(bounding_points: list) -> list:
    p1x, p1y, p2x, p2y = bounding_points[0][0], bounding_points[0][1], bounding_points[1][0], bounding_points[1][1]
    if p1x < p2x:
        left_top_x = p1x
        right_bottom_x = p2x
    else:
        if p1x > p2x:
            left_top_x = p2x
            right_bottom_x = p1x
        else:
            print(f'p1x:{p1x},p1y:{p1y},p2x:{p2x},p2y:{p2y}')
            assert False

    if p1y < p2y:
        left_top_y = p1y
        right_bottom_y = p2y
    else:
        if p1y > p2y:
            left_top_y = p2y
            right_bottom_y = p1y
        else:
            print(f'p1x:{p1x},p1y:{p1y},p2x:{p2x},p2y:{p2y}')
            assert False
    return [[left_top_x, left_top_y], [right_bottom_x, right_bottom_y]]


'点到直线的距离，直线由两点表示'


def point_to_two_points_line_distance(point0: np.array, point1_on_line: np.array,
                                      point2_on_line: np.array) -> float:
    vect1 = point1_on_line - point0
    vect2 = point2_on_line - point0
    distance = np.abs(np.cross(vect1, vect2)) / np.linalg.norm(point2_on_line - point1_on_line)
    return distance


def generated_rotated_landmark_sets():
    pass
