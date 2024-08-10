import numpy.linalg
# import modelTools
import pathlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
from torchvision import transforms
from torchvision import ops
import numpy as np
import json
from utiles import utils_fish_landmark_detection

'先输出聚合标记图像进行人工检测'
'人工移除或修改错误图像'
'最后输出图片聚合标记的csv图片'
'dataset 读取数据'


'以头部作为坐标原点，图像坐标轴为坐标系方向，堆叠在一起显示全部的关键点'
'如何解决哪四个点属于同一条鱼的问题？尝试判断头特征点是否在方框内？'
'如果同时在两个方框内，如何处理'
'如果头在方框外如何处理'
HEAD_INDEX = 0
TAIL_INDEX = 1
RIGHT_PECTORAL_FIN_INDEX = 2
LEFT_PECTORAL_FIN_INDEX = 3
LABEL_FISH = 'fish'
LABEL_FISH_MOUTH = 'fish_mouth'
LABEL_TAIL = 'fish_tail'
LABEL_FISH_PECTORAL_FIN = 'fish_pectoral fin'
train_data_key_points = []
line_sample_counts = 100


def create_a_polygon_dict(points):
    polygon_dict = {'label': 'fish_landmarks', 'group_id': None, 'points': points, 'shape_type': 'polygon', 'flags': {}}
    return polygon_dict


def labelme_landmark_label_to_labelme_polygon_label(labelme_label_files, polygon_label_save_path):

    labeled_landmark_fish_count = 0
    valid_labeled_landmark_fish_count = 0
    for a_label_file_name in labelme_label_files:
        a_landmark_label_content = utils_fish_landmark_detection.read_json(a_label_file_name)
        a_polygon_label_content = a_landmark_label_content.copy()
        #移除掉landmark label的所有landmark标记
        shape_with_out_landmarks =[]
        for a_label_key_point in a_polygon_label_content['shapes']:
            if a_label_key_point['label'] == LABEL_FISH:
                shape_with_out_landmarks.append(a_label_key_point)
        a_polygon_label_content['shapes'] = shape_with_out_landmarks
        fish_mouth_points = []
        fish_tail_points = []
        fish_boundingBox = []
        fish_pectoral_fin_points = []
        for a_label_key_point in a_landmark_label_content['shapes']:
            if a_label_key_point['label'] == LABEL_FISH:
                fish_boundingBox.append(a_label_key_point['points'])
            if a_label_key_point['label'] == LABEL_FISH_MOUTH:
                fish_mouth_points.append(a_label_key_point['points'])
            if a_label_key_point['label'] == LABEL_TAIL:
                fish_tail_points.append(a_label_key_point['points'])
            if a_label_key_point['label'] == LABEL_FISH_PECTORAL_FIN:
                fish_pectoral_fin_points.append(a_label_key_point['points'])
        len_fish_mouth_points, len_fish_tail_points, len_fish_boundingBox = len(fish_mouth_points), len(
            fish_tail_points), len(fish_boundingBox)
        assert len_fish_mouth_points == len_fish_tail_points
        labeled_landmark_fish_count = labeled_landmark_fish_count + len_fish_mouth_points

        tail_mouth_correlation_matrix = np.zeros((len_fish_mouth_points, len_fish_tail_points, len_fish_boundingBox))
        #计算头-尾-bbox相关性矩阵得分
        for fish_mouth_index in range(len_fish_mouth_points):
            a_fish_mouth_points = np.array(fish_mouth_points[fish_mouth_index]).squeeze()
            for fish_tail_index in range(len_fish_tail_points):
                a_fish_tail_points = np.array(fish_tail_points[fish_tail_index]).squeeze()
                for fish_boundingBox_index in range(len_fish_boundingBox):
                    fish_boundingBox[fish_boundingBox_index] = utils_fish_landmark_detection.two_bounding_point_to_TL_BR_pattern(
                        fish_boundingBox[fish_boundingBox_index])
                    a_fish_boundingBox_points = np.array(fish_boundingBox[fish_boundingBox_index]).reshape(-1,
                                                                                                           4).squeeze()
                    line_sample_points = utils_fish_landmark_detection.line_split_evenly(a_fish_mouth_points, a_fish_tail_points, 100)
                    points_in_box_count = 0
                    for a_line_point in line_sample_points:
                        if utils_fish_landmark_detection.is_point_in_boundingBox(a_line_point, a_fish_boundingBox_points):
                            points_in_box_count = points_in_box_count + 1
                    p1x, p1y, p2x, p2y = fish_boundingBox[fish_boundingBox_index][0][0], \
                        fish_boundingBox[fish_boundingBox_index][0][1], fish_boundingBox[fish_boundingBox_index][1][0], \
                        fish_boundingBox[fish_boundingBox_index][1][1]
                    boundingBox_diagonal_distance = np.linalg.norm(np.array([p1x, p1y]) - np.array([p2x, p2y]))
                    mouth_tail_distance = np.linalg.norm(a_fish_tail_points - a_fish_mouth_points)
                    tail_mouth_correlation_matrix[fish_mouth_index][fish_tail_index][
                        fish_boundingBox_index] = points_in_box_count / line_sample_counts + 0.01 * mouth_tail_distance / boundingBox_diagonal_distance
        tail_mouth_correlation_matrix_1_dim = tail_mouth_correlation_matrix.reshape(-1, 1)
        if len_fish_mouth_points == 1:
            sorted_index_tail_mouth_correlation = [0]
        else:
            sorted_index_tail_mouth_correlation = np.argsort(tail_mouth_correlation_matrix_1_dim, 0).squeeze()
        '鱼嘴尾错误检查'
        is_aggregated_fish_label_valid = True
        error_type = 0
        mouth_tail_check_table = np.zeros([len_fish_mouth_points, 1])
        tail_boundingBox_rate = len_fish_tail_points * len_fish_boundingBox
        for priority_mouth_tail_match_index in range(len_fish_mouth_points):
            a_index_1_dim = sorted_index_tail_mouth_correlation[-(priority_mouth_tail_match_index + 1)]
            a_check_table_index = a_index_1_dim // tail_boundingBox_rate
            mouth_tail_check_table[a_check_table_index] = mouth_tail_check_table[a_check_table_index] + 1
        for a_data_in_check_tabel in mouth_tail_check_table:
            if int(a_data_in_check_tabel) != 1:
                is_aggregated_fish_label_valid = False
                error_type = 1
        print(f'sorted_index_tail_mouth_correlation length is {len(sorted_index_tail_mouth_correlation)}')
        '?有几条鱼就取前几的匹配分数''更好的方式的采用非极大值抑制'
        aggregated_fish_bmt_labels_a_image = []
        for priority_mouth_tail_match_index in range(len_fish_mouth_points):
            a_index_1_dim = sorted_index_tail_mouth_correlation[-(priority_mouth_tail_match_index + 1)]
            index_fish_mouth = a_index_1_dim // tail_boundingBox_rate
            remain_count = a_index_1_dim % tail_boundingBox_rate
            index_fish_tail = remain_count // len_fish_boundingBox
            index_fish_boundingBox = remain_count % len_fish_boundingBox - 1
            a_aggregated_fish_label = [fish_boundingBox[index_fish_boundingBox], fish_mouth_points[index_fish_mouth],
                                       fish_tail_points[index_fish_tail]]
            aggregated_fish_bmt_labels_a_image.append(a_aggregated_fish_label)
        '鱼鳍聚合'
        len_aggregated_fish_bmt_labels_a_image = len(aggregated_fish_bmt_labels_a_image)
        len_fish_pectoral_fin = len(fish_pectoral_fin_points)
        body_fin_correlation_matrix = np.zeros((len_aggregated_fish_bmt_labels_a_image, len_fish_pectoral_fin))
        for a_mouth_tail_index in range(len_aggregated_fish_bmt_labels_a_image):
            for a_fish_pectoral_fin_index in range(len_fish_pectoral_fin):
                a_mouth_point = np.array(aggregated_fish_bmt_labels_a_image[a_mouth_tail_index][1]).squeeze()
                a_tail_point = np.array(aggregated_fish_bmt_labels_a_image[a_mouth_tail_index][2]).squeeze()
                a_pectoral_fin_point = np.array(fish_pectoral_fin_points[a_fish_pectoral_fin_index]).squeeze()
                a_fin_to_body_distance = utils_fish_landmark_detection.point_to_two_points_line_distance(a_pectoral_fin_point,
                                                                                      a_mouth_point, a_tail_point)
                a_fin_to_mouth_distance = np.linalg.norm(a_pectoral_fin_point - a_mouth_point)
                a_fin_to_tail_distance = np.linalg.norm(a_pectoral_fin_point - a_tail_point)
                body_fin_correlation_matrix[a_mouth_tail_index][
                    a_fish_pectoral_fin_index] = a_fin_to_body_distance + a_fin_to_mouth_distance + a_fin_to_tail_distance

        body_fin_correlation_matrix_1_dim = body_fin_correlation_matrix.reshape(-1, 1)
        sorted_index_body_fin_correlation_matrix = np.argsort(body_fin_correlation_matrix_1_dim, 0).squeeze()
        for priority_body_fin_correlation in range(len_fish_mouth_points * 2):
            a_index_1_dim = sorted_index_body_fin_correlation_matrix[priority_body_fin_correlation]
            index_fish_body = a_index_1_dim // len_fish_pectoral_fin
            index_pectoral_fin = a_index_1_dim % len_fish_pectoral_fin
            aggregated_fish_bmt_labels_a_image[index_fish_body].append(fish_pectoral_fin_points[index_pectoral_fin])


        '修正鱼鳍顺时针或逆时针'
        for a_aggregated_fish_label in aggregated_fish_bmt_labels_a_image:
            if len(a_aggregated_fish_label) != 5:
                is_aggregated_fish_label_valid = False
                if int(error_type) == 1:
                    error_type = 3
                else:
                    error_type = 2
                continue
            # xywh_bboxes = ops.box_convert(torch.tensor(a_aggregated_fish_label[0]).reshape(-1, 4), in_fmt='xyxy', out_fmt='xywh')
            # x, y, w, h = xywh_bboxes.squeeze().np()
            # rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
            # axes.add_patch(rect)
            # axes.scatter(a_aggregated_fish_label[1][0][0], a_aggregated_fish_label[1][0][1],marker='.',c='r')
            # axes.scatter(a_aggregated_fish_label[2][0][0], a_aggregated_fish_label[2][0][1],marker='.',c='g')
            # axes.scatter(a_fin_point[0][0], a_fin_point[0][1], marker='.', c='b')
            # a_fin_point[0][0], a_fin_point[0][1], marker='.', c='y')
            a_mouth_point = np.array(a_aggregated_fish_label[1]).squeeze()
            a_tail_point = np.array(a_aggregated_fish_label[2]).squeeze()
            a_fin_point1 = np.array(a_aggregated_fish_label[3]).squeeze()
            a_fin_point2 = np.array(a_aggregated_fish_label[4]).squeeze()
            # tail_mouth = a_tail_point - a_mouth_point
            fin1_mouth = a_fin_point1 - a_mouth_point
            fin2_mouth = a_fin_point2 - a_mouth_point
            c0 = np.cross(fin1_mouth, fin2_mouth)

            if c0 >= 0:
                a_temp = a_aggregated_fish_label[3]
                a_aggregated_fish_label[3] = a_aggregated_fish_label[4]
                a_aggregated_fish_label[4] = a_temp
            fish_land_mark = [a_aggregated_fish_label[1][0], a_aggregated_fish_label[3][0], a_aggregated_fish_label[2][0], a_aggregated_fish_label[4][0]]
            fish_land_mark_dict = create_a_polygon_dict(fish_land_mark)
            valid_labeled_landmark_fish_count = valid_labeled_landmark_fish_count + 1
            a_polygon_label_content['shapes'].append(fish_land_mark_dict)

        polygon_label_json_file_name = polygon_label_save_path / (
                    a_label_file_name.stem + '_polygon' + '.json')
        keypoints_object_json_file = open(str(polygon_label_json_file_name), 'w')
        keypoints_object_json_file.write(json.dumps(a_polygon_label_content))
        keypoints_object_json_file.close()
    print(f'labeled_landmark_fish_count is {labeled_landmark_fish_count}')
    print(f'valid_labeled_landmark_fish_count is {valid_labeled_landmark_fish_count}')
    pass



if __name__ == '__main__':
    labelme_label_file_path = pathlib.Path('H:/code/python/IRFishDetection2.0.0/dataset2.1/anji_process/train_label_simple_checked_combined')
    simple_label_save_path = pathlib.Path('H:/code/python/IRFishDetection2.0.0/dataset2.1/anji_process/polygon_label/train_label_simple_checked_combined')
    labelme_label_files = utils_fish_landmark_detection.get_filenames_of_path(labelme_label_file_path)
    labelme_landmark_label_to_labelme_polygon_label(labelme_label_files, simple_label_save_path)
    pass
