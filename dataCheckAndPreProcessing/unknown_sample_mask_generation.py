import pathlib
from utiles import utils_fish_landmark_detection
from utiles import utiles_parameters
from utiles import displayTool
import cv2
import numpy as np
import itertools
import torch
import math
from model.retinafaceDetection.RetinaFaceFishLandmarkPCADetectionNet2Layer import RetinaFaceLandmarkPCADetectionNet2Layer
from params import params_objects
from model.retinafaceDetection.RetinaFaceFishLandmarkPCADetectionNet3Layer import RetinaFaceLandmarkPCADetectionNet3Layer


color_green = (0, 255, 0)
color_red = (255, 0, 0)
color_blue = (0, 0, 255)
color_white = (255, 255, 255)
color_yellow = (255, 255, 0)
if __name__ == '__main__':
    user_defined_json_files_path = pathlib.Path(params_objects.data_path_hu_s_and_c_v1.all_label_path_str)

    results_save_path = pathlib.Path(params_objects.other_results_VGG16_FPN2L_DR8_16_save_path_v1.unknown_mask_index_path_str)

    # user_defined_json_files_path = pathlib.Path('H:/code/python/IRFishDetection2.0.0/dataset2.1/someTest'
    #                                        '/single_label_file_for_classification_test_2/landmarks')
    # results_save_path = pathlib.Path('H:/code/python/IRFishDetection2.0.0/dataset2.1/someTest'
    #                                        '/single_label_file_for_classification_test_2/mask'
    network_params = params_objects.VGG16_FPN2L_DR8_16_v1
    json_files = utils_fish_landmark_detection.get_filenames_of_path(user_defined_json_files_path)

    detection_net = RetinaFaceLandmarkPCADetectionNet2Layer(network_params)
    test_input_tensor = torch.zeros([1, 3, 1080, 1920])
    f_size = detection_net.feature_size_test_forward(test_input_tensor)
    sample_size_in_a_grid = 16
    f_grid_size = [0]
    for index, a_f_size in enumerate(f_size[1]):
        f_size_x = a_f_size[0]
        f_size_y = a_f_size[1]
        grid_size = f_size_x * f_size_y * sample_size_in_a_grid
        cur_grid_size = grid_size + f_grid_size[index]
        f_grid_size.append(cur_grid_size)

    total_prior_size = f_grid_size[-1]
    test_img_size = 10
    test_img_count = 0
    for a_landmark_json_file in json_files:
        # if test_img_count == test_img_size:
        #     break
        test_img_count = test_img_count + 1
        json_content = utils_fish_landmark_detection.read_json(a_landmark_json_file)
        img_name = json_content['img_name']
        bbox_and_landmarks = json_content['points']
        unknown_sample_index = []
        unknown_sample_coordinate = []
        unknown_sample_index_mask = torch.zeros(total_prior_size,dtype=torch.bool)
        # prior_selected_mask = np.zeros(total_prior_size)
        coor_masks_in_f = []

        for a_f_size in f_size[1]:
            a_mask = np.zeros((a_f_size[0],a_f_size[1]))
            coor_masks_in_f.append(a_mask)
        for a_bbox_and_landmark in bbox_and_landmarks:
            if a_bbox_and_landmark[-1] != 0:
                continue
            b1 = round(a_bbox_and_landmark[0])
            b2 = round(a_bbox_and_landmark[1])
            b3 = round(a_bbox_and_landmark[2])
            b4 = round(a_bbox_and_landmark[3])
            start_x = min(b1, b3)
            end_x = max(b1, b3)
            start_y = min(b2, b4)
            end_y = max(b2, b4)
            x_len_quarter = round((end_x - start_x) / 3)
            y_len_quarter = round((end_y - start_y) / 3)
            start_x_quarter = start_x + x_len_quarter
            end_x_quarter = end_x - x_len_quarter
            start_y_quarter = start_y + y_len_quarter
            end_y_quarter = end_y - y_len_quarter
            # x_range_orig = np.linspace(start_x_quarter, end_x_quarter, end_x_quarter - start_x_quarter + 1)
            # y_range_orig = np.linspace(start_y_quarter, end_y_quarter, end_y_quarter - start_y_quarter + 1)
            x_range_orig = np.linspace(start_x, end_x, end_x - start_x + 1)
            y_range_orig = np.linspace(start_y, end_y, end_y - start_y + 1)
            size_x_range_orig = x_range_orig.shape[0]
            size_y_range_orig = y_range_orig.shape[0]
            #total_size = x_range_orig.shape[0] * y_range_orig.shape[0]
            max_xy_orig = []
            for a_scale in f_size[2]:
                max_index_x = int(np.floor(size_x_range_orig / a_scale) * a_scale - 1)
                max_index_y = int(np.floor(size_y_range_orig / a_scale) * a_scale - 1)
                a_max_xy_orig = [x_range_orig[max_index_x], y_range_orig[max_index_y]]
                max_xy_orig.append(a_max_xy_orig)
           # print(f'total_size: {total_size}')
            load_size = 0
            for x_orig, y_orig in itertools.product(x_range_orig, y_range_orig):
                for index_f_map, a_scale in enumerate(f_size[2]):
                    cur_max_x_in_orig = max_xy_orig[index_f_map][0]
                    cur_max_y_in_orig = max_xy_orig[index_f_map][1]
                    if x_orig >= cur_max_x_in_orig or y_orig >= cur_max_y_in_orig:
                        continue
                    if len(x_range_orig) > int(a_scale/3) and len(y_range_orig) > int(a_scale/3):
                        if x_orig < x_range_orig[int(a_scale/3)] or x_orig > x_range_orig[-int(a_scale/3)]:
                            continue
                        if y_orig < y_range_orig[int(a_scale/3)] or y_orig > y_range_orig[-int(a_scale/3)]:
                            continue
                    cur_mask = coor_masks_in_f[index_f_map]
                    # if x_orig == x_range_orig[-1]:
                    #     x_in_f_map = int(np.floor(x_orig / a_scale))
                    # else:
                    #     x_in_f_map = int(np.ceil(x_orig / a_scale))
                    # if y_orig == y_range_orig[-1]:
                    #     y_in_f_map = int(np.floor(y_orig / a_scale))
                    # else:
                    #     y_in_f_map = int(np.ceil(y_orig / a_scale))
                    x_in_f_map = int(np.round(x_orig / a_scale))
                    y_in_f_map = int(np.round(y_orig / a_scale))
                    x_in_f_map = min(x_in_f_map, cur_mask.shape[1]-1)
                    y_in_f_map = min(y_in_f_map, cur_mask.shape[0]-1)
                    if cur_mask[y_in_f_map][x_in_f_map] == 1:
                        continue
                    cur_mask[y_in_f_map][x_in_f_map] = 1
                    cur_f_col_size = f_size[1][index_f_map][1]
                    cur_f_row_size = f_size[1][index_f_map][0]
                    a_unknown_sample_coordinate = [x_in_f_map, y_in_f_map, index_f_map]
                    unknown_sample_coordinate.append(a_unknown_sample_coordinate)
                    a_start_unknown_sample_index = int(
                        (y_in_f_map * cur_f_col_size + x_in_f_map) * sample_size_in_a_grid + f_grid_size[index_f_map])
                    for a_index in range(a_start_unknown_sample_index,
                                         a_start_unknown_sample_index + sample_size_in_a_grid):
                        #unknown_sample_index.append([a_index, index_f_map, x_orig, y_orig])
                        unknown_sample_index.append([a_index, index_f_map])
                        unknown_sample_index_mask[a_index] = True
                        # if prior_selected_mask[a_index] == 0:
                        #     unknown_sample_index.append([a_index, index_f_map])
                        #     prior_selected_mask[a_index] = 1
                        #     load_size = load_size + 1
            print(f'load_size: {load_size}')

        unknown_sample_index_tensor = torch.from_numpy(np.array(unknown_sample_index))
        unknown_sample_coordinate_tensor = torch.from_numpy((np.array(unknown_sample_coordinate)))
        print(f'unknown_sample_size is {unknown_sample_index_tensor.shape[0]}')
        #file_name = results_save_path / f'{img_name}_unknownIndex.pt'
        file_name = results_save_path / f'{img_name}_unknownIndexMask.pt'
        # file_name_coordinate = results_save_path / f'{img_name}_unknownCoordinate.pt'
        #torch.save(unknown_sample_index_tensor, file_name)
        torch.save(unknown_sample_index_mask, file_name)
        #torch.save(unknown_sample_coordinate_tensor, file_name_coordinate)
        pass
