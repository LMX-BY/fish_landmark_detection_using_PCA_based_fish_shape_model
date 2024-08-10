import pathlib
from utiles import utils_fish_landmark_detection
from utiles import displayTool
import cv2
import numpy as np
import itertools
import torch
import math
from model.retinafaceDetection.RetinaFaceFishLandmarkPCADetectionNet2Layer import RetinaFaceLandmarkPCADetectionNet2Layer
from model.retinafaceDetection.RetinaFaceFishLandmarkPCADetectionNet3Layer import RetinaFaceLandmarkPCADetectionNet3Layer
from params import params_objects
import matplotlib.pyplot as plt
color_green = (0, 255, 0)
color_red = (255, 0, 0)
color_blue = (0, 0, 255)
color_white = (255, 255, 255)
color_yellow = (255, 255, 0)


def unknown_sample_index_to_orig_coordinate(index, f_size, scale, sample_size_in_a_grid):
    col_size_f = f_size[1]
    row_size_f = f_size[0]
    grid_index_in_f = math.floor(index / sample_size_in_a_grid)
    y_in_f = math.floor(grid_index_in_f / col_size_f)
    x_in_f = grid_index_in_f % col_size_f
    start_x_in_orig = x_in_f * scale
    start_y_in_orig = y_in_f * scale
    orig_coordinate = []
    for a_x_in_orig in range(int(start_x_in_orig), int(start_x_in_orig + scale)):
        for a_y_in_orig in range(int(start_y_in_orig), int(start_y_in_orig + scale)):
            a_orig_coordinate = [a_x_in_orig, a_y_in_orig]
            orig_coordinate.append(a_orig_coordinate)
    return orig_coordinate


if __name__ == '__main__':
    img_path_str = params_objects.data_path_hu_s_and_c_v1.train_img_path_str
    unknown_index_file_path = pathlib.Path(params_objects.other_results_VGG16_FPN2L_DR8_16_save_path_v1.unknown_mask_index_path_str)
    unknown_index_files = utils_fish_landmark_detection.get_filenames_of_path(unknown_index_file_path)
    results_save_path = params_objects.other_results_VGG16_FPN2L_DR8_16_save_path_v1.unknown_mask_display_path_str

    detection_net = RetinaFaceLandmarkPCADetectionNet2Layer(params_objects.VGG16_FPN2L_DR8_16_v1)
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
    f_map_num = len(f_size[1])

    for a_unknown_index_files in unknown_index_files:
        name_mark = a_unknown_index_files.stem.split('_')[-1]
        img_name = '_'.join(a_unknown_index_files.stem.split('_')[0:-1])
        img_file_name = f'{img_path_str}/{img_name}.bmp'
        img = cv2.imread(img_file_name)
        display_img_list = []

        for copy_img_index in range(f_map_num):
           a_copy_image = img.copy()
           display_img_list.append(a_copy_image)

        unknown_index = torch.load(a_unknown_index_files)
        # fig_f1 = plt.figure(f'{img_name}_unknown_index_f1')
        # axes_f1 = fig_f1.add_subplot(111)
        # fig_f2 = plt.figure(f'{img_name}_unknown_index_f2')
        # axes_f2 = fig_f2.add_subplot(111)

        # axes_f1.imshow(img)
        # axes_f2.imshow(img)
        #
        # axes_list = [axes_f1, axes_f2]
        # img_display = [img1_display, img2_display]
        if name_mark == 'unknownCoordinate':
            for a_unknown_coordinate in unknown_index:
                cur_f_index = int(a_unknown_coordinate[2])
                cur_f_size = f_size[1][cur_f_index]
                cur_f_scale = f_size[2][cur_f_index]
                cur_sample_x_f = a_unknown_coordinate[0]
                cur_sample_y_f = a_unknown_coordinate[1]
                cur_sample_x_ori = cur_sample_x_f * cur_f_scale
                cur_sample_y_ori = cur_sample_y_f * cur_f_scale
                #axes_show = axes_list[cur_f_index]
                cur_img_display = display_img_list[cur_f_index]
                display_size = 1
                # axes_show.scatter(cur_sample_x_ori, cur_sample_y_ori, s=display_size,

                test1,test2 = (int(cur_sample_x_ori), int(cur_sample_y_ori))

                cv2.circle(cur_img_display, (int(cur_sample_x_ori), int(cur_sample_y_ori)), 1, (0, 0, 255), 1)
        elif name_mark == 'unknownIndexMask':
            count = 0
            for cur_index, a_unknown_value in enumerate(unknown_index):
                prior_index = int(cur_index)
                if a_unknown_value == 0:
                    continue
                cur_f_index = 0
                for a_f_index, a_start_f_index in enumerate(f_grid_size[0:-1]):
                    a_end_f_index = f_grid_size[a_f_index+1]
                    if a_end_f_index >= cur_index >= a_start_f_index:
                        cur_f_index = a_f_index
                cur_f_scale = f_size[2][cur_f_index]
                cur_f_size = f_size[1][cur_f_index]
                start_index_f = f_grid_size[cur_f_index]
                prior_index_in_f = prior_index - start_index_f
                unknown_index_in_orig = unknown_sample_index_to_orig_coordinate(prior_index_in_f, cur_f_size, cur_f_scale, sample_size_in_a_grid)
                # axes_show = axes_list[cur_f_index]
                cur_img_display = display_img_list[cur_f_index]
                # coordinate_orig = [a_unknown_index[2],a_unknown_index[3]]
                # unknown_index_in_orig.append(coordinate_orig)
                unknown_index_in_orig_np = np.array(unknown_index_in_orig)
                for a_unknown_index_in_orig_np in unknown_index_in_orig_np:
                    cv2.circle(cur_img_display,
                               (int(a_unknown_index_in_orig_np[0]), int(a_unknown_index_in_orig_np[1])), 1, (0, 0, 255),
                               1)
                display_size = 1
                # axes_show.scatter(unknown_index_in_orig_np[:,0], unknown_index_in_orig_np[:,1], s=display_size, marker='.', c='blue')
                count = count + 1
        else:
            count = 0
            for a_unknown_index in unknown_index:
                cur_f_index = int(a_unknown_index[1])
                cur_f_size = f_size[1][cur_f_index]
                cur_f_scale = f_size[2][cur_f_index]
                prior_index = int(a_unknown_index[0])
                start_index_f = f_grid_size[cur_f_index]
                prior_index_in_f = prior_index - start_index_f
                unknown_index_in_orig = unknown_sample_index_to_orig_coordinate(prior_index_in_f, cur_f_size, cur_f_scale, sample_size_in_a_grid)
                #axes_show = axes_list[cur_f_index]
                cur_img_display = display_img_list[cur_f_index]
                # coordinate_orig = [a_unknown_index[2],a_unknown_index[3]]
                #unknown_index_in_orig.append(coordinate_orig)
                unknown_index_in_orig_np = np.array(unknown_index_in_orig)
                for a_unknown_index_in_orig_np in unknown_index_in_orig_np:
                    cv2.circle(cur_img_display, (int(a_unknown_index_in_orig_np[0]), int(a_unknown_index_in_orig_np[1])), 1, (0, 0, 255), 1)
                display_size = 1
                #axes_show.scatter(unknown_index_in_orig_np[:,0], unknown_index_in_orig_np[:,1], s=display_size, marker='.', c='blue')
                count = count + 1
            # if count > 200:
            #     break

        # fig_f1.savefig(f'{results_save_path}/{img_name}_unknown_index_fig_1.png')
        # fig_f2.savefig(f'{results_save_path}/{img_name}_unknown_index_fig_2.png')
        for display_img_index, a_display_img in enumerate(display_img_list):
            cv2.imwrite(f'{results_save_path}/{img_name}_unknown_index_img{display_img_index}.png', a_display_img)
        pass
