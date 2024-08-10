import pathlib
from utiles import utils_fish_landmark_detection
from utiles import displayTool
import json
import numpy as np
import matplotlib.pyplot as plt
import math
import torch
import shutil


def principal_axis_sampling(mean, variance, principal_axis, sample_variance_points):
    samples = []
    for a_sample_variance_points in sample_variance_points:
        a_sample = mean + a_sample_variance_points * variance * principal_axis
        samples.append(np.squeeze(a_sample))
    return samples

#一套数据集处理输入输出路径
user_defined_json_path_hu_supplemented_and_display_checked = pathlib.Path('H:/code/python/IRFishDetection2.0.0/dataset2.2/'
                                                              'anji_process/polygon_label1.0_hudingpeng_supplemented/'
                                                              'lableme_polygon_label_1.0_filterd_small_fish_hu_supplemented_and_display_checked/'
                                                              'user_defined_label')
display_figure_save_path_hu_supplemented_and_display_checked = pathlib.Path('C:/Users/DELL/Desktop/test/display')
outlier_file_copy_path_hu_supplemented_and_display_checked = pathlib.Path('H:/code/python/IRFishDetection2.0.0'
                                                                          '/dataset2.2/anji_process/polygon_label1'
                                                                          '.0_hudingpeng_supplemented'
                                                                          '/lableme_polygon_label_1.0_filterd_small_fish_hu_supplemented_and_display_checked/user_defined_label/outlier_file_checked_by_svd')

outlier_file_copy_path_hu_supplemented_and_display_checked = pathlib.Path('C:/Users/DELL/Desktop/test')

label_me_polygon_files_path_hu_supplemented_and_display_checked = pathlib.Path('H:/code/python/IRFishDetection2.0.0/dataset2.2/'
                                                                               'anji_process/polygon_label1.0_hudingpeng_supplemented/'
                                                                               'lableme_polygon_label_1.0_filterd_small_fish_hu_supplemented_and_display_checked')
#一套数据集处理输入输出路径




if __name__ == '__main__':
    colorMap = {0: 'red', 1: 'orange', 2: 'yellow', 3: 'green'}
    # user_defined_json_files_path = pathlib.Path('H:/code/python/IRFishDetection2.0.0/dataset2.1'
    #                                             '/poly_and_used_defined_label/all_checked_label/user_defined_json_1.0_2.0')
    # label_me_polygon_files_path = pathlib.Path('H:/code/python/IRFishDetection2.0.0/dataset2.1'
    #                                            '/poly_and_used_defined_label/all_checked_label/label_me_polygon_1.0_2.0')
    # display_figure_save_path = pathlib.Path('H:/code/python/IRFishDetection2.0.0/dataset2.1'
    #                                         '/poly_and_used_defined_label/all_checked_label/display_1.0_2.0')
    # invalid_file_copy_path = pathlib.Path('H:/code/python/IRFishDetection2.0.0/dataset2.1/poly_and_used_defined_label'
    #                                       '/all_checked_label/outlier_file_name_1.0_2.0')
    user_defined_json_files_path = user_defined_json_path_hu_supplemented_and_display_checked
    label_me_polygon_files_path = label_me_polygon_files_path_hu_supplemented_and_display_checked
    display_figure_save_path = display_figure_save_path_hu_supplemented_and_display_checked
    invalid_file_copy_path = outlier_file_copy_path_hu_supplemented_and_display_checked
    json_files = utils_fish_landmark_detection.get_filenames_of_path(user_defined_json_files_path)
    landmark_polygon_list = []
    aligned_landmark_list = []
    file_name_list = []

    for a_json_file in json_files:
        json_content = utils_fish_landmark_detection.read_json(a_json_file)
        bbox_and_landmarks = json_content['points']
        for a_bbox_and_landmarks in bbox_and_landmarks:
            if a_bbox_and_landmarks[6] == 0:
                continue
            a_landmark_set = [0] * 8
            a_aligned_landmark_set = [0] * 8
            a_landmark_set[0:2] = a_bbox_and_landmarks[4:6]
            a_landmark_set[2:4] = a_bbox_and_landmarks[7:9]
            a_landmark_set[4:6] = a_bbox_and_landmarks[10:12]
            a_landmark_set[6:8] = a_bbox_and_landmarks[13:15]
            landmark_polygon_list.append(a_landmark_set)
            reference_vector = np.array([0, -1])
            tail_to_mouth_vector = np.array(
                [a_landmark_set[0] - a_landmark_set[4], a_landmark_set[1] - a_landmark_set[5]])
            r_matrix = utils_fish_landmark_detection.rotation_matrix_between_two_2d_vector(tail_to_mouth_vector,
                                                                                           reference_vector)
            if math.isnan(r_matrix[0][0]):
                print(f'nan error occur!')
            a_aligned_landmark_set = a_landmark_set.copy()
            for index in range(0, 8, 2):
                a_aligned_landmark_set[index] = a_aligned_landmark_set[index] - a_landmark_set[0]
                a_aligned_landmark_set[index + 1] = a_aligned_landmark_set[index + 1] - a_landmark_set[1]
                a_aligned_landmark_set[index:index + 2] = r_matrix @ a_aligned_landmark_set[index:index + 2]
            # if a_aligned_landmark_set[7] > 5.6 and a_aligned_landmark_set[7] < 10.8:
            #     print(f'{a_json_file.name}')
            #     break
            aligned_landmark_list.append(a_aligned_landmark_set)
            file_name_list.append(a_json_file)

    # 保存aligned_landmark图片
    # x_major_tricks = np.linspace(-60, 40, 10)
    # y_major_tricks = np.linspace(-20, 180, 20)
    # aligned_landmark_fig = plt.figure('aligned_landmark_fig')
    # axes = aligned_landmark_fig.add_subplot(111)
    # axes.set_xticks(x_major_tricks)
    # axes.set_yticks(y_major_tricks)
    # axes.grid(which='major', alpha=0.3)
    # displayTool.display_key_points(aligned_landmark_list[0:200], axes, color='blue', pointColorMap=colorMap)
    # aligned_landmark_fig.savefig(str(display_figure_save_path / 'aligned_landmarks.svg'), dpi=600, format='svg')

    # svd
    aligned_landmark_np = np.array(aligned_landmark_list)
    data_set_size = aligned_landmark_np.shape[0]
    aligned_landmark_mean = np.expand_dims(np.mean(aligned_landmark_np, axis=0), 0)
    aligned_landmark_mean_expanded = np.repeat(aligned_landmark_mean, aligned_landmark_np.shape[0], 0)
    mean_normalised_aligned_landmark_np = aligned_landmark_np - aligned_landmark_mean_expanded
    mean_normalised_aligned_landmark_np = mean_normalised_aligned_landmark_np.transpose()
    U, Sigmal, V = np.linalg.svd(mean_normalised_aligned_landmark_np)
    sample_variance_points = np.linspace(-2, 2, 10)
    for component_index in range(8):
        shape_generate_with_principal_axis = principal_axis_sampling(aligned_landmark_mean,
                                                                     math.sqrt(Sigmal[component_index]),
                                                                     np.expand_dims(U[:, component_index], 0),
                                                                     sample_variance_points)
        fig = plt.figure(f'principal_display_{component_index}')
        axes = fig.add_subplot(111)
        displayTool.display_key_points(shape_generate_with_principal_axis, axes, color='blue', pointColorMap=colorMap)
        fig.savefig(f'{str(display_figure_save_path)}/principal_display_{component_index}.png')
        plt.close(fig)
        # for shape_index, a_shape in enumerate(shape_generate_with_principal_axis):
        #     fig = plt.figure(f'principal_display_{component_index}_{shape_index}')
        #     axes = fig.add_subplot(111)
        #     a_shape = np.expand_dims(a_shape,0)
        #     displayTool.display_key_points(a_shape, axes, color='blue', pointColorMap=colorMap)
        #     fig.savefig(f'{str(display_figure_save_path)}/principal_display_{component_index}_shape_{shape_index}.png')
    outlier_count = 0
    v_r = 2
    for data_index, a_mean_normalised_aligned_landmark_np in enumerate(mean_normalised_aligned_landmark_np.transpose()):
        for component_index, a_eigen_landmark_set in enumerate(U.transpose()):
            f1 = np.expand_dims(a_mean_normalised_aligned_landmark_np, 0)
            f2 = np.expand_dims(a_eigen_landmark_set, 1)
            w = np.dot(f1, f2)
            variance = math.sqrt(Sigmal[component_index])
            if w > v_r * variance or w < -v_r * variance:
                outlier_count = outlier_count + 1
                fig = plt.figure(f'outlier_display_{outlier_count}')
                axes = fig.add_subplot(111)
                outlier_file_name = file_name_list[data_index].name
                displayTool.display_key_points(np.expand_dims(aligned_landmark_np[data_index], 0), axes, color='blue',
                                               pointColorMap=colorMap)
                fig.savefig(f'{str(display_figure_save_path)}/outlier_display_{file_name_list[data_index].stem}_oc_{outlier_count}.png')
                plt.close(fig)
                # outlier_label_me_polygon_file_name = '_'.join(outlier_file_name.split('_')[:-1]) + '_polygon.json'
                # shutil.copy(f'{str(label_me_polygon_files_path)}/{outlier_label_me_polygon_file_name}',
                #             f'{str(invalid_file_copy_path)}/{outlier_label_me_polygon_file_name}')
                shutil.copy(str(file_name_list[data_index]),
                            f'{str(invalid_file_copy_path)}/{outlier_file_name}')
                print(f'outlier file name: {outlier_file_name}')
                break

    print(f'outlier_count is {outlier_count}')
    print(f'data size is {data_set_size}')
pass
# 对齐
# aligned_landmark = []
# for a_landmark_set
