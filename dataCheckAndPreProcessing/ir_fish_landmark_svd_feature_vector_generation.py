import pathlib
from utiles import utils_fish_landmark_detection
from utiles import displayTool
from utiles import utiles_files
import json
import numpy as np
import matplotlib.pyplot as plt
import math
import torch
import shutil
import torch
from params import params_objects

def principal_axis_sampling(mean, variance, principal_axis, sample_variance_points):
    samples = []
    for a_sample_variance_points in sample_variance_points:
        a_sample = mean + a_sample_variance_points * variance * principal_axis
        samples.append(np.squeeze(a_sample))
    return samples


def generate_rotated_and_translation_landmarks_4(landmarks, orientations, delta_x, delta_y):
    size_landmarks = landmarks.shape[0]
    rotation_size = orientations.shape[0]
    size_x = delta_x.shape[0]
    size_y = delta_y.shape[0]
    rotated_landmark_size = size_landmarks * rotation_size * size_x * size_y
    rotated_landmarks = np.zeros((rotated_landmark_size, 8))
    count = 0
    for a_orientations in orientations:
        for a_delta_x in delta_x:
            for a_delta_y in delta_y:
                rm = utils_fish_landmark_detection.rotation_matrix(a_orientations)
                for a_landmark_set in landmarks:
                    p1 = np.array([[a_landmark_set[0]], [a_landmark_set[1]]])
                    p2 = np.array([[a_landmark_set[2]], [a_landmark_set[3]]])
                    p3 = np.array([[a_landmark_set[4]], [a_landmark_set[5]]])
                    p4 = np.array([[a_landmark_set[6]], [a_landmark_set[7]]])
                    tran = np.array([a_delta_x, a_delta_y])
                    r_p1 = np.dot(rm, p1).squeeze() + tran
                    r_p2 = np.dot(rm, p2).squeeze() + tran
                    r_p3 = np.dot(rm, p3).squeeze() + tran
                    r_p4 = np.dot(rm, p4).squeeze() + tran
                    a_rotated_landmark_set = np.array([r_p1[0], r_p1[1], r_p2[0], r_p2[1], r_p3[0], r_p3[1], r_p4[0], r_p4[1]])
                    rotated_landmarks[count] = a_rotated_landmark_set
                    count = count + 1
    assert count == rotated_landmark_size
    return rotated_landmarks


if __name__ == '__main__':
    colorMap = {0: 'red', 1: 'orange', 2: 'blue', 3: 'green'}
    user_defined_json_files_path = pathlib.Path(params_objects.data_path_hu_s_and_c_v1.all_label_path_str)
    # label_me_polygon_files_path = pathlib.Path('H:/code/python/IRFishDetection2.0.0/dataset2.1'
    #                                            '/poly_and_used_defined_label/all_checked_label/label_me_polygon_1.0_2.0')
    results_save_path = pathlib.Path(params_objects.other_results_VGG16_FPN2L_DR8_16_save_path_v1.svd_feature_path)
    pca_feature_size = params_objects.VGG16_FPN2L_DR8_16_v1.landmark_dim
    #pca_feature_size = 4

    # invalid_file_copy_path = pathlib.Path('H:/code/python/IRFishDetection2.0.0/dataset2.1/poly_and_used_defined_label'
    #                                       '/all_checked_label/outlier_file_name_1.0_2.0')
    json_files = utiles_files.get_filenames_of_path(user_defined_json_files_path)
    landmark_polygon_list = []
    aligned_landmark_list = []
    file_name_list = []
    bbox_fish_count = 0
    for a_json_file in json_files:
        json_content = utiles_files.read_json(a_json_file)
        bbox_and_landmarks = json_content['points']
        for a_bbox_and_landmarks in bbox_and_landmarks:
            bbox_fish_count = bbox_fish_count + 1
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
    # x_major_tricks = np.linspace(-60, 40, 20)
    # y_major_tricks = np.linspace(-20, 180, 40)
    # aligned_landmark_fig = plt.figure('aligned_landmark_fig')
    # axes = aligned_landmark_fig.add_subplot(111)
    # axes.set_xticks(x_major_tricks)
    # axes.set_yticks(y_major_tricks)
    # axes.grid(which='major', alpha=0.3)
    # displayTool.display_key_points( aligned_landmark_list, axes, color='blue', pointColorMap=colorMap)
    # aligned_landmark_fig.savefig(str(display_figure_save_path / 'aligned_landmarks.png'))

    # 产生旋转样本
    aligned_landmark_np = np.array(aligned_landmark_list)
    # start_orientation = np.deg2rad(-10)
    # end_orientation = np.deg2rad(10)
    # interval = np.deg2rad(22.5)
    # orientation_sample_size = int((end_orientation - start_orientation) / interval) + 1
    # orientations = np.linspace(start_orientation, end_orientation, 11)
    # delta_x = np.linspace(-4,4,5)
    # delta_y = np.linspace(-4,4,5)
    # transformed_aligned_landmark_np = generate_rotated_and_translation_landmarks_4(aligned_landmark_np, orientations, delta_x, delta_y)

    # svd
    data_set_size = aligned_landmark_np.shape[0]
    aligned_landmark_mean = np.expand_dims(np.mean(aligned_landmark_np, axis=0), 0)
    aligned_landmark_mean_expanded = np.repeat(aligned_landmark_mean, aligned_landmark_np.shape[0], 0)
    mean_normalised_aligned_landmark_np = aligned_landmark_np - aligned_landmark_mean_expanded
    mean_normalised_aligned_landmark_np = mean_normalised_aligned_landmark_np.transpose()
    U, Sigmal, V = np.linalg.svd(mean_normalised_aligned_landmark_np)
    sample_variance_points = np.linspace(-3, 3, 11)
    normalized_sigmal = Sigmal / np.sqrt(data_set_size - 1)

    # transformed_data_set_size = transformed_aligned_landmark_np.shape[0]
    # transformed_aligned_landmark_mean = np.expand_dims(np.mean(transformed_aligned_landmark_np, axis=0), 0)
    # transformed_aligned_landmark_mean_expanded = np.repeat(transformed_aligned_landmark_mean, transformed_aligned_landmark_np.shape[0], 0)
    # mean_normalised_transformed_aligned_landmark_np = transformed_aligned_landmark_np - transformed_aligned_landmark_mean_expanded
    # mean_normalised_transformed_aligned_landmark_np = mean_normalised_transformed_aligned_landmark_np.transpose().astype(np.float32)
    # transformed_U, transformed_Sigmal, transformed_V = np.linalg.svd(mean_normalised_transformed_aligned_landmark_np)
    # sample_variance_points = np.linspace(-2, 2, 10)
    x_major_tricks_sigmal = np.linspace(1, 8, 7)
    y_major_tricks_sigmal = np.linspace(0, 2000, 10)
    fig_sigmal = plt.figure(f'sigmal_display')
    axes = fig_sigmal.add_subplot(111)
    # axes.set_xticks(x_major_tricks_sigmal)
    # axes.set_yticks(y_major_tricks_sigmal)
    axes.set_xlabel('Index')
    axes.set_ylabel('Normalized Singular Value')
    sigmal_size = normalized_sigmal.shape[0]
    axes.plot(range(1,sigmal_size+1),normalized_sigmal)
    #axes.legend()
    np.save(f"{str(results_save_path)}\\ori_sigmal.npy", normalized_sigmal)
    fig_sigmal.savefig(f'{str(results_save_path)}/sigmal_display.svg', dpi=600, format='svg')
    for component_index in range(8):
        # shape_generate_with_principal_axis = principal_axis_sampling(aligned_landmark_mean,
        #                                                              math.sqrt(Sigmal[component_index]),
        #                                                              np.expand_dims(U[:, component_index], 0),
        #                                                              sample_variance_points)
        shape_generate_with_principal_axis = principal_axis_sampling(aligned_landmark_mean,
                                                                     normalized_sigmal[component_index],
                                                                     np.expand_dims(U[:, component_index], 0),
                                                                     sample_variance_points)
        fig = plt.figure(f'principal_display_{component_index}')
        axes = fig.add_subplot(111)
        axes.set_xlabel('X Pixel Coordinates')
        axes.set_ylabel('Y Pixel Coordinates')
        winter_color_map = plt.cm.get_cmap('Set3')

        displayTool.display_key_points_counter_color_map(shape_generate_with_principal_axis, axes, winter_color_map, colorMap)
        fig.savefig(f'{str(results_save_path)}/principal_display_{component_index}.svg', dpi=600, format='svg')
        plt.close(fig)

        # for shape_index, a_shape in enumerate(shape_generate_with_principal_axis):
        #     fig = plt.figure(f'principal_display_{component_index}_{shape_index}')
        #     axes = fig.add_subplot(111)
        #     a_shape = np.expand_dims(a_shape, 0)
        #     displayTool.display_key_points_counter_color_map(a_shape, axes, winter_color_map, colorMap)
        #     fig.savefig(f'{str(results_save_path)}/principal_display_{component_index}_shape_{shape_index}.svg', dpi=600, format='svg')

    # 结果保存
    aligned_landmark_mean_tensor = torch.from_numpy(aligned_landmark_mean)
    svd_feature_vectors = torch.from_numpy(U[:, 0:pca_feature_size].transpose())
    #important_singular_values = torch.from_numpy(Sigmal[0:pca_feature_size])
    # save_dict = {'mean_feature': aligned_landmark_mean_tensor, 'svd_feature': svd_feature_vectors,
    #              'normalized_singular_values': important_singular_values/np.sqrt(data_set_size-1), 'sample_size': data_set_size}
    save_dict = {'mean_feature': aligned_landmark_mean_tensor, 'svd_feature': svd_feature_vectors,
                 'normalized_singular_values': normalized_sigmal}
    save_file_name = results_save_path / f'svd_features'
    torch.save(save_dict, str(save_file_name))
    load_dict = torch.load(str(save_file_name))

    for index_svd, a_svd_feature in enumerate(svd_feature_vectors):
        a_svd_feature = a_svd_feature.unsqueeze(0)
        fig = plt.figure(f'feature_svd_{index_svd}')
        axes = fig.add_subplot(111)
        displayTool.display_key_points(a_svd_feature, axes, color='blue', pointColorMap=colorMap)
        fig.savefig(f'{str(results_save_path)}/feature_svd_{index_svd}.png')

    # 显示读取的结果
    fig = plt.figure(f'load_feature_mean')
    axes = fig.add_subplot(111)
    loaded_feature_mean = load_dict['mean_feature']
    displayTool.display_key_points(loaded_feature_mean, axes, color='blue', pointColorMap=colorMap)
    fig.savefig(f'{str(results_save_path)}/load_feature_mean.png')

    for index_svd, a_loaded_svd_feature in enumerate(load_dict['svd_feature']):
        a_loaded_svd_feature = a_loaded_svd_feature.unsqueeze(0)
        fig = plt.figure(f'load_feature_svd_{index_svd}')
        axes = fig.add_subplot(111)
        displayTool.display_key_points(a_loaded_svd_feature, axes, color='blue', pointColorMap=colorMap)
        fig.savefig(f'{str(results_save_path)}/load_feature_svd_{index_svd}.png')
        pass

    pass
