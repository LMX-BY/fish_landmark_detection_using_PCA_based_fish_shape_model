from params import params_objects
import torch
import matplotlib.pyplot as plt
import numpy as np
from dataCheckAndPreProcessing.ir_fish_landmark_svd_feature_vector_generation import generate_rotated_and_translation_landmarks_4
from utiles import displayTool
landmark_points_colormap = {0: 'red', 1: 'orange', 2: 'blue', 3: 'green'}
if __name__ == '__main__':
    # path and parameter
    # ut_params = params_objects.unscented_svd_augmented_1_1_2p5_large_size_v1
    # ut_params = params_objects.unscented_svd_augmented_5_5_10_large_size_v1
    #ut_params = params_objects.unscented_svd_augmented_2p5_2p5_5_large_size_v1
   # ut_params = params_objects.unscented_svd_augmented_2p5_2p5_5_large_size_v2
    #ut_params = params_objects.unscented_svd_augmented_1_1_2p5_large_size_v2
    ut_params = params_objects.unscented_svd_augmented_5_5_10_large_size_v2
    original_svd_feature_path_str = params_objects.data_path_svd_origin_VGG16_FPN2L_DR8_16_v1.svd_param_file_path_str
    results_save_path = ut_params.augmented_svd_param_file_path_str
    bias_x = ut_params.bias_x
    bias_y = ut_params.bias_y
    bias_orientation = ut_params.bias_orientation
    size_x = ut_params.size_x
    size_y = ut_params.size_y
    size_orientation = ut_params.size_orientation
    unscented_k = ut_params.unscented_k

    # load input data
    original_svd_features = torch.load(original_svd_feature_path_str)

    # get original unscented transformed svd features
    original_unscented_samples_dict = {}
    mean_feature = original_svd_features['mean_feature']
    # row vector is svd feature vector
    svd_features = original_svd_features['svd_feature']
    svd_singular_value = original_svd_features['normalized_singular_values']
    #svd_sample_size = original_svd_features['sample_size']
#    sqrt_singular = torch.sqrt(svd_singular_value)
    # generate rotated and translated samples of original svd features
    for i in range(9):
        if i == 0:
            original_unscented_samples_dict["mean"] = mean_feature
        else:
            idx = i - 1
            original_unscented_samples_dict[f"mean_plus_{i}"] = mean_feature + np.sqrt((8+unscented_k)) * svd_singular_value[idx] * svd_features[idx, :]
            original_unscented_samples_dict[f"mean_minus_{i}"] = mean_feature - np.sqrt((8+unscented_k)) * svd_singular_value[idx] * svd_features[idx, :]
            # original_unscented_samples_dict[f"mean_plus_{i}"] = mean_feature + svd_singular_value[idx] * svd_features[idx, :]
            # original_unscented_samples_dict[f"mean_minus_{i}"] = mean_feature - svd_singular_value[idx] * svd_features[idx, :]

    # unscented transformed svd features display
    for k_ind in original_unscented_samples_dict.keys():
        fig, axes = plt.subplots()
        print(f'k_ind is {k_ind}')
        test_1 = original_unscented_samples_dict[k_ind]
        test_2 = test_1[0, 0:8:2]
        axes.fill(original_unscented_samples_dict[k_ind][0, 0:8:2], original_unscented_samples_dict[k_ind][0, 1:9:2], alpha=0.1, c="blue")
        for point_index in range(0, 4):
            axes.scatter(original_unscented_samples_dict[k_ind][0, point_index * 2], original_unscented_samples_dict[k_ind][0, point_index * 2 + 1],
                         marker='.',
                         color=landmark_points_colormap[point_index])
        fig.savefig(f"{results_save_path}\\ori_ut_svd_features_{k_ind}.png")
        plt.close()

    fig, axes = plt.subplots()
    for k_all in original_unscented_samples_dict.keys():
        axes.fill(original_unscented_samples_dict[k_all][0, 0:8:2], original_unscented_samples_dict[k_all][0, 1:9:2], alpha=0.1, c="blue")
        for point_index in range(0, 4):
            axes.scatter(original_unscented_samples_dict[k_all][0, point_index * 2], original_unscented_samples_dict[k_all][0, point_index * 2 + 1],
                         marker='.',
                         color=landmark_points_colormap[point_index])
    fig.savefig(f"{results_save_path}\\ori_ut_svd_features_all.png")
    plt.close()

    # svd for transformed samples


    start_orientation = np.deg2rad(-bias_orientation)
    end_orientation = np.deg2rad(bias_orientation)
    orientations = np.linspace(start_orientation, end_orientation, size_orientation)
    delta_x = np.linspace(-bias_x, bias_x, size_x)
    delta_y = np.linspace(-bias_y, bias_y, size_y)

    rt_transformed_samples_dict = {}
    for ky, val in original_unscented_samples_dict.items():
        rt_transformed_samples_dict[ky] = generate_rotated_and_translation_landmarks_4(
            val, orientations, delta_x, delta_y).tolist()

    # w = k/(n+k), i = 0
    # w = 1/2(n+k), i = 1, 2, ..., 2n
    # sample augmentation
    w_mean = unscented_k / (8 + unscented_k)
    w_others = 1 / (2 * (8 + unscented_k))
    if w_mean > w_others:
        w = int(round(w_mean / w_others))
        rt_transformed_samples_dict["mean"] = rt_transformed_samples_dict["mean"] * w
    else:
        w = int(round(w_others / w_mean))
        for ky in rt_transformed_samples_dict.keys():
            if ky != "mean":
                rt_transformed_samples_dict[ky] = rt_transformed_samples_dict[ky] * w

   # compute svd features with resampling samples
    augmented_samples_list = []
    for vl in rt_transformed_samples_dict.values():
        augmented_samples_list.extend(vl)

    augmented_samples_np = np.array(augmented_samples_list)


    aligned_landmark_mean = np.expand_dims(np.mean(augmented_samples_np, axis=0), 0)
    augmented_sample_size = augmented_samples_np.shape[0]
    aligned_landmark_mean_expanded = np.repeat(aligned_landmark_mean, augmented_samples_np.shape[0], 0)
    mean_normalised_aligned_landmark_np = augmented_samples_np - aligned_landmark_mean_expanded
    mean_normalised_aligned_landmark_np = mean_normalised_aligned_landmark_np.transpose()
    U_np_arr, Sigma_np_arr, V_np_arr = np.linalg.svd(mean_normalised_aligned_landmark_np)
    normalized_Sigma_np_arr = Sigma_np_arr / np.sqrt(augmented_sample_size-1)
    #sqrt_Sigma_np_arr = np.sqrt(Sigma_np_arr)
    mean_np_arr = aligned_landmark_mean[0]


    # results save
    unscented_transformed_svd_param_dict = {}
    aligned_landmark_mean_tensor = torch.from_numpy(aligned_landmark_mean)
    unscented_transformed_svd_feature_tensor = torch.from_numpy(U_np_arr.transpose())
    unscented_transformed_normalized_singular_tensor = torch.from_numpy(normalized_Sigma_np_arr)
    unscented_transformed_svd_param_dict['mean_feature'] = aligned_landmark_mean_tensor
    unscented_transformed_svd_param_dict['svd_feature'] = unscented_transformed_svd_feature_tensor
    unscented_transformed_svd_param_dict['normalized_singular_values'] = unscented_transformed_normalized_singular_tensor
    #unscented_transformed_svd_param_dict['sample_size'] = svd_sample_size
    torch.save(unscented_transformed_svd_param_dict, f'{results_save_path}/unscented_transformed_svd_param')
    # results display
    fig_sigmal = plt.figure(f'sigmal_display')
    axes = fig_sigmal.add_subplot(111)
    # axes.set_xticks(x_major_tricks_sigmal)
    # axes.set_yticks(y_major_tricks_sigmal)
    axes.set_xlabel('Index')
    axes.set_ylabel('Normalized Singular Value')
    sigmal_size = Sigma_np_arr.shape[0]
    axes.plot(range(1, sigmal_size + 1), normalized_Sigma_np_arr)
    axes.legend()
    fig_sigmal.savefig(f'{results_save_path}/sigmal_display.svg', dpi=600, format='svg')
    # np.save(f"{results_save_path}\\sigmal.npy", Sigma_np_arr)
    # np.save(f"{result_fdr_dir}\\svdFeature_mean.npy", mean_np_arr)
    # np.save(f"{result_fdr_dir}\\svdFeature_U.npy", U_np_arr)
    # np.save(f"{result_fdr_dir}\\svdFeature_Sigma.npy", Sigma_np_arr)
    # np.save(f"{result_fdr_dir}\\svdFeature_sqrtSigma.npy", sqrt_Sigma_np_arr)

    # interval_list = [3, 2.5, 2, 1.75, 1.5, 1.25, 1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    variance_points = np.linspace(-3, 3, 11)
    samples_alone_principal_axis_dict = {}
    svd_feature_generated_fish_landmarks = []
    for i in range(1, 9):
        a_feature_generated_fish_landmarks = []
        samples_alone_principal_axis_dict[f"principal_{i}_samples"] = {}
        samples_alone_principal_axis_dict[f"principal_{i}_samples"]["mean"] = mean_np_arr
        for interval in variance_points:
            idx = i - 1
            a_generated_fish_landmarks = mean_np_arr + normalized_Sigma_np_arr[idx] * interval * U_np_arr[:, idx]
            samples_alone_principal_axis_dict[f"principal_{i}_samples"][f"mean_{interval}_sigma"] = a_generated_fish_landmarks
            a_feature_generated_fish_landmarks.append(a_generated_fish_landmarks)
        svd_feature_generated_fish_landmarks.append(a_feature_generated_fish_landmarks)

    # with open(f"{result_fdr_dir}\\svdSampling_samples.json", "w+") as f_svd_sampling:
    #     json.dump(sampled_samples_dict_dict, f_svd_sampling)

    # for k_sigma, v_dict in sampled_samples_dict_dict.items():
    #     sigma_idx = k_sigma.split("_")[0][-1]
    #
    #     for k_ind in v_dict.keys():
    #         fig, axes = plt.subplots()
    #         axes.fill(v_dict[k_ind][0:8:2], v_dict[k_ind][1:9:2], alpha=0.1, c="blue")
    #         for point_index in range(0, 4):
    #             axes.scatter(v_dict[k_ind][point_index * 2],
    #                          v_dict[k_ind][point_index * 2 + 1],
    #                          marker='.',
    #                          color={0: 'red', 1: 'orange', 2: 'yellow', 3: 'green'}[point_index])
    #         fig.savefig(f"{result_fdr_dir}\\svdSamplingPlotting_sigma{sigma_idx}_{k_ind}.png")
    #         plt.close()
    #
    #     fig, axes = plt.subplots()
    #     for k_all in v_dict.keys():
    #         axes.fill(v_dict[k_all][0:8:2], v_dict[k_all][1:9:2], alpha=0.1, c="blue")
    #
    #         for point_index in range(0, 4):
    #             axes.scatter(v_dict[k_all][point_index * 2],
    #                          v_dict[k_all][point_index * 2 + 1],
    #                          marker='.',
    #                          color={0: 'red', 1: 'orange', 2: 'yellow', 3: 'green'}[point_index])
    #     fig.savefig(f"{result_fdr_dir}\\svdSamplingPlotting_sigma{sigma_idx}_all.png")
    #     plt.close()

    for component_index, a_feature_generated_fish_landmarks in enumerate(svd_feature_generated_fish_landmarks):
        fig = plt.figure(f'principal_display_{component_index}')
        axes = fig.add_subplot(111)
        axes.set_xlabel('X Pixel Coordinates')
        axes.set_ylabel('Y Pixel Coordinates')
        winter_color_map = plt.colormaps.get_cmap('Set3')
        displayTool.display_key_points_counter_color_map(a_feature_generated_fish_landmarks, axes, winter_color_map,
                                                         landmark_points_colormap)
        fig.savefig(f'{results_save_path}/principal_display_{component_index}.svg', dpi=600, format='svg')
        plt.close(fig)
    print('gaga')

    # save results
