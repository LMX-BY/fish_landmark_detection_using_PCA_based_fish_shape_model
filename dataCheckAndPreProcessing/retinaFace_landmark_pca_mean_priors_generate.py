import torch
from params import params_objects
from model.retinafaceDetection import priorLandmarkGenerator
from model.retinafaceDetection.RetinaFaceFishLandmarkPCADetectionNet3Layer import RetinaFaceLandmarkPCADetectionNet3Layer
from model.retinafaceDetection.RetinaFaceFishLandmarkPCADetectionNet2Layer import RetinaFaceLandmarkPCADetectionNet2Layer
import pathlib
from utiles import utils_fish_landmark_detection


if __name__ == '__main__':
    detection_net = RetinaFaceLandmarkPCADetectionNet2Layer(params_objects.VGG16_FPN2L_DR8_16_v1)
    test_input_tensor = torch.zeros([1,3,1080,1920])
    f_size = detection_net.feature_size_test_forward(test_input_tensor)
    original_img_size = f_size[0]
    # feature_map_size_and_scale = [[f_size[1][0][0], f_size[1][0][1], 1],
    #                               [f_size[1][1][0], f_size[1][1][1], 1],
    #                               [f_size[1][2][0],f_size[1][2][1], 1]]
    # feature_map_size_and_scale = [[f_size[1][0][0], f_size[1][0][1], 1],
    #                               [f_size[1][1][0], f_size[1][1][1], 1]]

    feature_map_size_and_scale = [a_map_size + [1] for a_map_size in f_size[1]]
    # feature_map_size_and_scale = [[f_size[1][0][0], f_size[1][0][1], 1],
    #                               [f_size[1][1][0], f_size[1][1][1], 1]]

    pca_param_file_path = params_objects.data_path_svd_origin_VGG16_FPN2L_DR8_16_v1.svd_param_file_path_str
    #pca_param_file_path = params_objects.data_path_unscented_svd_augmented_5_5_10_large_size_VGG16_FPN2L_DR8_16_v1.svd_param_file_path_str
    #pca_param_file_path = params_objects.data_path_unscented_svd_augmented_2p5_2p5_5_large_size_VGG16_FPN2L_DR8_16_v1.svd_param_file_path_str
    #pca_param_file_path = params_objects.data_path_unscented_svd_augmented_1_1_2p5_large_size_VGG16_FPN2L_DR8_16_v1.svd_param_file_path_str


    #save_file_name = pathlib.Path(params_objects.data_path_svd_origin_VGG16_FPN2L_DR8_16_v1.landmark_priors_file_path_str)
    save_file_name = pathlib.Path(
        params_objects.data_path_unscented_svd_augmented_5_5_10_large_size_VGG16_FPN2L_DR8_16_v1.landmark_priors_file_path_str)
    # save_file_name = pathlib.Path(
    #     params_objects.data_path_unscented_svd_augmented_2p5_2p5_5_large_size_VGG16_FPN2L_DR8_16_v1.landmark_priors_file_path_str)
    # save_file_name = pathlib.Path(
    #     params_objects.data_path_unscented_svd_augmented_1_1_2p5_large_size_VGG16_FPN2L_DR8_16_v1.landmark_priors_file_path_str)

    load_dict = torch.load(pca_param_file_path)
    pca_mean = load_dict['mean_feature'].to(torch.float32)
    pca_features = load_dict['svd_feature'].to(torch.float32)
    pca_singular_value = load_dict['singular_values'].to(torch.float32)

    anc_key_pts_orientated_a_cell = utils_fish_landmark_detection.generate_orientated_anchor_key_points(16,
                                                                                                        pca_mean)
    prior_landmarks_tensor, recover_landmarks_parameters_tensor = priorLandmarkGenerator.generate_prior_fish_4_Landmarks(
        anc_key_pts_orientated_a_cell, original_img_size, feature_map_size_and_scale)
    save_dict = {'prior_landmarks': prior_landmarks_tensor,
                 'recover_landmarks_parameters': recover_landmarks_parameters_tensor}
    priorLandmarkGenerator.save_prior_fish_4_landmarks(save_file_name, save_dict)
    load_dict = priorLandmarkGenerator.load_prior_fish_4_landmarks(save_file_name)
    pass