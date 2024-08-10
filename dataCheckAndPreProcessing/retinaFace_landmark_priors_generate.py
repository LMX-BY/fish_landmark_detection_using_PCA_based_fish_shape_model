import torch
from model.retinafaceDetection import priorLandmarkGenerator
from model.retinafaceDetection.RetinaFaceFishLandmarkDetectionNet3Layer import RetinaFaceLandmarkDetectionNet3Layer
import pathlib
from utiles import utils_fish_landmark_detection


if __name__ == '__main__':
    detection_net = RetinaFaceLandmarkDetectionNet3Layer(config_retinaface_net.cfg_vgg16)
    test_input_tensor = torch.zeros([1,3,1080,1920])
    f_size = detection_net.feature_size_test_forward(test_input_tensor)
    original_img_size = f_size[0]
    feature_map_size_and_scale = [[f_size[1][0][0], f_size[1][0][1], 1],
                                  [f_size[1][1][0], f_size[1][1][1], 1],
                                  [f_size[1][2][0],f_size[1][2][1],1]]

    anc_key_pts_single_orientation_file_path = \
        pathlib.Path(
            'H:/code/python/AboveWaterIRFishDetection1.0.0/data/experimentData1.0.0/valid_keypoints_object_label'
            '/train/kmeans_key_points_display/single_orientation_anchor_key_points2023-05-31-15-33-48.json')

    save_file_name = pathlib.Path(
        'H:/code/python/IRFishDetection2.0.0/dataset2.1/priors/retina_face_vgg16_priors.pt')

    anc_key_pts_single_orientation = torch.tensor(
        utils_fish_landmark_detection.read_json(anc_key_pts_single_orientation_file_path))
    anc_key_pts_orientated_a_cell = utils_fish_landmark_detection.generate_orientated_anchor_key_points(8,
                                                                                                        anc_key_pts_single_orientation)
    prior_landmarks_tensor, recover_landmarks_parameters_tensor = priorLandmarkGenerator.generate_prior_fish_4_Landmarks(
        anc_key_pts_orientated_a_cell, original_img_size, feature_map_size_and_scale)
    save_dict = {'prior_landmarks': prior_landmarks_tensor,
                 'recover_landmarks_parameters': recover_landmarks_parameters_tensor}
    priorLandmarkGenerator.save_prior_fish_4_landmarks(save_file_name, save_dict)
    load_dict = priorLandmarkGenerator.load_prior_fish_4_landmarks(save_file_name)
    pass