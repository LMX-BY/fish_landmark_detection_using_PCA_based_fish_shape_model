import pathlib
import numpy as np
import itertools
import torch
from utiles import utils_fish_landmark_detection


def generate_prior_fish_4_Landmarks(landmarks_in_a_cell, original_img_size, feature_map_size_and_scale):
    num_priors_in_a_cell = len(landmarks_in_a_cell)
    num_of_priors_cell = 0
    for a_feature_map_size in feature_map_size_and_scale:
        num_of_cell_in_a_feature_map = a_feature_map_size[0] * a_feature_map_size[1]
        num_of_priors_cell = num_of_priors_cell + num_of_cell_in_a_feature_map
    num_priors = num_priors_in_a_cell * num_of_priors_cell

    recover_landmarks_parameters = np.zeros((num_priors, 2))
    prior_landmarks = np.zeros((num_priors, 8))

    total_priors_count = 0

    for k, a_size_and_scale in enumerate(feature_map_size_and_scale):
        for row, col in itertools.product(range(a_size_and_scale[0]), range(a_size_and_scale[1])):
            row_size = a_size_and_scale[0]
            col_size = a_size_and_scale[1]
            landmark_scale = a_size_and_scale[2]
            cx = (col + 0.5) / col_size
            cy = (row + 0.5) / row_size
            is_col = True
            for a_landmark_set in landmarks_in_a_cell:
                a_landmark_set_normalized = a_landmark_set.numpy().copy()
                a_landmark_set_normalized[0:8:2] = a_landmark_set_normalized[0:8:2] * landmark_scale / original_img_size[1] + cx
                a_landmark_set_normalized[1:9:2] = a_landmark_set_normalized[1:9:2] * landmark_scale / \
                                                   original_img_size[0] + cy
                a_scale_parameters = a_size_and_scale[2]
                recover_landmarks_parameters[total_priors_count] = np.array([a_scale_parameters, k])
                prior_landmarks[total_priors_count] = a_landmark_set_normalized
                total_priors_count = total_priors_count + 1

    prior_landmarks_tensor = torch.Tensor(prior_landmarks)
    recover_landmarks_parameters_tensor = torch.Tensor(recover_landmarks_parameters)
    return prior_landmarks_tensor, recover_landmarks_parameters_tensor


def save_prior_fish_4_landmarks(file_name: pathlib.Path, prior_landmarks_dict):
    torch.save(prior_landmarks_dict, str(file_name))


def load_prior_fish_4_landmarks(file_name:pathlib):
    return torch.load(str(file_name))


if __name__ == '__main__':
    anc_key_pts_single_orientation_file_path = \
        pathlib.Path(
            'H:/code/python/AboveWaterIRFishDetection1.0.0/data/experimentData1.0.0/valid_keypoints_object_label'
            '/train/kmeans_key_points_display/single_orientation_anchor_key_points2023-05-31-15-33-48.json')

    save_file_name = pathlib.Path('H:/code/python/IRFishDetection2.0.0/dataset2.1/someTest/priorLandmarkGenerator/priorLandmarks.pt')
    original_img_size = [1080, 1920]
    feature_map_size_and_scale = [[64,64,1], [32,32,1]]

    anc_key_pts_single_orientation = torch.tensor(utils_fish_landmark_detection.read_json(anc_key_pts_single_orientation_file_path))
    anc_key_pts_orientated_a_cell = utils_fish_landmark_detection.generate_orientated_anchor_key_points(8, anc_key_pts_single_orientation)
    prior_landmarks_tensor, recover_landmarks_parameters_tensor = generate_prior_fish_4_Landmarks(anc_key_pts_orientated_a_cell, original_img_size, feature_map_size_and_scale)
    save_dict = {'prior_landmarks':prior_landmarks_tensor,'recover_landmarks_parameters': recover_landmarks_parameters_tensor}
    save_prior_fish_4_landmarks(save_file_name, save_dict)
    load_dict = load_prior_fish_4_landmarks(save_file_name)
    pass

