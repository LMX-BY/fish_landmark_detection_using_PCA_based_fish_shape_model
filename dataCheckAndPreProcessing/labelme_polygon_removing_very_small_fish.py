import pathlib
import torch
from utiles import utils_fish_landmark_detection
import cv2
import numpy as np
import json
import shutil
from utiles import displayTool


area_threshold_for_bbox = 3000
area_threshold_for_landmarks = 1800
def label_me_json_polygon_removing_very_small_fish(label_me_json_files, results_save_path):
    filtered_fish_count = 0
    filterd_fish_landmarks_count = 0
    for a_label_me_json_file in label_me_json_files:
        has_confused = False
        results_dict = {}
        json_content = utils_fish_landmark_detection.read_json(a_label_me_json_file)
        label_content = json_content['shapes']
        small_fish_filterd_label_content = []
        results_dict['img_name'] = '_'.join(a_label_me_json_file.stem.split('_')[:-1])
        # if results_dict['img_name'] == 'ch05_20190703215846_115':
        #     pass
        results_dict['img_size'] = [json_content['imageHeight'],json_content['imageWidth']]
        for a_label_content in label_content:
            label_name = a_label_content['label']
            points = a_label_content['points']
            if label_name == 'fish':
                bbox_length = abs(points[0][0]-points[1][0])
                bbox_height = abs(points[0][1]-points[1][1])
                areas = bbox_length * bbox_height
                if areas > area_threshold_for_bbox:
                    small_fish_filterd_label_content.append(a_label_content)
                else:
                    filtered_fish_count = filtered_fish_count + 1
            if label_name == 'fish_landmarks':
                fish_landmarks_np = np.array(points)
                fish_landmarks_maxX = np.max(fish_landmarks_np[:,0])
                fish_landmarks_minX = np.min(fish_landmarks_np[:,0])
                fish_landmarks_maxY = np.max(fish_landmarks_np[:,1])
                fish_landmarks_minY = np.min(fish_landmarks_np[:,1])
                bbox_length = fish_landmarks_maxX - fish_landmarks_minX
                bbox_height = fish_landmarks_maxY - fish_landmarks_minY
                assert bbox_length > 0
                assert bbox_height > 0
                areas = bbox_length * bbox_height
                if areas > area_threshold_for_landmarks:
                    small_fish_filterd_label_content.append(a_label_content)
                else:
                    filterd_fish_landmarks_count = filterd_fish_landmarks_count + 1
        json_content['shapes'] = small_fish_filterd_label_content
        filterd_polygon_label_json_file_name =  results_save_path / (
                a_label_me_json_file.stem  + '.json')
        filterd_polygon_label_json_file = open(str(filterd_polygon_label_json_file_name), 'w')
        filterd_polygon_label_json_file.write(json.dumps(json_content))
        filterd_polygon_label_json_file.close()
    print(f'filtered_fish_count is {filtered_fish_count}')
    print(f'filterd_fish_landmarks_count is {filterd_fish_landmarks_count}')





if __name__ == '__main__':
    # label_me_json_file_path = pathlib.Path('H:/code/python/IRFishDetection2.0.0/dataset2.1'
    #                                        '/poly_and_used_defined_label/all_checked_label/label_me_polygon_1.0_2.0')
    # results_save_path = pathlib.Path('H:/code/python/IRFishDetection2.0.0/dataset2.1/poly_and_used_defined_label'
    #                                  '/all_checked_label/user_defined_json_1.0_2.0')
    # invalid_file_copy_path = pathlib.Path('H:/code/python/IRFishDetection2.0.0/dataset2.1/poly_and_used_defined_label'
    #                                  '/all_checked_label/invalid_file_1.0_2.0')
    label_me_json_file_path = pathlib.Path('H:/code/python/IRFishDetection2.0.0/dataset2.2/anji_process'
                                           '/polygon_label1.0_hudingpeng_supplemented')
    results_save_path = pathlib.Path('H:/code/python/IRFishDetection2.0.0/dataset2.2/anji_process/polygon_label1.0'
                                     '_hudingpeng_supplemented/filterd_small_fish_labelme_polygon')

    # display_test_save_path = [pathlib.Path('H:/code/python/IRFishDetection2.0.0/dataset2.1/imgs')
    #                           ,pathlib.Path('H:/code/python/IRFishDetection2.0.0/dataset2.1/anji_process/polygon_label'
    #                                       '/train_label_simple_checked_combined/display_test')]
    display_test_save_path = None
    label_me_json_files = utils_fish_landmark_detection.get_filenames_of_path(label_me_json_file_path)
    label_me_json_polygon_removing_very_small_fish(label_me_json_files, results_save_path)
