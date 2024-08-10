import pathlib
from utiles import utils_fish_landmark_detection
from utiles import displayTool
from utiles import utiles_files
import cv2
import numpy as np
color_green = (0, 255, 0)
color_red = (255, 0, 0)
color_blue = (0, 0, 255)
color_white = (255, 255, 255)
color_yellow = (255, 255, 0)
color_orange = (255, 140, 0)
color_gold = (255, 215, 0)
path_1_2_zhidan_relabeled_user_defined_label = ('H:/code/python/IRFishDetection2.0.0/dataset2.1/poly_and_used_defined_label/all_checked_label'
            '/user_defined_json_1.0_2.0')
path_1_xiaohu_relabled_user_defined_label = ('H:/code/python/IRFishDetection2.0.0/dataset2.2'
                                             '/poly_and_used_defined_label/all_checked_label/user_defined_json_1.0_2'
                                             '.0_hu_s_and_c')

results_save_path_1_2_zhidan_relabeled_user_defined_label = ('H:/code/python/IRFishDetection2.0.0/dataset2.1/poly_and_used_defined_label'
                                         '/all_checked_label/user_defined_json_1.0_2.0/display')
results_save_path_1_xiaohu_relabled_user_defined_label = ('H:/code/python/IRFishDetection2.0.0/dataset2.2'
                                                          '/poly_and_used_defined_label/all_checked_label/user_defined_json_1.0_2.0_hu_s_and_c/display')
results_save_path_1_display_for_paper = ('H:/code/python/IRFishDetection2.0.0/results/label_display_for_paper')
if __name__ == '__main__':
    img_path = pathlib.Path('H:/code/python/IRFishDetection2.0.0/dataset2.2/img')
    landmark_json_file_path = pathlib.Path(path_1_xiaohu_relabled_user_defined_label)
    landmark_json_files = utiles_files.get_filenames_of_path(landmark_json_file_path)
    display_img_save_path = pathlib.Path(results_save_path_1_display_for_paper)

    for a_landmark_json_file in landmark_json_files:
        json_content = utiles_files.read_json(a_landmark_json_file)
        img_file_name = f'{str(img_path)}/{json_content["img_name"]}.bmp'
        img_cv = cv2.imread(img_file_name)
        bbox_and_landmarks = json_content['points']
        bbox_and_landmarks_np = np.array(bbox_and_landmarks)
        displayTool.display_bbox_landmarks_with_mark_num_16(bbox_and_landmarks_np, img_cv, color_green, color_red, color_red)
        display_img = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        save_img_name = f'{str(display_img_save_path)}/{a_landmark_json_file.stem}_display.bmp'
        cv2.imwrite(save_img_name, display_img)


