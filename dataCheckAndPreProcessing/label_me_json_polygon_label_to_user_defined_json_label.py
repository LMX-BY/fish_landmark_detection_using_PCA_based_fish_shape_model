import pathlib
import torch
from utiles import utils_fish_landmark_detection
import cv2
import numpy as np
import json
import shutil
from utiles import displayTool

color_green = (0, 255, 0)
color_red = (255, 0, 0)
color_blue = (0, 0, 255)
color_white = (255, 255, 255)
color_yellow = (255, 255, 0)
def label_me_json_polygon_label_to_user_defined_json_label(label_me_json_files, results_save_path, invalid_file_copy_path = None, display_path = None):
    invaild_file_count = 0
    for a_label_me_json_file in label_me_json_files:
        has_confused = False
        results_dict = {}
        json_content = utils_fish_landmark_detection.read_json(a_label_me_json_file)
        label_content = json_content['shapes']
        results_dict['img_name'] = '_'.join(a_label_me_json_file.stem.split('_')[:-1])
        # if results_dict['img_name'] == 'ch05_20190703215846_115':
        #     pass
        results_dict['img_size'] = [json_content['imageHeight'],json_content['imageWidth']]
        bbox_list = []
        landmark_list = []
        for a_label_content in label_content:
            label_name = a_label_content['label']
            points = a_label_content['points']
            if label_name == 'fish':
                bbox_list.append(points)
            if label_name == 'fish_landmarks':
                landmark_list.append(points)

        size_bbox = len(bbox_list)
        size_landmark = len(landmark_list)
        matched_label_list = []
        bbox_tensor = torch.Tensor(bbox_list).reshape(-1, 4)
        try:
            test = np.array(landmark_list)
        except:
            pass
        #print(f'test size: {test.shape}')
        try:
            a_landmark_set_tensor = torch.from_numpy(np.array(landmark_list))
        except:
            print(f'exception')
        if size_landmark != 0:
            landmark_bbox_xyxy_tensor = torch.ones([size_landmark,4])
            for index, a_landmark_set in enumerate(landmark_list):
                a_landmark_set_np = np.array(a_landmark_set)
                test = np.int32([a_landmark_set_np])
                landmark_bbox_cwh = np.array(cv2.boundingRect(np.int32([a_landmark_set_np])))
                landmark_bbox_cwh[2:4] = landmark_bbox_cwh[0:2] + landmark_bbox_cwh[2:4]
                landmark_bbox_xyxy_tensor[index] = torch.from_numpy(landmark_bbox_cwh)
            if display_path is not  None:
                img_path = display_path[0]
                landmark_tensor_reshaped = a_landmark_set_tensor.reshape(-1,8)
                combined_bbox_and_landmark_tensor = torch.cat((landmark_bbox_xyxy_tensor,landmark_tensor_reshaped),1)
                img_display = cv2.imread(f'{str(img_path)}/{results_dict["img_name"]}.bmp')
                displayTool.display_bbox(bbox_tensor[0].unsqueeze(0), img_display, color_green)
                displayTool.display_bbox_landmarks_with_mark_num_12(combined_bbox_and_landmark_tensor, img_display, color_blue, color_red, color_yellow)
                cv2.imwrite(f'{display_path[1]}/{results_dict["img_name"]}_test.bmp',img_display)
            #计算landmarks 的包围框
            match_count = 0
            match_matrix_bbox_to_landmark = torch.zeros((size_bbox,size_landmark))
            for bbox_index, a_bbox_tensor in enumerate(bbox_tensor):
                maxX = max(a_bbox_tensor[0:4:2])
                maxY = max(a_bbox_tensor[1:5:2])
                minX = min(a_bbox_tensor[0:4:2])
                minY = min(a_bbox_tensor[1:5:2])
                a_bbox_tensor = torch.Tensor([minX,minY,maxX,maxY])
                bbox_tensor[bbox_index] = a_bbox_tensor
                score_bbox_to_landmark = utils_fish_landmark_detection.jaccard(a_bbox_tensor.unsqueeze(0), landmark_bbox_xyxy_tensor)
                match_matrix_bbox_to_landmark[bbox_index] = score_bbox_to_landmark
            max_score_value, max_score_index = match_matrix_bbox_to_landmark.max(0)
            bbox_match_count_record = torch.zeros(size_bbox).int()
            bbox_matched_landmark_index = torch.ones(size_bbox)*-1
            bbox_matched_landmark_index = bbox_matched_landmark_index.int()
            for landmark_index, a_matched_bbox_index in enumerate(max_score_index):
                bbox_matched_landmark_index[a_matched_bbox_index] = landmark_index
                bbox_match_count_record[a_matched_bbox_index] = bbox_match_count_record[a_matched_bbox_index] + 1
            for bbox_index, matched_landmark_index in enumerate(bbox_matched_landmark_index):
                if matched_landmark_index == -1 or bbox_match_count_record[bbox_index] > 1:
                    a_match_label = torch.ones(16) * 0
                    a_match_label[0:4] = bbox_tensor[bbox_index]
                    # a_match_label[6] = 0
                    # a_match_label[9] = 0
                    # a_match_label[12] = 0
                    # a_match_label[15] = 0
                    a_match_label_list = a_match_label.tolist()
                    matched_label_list.append(a_match_label_list)
                else:
                    match_count = match_count + 1
                    a_match_label = torch.ones(16)
                    a_match_label[0:4] = bbox_tensor[bbox_index]
                    a_match_label[4:6] = a_landmark_set_tensor[matched_landmark_index][0]
                    a_match_label[7:9] = a_landmark_set_tensor[matched_landmark_index][1]
                    a_match_label[10:12] = a_landmark_set_tensor[matched_landmark_index][2]
                    a_match_label[13:15] = a_landmark_set_tensor[matched_landmark_index][3]
                    a_match_label_list = a_match_label.tolist()
                    matched_label_list.append(a_match_label_list)
        else:
            for bbox_index in range(size_bbox):
                a_match_label = torch.ones(16) * 0
                a_match_label[0:4] = bbox_tensor[bbox_index]
                a_match_label_list = a_match_label.tolist()
                matched_label_list.append(a_match_label_list)

            # sorted_score, sorted_score_index = score_bbox_to_landmark.sort(descending=True)
            # sorted_score = sorted_score.squeeze()
            # sorted_score_index = sorted_score_index.squeeze()
            #
            # if sorted_score.dim() ==0 and sorted_score.item() == 0:
            #     continue
            # # print(f'sorted_score length is {sorted_score.shape[0]}')
            # # if sorted_score.shape[0] == 0:
            # #     continue
            # len_sorted_score = sorted_score.size()
            # if sorted_score.dim() != 0:
            #     #print(f'len_sorted_score is {len_sorted_score}')
            #     if sorted_score.size(0)>1:
            #         if sorted_score[1] > 0.5 and (sorted_score[0] - sorted_score[1]) < 0.2:
            #             has_confused = True
            #     else:
            #         has_confused = False
            # else:
            #     sorted_score = sorted_score.unsqueeze(0)
            #     sorted_score_index = sorted_score_index.unsqueeze(0)
            # if has_confused is True:
            #     print(f'confuse occur!')
            #     break
            # if sorted_score[0] < 0.1:
            #     #没有匹配
            #     a_match_label = torch.ones(16) * -1
            #     a_match_label[0:4] = a_bbox_tensor
            #     a_match_label_list = a_match_label.tolist()
            #     matched_label_list.append(a_match_label_list)
            # else:
            #     match_count = match_count + 1
            #     a_match_label = torch.ones(16)
            #     a_matched_index = sorted_score_index[0].item()
            #     a_match_label[0:4] = a_bbox_tensor
            #     a_match_label[4:6] = a_landmark_set_tensor[a_matched_index][0]
            #     a_match_label[7:9] = a_landmark_set_tensor[a_matched_index][1]
            #     a_match_label[10:12] = a_landmark_set_tensor[a_matched_index][2]
            #     a_match_label[13:15] = a_landmark_set_tensor[a_matched_index][3]
            #     a_match_label_list = a_match_label.tolist()
            #     matched_label_list.append(a_match_label_list)

        matched_label_size = len(matched_label_list)
        print(f'{a_label_me_json_file.stem}: size_landmark :{size_landmark}, match_count: {match_count}')
        if size_landmark != match_count:
            has_confused = True
        results_dict['points'] = matched_label_list
        results_file_name = f'{str(results_save_path)}/{results_dict["img_name"]}_landmarks.json'
        results_file = open(str(results_file_name), 'w')
        results_file.write(json.dumps(results_dict))
        results_file.close()
        if has_confused:
            invaild_file_count = invaild_file_count + 1
            if invalid_file_copy_path is not None:
                shutil.copy(str(a_label_me_json_file), str(invalid_file_copy_path / str(a_label_me_json_file.name)))
    print(f'invalid file count: {invaild_file_count}')





if __name__ == '__main__':
    # label_me_json_file_path = pathlib.Path('H:/code/python/IRFishDetection2.0.0/dataset2.1'
    #                                        '/poly_and_used_defined_label/all_checked_label/label_me_polygon_1.0_2.0')
    # results_save_path = pathlib.Path('H:/code/python/IRFishDetection2.0.0/dataset2.1/poly_and_used_defined_label'
    #                                  '/all_checked_label/user_defined_json_1.0_2.0')
    # invalid_file_copy_path = pathlib.Path('H:/code/python/IRFishDetection2.0.0/dataset2.1/poly_and_used_defined_label'
    #                                  '/all_checked_label/invalid_file_1.0_2.0')
    label_me_json_file_path = pathlib.Path('H:/code/python/IRFishDetection2.0.0/dataset2.2/anji_process/'
                                           'polygon_label1.0_hudingpeng_supplemented/'
                                           'lableme_polygon_label_1.0_filterd_small_fish_hu_supplemented_and_display_checked')
    results_save_path = pathlib.Path('H:/code/python/IRFishDetection2.0.0/dataset2.2/anji_process/polygon_label1.0_hudingpeng_supplemented/'
                                     'lableme_polygon_label_1.0_filterd_small_fish_hu_supplemented_and_display_checked'
                                     '/user_defined_label')
    invalid_file_copy_path = None
    invalid_file_copy_path = pathlib.Path('H:/code/python/IRFishDetection2.0.0/dataset2.2/anji_process'
                                           '/polygon_label1.0_hudingpeng_supplemented/'
                                          'lableme_polygon_label_1.0_filterd_small_fish_hu_supplemented_and_display_checked/invalid_user_defined_label')
    # display_test_save_path = [pathlib.Path('H:/code/python/IRFishDetection2.0.0/dataset2.1/imgs')
    #                           ,pathlib.Path('H:/code/python/IRFishDetection2.0.0/dataset2.1/anji_process/polygon_label'
    #                                       '/train_label_simple_checked_combined/display_test')]
    display_test_save_path = None
    label_me_json_files = utils_fish_landmark_detection.get_filenames_of_path(label_me_json_file_path)
    label_me_json_polygon_label_to_user_defined_json_label(label_me_json_files, results_save_path, invalid_file_copy_path, display_test_save_path)
