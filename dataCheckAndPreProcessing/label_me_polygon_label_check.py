import pathlib
from utiles import utils_fish_landmark_detection
import datetime
import shutil


# 针对水面红外多点标注错误检测函数
def labelme_fish_polygon_label_check(labelme_json_files, error_log_file, error_file_save_path=None,
                                valid_file_save_path=None):
    valid_label_name = {'fish': 'rectangle', 'fish_landmarks': 'polygon'}
    total_error_count = 0
    error_file_name = []
    valid_file_name = []
    for a_label_file_name in labelme_json_files:
        a_label_content = utils_fish_landmark_detection.read_json(a_label_file_name)
        a_label_shapes = a_label_content['shapes']
        label_count_dict = {'fish': 0, 'fish_landmarks': 0}
        has_error = False
        '检查标记名称和类型是否合法'
        for a_a_lable_shapes in a_label_shapes:
            label_name = a_a_lable_shapes['label']
            label_type = a_a_lable_shapes['shape_type']
            if label_name not in valid_label_name:
                '标签名不合法'
                total_error_count = total_error_count + 1
                error_log_file.write(
                    f'label name error at {str(a_label_file_name)}: label_name is {label_name},label_type is {label_type}' + '\n')
                print(
                    f'label name error at {str(a_label_file_name)}: label_name is {label_name},label_type is {label_type}')
                has_error = True
                continue
            if valid_label_name[label_name] != label_type:
                '标签名和标签类型不匹配'
                total_error_count = total_error_count + 1
                error_log_file.write(
                    f'label type error at {str(a_label_file_name)}: label_name is {label_name},label_type is {label_type}' + '\n')
                print(
                    f'label type error at {str(a_label_file_name)}: label_name is {label_name},label_type is {label_type}')
                has_error = True
                continue
            label_count_dict[label_name] = label_count_dict[label_name] + 1
        '检查标记数量是否合法，一条鱼对应四个点'
        fish_count = label_count_dict['fish']
        fish_landmarks_count = label_count_dict['fish_landmarks']
        if fish_landmarks_count > fish_count:
            total_error_count = total_error_count + 1
            has_error = True
            error_log_file.write(f'label count constrain error at {str(a_label_file_name)}: \
                             fish is {fish_count},fish landmarks is {fish_landmarks_count}' + '\n')
            print(f'label count constrain error at {str(a_label_file_name)}: fish is {fish_count}, fish landmarks is {fish_landmarks_count}')
        if has_error:
            error_file_name.append(a_label_file_name)
        else:
            valid_file_name.append(a_label_file_name)

    if error_file_save_path is not None:
        for a_error_file_name in error_file_name:
            shutil.copy(str(a_error_file_name), str(error_file_save_path / str(a_error_file_name.name)))
    if valid_file_save_path is not None:
        for a_valid_file_name in valid_file_name:
            shutil.copy(str(a_valid_file_name), str(valid_file_save_path / str(a_valid_file_name.name)))

    return total_error_count


if __name__ == '__main__':
    json_path1 = pathlib.Path('H:/code/python/IRFishDetection2.0.0/dataset2.2/anji_process/polygon_label1.0（hudingpeng_supplemented）')
    # json_path2 = pathlib.Path('H:/code/python/IRFishDetection2.0.0/dataset2.1/lablemeJson/test_label')
    # json_path3 = pathlib.Path('H:/code/python/IRFishDetection2.0.0/dataset2.1/lablemeJson/train_label')
    json_files = utils_fish_landmark_detection.get_filenames_of_path(json_path1)
    # json_files = json_files + utils_fish_landmark_detection.get_filenames_of_path(json_path2)
    # json_files = json_files + utils_fish_landmark_detection.get_filenames_of_path(json_path3)

    # json_path = pathlib.Path('G:/dataSet/aboveWaterIR/data-for-experiments/experimentData1.0.0/train_label_20230522')
    # json_files = utils_fish_landmark_detection.get_filenames_of_path(json_path)

    error_log_file_name = pathlib.Path(
        str(json_path1 / 'error_files') +'/error_log_' + datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + '.txt')
    error_log_file = open(str(error_log_file_name), 'w')

    error_file_save_path = None
    # valid_file_save_path = pathlib.Path('H:/code/python/IRFishDetection2.0.0/dataset2.1/tempErrorCheckedLabelmeFile')
    valid_file_save_path = None

    # error_file_save_path = None
    # valid_file_save_path = None

    total_error_count = labelme_fish_polygon_label_check(json_files, error_log_file, error_file_save_path,
                                                    valid_file_save_path)
    print(f'total error count is {total_error_count}')
    error_log_file.close()
