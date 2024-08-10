import pathlib
import random
import shutil
from utiles import utils_fish_landmark_detection
if __name__ == '__main__':
    label_file_path = pathlib.Path('H:/code/python/IRFishDetection2.0.0/dataset2.1/poly_and_used_defined_label/all_checked_label/user_defined_json_1.0_2.0')
    train_file_path = pathlib.Path('H:/code/python/IRFishDetection2.0.0/dataset2.1/poly_and_used_defined_label/all_checked_label/user_defined_json_1.0_2.0/train')
    test_file_path = pathlib.Path('H:/code/python/IRFishDetection2.0.0/dataset2.1/poly_and_used_defined_label/all_checked_label/user_defined_json_1.0_2.0/test')
    train_test_rate = 3
    label_files = utils_fish_landmark_detection.get_filenames_of_path(label_file_path)
    decision_value = 10/(train_test_rate+1)
    train_file_size = 0
    test_file_size = 0
    for a_file in label_files:
        random_v = random.randint(1,10)
        file_name = a_file.name
        if random_v < decision_value:
            #test label
            shutil.copy(str(a_file),
                        f'{str(test_file_path)}/{file_name}')
            test_file_size = test_file_size + 1
        else:
            #train label
            shutil.copy(str(a_file),
                        f'{str(train_file_path)}/{file_name}')
            train_file_size = train_file_size + 1
