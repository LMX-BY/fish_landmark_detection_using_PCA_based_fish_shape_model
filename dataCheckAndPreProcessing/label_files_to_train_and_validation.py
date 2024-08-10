import pathlib
from utiles import utiles_files
import random
import numpy as np
import shutil
label_file_path_str = ('H:/code/python/IRFishDetection2.0.0/dataset2.2/poly_and_used_defined_label/all_checked_label'
                       '/user_defined_json_1.0_2.0_hu_s_and_c/train')
results_train_files = ('H:/code/python/IRFishDetection2.0.0/dataset2.2/poly_and_used_defined_label/all_checked_label'
                       '/user_defined_json_1.0_2.0_hu_s_and_c/train_1')
results_validation_files = ('H:/code/python/IRFishDetection2.0.0/dataset2.2/poly_and_used_defined_label'
                            '/all_checked_label/user_defined_json_1.0_2.0_hu_s_and_c/validation')
train_rate = 0.9
if __name__ == '__main__':
    label_file_path = pathlib.Path(label_file_path_str)
    all_file_names = utiles_files.get_filenames_of_path(label_file_path)
    random.shuffle(all_file_names)
    all_file_size = len(all_file_names)
    train_size = int(np.ceil(all_file_size * train_rate))
    train_label_files = all_file_names[0:train_size]
    valid_label_files = all_file_names[train_size:all_file_size]
    for a_files in train_label_files:
        file_name = a_files.name
        shutil.copy(str(a_files), results_train_files + '/' + file_name)
    for a_files in valid_label_files:
        file_name = a_files.name
        shutil.copy(str(a_files), results_validation_files + '/' + file_name)
