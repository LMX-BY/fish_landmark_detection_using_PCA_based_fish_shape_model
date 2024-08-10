from torch.utils.data import Dataset
import pathlib
from typing import Dict, List
import cv2 as cv
from torchvision import transforms
from utiles import utils_fish_landmark_detection
class IRFishDataSetFromLabelmeJson(Dataset):
    def __init__(
        self,
        img_path: pathlib.Path,
        label_files: List[pathlib.Path],
        img_name_as_key:bool,
        transforms: transforms = None,
        mapping: Dict = None):
            self.img_path = img_path
            self.label_files = label_files
            self.img_name_as_key = img_name_as_key
            self.transforms = transforms
            self.mapping = mapping
            if img_name_as_key:
                self.file_name_list = utils_fish_landmark_detection.get_filenames_of_path(self.img_path)
            else:
                self.file_name_list = utils_fish_landmark_detection.get_filenames_of_path(self.label_path)

    def __len__(self):
        return len(self.file_name_list)

    def __getitem__(self, item:int):
        a_file_name = self.file_name_list[item]
        '文件名关联'
        if self.img_name_as_key:
            img_file_name = a_file_name
            label_file_name = utils_fish_landmark_detection.img_to_json_suffix(img_file_name, self.label_path)
        else:
            label_file_name = a_file_name
            img_file_name = utils_fish_landmark_detection.json_to_bmp_suffix(label_file_name,self.img_path)
        img = cv.imread(str(img_file_name))

        if self.transforms is not None:
            img = self.transforms(img)
        labelDict = utils_fish_landmark_detection.read_json(label_file_name)
        return {
            'img':img,
            'labelDict':labelDict['shapes'],
            'img_file_name':img_file_name,
            'label_file_name':label_file_name
        }

# 'KeyPointAndObjectDataSet Test'
# img_file_path = pathlib.Path('D:/code/python/AboveWaterIRFishDetection/data/experimentData1.0.0/train')
# label_file_path = pathlib.Path('D:/code/python/AboveWaterIRFishDetection/data/experimentData1.0.0/train_label')
# img_file_names = utils.get_filenames_of_path(img_file_path)
#
#
# testDataSet = KeyPointAndObjectDataSet(img_file_names,label_file_path)
# a_label_content = testDataSet[0]
# '显示图像'
# cv.imshow('test0',a_label_content['img'])
# '显示标记'
# for a_label in a_label_content['labelDict']:
#     a_label_name = a_label['label']
#     if a_label_name == 'fish':
#         continue
#     if a_label_name == 'fish_pe'
#     print(a_label)
#
# cv.waitKey(0)
#
# print(a_label['labelDict'])