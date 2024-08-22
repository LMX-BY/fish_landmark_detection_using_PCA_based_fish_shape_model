## Introduction
This project implements a fish landmark detection network with PCA-based shape model, whose landmark regression heads output the coefficients of the principal components, thus ensuring that the output landmarks conform to a valid fish shape.
## Data preparation
Download images and annotation files from [TIANCHI](https://tianchi.aliyun.com/dataset/184944).
## Training
1、Set path parameters related to dataset path(class DataPath).  
2、Set path parameters related to fish shape model(class SVDRelatedDataPath). The fish shape model can be found in the folder 'hu_s_and_c'.  
3、Set path paramters reltaed to output results(class TrainResultsSavePath).  
4、Run train_retinafacenet_fish_landmark_svd_based_error_coco_debug.py script.  
## Results Visulization
![](result_samples_display/ch05_20190703215846_117_train_0_batch_index_84.bmp)
![](result_samples_display/ch05_20190703215846_1201_train_0_batch_index_25.bmp)
