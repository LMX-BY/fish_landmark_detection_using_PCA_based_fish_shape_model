import utiles.utiles_parameters as UP

# path
data_path_hu_s_and_c_v1 = UP.DataPath()
data_path_hu_s_and_c_v1.train_label_path_str = ('/root/LG_WS/IRFishDetection2.0.0/dataset2.2/'
                                                'poly_and_used_defined_label/'
                                                'all_checked_label/user_defined_json_1.0_2.0_hu_s_and_c/train_1')
data_path_hu_s_and_c_v1.validation_label_path_str = ('/root/LG_WS/IRFishDetection2.0.0/dataset2.2/'
                                                     'poly_and_used_defined_label/'
                                                     'all_checked_label/user_defined_json_1.0_2.0_hu_s_and_c/validation')
data_path_hu_s_and_c_v1.test_label_path_str = ('/root/LG_WS/IRFishDetection2.0.0/dataset2.2/'
                                               'poly_and_used_defined_label/'
                                               'all_checked_label/user_defined_json_1.0_2.0_hu_s_and_c/test')
data_path_hu_s_and_c_v1.test_img_path_str = '/root/LG_WS/IRFishDetection2.0.0/dataset2.2/img'
data_path_hu_s_and_c_v1.train_img_path_str = '/root/LG_WS/IRFishDetection2.0.0/dataset2.2/img'
data_path_hu_s_and_c_v1.all_label_path_str = ('/root/LG_WS/IRFishDetection2.0.0/dataset2.2'
                                              '/poly_and_used_defined_label/all_checked_label/user_defined_json_1.0_2'
                                              '.0_hu_s_and_c')

data_path_small_test_v1 = UP.DataPath()
data_path_small_test_v1.train_label_path_str = ('/root/LG_WS/IRFishDetection2.0.0/dataset2.2/'
                                                'poly_and_used_defined_label/'
                                                'all_checked_label/small_set_for_test/train')
data_path_small_test_v1.test_label_path_str = ('/root/LG_WS/IRFishDetection2.0.0/dataset2.2/'
                                               'poly_and_used_defined_label/'
                                               'all_checked_label/small_set_for_test/test')
data_path_small_test_v1.test_img_path_str = '/root/LG_WS/IRFishDetection2.0.0/dataset2.2/img'
data_path_small_test_v1.train_img_path_str = '/root/LG_WS/IRFishDetection2.0.0/dataset2.2/img'

results_path_bbox_reg_v1 = UP.TrainResultsSavePath()
results_path_bbox_reg_v1.display_results_test_save_path_str = ('/root/LG_WS/IRFishDetection2.0.0/results/bboxLoss'
                                                               '/test')
results_path_bbox_reg_v1.display_results_train_save_path_str = ('/root/LG_WS/IRFishDetection2.0.0/results/'
                                                                'bboxLoss/train')
results_path_bbox_reg_v1.display_results_validation_save_path_str = ('/root/LG_WS/IRFishDetection2.0.0/results'
                                                                     '/bboxLoss/valid')
results_path_bbox_reg_v1.display_results_pos_sample_save_path_str = ('/root/LG_WS/IRFishDetection2.0.0/results'
                                                                     '/bboxLoss/pos_sample')
results_path_bbox_reg_v1.ckp_save_path_str = '/root/LG_WS/IRFishDetection2.0.0/results/bboxLoss/ckp/train'
results_path_bbox_reg_v1.best_model_save_path = ('/root/LG_WS/IRFishDetection2.0.0/results/bboxLoss/ckp/train'
                                                 '/best_models')

results_path_bbox_reg_small_test_v1 = UP.TrainResultsSavePath()
results_path_bbox_reg_small_test_v1.display_results_test_save_path_str = ('/root/LG_WS/IRFishDetection2.0.0/results'
                                                                          '/bboxLoss/small_test')
results_path_bbox_reg_small_test_v1.display_results_train_save_path_str = ('/root/LG_WS/IRFishDetection2.0.0'
                                                                           '/results/bboxLoss/small_train')
results_path_bbox_reg_small_test_v1.display_results_validation_save_path_str = ('/root/LG_WS/IRFishDetection2.0.0'
                                                                                '/results/bboxLoss/small_valid')
results_path_bbox_reg_small_test_v1.display_results_pos_sample_save_path_str = ('/root/LG_WS/IRFishDetection2.0.0'
                                                                                '/results/bboxLoss/small_pos_sample')
results_path_bbox_reg_small_test_v1.ckp_save_path_str = ('/root/LG_WS/IRFishDetection2.0.0/results/bboxLoss/ckp'
                                                         '/small_test')
results_path_bbox_reg_small_test_v1.best_model_save_path = ('/root/LG_WS/IRFishDetection2.0.0/results/bboxLoss/ckp'
                                                            '/small_test/best_models')

results_path_anchor_landmarks_small_test_v1 = UP.TrainResultsSavePath()
results_path_anchor_landmarks_small_test_v1.display_results_test_save_path_str = ('/root/LG_WS/IRFishDetection2.0.0'
                                                                                  '/results/anchorLandmarksLoss'
                                                                                  '/small_test')
results_path_anchor_landmarks_small_test_v1.display_results_train_save_path_str = ('/root/LG_WS/IRFishDetection2.0.0'
                                                                                   '/results/anchorLandmarksLoss'
                                                                                   '/small_train')
results_path_anchor_landmarks_small_test_v1.display_results_validation_save_path_str = (
    '/root/LG_WS/IRFishDetection2.0'
    '.0/results/anchorLandmarksLoss'
    '/small_validation')
results_path_anchor_landmarks_small_test_v1.display_results_pos_sample_save_path_str = (
    '/root/LG_WS/IRFishDetection2.0'
    '.0/results/anchorLandmarksLoss'
    '/small_pos_landmarks')
results_path_anchor_landmarks_small_test_v1.ckp_save_path_str = (
    '/root/LG_WS/IRFishDetection2.0.0/results/anchorLandmarksLoss'
    '/ckp/small_test')
results_path_anchor_landmarks_small_test_v1.best_model_save_path = ''

results_path_anchor_landmarks_v1 = UP.TrainResultsSavePath()
results_path_anchor_landmarks_v1.display_results_test_save_path_str = ('/root/LG_WS/IRFishDetection2.0.0'
                                                                       '/results/anchorLandmarksLoss/test')
results_path_anchor_landmarks_v1.display_results_train_save_path_str = ('/root/LG_WS/IRFishDetection2.0.0'
                                                                        '/results/anchorLandmarksLoss/train')
results_path_anchor_landmarks_v1.display_results_validation_save_path_str = ('/root/LG_WS/IRFishDetection2.0'
                                                                             '.0/results/anchorLandmarksLoss'
                                                                             '/validation')
results_path_anchor_landmarks_v1.display_results_pos_sample_save_path_str = ('/root/LG_WS/IRFishDetection2.0'
                                                                             '.0/results/anchorLandmarksLoss'
                                                                             '/pos_landmarks')
results_path_anchor_landmarks_v1.ckp_save_path_str = ('/root/LG_WS/IRFishDetection2.0.0/results/anchorLandmarksLoss'
                                                      '/ckp/train')
# results_path_anchor_landmarks_v1.best_model_save_path = ('/root/LG_WS/IRFishDetection2.0.0/results/svdEncodedLoss'
#                                                          '/ckp/small_test/best_models')

results_path_svd_encoded_small_test_v1 = UP.TrainResultsSavePath()
results_path_svd_encoded_small_test_v1.display_results_test_save_path_str = ('/root/LG_WS/IRFishDetection2.0.0'
                                                                             '/results/svdEncodedLoss/small_test')
results_path_svd_encoded_small_test_v1.display_results_train_save_path_str = ('/root/LG_WS/IRFishDetection2.0.0'
                                                                              '/results/svdEncodedLoss/small_train')
results_path_svd_encoded_small_test_v1.display_results_validation_save_path_str = ('/root/LG_WS/IRFishDetection2.0'
                                                                                   '.0/results/svdEncodedLoss'
                                                                                   '/small_validation')
results_path_svd_encoded_small_test_v1.display_results_pos_sample_save_path_str = ('/root/LG_WS/IRFishDetection2.0'
                                                                                   '.0/results/svdEncodedLoss'
                                                                                   '/small_pos_landmarks')
results_path_svd_encoded_small_test_v1.ckp_save_path_str = ('/root/LG_WS/IRFishDetection2.0.0/results/svdEncodedLoss'
                                                            '/ckp/small_test')

results_path_svd_encoded_small_test_v1.best_model_save_path = ('/root/LG_WS/IRFishDetection2.0.0/results'
                                                               '/svdEncodedLoss/ckp/small_test/best_models')

results_path_svd_encoded_v1 = UP.TrainResultsSavePath()
results_path_svd_encoded_v1.display_results_test_save_path_str = ('/root/LG_WS/IRFishDetection2.0.0'
                                                                  '/results/svdEncodedLoss/test')
results_path_svd_encoded_v1.display_results_train_save_path_str = ('/root/LG_WS/IRFishDetection2.0.0'
                                                                   '/results/svdEncodedLoss/train')
results_path_svd_encoded_v1.display_results_validation_save_path_str = ('/root/LG_WS/IRFishDetection2.0'
                                                                        '.0/results/svdEncodedLoss'
                                                                        '/validation')
results_path_svd_encoded_v1.display_results_pos_sample_save_path_str = ('/root/LG_WS/IRFishDetection2.0'
                                                                        '.0/results/svdEncodedLoss'
                                                                        '/pos_landmarks')
results_path_svd_encoded_v1.ckp_save_path_str = ('/root/LG_WS/IRFishDetection2.0.0/results/svdEncodedLoss'
                                                 '/ckp/train')

results_path_svd_encoded_v1.best_model_save_path = ('/root/LG_WS/IRFishDetection2.0.0/results/svdEncodedLoss/ckp'
                                                    '/train/best_models')

results_path_svd_encoded_no_display_v1 = UP.TrainResultsSavePath()
results_path_svd_encoded_no_display_v1.display_results_test_save_path_str = None
results_path_svd_encoded_no_display_v1.display_results_train_save_path_str = None
results_path_svd_encoded_no_display_v1.display_results_validation_save_path_str = None
results_path_svd_encoded_no_display_v1.display_results_pos_sample_save_path_str = None
results_path_svd_encoded_no_display_v1.ckp_save_path_str = ('/root/LG_WS/IRFishDetection2.0.0/results/svdEncodedLoss'
                                                 '/ckp/train')

results_path_svd_encoded_no_display_v1.best_model_save_path = ('/root/LG_WS/IRFishDetection2.0.0/results/svdEncodedLoss/ckp'
                                                    '/train/best_models')

other_results_VGG16_FPN2L_DR8_16_save_path_v1 = UP.OtherResultsSavePath()
other_results_VGG16_FPN2L_DR8_16_save_path_v1.unknown_mask_index_path_str = ('/root/LG_WS/IRFishDetection2.0.0'
                                                                             '/results/other_results'
                                                                             '/hu_s_and_c'
                                                                             '/unknown_mask/VGG16_FPN2_DR8_16_v1')
other_results_VGG16_FPN2L_DR8_16_save_path_v1.unknown_mask_display_path_str = ('/root/LG_WS/IRFishDetection2.0.0'
                                                                               '/results/other_results'
                                                                               '/hu_s_and_c/'
                                                                               'unknown_mask_display'
                                                                               '/VGG16_FPN2_DR8_16_v1')
other_results_VGG16_FPN2L_DR8_16_save_path_v1.svd_feature_path = ('/root/LG_WS/IRFishDetection2.0.0/results'
                                                                  '/other_results/hu_s_and_c'
                                                                  '/svd_params')
# svd related
data_path_svd_origin_VGG16_FPN2L_DR8_16_v1 = UP.SVDRelatedDataPath()
data_path_svd_origin_VGG16_FPN2L_DR8_16_v1.unknown_mask_path_str = ('/root/LG_WS/IRFishDetection2.0.0/results'
                                                                    '/other_results/hu_s_and_c/unknow_masks'
                                                                    '/VGG16_FPN2_DR8_16_v1')
data_path_svd_origin_VGG16_FPN2L_DR8_16_v1.svd_param_file_path_str = ('/root/LG_WS/IRFishDetection2.0.0/results'
                                                                      '/other_results/hu_s_and_c/svd_params'
                                                                      '/svd_features')
data_path_svd_origin_VGG16_FPN2L_DR8_16_v1.landmark_priors_file_path_str = ('/root/LG_WS/IRFishDetection2.0.0'
                                                                            '/results/other_results/hu_s_and_c'
                                                                            '/svd_landmark_priors'
                                                                            '/VGG16_FPN2L_DR8_16_svd_origin_landmark_priors_v1')

data_path_unscented_svd_augmented_1_1_2p5_large_size_VGG16_FPN2L_DR8_16_v1 = UP.SVDRelatedDataPath()
data_path_unscented_svd_augmented_1_1_2p5_large_size_VGG16_FPN2L_DR8_16_v1.unknown_mask_path_str = ('/root/LG_WS/IRFishDetection2.0.0/results'
                                                                                                    '/other_results/hu_s_and_c/unknow_masks'
                                                                                                    '/VGG16_FPN2_DR8_16_v1')
data_path_unscented_svd_augmented_1_1_2p5_large_size_VGG16_FPN2L_DR8_16_v1.svd_param_file_path_str = ('/root/LG_WS/IRFishDetection2.0.0/results'
                                                                                                      '/other_results/hu_s_and_c/augmented_svd_params'
                                                                                                      '/unscented_svd_augmented_1_1_2p5_large_size_v1'
                                                                                                      '/unscented_transformed_svd_param')
data_path_unscented_svd_augmented_1_1_2p5_large_size_VGG16_FPN2L_DR8_16_v1.landmark_priors_file_path_str = ('/root/LG_WS/IRFishDetection2.0.0'
                                                                                                            '/results/other_results/hu_s_and_c'
                                                                                                            '/svd_landmark_priors'
                                                                                                            '/VGG16_FPN2L_DR8_16_svd_origin_landmark_priors_v1')

data_path_unscented_svd_augmented_2p5_2p5_5_large_size_VGG16_FPN2L_DR8_16_v1 = UP.SVDRelatedDataPath()
data_path_unscented_svd_augmented_2p5_2p5_5_large_size_VGG16_FPN2L_DR8_16_v1.unknown_mask_path_str = ('/root/LG_WS/IRFishDetection2.0.0/results'
                                                                                                    '/other_results/hu_s_and_c/unknow_masks'
                                                                                                    '/VGG16_FPN2_DR8_16_v1')
data_path_unscented_svd_augmented_2p5_2p5_5_large_size_VGG16_FPN2L_DR8_16_v1.svd_param_file_path_str = ('/root/LG_WS/IRFishDetection2.0.0/results'
                                                                                                      '/other_results/hu_s_and_c/augmented_svd_params'
                                                                                                      '/unscented_svd_augmented_2p5_2p5_5_large_size_v1'
                                                                                                      '/unscented_transformed_svd_param')
data_path_unscented_svd_augmented_2p5_2p5_5_large_size_VGG16_FPN2L_DR8_16_v1.landmark_priors_file_path_str = ('/root/LG_WS/IRFishDetection2.0.0'
                                                                                                            '/results/other_results/hu_s_and_c'
                                                                                                            '/svd_landmark_priors'
                                                                                                            '/VGG16_FPN2L_DR8_16_svd_origin_landmark_priors_v1')

data_path_unscented_svd_augmented_5_5_10_large_size_VGG16_FPN2L_DR8_16_v1 = UP.SVDRelatedDataPath()
data_path_unscented_svd_augmented_5_5_10_large_size_VGG16_FPN2L_DR8_16_v1.unknown_mask_path_str =  ('/root/LG_WS/IRFishDetection2.0.0/results'
                                                                                                    '/other_results/hu_s_and_c/unknow_masks'
                                                                                                    '/VGG16_FPN2_DR8_16_v1')
data_path_unscented_svd_augmented_5_5_10_large_size_VGG16_FPN2L_DR8_16_v1.svd_param_file_path_str = ('/root/LG_WS/IRFishDetection2.0.0/results'
                                                                                                      '/other_results/hu_s_and_c/augmented_svd_params'
                                                                                                      '/unscented_svd_augmented_5_5_10_large_size_v1'
                                                                                                      '/unscented_transformed_svd_param')
data_path_unscented_svd_augmented_5_5_10_large_size_VGG16_FPN2L_DR8_16_v1.landmark_priors_file_path_str = ('/root/LG_WS/IRFishDetection2.0.0'
                                                                                                            '/results/other_results/hu_s_and_c'
                                                                                                            '/svd_landmark_priors'
                                                                                                            '/VGG16_FPN2L_DR8_16_svd_origin_landmark_priors_v1')

data_path_unscented_svd_augmented_1_1_2p5_large_size_VGG16_FPN2L_DR8_16_v2 = UP.SVDRelatedDataPath()
data_path_unscented_svd_augmented_1_1_2p5_large_size_VGG16_FPN2L_DR8_16_v2.unknown_mask_path_str = ('/root/LG_WS/IRFishDetection2.0.0/results'
                                                                                                    '/other_results/hu_s_and_c/unknow_masks'
                                                                                                    '/VGG16_FPN2_DR8_16_v1')
data_path_unscented_svd_augmented_1_1_2p5_large_size_VGG16_FPN2L_DR8_16_v2.svd_param_file_path_str = ('/root/LG_WS/IRFishDetection2.0.0/results'
                                                                                                      '/other_results/hu_s_and_c/augmented_svd_params'
                                                                                                      '/unscented_svd_augmented_1_1_2p5_large_size_v2'
                                                                                                      '/unscented_transformed_svd_param')
data_path_unscented_svd_augmented_1_1_2p5_large_size_VGG16_FPN2L_DR8_16_v2.landmark_priors_file_path_str = ('/root/LG_WS/IRFishDetection2.0.0'
                                                                                                            '/results/other_results/hu_s_and_c'
                                                                                                            '/svd_landmark_priors'
                                                                                                            '/VGG16_FPN2L_DR8_16_svd_origin_landmark_priors_v1')

data_path_unscented_svd_augmented_2p5_2p5_5_large_size_VGG16_FPN2L_DR8_16_v2 = UP.SVDRelatedDataPath()
data_path_unscented_svd_augmented_2p5_2p5_5_large_size_VGG16_FPN2L_DR8_16_v2.unknown_mask_path_str = ('/root/LG_WS/IRFishDetection2.0.0/results'
                                                                                                    '/other_results/hu_s_and_c/unknow_masks'
                                                                                                    '/VGG16_FPN2_DR8_16_v1')
data_path_unscented_svd_augmented_2p5_2p5_5_large_size_VGG16_FPN2L_DR8_16_v2.svd_param_file_path_str = ('/root/LG_WS/IRFishDetection2.0.0/results'
                                                                                                      '/other_results/hu_s_and_c/augmented_svd_params'
                                                                                                      '/unscented_svd_augmented_2p5_2p5_5_large_size_v2'
                                                                                                      '/unscented_transformed_svd_param')
data_path_unscented_svd_augmented_2p5_2p5_5_large_size_VGG16_FPN2L_DR8_16_v2.landmark_priors_file_path_str = ('/root/LG_WS/IRFishDetection2.0.0'
                                                                                                            '/results/other_results/hu_s_and_c'
                                                                                                            '/svd_landmark_priors'
                                                                                                            '/VGG16_FPN2L_DR8_16_svd_origin_landmark_priors_v1')

data_path_unscented_svd_augmented_5_5_10_large_size_VGG16_FPN2L_DR8_16_v2 = UP.SVDRelatedDataPath()
data_path_unscented_svd_augmented_5_5_10_large_size_VGG16_FPN2L_DR8_16_v2.unknown_mask_path_str =  ('/root/LG_WS/IRFishDetection2.0.0/results'
                                                                                                    '/other_results/hu_s_and_c/unknow_masks'
                                                                                                    '/VGG16_FPN2_DR8_16_v1')
data_path_unscented_svd_augmented_5_5_10_large_size_VGG16_FPN2L_DR8_16_v2.svd_param_file_path_str = ('/root/LG_WS/IRFishDetection2.0.0/results'
                                                                                                      '/other_results/hu_s_and_c/augmented_svd_params'
                                                                                                      '/unscented_svd_augmented_5_5_10_large_size_v2'
                                                                                                      '/unscented_transformed_svd_param')
data_path_unscented_svd_augmented_5_5_10_large_size_VGG16_FPN2L_DR8_16_v2.landmark_priors_file_path_str = ('/root/LG_WS/IRFishDetection2.0.0'
                                                                                                            '/results/other_results/hu_s_and_c'
                                                                                                            '/svd_landmark_priors'
                                                                                                            '/VGG16_FPN2L_DR8_16_svd_origin_landmark_priors_v1')



# train param
trainingParams_bboxloss_v1 = UP.TrainingParams()
trainingParams_bboxloss_v1.batch_size = 1
trainingParams_bboxloss_v1.num_workers = 1
trainingParams_bboxloss_v1.CPU = False
trainingParams_bboxloss_v1.lr = 0.001
trainingParams_bboxloss_v1.momentum = 0.1
trainingParams_bboxloss_v1.weight_decay = 5e-4
trainingParams_bboxloss_v1.factor = 0.75
trainingParams_bboxloss_v1.patience = 3
trainingParams_bboxloss_v1.min_lr = 0
trainingParams_bboxloss_v1.gamma = 0.1
trainingParams_bboxloss_v1.precision = 32  # 未用
trainingParams_bboxloss_v1.num_of_class = 2
trainingParams_bboxloss_v1.num_of_landmarks = 4
trainingParams_bboxloss_v1.seed = 42
trainingParams_bboxloss_v1.max_epochs = 350
trainingParams_bboxloss_v1.patience = 50
trainingParams_bboxloss_v1.img_height = int(1080)
trainingParams_bboxloss_v1.img_width = int(1920)
trainingParams_bboxloss_v1.train_or_test = "Train"
trainingParams_bboxloss_v1.conf_thresh_for_test = 0.51
trainingParams_bboxloss_v1.nms_thresh_for_test = 0.3
trainingParams_bboxloss_v1.variance = [1, 1]
trainingParams_bboxloss_v1.overlap_threshold = 0.5
trainingParams_bboxloss_v1.neg_pos_ratio = 2
trainingParams_bboxloss_v1.landmark_size = 4
trainingParams_bboxloss_v1.train_display_results_save_interval = 100
trainingParams_bboxloss_v1.validation_display_results_save_interval = 50
trainingParams_bboxloss_v1.bbox_regression_weight = 1
trainingParams_bboxloss_v1.classification_weight = 6
trainingParams_bboxloss_v1.landmark_regression_weight = 6
trainingParams_bboxloss_v1.svd_error_weight = 0
trainingParams_bboxloss_v1.pos_sample_score_threshold = 30

trainingParams_svdloss_v1 = UP.TrainingParams()
trainingParams_svdloss_v1.batch_size = 1
trainingParams_svdloss_v1.num_workers = 1
trainingParams_svdloss_v1.CPU = False
trainingParams_svdloss_v1.lr = 0.001
trainingParams_svdloss_v1.momentum = 0.1
trainingParams_svdloss_v1.weight_decay = 5e-4
trainingParams_svdloss_v1.factor = 0.75
trainingParams_svdloss_v1.patience = 3
trainingParams_svdloss_v1.min_lr = 0
trainingParams_svdloss_v1.gamma = 0.1
trainingParams_svdloss_v1.precision = 32  # 未用
trainingParams_svdloss_v1.num_of_class = 2
trainingParams_svdloss_v1.num_of_landmarks = 4
trainingParams_svdloss_v1.seed = 42
trainingParams_svdloss_v1.max_epochs = 500
trainingParams_svdloss_v1.patience = 50
trainingParams_svdloss_v1.img_height = int(1080)
trainingParams_svdloss_v1.img_width = int(1920)
trainingParams_svdloss_v1.train_or_test = "Train"
trainingParams_svdloss_v1.conf_thresh_for_test = 0.51
trainingParams_svdloss_v1.nms_thresh_for_test = 0.05
trainingParams_svdloss_v1.variance = [1, 1]
trainingParams_svdloss_v1.overlap_threshold = 0.5
trainingParams_svdloss_v1.neg_pos_ratio = 5
trainingParams_svdloss_v1.landmark_size = 4
trainingParams_svdloss_v1.train_display_results_save_interval = 100
trainingParams_svdloss_v1.validation_display_results_save_interval = 40
trainingParams_svdloss_v1.bbox_regression_weight = 0
trainingParams_svdloss_v1.classification_weight = 2
trainingParams_svdloss_v1.landmark_regression_weight = 1
trainingParams_svdloss_v1.svd_error_weight = 0.2
trainingParams_svdloss_v1.pos_sample_score_threshold = 30

# model param
VGG16_FPN2L_DR8_16_v1 = UP.NetworkParams()

VGG16_FPN2L_DR8_16_v1.backbone = 'vgg16'
VGG16_FPN2L_DR8_16_v1.pretrain = True
VGG16_FPN2L_DR8_16_v1.prior_box_sizes = [
    [[93, 84], [62, 97], [44, 73], [64, 27], [142, 46], [77, 55], [128, 66], [104, 37]],
    [[93, 84], [62, 97], [44, 73], [64, 27], [142, 46], [77, 55], [128, 66], [104, 37]]]
VGG16_FPN2L_DR8_16_v1.steps = [8, 16]
VGG16_FPN2L_DR8_16_v1.clip = False
VGG16_FPN2L_DR8_16_v1.return_layers = {'16': 1, '23': 2}
VGG16_FPN2L_DR8_16_v1.in_channels_list = [256, 512]
VGG16_FPN2L_DR8_16_v1.in_channel = 256
VGG16_FPN2L_DR8_16_v1.out_channel = 256
VGG16_FPN2L_DR8_16_v1.landmark_dim = 8
VGG16_FPN2L_DR8_16_v1.prior_num_in_a_cell = 16
VGG16_FPN2L_DR8_16_v1.pca_feature_size = 8

unscented_svd_augmented_5_5_10_large_size_v1 = UP.UnscentedSVDAugmentationPara()
unscented_svd_augmented_5_5_10_large_size_v1.bias_x = 5
unscented_svd_augmented_5_5_10_large_size_v1.bias_y = 5
unscented_svd_augmented_5_5_10_large_size_v1.bias_orientation = 10
unscented_svd_augmented_5_5_10_large_size_v1.size_x = 11
unscented_svd_augmented_5_5_10_large_size_v1.size_y = 11
unscented_svd_augmented_5_5_10_large_size_v1.size_orientation = 21
unscented_svd_augmented_5_5_10_large_size_v1.unscented_k = 2
unscented_svd_augmented_5_5_10_large_size_v1.augmented_svd_param_file_path_str = ('/root/LG_WS/IRFishDetection2.0.0'
                                                                                  '/results/other_results/hu_s_and_c'
                                                                                  '/augmented_svd_params'
                                                                                  '/unscented_svd_augmented_5_5_10_large_size_v1')

unscented_svd_augmented_2p5_2p5_5_large_size_v1 = UP.UnscentedSVDAugmentationPara()
unscented_svd_augmented_2p5_2p5_5_large_size_v1.bias_x = 2.5
unscented_svd_augmented_2p5_2p5_5_large_size_v1.bias_y = 2.5
unscented_svd_augmented_2p5_2p5_5_large_size_v1.bias_orientation = 5
unscented_svd_augmented_2p5_2p5_5_large_size_v1.size_x = 11
unscented_svd_augmented_2p5_2p5_5_large_size_v1.size_y = 11
unscented_svd_augmented_2p5_2p5_5_large_size_v1.size_orientation = 21
unscented_svd_augmented_2p5_2p5_5_large_size_v1.unscented_k = 2
unscented_svd_augmented_2p5_2p5_5_large_size_v1.augmented_svd_param_file_path_str = ('/root/LG_WS/IRFishDetection2.0.0'
                                                                                     '/results/other_results/hu_s_and_c'
                                                                                     '/augmented_svd_params'
                                                                                     '/unscented_svd_augmented_2p5_2p5_5_large_size_v1')

unscented_svd_augmented_1_1_2p5_large_size_v1 = UP.UnscentedSVDAugmentationPara()
unscented_svd_augmented_1_1_2p5_large_size_v1.bias_x = 1
unscented_svd_augmented_1_1_2p5_large_size_v1.bias_y = 1
unscented_svd_augmented_1_1_2p5_large_size_v1.bias_orientation = 2.5
unscented_svd_augmented_1_1_2p5_large_size_v1.size_x = 11
unscented_svd_augmented_1_1_2p5_large_size_v1.size_y = 11
unscented_svd_augmented_1_1_2p5_large_size_v1.size_orientation = 21
unscented_svd_augmented_1_1_2p5_large_size_v1.unscented_k = 2
unscented_svd_augmented_1_1_2p5_large_size_v1.augmented_svd_param_file_path_str = ('/root/LG_WS/IRFishDetection2.0.0'
                                                                                   '/results/other_results/hu_s_and_c'
                                                                                   '/augmented_svd_params'
                                                                                   '/unscented_svd_augmented_1_1_2p5_large_size_v1')

unscented_svd_augmented_5_5_10_large_size_v2 = UP.UnscentedSVDAugmentationPara()
unscented_svd_augmented_5_5_10_large_size_v2.bias_x = 5
unscented_svd_augmented_5_5_10_large_size_v2.bias_y = 5
unscented_svd_augmented_5_5_10_large_size_v2.bias_orientation = 10
unscented_svd_augmented_5_5_10_large_size_v2.size_x = 11
unscented_svd_augmented_5_5_10_large_size_v2.size_y = 11
unscented_svd_augmented_5_5_10_large_size_v2.size_orientation = 21
unscented_svd_augmented_5_5_10_large_size_v2.unscented_k = 2
unscented_svd_augmented_5_5_10_large_size_v2.augmented_svd_param_file_path_str = ('/root/LG_WS/IRFishDetection2.0.0'
                                                                                  '/results/other_results/hu_s_and_c'
                                                                                  '/augmented_svd_params'
                                                                                  '/unscented_svd_augmented_5_5_10_large_size_v2')

unscented_svd_augmented_2p5_2p5_5_large_size_v2 = UP.UnscentedSVDAugmentationPara()
unscented_svd_augmented_2p5_2p5_5_large_size_v2.bias_x = 2.5
unscented_svd_augmented_2p5_2p5_5_large_size_v2.bias_y = 2.5
unscented_svd_augmented_2p5_2p5_5_large_size_v2.bias_orientation = 5
unscented_svd_augmented_2p5_2p5_5_large_size_v2.size_x = 11
unscented_svd_augmented_2p5_2p5_5_large_size_v2.size_y = 11
unscented_svd_augmented_2p5_2p5_5_large_size_v2.size_orientation = 21
unscented_svd_augmented_2p5_2p5_5_large_size_v2.unscented_k = 2
unscented_svd_augmented_2p5_2p5_5_large_size_v2.augmented_svd_param_file_path_str = ('/root/LG_WS/IRFishDetection2.0.0'
                                                                                     '/results/other_results/hu_s_and_c'
                                                                                     '/augmented_svd_params'
                                                                                     '/unscented_svd_augmented_2p5_2p5_5_large_size_v2')

unscented_svd_augmented_1_1_2p5_large_size_v2 = UP.UnscentedSVDAugmentationPara()
unscented_svd_augmented_1_1_2p5_large_size_v2.bias_x = 1
unscented_svd_augmented_1_1_2p5_large_size_v2.bias_y = 1
unscented_svd_augmented_1_1_2p5_large_size_v2.bias_orientation = 2.5
unscented_svd_augmented_1_1_2p5_large_size_v2.size_x = 11
unscented_svd_augmented_1_1_2p5_large_size_v2.size_y = 11
unscented_svd_augmented_1_1_2p5_large_size_v2.size_orientation = 21
unscented_svd_augmented_1_1_2p5_large_size_v2.unscented_k = 2
unscented_svd_augmented_1_1_2p5_large_size_v2.augmented_svd_param_file_path_str = ('/root/LG_WS/IRFishDetection2.0.0'
                                                                                   '/results/other_results/hu_s_and_c'
                                                                                   '/augmented_svd_params'
                                                                                   '/unscented_svd_augmented_1_1_2p5_large_size_v2')

if __name__ == '__main__':
    # path
    data_path_hu_s_and_c_v1_json_file = ('/root/LG_WS/IRFishDetection2.0.0/parameters/'
                                         'data_path_hu_s_and_c_v1.json')
    UP.save_params_to_json(data_path_hu_s_and_c_v1, data_path_hu_s_and_c_v1_json_file)

    data_path_small_test_v1_json_file = ('/root/LG_WS/IRFishDetection2.0.0/parameters/'
                                         'data_path_small_test_v1.json')
    UP.save_params_to_json(data_path_small_test_v1, data_path_small_test_v1_json_file)

    # train param
    trainingParams_bboxloss_v1_json_file = ('/root/LG_WS/IRFishDetection2.0.0/parameters/'
                                   'training_v1.json')
    UP.save_params_to_json(trainingParams_bboxloss_v1, trainingParams_bboxloss_v1_json_file)

    # model param
    VGG16_FPN2L_DR8_16_v1_json_file = ('/root/LG_WS/IRFishDetection2.0.0/parameters/'
                                       'VGG16_FPN2L_DR8_16_v1.json')
    UP.save_params_to_json(VGG16_FPN2L_DR8_16_v1, VGG16_FPN2L_DR8_16_v1_json_file)
