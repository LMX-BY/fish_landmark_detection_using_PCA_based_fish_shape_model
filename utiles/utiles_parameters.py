import json


# None 初始化方便知道在赋值后哪些参数未赋值
class TrainingParams:
    def __init__(self):
        self.batch_size = None
        self.num_workers = None
        self.CPU = None
        self.lr = None
        self.momentum = None
        self.weight_decay = None
        self.factor = None
        self.patience = None
        self.min_lr = None
        self.gamma = None
        self.precision = None  # 未用
        self.num_of_class = None
        self.num_of_landmarks = None
        self.seed = None
        self.max_epochs = None
        self.patience = None
        self.img_height = None
        self.img_width = None
        self.train_or_test = None
        self.conf_thresh_for_test = None
        self.nms_thresh_for_test = None
        self.variance = None
        self.overlap_threshold = None
        self.neg_pos_ratio = None
        self.landmark_size = None
        self.train_display_results_save_interval = None
        self.validation_display_results_save_interval = None
        self.bbox_regression_weight = None
        self.classification_weight = None
        self.landmark_regression_weight = None
        self.svd_error_weight = None
        self.pos_sample_score_threshold = None


class NetworkParams:
    def __init__(self):
        self.backbone = None
        self.pretrain = None
        self.prior_box_sizes = None
        self.steps = None
        self.clip = None
        self.return_layers = None
        self.in_channels_list = None
        self.in_channel = None
        self.out_channel = None
        self.landmark_dim = None
        self.prior_num_in_a_cell = None
        self.pca_feature_size = None


class UnscentedSVDAugmentationPara:
    def __init__(self):
        self.bias_orientation = None
        self.size_orientation = None
        self.bias_x = None
        self.size_x = None
        self.bias_y = None
        self.size_y = None
        self.unscented_k = None
        self.augmented_svd_param_file_path_str = None


class DataPath:
    def __init__(self):
        self.train_label_path_str = None
        self.validation_label_path_str = None
        self.test_label_path_str = None
        self.train_img_path_str = None
        self.test_img_path_str = None
        self.all_label_path_str = None


class TrainResultsSavePath:
    def __init__(self):
        self.ckp_save_path_str = None
        self.best_model_save_path = None
        self.display_results_validation_save_path_str = None
        self.display_results_test_save_path_str = None
        self.display_results_train_save_path_str = None
        self.display_results_pos_sample_save_path_str = None


class OtherResultsSavePath:
    def __init__(self):
        self.unknown_mask_index_path_str = None
        self.unknown_mask_display_path_str = None
        self.svd_feature_path = None


# pca related
class SVDRelatedDataPath:
    def __init__(self):
        self.unknown_mask_path_str = None
        self.landmark_priors_file_path_str = None
        self.svd_param_file_path_str = None

        # self.pos_landmarks_results_save_path = None
        # self.csv_results_file_path_for_train = None


# !! 将参数对象写入json，无异常捕获
def save_params_to_json(param_object, file_name):
    file = open(file_name, 'w')
    file.write(json.dumps(param_object.__dict__))


# !! 将json数据写入参数对象，无异常捕获
def load_params_from_json(param_object, file_name):
    file = open(file_name, 'r')
    param_dict = json.load(file)
    param_object.__dict__ = param_dict
