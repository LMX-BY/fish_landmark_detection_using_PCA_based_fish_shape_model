from __future__ import print_function
import os
import torch
import torch.utils.data as data
from torchvision import transforms
import time
import datetime
import math
import cv2
from model.retinafaceDetection.FishLandmarkRefinementWithBBoxnms import FishLandmarkRefinementWithBBoxnms
from model.retinafaceDetection.RetinaFaceFishLandmarkDetectionNet3Layer import RetinaFaceLandmarkDetectionNet3Layer
from model.retinafaceDetection.RetinaFaceFishLandmarkDetectionNet2Layer import RetinaFaceLandmarkDetectionNet2Layer
from model.retinafaceDetection.priorBBoxGenerator import IRFishDetectionBBoxPriors
from model.retinafaceDetection.RetinaFaceFishBBoxLoss import RetinaFaceFishBBoxLoss
from utiles import utiles_weight_initialization
from utiles import utils_fish_landmark_detection
from utiles import utiles_parameters
from dataset import IRFishDatasetCoco_16
from utiles import displayTool
from utiles import coco_utils
from utiles import coco_eval
from utiles import user_def_transform
from utiles import utiles_files
from params import params_objects
import pathlib
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks import ModelCheckpoint
from torchmetrics.detection import MeanAveragePrecision
from pytorch_lightning.callbacks import LearningRateMonitor
import torch.nn.functional as F

color_green = (0, 255, 0)
color_red = (0, 0, 255)
color_blue = (0, 0, 255)
color_white = (255, 255, 255)


class LitIRFishKeyPtsDetection(pl.LightningModule):
    def __init__(self, backbone, loss, priors, detector, params, coco_val_data_api, coco_test_data_api, path):
        super().__init__()
        self.backbone = backbone
        self.loss = loss
        self.priors = priors
        self.detector = detector
        self.pointColorMap = {0: 'red', 1: 'orange', 2: 'yellow', 3: 'green'}
        self.train_count = 0
        self.pytorch_device = torch.device("cpu" if params.CPU else "cuda")

        self.results_save_path_for_test = path.display_results_test_save_path_str
        self.results_save_path_for_valid = path.display_results_validation_save_path_str
        self.results_save_path_for_train = path.display_results_train_save_path_str
        self.results_save_path_for_pos_sample = path.display_results_pos_sample_save_path_str

        self.params = params
        self.train_count = 0
        self.validation_count = 0
        self.detection_metric_for_valid = MeanAveragePrecision(iou_type="bbox")
        self.train_results_save_interval = params.train_display_results_save_interval
        self.validation_results_save_interval = params.validation_display_results_save_interval
        self.max_map50_score_validation = 0

        self.b_w = params.bbox_regression_weight
        self.c_w = params.classification_weight
        self.l_w = params.landmark_regression_weight
        assert self.b_w > 0
        assert self.c_w > 0
        assert self.l_w > 0

        self.iou_types = ['bbox', 'keypoints']
        self.coco_val_data_api = coco_val_data_api
        self.coco_test_data_api = coco_test_data_api
        self.val_coco_evaluator = coco_eval.CocoEvaluator(self.coco_val_data_api, self.iou_types)
        self.test_coco_evaluator = coco_eval.CocoEvaluator(self.coco_test_data_api, self.iou_types)

    def training_step(self, batch, batch_idx):
        # print(f'self.current_epoch is {self.current_epoch}')
        self.backbone.set_phase_train()
        data, label_coco, source_imgs, img_file_names = batch
        img_height = data.size(2)
        img_width = data.size(3)
        label_local = self.label_coco_to_local(label_coco, img_height, img_width)
        out = self.backbone(data)
        loss_l, loss_c, loss_landm, pos_sample_index = self.loss(out, self.priors, label_local)
        loss_l = loss_l * self.b_w
        loss_c = loss_c * self.c_w
        loss_landm = loss_landm * self.l_w
        loss = loss_l + loss_c + loss_landm
        #loss = loss_c + loss_l
        self.log("train_loss_l", loss_l)
        self.log("train_loss_c", loss_c)
        self.log("train_loss_landm", loss_landm)
        self.log("train_loss", loss)
        print(f'loss score is {loss},loss_l is {loss_l},loss_c is {loss_c},loss_landm is {loss_landm}')
        self.train_count = self.train_count + 1

        # 显示gt和正样本,及正样本对应landmark的回归情况
        if self.current_epoch % self.train_results_save_interval == 0:
            if self.results_save_path_for_pos_sample is not None:
                with torch.no_grad():
                    for index_batch, a_img in enumerate(source_imgs):
                        pos_sample_index_batch = pos_sample_index[index_batch]
                        pos_reg_encoded_bbox = out[0][index_batch][
                            pos_sample_index_batch].detach().cpu().clone().squeeze()
                        pos_reg_encoded_landmarks = out[2][index_batch][
                            pos_sample_index_batch].detach().cpu().clone().squeeze()
                        pos_prior_bbox = self.priors[pos_sample_index_batch].detach().cpu().clone().squeeze()
                        pos_reg_decoded_bbox = utils_fish_landmark_detection.decode_bbox(
                            pos_reg_encoded_bbox, pos_prior_bbox, self.params.variance)
                        pos_reg_decoded_landmarks = utils_fish_landmark_detection.decode_landm_with_bbox(
                            pos_reg_encoded_landmarks, pos_prior_bbox, self.params.variance)
                        display_img = a_img.copy()
                        batch_label = label_local[index_batch].detach().clone()
                        # pos_score = out[1][index_batch][pos_sample_index].detach().cpu().clone().squeeze()
                        # for a_pos_sample_index in pos_sample_index:
                        #     a_pos_score = out[1][index_batch][a_pos_sample_index]
                        #     print('*' * 10)
                        #     print(f'posIndex: {a_pos_sample_index}')
                        #     print('pos sample score:')
                        #     print(a_pos_score)
                        #     print('after softmax:')
                        #     print(F.softmax(a_pos_score))
                        # 显示gt
                        displayTool.display_normalized_bbox(batch_label[:, 0:4], display_img, color_green)
                        displayTool.display_normalized_landmark_in_polygon_in_img(batch_label[:, 4:12], display_img,
                                                                                  color_green)
                        # 显示正样本先验
                        displayTool.display_normalized_bbox(utils_fish_landmark_detection.point_form(pos_prior_bbox),
                                                            display_img, color_red)
                        # 显示正样本栅格对应的预测值
                        displayTool.display_normalized_bbox(pos_reg_decoded_bbox, display_img, color_blue)
                        displayTool.display_normalized_landmark_in_polygon_in_img(pos_reg_decoded_landmarks,
                                                                                  display_img, color_blue)
                        img_name = img_file_names[index_batch]
                        display_img = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
                        save_img_name = self.results_save_path_for_pos_sample + '/' + img_name + '_epoch_' + str(
                            self.current_epoch) + '_batchIndex_' + str(batch_idx) + '.bmp'
                        cv2.imwrite(save_img_name, display_img)

            # 保存训练集的训练结果
            if self.results_save_path_for_train is not None:
                data_for_show = data.detach().clone()
                self.backbone.set_phase_eval()
                with torch.no_grad():
                    loc, conf, landms = self.backbone(data_for_show)
                    for index in range(loc.size(0)):
                        a_loc = loc[index]
                        a_conf = conf[index]
                        a_landms = landms[index]
                        a_gt = label_local[index]
                        a_img_data = source_imgs[index]
                        display_img = a_img_data.copy()
                        detected_result = self.detector.forward(a_loc, a_conf, a_landms, self.priors)
                        a_img_file_path = img_file_names[index]
                        a_img_name = a_img_file_path + 'train_' + str(self.train_count) + 'batch_index_' + str(
                            batch_idx)
                        # a_img_data = (a_img_data.detach().clone().cpu().int().numpy().transpose(1, 2, 0))
                        # a_img_data = np.ascontiguousarray(a_img_data)
                        # gt
                        displayTool.display_normalized_bbox(a_gt[:, 0:4], display_img, color_green)
                        displayTool.display_normalized_landmark_in_polygon_in_img(a_gt[:, 4:12], display_img,
                                                                                  color_green)
                        # detection results
                        displayTool.display_bbox_and_landmark_in_polygon_and_score(detected_result, display_img,
                                                                                   color_red, color_white)
                        save_img_name = self.results_save_path_for_train + '/' + a_img_name + '.bmp'
                        display_img = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
                        cv2.imwrite(save_img_name, display_img)

        return loss

    def validation_step(self, batch, batch_idx):
        self.validation_count = self.validation_count + 1
        if batch_idx == 0:
            self.detection_metric_for_valid.reset()
        self.backbone.set_phase_eval()
        print(self.backbone.phase)
        data, label_coco, source_imgs, img_file_path_list = batch
        img_height = data.size(2)
        img_width = data.size(3)
        label_local = self.label_coco_to_local(label_coco, img_height, img_width)
        loc, conf, landms = self.backbone(data)

        # 获得验证结果
        for index in range(loc.size(0)):
            a_loc = loc[index]
            a_conf = conf[index]
            a_landms = landms[index]
            a_img_data = source_imgs[index]
            a_gt = label_local[index]
            a_label_coco = [label_coco[index]]
            img_height = a_img_data.shape[0]
            img_width = a_img_data.shape[1]
            detected_result = self.detector.forward(a_loc, a_conf, a_landms, self.priors)
            detected_result_coco = self.detected_result_local_to_coco(detected_result)
            detected_result_coco = [detected_result_coco]
            # 計算評價
            cpu_device = torch.device("cpu")
            detected_result_coco = [{k: v.to(cpu_device) for k, v in t.items()} for t in detected_result_coco]
            res = {target["image_id"].item(): output for target, output in zip(a_label_coco, detected_result_coco)}
            self.val_coco_evaluator.update(res)
            if self.current_epoch % self.validation_results_save_interval == 0:
                if self.results_save_path_for_valid is not None:
                    a_img_file_path = img_file_path_list[index]
                    a_img_name = a_img_file_path + '_train_' + str(self.validation_count) + '_batch_index_' + str(
                        batch_idx)
                    # a_img_data = (a_img_data.detach().clone().cpu().int().numpy().transpose(1, 2, 0))
                    # a_img_data = np.ascontiguousarray(a_img_data)
                    # detected results
                    displayTool.display_bbox_and_landmark_in_polygon_and_score(detected_result, a_img_data,
                                                                               color_red, color_white)
                    # gt
                    displayTool.display_normalized_bbox(a_gt[:, 0:4], a_img_data, color_green)
                    displayTool.display_normalized_landmark_in_polygon_in_img(a_gt[:, 4:12], a_img_data,
                                                                              color_green)
                    save_img_name = self.results_save_path_for_valid + '/' + a_img_name + '.bmp'
                    cv2.imwrite(save_img_name, a_img_data)

        if batch_idx == (self.trainer.num_val_batches[0] - 1):
            self.val_coco_evaluator.synchronize_between_processes()
            # accumulate predictions from all images
            self.val_coco_evaluator.accumulate()
            self.val_coco_evaluator.summarize()
            coco_info_bbox = self.val_coco_evaluator.coco_eval[self.iou_types[0]].stats.tolist()
            coco_info_landmarks = self.val_coco_evaluator.coco_eval[self.iou_types[1]].stats.tolist()
            self.log("bbox_AP_0.5:0.95", coco_info_bbox[0])
            self.log("bbox_AP_0.5", coco_info_bbox[1])
            self.log("bbox_AP_0.75", coco_info_bbox[2])
            self.log("bbox_AR_0.5:0.95", coco_info_bbox[5])
            self.log("bbox_AR_0.5", coco_info_bbox[6])
            self.log("bbox_AR_0.75", coco_info_bbox[7])

            self.log("landmarks_AP_0.5:0.95", coco_info_landmarks[0])
            self.log("landmarks_AP_0.5", coco_info_landmarks[1])
            self.log("landmarks_AP_0.75", coco_info_landmarks[2])
            self.log("landmarks_AR_0.5:0.95", coco_info_landmarks[5])
            self.log("landmarks_AR_0.5", coco_info_landmarks[6])
            self.log("landmarks_AR_0.75", coco_info_landmarks[7])
            self.log("val_map50", coco_info_landmarks[1])
            print('*' * 10 + 'landmarks' + '*' * 10)
            print('*' * 10 + 'landmarks' + '*' * 10)
            coco_info_bbox = self.val_coco_evaluator.coco_eval[self.iou_types[1]].stats.tolist()
            # self.train_coco_evaluator.synchronize_between_processes()
            self.val_coco_evaluator = coco_eval.CocoEvaluator(self.coco_val_data_api, self.iou_types)

    def test_step(self, batch, batch_idx):
        if batch_idx == 0:
            self.detection_metric_for_valid.reset()
        self.backbone.set_phase_eval()
        print(self.backbone.phase)
        data, label_coco, source_imgs, img_file_path_list = batch
        img_height = data.size(2)
        img_width = data.size(3)
        label_local = self.label_coco_to_local(label_coco, img_height, img_width)
        loc, conf, landms = self.backbone(data)

        # 获得验证结果
        for index in range(loc.size(0)):
            a_loc = loc[index]
            a_conf = conf[index]
            a_landms = landms[index]
            a_img_data = source_imgs[index]
            a_gt = label_local[index]
            a_label_coco = [label_coco[index]]
            img_height = a_img_data.shape[0]
            img_width = a_img_data.shape[1]
            detected_result = self.detector.forward(a_loc, a_conf, a_landms, self.priors)
            detected_result_coco = self.detected_result_local_to_coco(detected_result)
            detected_result_coco = [detected_result_coco]
            # 計算評價
            cpu_device = torch.device("cpu")
            detected_result_coco = [{k: v.to(cpu_device) for k, v in t.items()} for t in detected_result_coco]
            res = {target["image_id"].item(): output for target, output in zip(a_label_coco, detected_result_coco)}
            self.test_coco_evaluator.update(res)

            if self.results_save_path_for_valid is not None:
                a_img_file_path = img_file_path_list[index]
                a_img_name = a_img_file_path + '_train_' + str(self.validation_count) + '_batch_index_' + str(
                    batch_idx)
                # a_img_data = (a_img_data.detach().clone().cpu().int().numpy().transpose(1, 2, 0))
                # a_img_data = np.ascontiguousarray(a_img_data)
                # detected results
                displayTool.display_bbox_and_landmark_in_polygon_and_score(detected_result, a_img_data,
                                                                           color_red, color_white)
                # gt
                #displayTool.display_normalized_bbox(a_gt[:, 0:4], a_img_data, color_green)
                displayTool.display_normalized_landmark_in_polygon_in_img(a_gt[:, 4:12], a_img_data,
                                                                          color_green)
                save_img_name = self.results_save_path_for_test + '/' + a_img_name + '.bmp'
                cv2.imwrite(save_img_name, a_img_data)

        if batch_idx == (self.trainer.num_test_batches[0] - 1):
            self.test_coco_evaluator.synchronize_between_processes()
            # accumulate predictions from all images

            self.test_coco_evaluator.accumulate()
            self.test_coco_evaluator.summarize()
            coco_info_bbox = self.test_coco_evaluator.coco_eval[self.iou_types[0]].stats.tolist()
            coco_info_landmarks = self.test_coco_evaluator.coco_eval[self.iou_types[1]].stats.tolist()
            self.log("bbox_AP_0.5:0.95", coco_info_bbox[0])
            self.log("bbox_AP_0.5", coco_info_bbox[1])
            self.log("bbox_AP_0.75", coco_info_bbox[2])
            self.log("bbox_AR_0.5:0.95", coco_info_bbox[5])
            self.log("bbox_AR_0.5", coco_info_bbox[6])
            self.log("bbox_AR_0.75", coco_info_bbox[7])

            self.log("landmarks_AP_0.5:0.95", coco_info_landmarks[0])
            self.log("landmarks_AP_0.5", coco_info_landmarks[1])
            self.log("landmarks_AP_0.75", coco_info_landmarks[2])
            self.log("landmarks_AR_0.5:0.95", coco_info_landmarks[5])
            self.log("landmarks_AR_0.5", coco_info_landmarks[6])
            self.log("landmarks_AR_0.75", coco_info_landmarks[7])
            print('*' * 10 + 'landmarks' + '*' * 10)
            print('*' * 10 + 'landmarks' + '*' * 10)
            coco_info_bbox = self.test_coco_evaluator.coco_eval[self.iou_types[1]].stats.tolist()
            # self.train_coco_evaluator.synchronize_between_processes()
            self.test_coco_evaluator = coco_eval.CocoEvaluator(self.coco_val_data_api, self.iou_types)

    def configure_optimizers(self):
        # ***********test************
        count = 0
        for name, parameter in self.backbone.body.named_parameters():
            if count < 3:
                parameter.requires_grad = False
            else:
                parameter.requires_grad = True
            count = count + 1
        for name, parameter in self.backbone.body.named_parameters():
            print(f'name is {name}')
            print(f'required grad is {parameter.requires_grad}')
        # ***********test************
        non_frozen_parameters = [p for p in self.backbone.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(
            non_frozen_parameters, lr=self.params.lr, momentum=self.params.momentum,
            weight_decay=self.params.weight_decay
        )
        # for a_param_group in optimizer.param_groups:
        #     a_param_group
        #     pass
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=self.params.factor, patience=self.params.patience, min_lr=self.params.min_lr
        )
        # return optimizer
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
            "monitor": "train_loss",
        }

    def label_coco_to_local(self, label, img_height, img_width):
        transformed_label = []
        scale_tensor = torch.Tensor([img_width, img_height, img_width, img_height,
                                     img_width, img_height, img_width, img_height,
                                     img_width, img_height, img_width, img_height, 1]).to(self.pytorch_device)
        label_batch_list = []
        for label_in_img in label:
            bboxes = label_in_img['boxes']
            landmarks = label_in_img['keypoints']
            label_in_img_list = []
            for a_bboxes, a_landmarks in zip(bboxes, landmarks):
                label_has_landmark = torch.Tensor([[1]]).to(self.pytorch_device)
                if a_landmarks[0][2] == 0:
                    label_has_landmark[0][0] = -1
                a_landmarks_reshape = a_landmarks[:, :2].reshape(1, -1)
                a_bboxes_reshape = a_bboxes.reshape(1, -1)
                a_label_tensor = torch.cat((a_bboxes_reshape, a_landmarks_reshape, label_has_landmark), 1)
                a_label_tensor = a_label_tensor / scale_tensor
                label_in_img_list.append(a_label_tensor)
            label_in_img_tensor = torch.cat(label_in_img_list, 0)
            label_batch_list.append(label_in_img_tensor)
        return label_batch_list

    def detected_result_local_to_coco(self, detected_result):
        pass
        key_points_with_visibility_tensor = torch.ones((detected_result.shape[0], 12))
        key_points_with_visibility_tensor[:, 0:2] = torch.from_numpy(detected_result[:, 5:7])
        key_points_with_visibility_tensor[:, 3:5] = torch.from_numpy(detected_result[:, 7:9])
        key_points_with_visibility_tensor[:, 6:8] = torch.from_numpy(detected_result[:, 9:11])
        key_points_with_visibility_tensor[:, 9:11] = torch.from_numpy(detected_result[:, 11:13])
        results_dict = {'boxes': torch.from_numpy(detected_result[:, 0:4]).to(self.pytorch_device),
                        'labels': torch.ones(detected_result.shape[0]).int().to(self.pytorch_device),
                        'scores': torch.from_numpy(detected_result[:, 4]).to(self.pytorch_device),
                        'keypoints': key_points_with_visibility_tensor}
        return results_dict


class FineTuningCallback(Callback):
    def __init__(self, params):
        self.params = params

    def on_train_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if trainer.current_epoch == 6:
            count = 0
            for name, parameter in pl_module.backbone.body.named_parameters():
                if count < 3:
                    parameter.requires_grad = False
                else:
                    parameter.requires_grad = True
                count = count + 1
            for name, parameter in pl_module.backbone.body.named_parameters():
                print(f'name is {name}')
                print(f'required grad is {parameter.requires_grad}')

            non_frozen_parameters = [p for p in pl_module.backbone.parameters() if p.requires_grad]
            optimizer = torch.optim.SGD(
                non_frozen_parameters, lr=self.params.lr, momentum=self.params.momentum,
                weight_decay=self.params.weight_decay
            )
            # for a_param_group in optimizer.param_groups:
            #     a_param_group
            #     pass
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", factor=self.params.factor, patience=self.params.patience, min_lr=self.params.min_lr
            )
            trainer.optimizers = [optimizer]
            trainer.lr_schedulers = [lr_scheduler]




if __name__ == '__main__':
    # 导入参数
    data_path_params = params_objects.data_path_hu_s_and_c_v1
    results_path_params = params_objects.results_path_bbox_reg_v1
    training_params = params_objects.trainingParams_bboxloss_v1
    network_params = params_objects.VGG16_FPN2L_DR8_16_v1
    # ckp_file_path = (
    #     'H:/code/python/IRFishDetection2.0.0/results/bboxLoss/ckp/train/lightning_logs/vgg16+Retina_Face+2_Layer_FPN'
    #     '+bbox_reg_1/checkpoints/epoch=249-step=25500.ckpt')

    best_model_path = utiles_files.create_folder_with_cur_time_info(results_path_params.best_model_save_path,
                                                                    'best_model')
    # ckp_file_path = ('H:/code/python/IRFishDetection2.0.0/results/svdEncodedLoss/ckp/train/lightning_logs/temp_results/epoch=499-step=101500.ckpt')
    ckp_file_path = (
        r'H:\code\models_for_pca_encoded_deep_learning\cq_A_032_44681\bboxLoss\best_model_20240725_1438_retina_vgg16_2FPN_bbox_reg\test-ccb-epoch=67-val_map50=0.46.ckpt')
    # 路径初始化

    train_label_path = pathlib.Path(data_path_params.train_label_path_str)
    validation_label_path = pathlib.Path(data_path_params.validation_label_path_str)
    train_img_path = pathlib.Path(data_path_params.train_img_path_str)
    test_label_path = pathlib.Path(data_path_params.test_label_path_str)
    test_img_path = pathlib.Path(data_path_params.test_img_path_str)

    #param = Params()

    train_label_files = utiles_files.get_filenames_of_path(train_label_path)
    valid_lable_files = utiles_files.get_filenames_of_path(validation_label_path)
    test_label_files = utiles_files.get_filenames_of_path(test_label_path)


    # transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize(utils_ir_fish_detection.ir_fish_img_means,
    #                          utils_ir_fish_detection.ir_fish_img_stds)
    # ])
    # data_transform = {
    #     "train": user_def_transform.Compose([user_def_transform.ToTensor(),
    #                                          user_def_transform.RandomHorizontalFlip(0.5)]),
    #     "val": user_def_transform.Compose([user_def_transform.ToTensor()])
    # }
    # 数据预处理方法初始化
    data_transform = {
        "train": user_def_transform.Compose([user_def_transform.ToTensor(),
                                             user_def_transform.Normalize(
                                                 utils_fish_landmark_detection.ir_fish_img_means,
                                                 utils_fish_landmark_detection.ir_fish_img_stds)]),
        "val": user_def_transform.Compose([user_def_transform.ToTensor(),
                                           user_def_transform.Normalize(utils_fish_landmark_detection.ir_fish_img_means,
                                                                        utils_fish_landmark_detection.ir_fish_img_stds)
                                           ])
    }
    # data_transform = {
    #     "train": user_def_transform.Compose([user_def_transform.ToTensor()]),
    #     "val": user_def_transform.Compose([user_def_transform.ToTensor()])
    # }

    # 导入数据集
    train_dataset = IRFishDatasetCoco_16.IRFishDatasetCoco_16(train_img_path, train_label_files,
                                                              training_params.num_of_landmarks, data_transform["train"])
    valid_dataset = IRFishDatasetCoco_16.IRFishDatasetCoco_16(train_img_path, valid_lable_files,
                                                              training_params.num_of_landmarks, data_transform["val"])
    test_dataset = IRFishDatasetCoco_16.IRFishDatasetCoco_16(test_img_path, test_label_files, training_params.num_of_landmarks,
                                                             data_transform["val"])

    train_data_loader = data.DataLoader(train_dataset, training_params.batch_size, shuffle=True, num_workers=training_params.num_workers,
                                        collate_fn=IRFishDatasetCoco_16.detection_collate)
    valid_data_loader = data.DataLoader(valid_dataset, training_params.batch_size, shuffle=True, num_workers=training_params.num_workers,
                                        collate_fn=IRFishDatasetCoco_16.detection_collate)
    test_data_loader = data.DataLoader(test_dataset, training_params.batch_size, shuffle=True, num_workers=training_params.num_workers,
                                       collate_fn=IRFishDatasetCoco_16.detection_collate)

    coco_val_data_api = coco_utils.get_coco_api_from_dataset(valid_data_loader.dataset)
    coco_test_data_api = coco_utils.get_coco_api_from_dataset(test_data_loader.dataset)

    # 构建网络
    #cfg = cfg_vgg16_2_layer_8_16
    device = torch.device("cpu" if training_params.CPU else "cuda")
    FeatureExtractor = RetinaFaceLandmarkDetectionNet2Layer(cfg=network_params)
    # # 权重初始化
    #FeatureExtractor.apply(utiles_weight_initialization.init_weights_kaiming_uniform)

    FeatureExtractor.print_paramter_grad_state()
    test_input_tensor = torch.zeros([1, 3, training_params.img_height, training_params.img_width])
    f_size = FeatureExtractor.feature_size_test_forward(test_input_tensor)
    loss = RetinaFaceFishBBoxLoss(training_params)
    # 构建先验框
    priorbox = IRFishDetectionBBoxPriors(network_params, training_params, f_size[1])
    detectionRefinement = FishLandmarkRefinementWithBBoxnms(training_params)
    with torch.no_grad():
        priors = priorbox.forward()
        priors = priors.cuda()

    # 训练任务
    ssd_key_pts_task = LitIRFishKeyPtsDetection(FeatureExtractor, loss, priors, detectionRefinement,
                                                training_params,
                                                coco_val_data_api,
                                                coco_test_data_api,
                                                results_path_params)
    fineTuningCallback = FineTuningCallback(training_params)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_callback = ModelCheckpoint(monitor='val_map50',
                                          dirpath=best_model_path,
                                          filename='test-ccb-{epoch:02d}-{val_map50:.2f}',
                                          save_top_k=5,
                                          mode='max',
                                          save_last=True)
    ssd_key_pts_trainer = Trainer(
        accelerator="gpu",
        precision=training_params.precision,  # try 16 with enable_pl_optimizer=False
        # callbacks=[checkpoint_callback, learningrate_callback, early_stopping_callback],
        callbacks=[lr_monitor,fineTuningCallback,checkpoint_callback],
        default_root_dir=results_path_params.ckp_save_path_str,  # where checkpoints are saved to
        # logger=neptune_logger,
        log_every_n_steps=1,
        num_sanity_val_steps=0,
        max_epochs=training_params.max_epochs,
    )

    # 训练

    #train_or_test = "Train"
    train_or_test = "Test"
    if train_or_test == "Train":
        ssd_key_pts_trainer.fit(
            model=ssd_key_pts_task, train_dataloaders=train_data_loader
            , val_dataloaders=valid_data_loader
            # ,ckpt_path=ckp_file_path
        )
    else:
        ssd_key_pts_trainer.test(model=ssd_key_pts_task, ckpt_path=ckp_file_path, dataloaders=test_data_loader)



