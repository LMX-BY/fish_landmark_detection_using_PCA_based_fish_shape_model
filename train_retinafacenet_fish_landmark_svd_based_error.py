from __future__ import print_function
import os
import torch
import torch.utils.data as data
from torchvision import transforms
import time
import datetime
import math
import cv2
from config.config_retinaface_net import cfg_vgg16_pca_2_layer, cfg_vgg16_pca_3_layer
from model.retinafaceDetection.RetinaFaceFishLandmarkPCADetectionNet2Layer import RetinaFaceLandmarkPCADetectionNet2Layer
from model.retinafaceDetection.RetinaFaceFishLandmarkPCADetectionNet3Layer import RetinaFaceLandmarkPCADetectionNet3Layer
from model.retinafaceDetection.RetinaFaceFishLandmarkDetectionPriors import RetinaFaceLandmarkDetectionPriors
from model.retinafaceDetection.RetinaFaceLandmarkPCALossWithUnknownMask import RetinaFaceLandmarkPCALossWithUnknownMask
from model.retinafaceDetection.FishLandmarkRefinementWithLandmarkPCAnms import FishLandmarkRefinementWithLandmarkPCAnms
from model.retinafaceDetection import priorLandmarkGenerator
from utiles import displayTool
import matplotlib.pyplot as plt
from utiles import user_def_transform
from utiles import utils_fish_landmark_detection
from utiles import coco_utils
from utiles import coco_eval
from dataset import IRFishDatasetCoco_16_unknown_mask
import pathlib
import numpy as np
import pytorch_lightning as pl
import torch.nn.functional as F
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import Callback
from torchmetrics.detection import MeanAveragePrecision
from pytorch_lightning.callbacks import LearningRateMonitor

gt_pointColorMap_rgb = {0: (255, 0, 0), 1: (255, 0, 0), 2: (255, 0, 0), 3: (255, 0, 0)}
pos_landmarks_pointColorMap_rgb = {0: (0, 255, 0), 1: (0, 255, 0), 2: (0, 255, 0), 3: (0, 255, 0)}

color_green = (0, 255, 0)
color_red = (255, 0, 0)
color_blue = (0, 0, 255)
color_white = (255, 255, 255)
color_yellow = (255, 255, 0)


class LitIRFishKeyPtsDetection(pl.LightningModule):
    def __init__(self, backbone, loss, priors, detector,
                 params, pca_param_dict,
                 coco_val_data_api, coco_test_data_api,
                 results_save_path_for_test=None, results_save_path_for_valid=None,
                 results_save_path_for_train=None, results_save_path_for_pos_landmarks=None,
                 csv_results_file_path_for_train=None):
        super().__init__()
        self.backbone = backbone
        self.loss = loss
        self.priors = priors
        self.detector = detector
        self.pointColorMap = {0: 'red', 1: 'orange', 2: 'yellow', 3: 'green'}
        self.train_count = 0
        self.pytorch_device = torch.device("cpu" if param.CPU else "cuda")
        self.results_save_path_for_test = results_save_path_for_test
        self.results_save_path_for_valid = results_save_path_for_valid
        self.results_save_path_for_train = results_save_path_for_train
        self.results_save_path_for_pos_landmarks = results_save_path_for_pos_landmarks
        self.csv_results_file_path_for_train = csv_results_file_path_for_train
        self.params = params
        self.train_count = 0
        self.validation_count = 0
        self.detection_metric_for_valid = MeanAveragePrecision(iou_type="bbox")
        self.train_results_save_interval = 40
        self.validation_results_save_interval = 40
        self.max_map50_score_validation = 0
        self.csv_results_file = None
        self.pca_feature_size = params.pca_feature_size
        self.pca_mean = pca_param_dict['mean_feature'].to(self.pytorch_device).to(torch.float32)
        self.pca_features = pca_param_dict['svd_feature'].to(self.pytorch_device).to(torch.float32)[0:self.pca_feature_size,:]
        img_size = torch.Tensor([params.img_width, params.img_height,
                                                  params.img_width, params.img_height,
                                                  params.img_width, params.img_height,
                                                  params.img_width, params.img_height]).to(self.pytorch_device)
        self.img_size_normalized_pca_features = self.pca_features / img_size
        self.singular_values = pca_param_dict['singular_values'].to(self.pytorch_device).to(torch.float32)
        self.pca_features.requires_grad = False
        self.singular_values.requires_grad = False
        self.pca_mean.requires_grad = False

        self.iou_types = ['keypoints']
        self.coco_val_data_api = coco_val_data_api
        self.coco_test_data_api = coco_test_data_api
        self.val_coco_evaluator = coco_eval.CocoEvaluator(self.coco_val_data_api, self.iou_types)
        self.test_coco_evaluator = coco_eval.CocoEvaluator(self.coco_test_data_api, self.iou_types)


        if self.csv_results_file_path_for_train is not None:
            file_name = self.csv_results_file_path_for_train + '/' + datetime.datetime.now().strftime(
                '%Y-%m-%d-%H-%M-%S') + '.csv'
            self.csv_results_file = open(file_name, 'w')

    def training_step(self, batch, batch_idx):
        # print(f'self.current_epoch is {self.current_epoch}')
        self.backbone.set_phase_train()
        data, label_coco, unknown_sample_mask, source_imgs, img_file_names = batch
        img_height = data.size(2)
        img_width = data.size(3)
        label_local = self.label_coco_to_local(label_coco, img_height, img_width)
        out = self.backbone(data)
        focal_loss, loss_c, loss_landm, loss_weight, pos_sample_index, neg_sample_index = self.loss(out, self.priors, label_local, unknown_sample_mask)
        if math.isnan(loss_landm):
            pass
        # pos_sample_index = pos_sample_index.detach().cpu().clone()
        #loss_l = loss_l
        loss_c = loss_c * 5
        loss_landm = loss_landm  # / 100
        # loss = loss_c + loss_landm
        #loss = loss_landm
        loss = loss_c + loss_landm + loss_weight
        #loss = loss_c
        #self.log("train_loss_l", loss_l)
        self.log("train_loss_c", loss_c)
        self.log("train_loss_landm", loss_landm)
        self.log("train_loss_weight", loss_weight)
        self.log("train_loss", loss)
        print(f'loss score is {loss},loss_c is {loss_c},loss_landm is {loss_landm}, loss_weight is {loss_weight}')
        self.train_count = self.train_count + 1

        # #显示gt和正样本,及正样本对应landmark的回归情况
        if self.current_epoch % self.train_results_save_interval == 0:
            if self.results_save_path_for_pos_landmarks is not None:
                for index_batch, a_img in enumerate(source_imgs):
                    pos_sample_index_batch = pos_sample_index[index_batch]
                    pos_reg_encoded_landmarks = out[2][index_batch][pos_sample_index_batch].detach().clone().squeeze()
                    zeros = torch.zeros(pos_reg_encoded_landmarks.shape).to(self.pytorch_device)
                    pos_prior_landmarks = self.priors['prior_landmarks'][
                        pos_sample_index_batch].detach().clone().squeeze()
                    if pos_reg_encoded_landmarks.dim() == 1:
                        pos_reg_encoded_landmarks = pos_reg_encoded_landmarks.unsqueeze(0)
                    pos_reg_decoded_landmarks = utils_fish_landmark_detection.decode_landmarks_with_pca_param(
                        pos_reg_encoded_landmarks, pos_prior_landmarks, self.img_size_normalized_pca_features)
                    # pos_reg_decoded_landmarks = utils_fish_landmark_detection.decode_landmarks_with_pca_param(
                    #     zeros, pos_prior_landmarks, self.img_size_normalized_pca_features)
                    display_img = a_img.copy()
                    gt_landmarks = label_local[index_batch].detach().clone()
                    # pos_score = out[1][index_batch][pos_sample_index].detach().cpu().clone().squeeze()
                    # for a_pos_sample_index in pos_sample_index:
                    #     a_pos_score = out[1][index_batch][a_pos_sample_index]
                    #     a_pos_score = a_pos_score.squeeze()
                    #     print('*' * 10)
                    #     print(f'posIndex: {a_pos_sample_index}')
                    #     print('pos sample score:')
                    #     print(a_pos_score)
                    #     print('after softmax:')
                    #     print(F.softmax(a_pos_score,dim = 1))
                    # for a_pos_score in pos_score:
                    #     print('*'*10)
                    #     print(a_pos_score)
                    #     print('after softmax:')
                    #     print(F.softmax(a_pos_score))
                    # gt
                    displayTool.display_normalized_landmark_in_polygon_in_img(gt_landmarks[:, 4:12], display_img, color_green)
                    # pos prior landmarks
                    # displayTool.display_normalized_landmark_in_polygon_in_img(pos_prior_landmarks, display_img, color_blue)
                    # pos reg decoded landmarks
                    displayTool.display_normalized_landmark_in_polygon_in_img(pos_reg_decoded_landmarks, display_img,
                                                                              color_red)
                    img_name = img_file_names[index_batch]
                    display_img = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
                    cv2.imwrite(self.results_save_path_for_pos_landmarks + '/' + img_name + '_epoch_' + str(
                        self.current_epoch) + '_batchIndex_' + str(batch_idx) + '.bmp', display_img)

                    if self.csv_results_file is not None:
                        pos_sample_index_np = pos_sample_index_batch.detach().cpu().numpy()
                        neg_sample_index_np = neg_sample_index.detach().cpu().numpy()
                        pos_sample_score_np = out[1][index_batch][pos_sample_index_batch].detach().cpu().numpy().squeeze()
                        neg_sample_score_np = out[1][index_batch][neg_sample_index].detach().cpu().numpy().squeeze()
                        pos_sample_index_np = pos_sample_index_np.transpose(1, 0)
                        neg_sample_index_np = neg_sample_index_np.transpose(1, 0)
                        pos_sample_score_np = pos_sample_score_np.transpose(1, 0)
                        neg_sample_score_np = neg_sample_score_np.transpose(1, 0)
                        np.savetxt(self.csv_results_file, pos_sample_index_np, delimiter=',', fmt='%.2f')
                        np.savetxt(self.csv_results_file, pos_sample_score_np, delimiter=',', fmt='%.2f')
                        np.savetxt(self.csv_results_file, neg_sample_index_np, delimiter=',', fmt='%.2f')
                        np.savetxt(self.csv_results_file, neg_sample_score_np, delimiter=',', fmt='%.2f')

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
                        a_gt = label_local[index].detach().clone()
                        a_img_data = source_imgs[index]
                        # 输出正样本的分类分数
                        # for a_pos_sample_index in pos_sample_index:
                        #     print('*'*10)
                        #     print(f'posIndex: {a_pos_sample_index}')
                        #     print('pos sample score:')
                        #     print(a_conf[a_pos_sample_index])
                        detected_result, detected_priors, index_tensor_for_prior = self.detector.forward(a_loc, a_conf,
                                                                                                         a_landms,
                                                                                                         self.priors)
                        img_name = img_file_names[index]
                        img_name = img_name + '_train_' + str(self.train_count) + '_batch_index_' + str(
                            batch_idx)
                        # a_img_data = (a_img_data.detach().clone().cpu().int().np().transpose(1, 2, 0))
                        # a_img_data = np.ascontiguousarray(a_img_data)
                        # gt
                        displayTool.display_normalized_landmark_in_polygon_in_img(a_gt[:, 4:12], a_img_data, color_green)
                        # detected results
                        displayTool.display_landmark_in_polygon_in_img(detected_result[:, 0:8], a_img_data, color_red)
                        # detected priors
                        # a_img_data_2 = a_img_data.copy()
                        # displayTool.display_normalized_landmark_in_polygon_in_img(detected_priors, a_img_data,
                        #                                                           color_blue)
                        # detected priors by index
                        # detected_priors_by_index = self.priors['prior_landmarks'][index_tensor_for_prior].detach().cpu().clone().squeeze()
                        # displayTool.display_normalized_landmark_in_polygon_in_img(detected_priors_by_index, a_img_data_2,
                        #                                                           color_yellow)
                        a_img_data = cv2.cvtColor(a_img_data, cv2.COLOR_BGR2RGB)
                        # a_img_data_2 = cv2.cvtColor(a_img_data_2, cv2.COLOR_BGR2RGB)
                        save_img_name = self.results_save_path_for_train + '/' + img_name + '.bmp'
                        # save_img_name_2 = self.results_save_path_for_train + '/' + a_img_name + '_2.bmp'
                        cv2.imwrite(save_img_name, a_img_data)
                        # cv2.imwrite(save_img_name_2, a_img_data_2)
                        if self.csv_results_file is not None:
                            detected_index_np = index_tensor_for_prior.detach().cpu().numpy()
                            detected_score_np = a_conf[index_tensor_for_prior].squeeze().detach().cpu().numpy()
                            detected_index_np = detected_index_np.transpose(1, 0)
                            detected_score_np = detected_score_np.transpose(1, 0)
                            np.savetxt(self.csv_results_file, detected_index_np, delimiter=',', fmt='%.2f')
                            np.savetxt(self.csv_results_file, detected_score_np, delimiter=',', fmt='%.2f')
                            seg_line = '*' * 10 + '\n'
                            self.csv_results_file.write(seg_line)

        return loss

    def validation_step(self, batch, batch_idx):
        self.validation_count = self.validation_count + 1
        if batch_idx == 0:
            self.detection_metric_for_valid.reset()
        self.backbone.set_phase_eval()
        print(self.backbone.phase)

        data, label_coco, unknown_sample_mask, source_imgs, img_file_names = batch
        img_height = data.size(2)
        img_width = data.size(3)
        label_local = self.label_coco_to_local(label_coco, img_height, img_width)
        loc, conf, landms = self.backbone(data)

        #获得验证结果
        for index in range(loc.size(0)):
            a_loc = loc[index]
            a_conf = conf[index]
            a_landms = landms[index]
            a_img_data = source_imgs[index]
            a_gt = label_local[index]
            a_label_coco = [label_coco[index]]
            img_height = a_img_data.shape[0]
            img_width = a_img_data.shape[1]
            detected_result, detected_priors, index_tensor_for_prior = self.detector.forward(a_loc, a_conf, a_landms, self.priors)
            #计算mAP
            detected_result_coco = self.detected_result_local_to_coco(detected_result)
            detected_result_coco = [detected_result_coco]
            # 計算評價
            cpu_device = torch.device("cpu")
            detected_result_coco = [{k: v.to(cpu_device) for k, v in t.items()} for t in detected_result_coco]
            res = {target["image_id"].item(): output for target, output in zip(a_label_coco, detected_result_coco)}
            self.val_coco_evaluator.update(res)

            if self.results_save_path_for_valid is not None:
                img_name = img_file_names[index]
                img_name = img_name + '_train_' + str(self.train_count) + '_batch_index_' + str(
                    batch_idx)
                # a_img_data = (a_img_data.detach().clone().cpu().int().np().transpose(1, 2, 0))
                # a_img_data = np.ascontiguousarray(a_img_data)
                # gt
                displayTool.display_normalized_landmark_in_polygon_in_img(a_gt[:, 4:12], a_img_data, color_green)
                # detected results
                displayTool.display_landmark_in_polygon_in_img(detected_result[:, 0:8], a_img_data, color_red)
                # detected priors
                # a_img_data_2 = a_img_data.copy()
                displayTool.display_normalized_landmark_in_polygon_in_img(detected_priors, a_img_data,
                                                                          color_blue)

                save_img_name = self.results_save_path_for_valid + '/' + img_name + '.bmp'
                if self.current_epoch % self.validation_results_save_interval == 0:
                    cv2.imwrite(save_img_name, a_img_data)

        if batch_idx == (self.trainer.num_val_batches[0] - 1):
            self.val_coco_evaluator.synchronize_between_processes()
            # accumulate predictions from all images
            self.val_coco_evaluator.accumulate()
            self.val_coco_evaluator.summarize()
            coco_info_bbox = self.val_coco_evaluator.coco_eval[self.iou_types[0]].stats.tolist()
            self.log("AP_0.5:0.95", coco_info_bbox[0])
            self.log("AP_0.5", coco_info_bbox[1])
            self.log("AP_0.75", coco_info_bbox[2])
            self.log("AR_0.5:0.95", coco_info_bbox[5])
            self.log("AR_0.5", coco_info_bbox[6])
            self.log("AR_0.75", coco_info_bbox[7])
            print('*' * 10 + 'landmarks' + '*' * 10)
            # coco_info_bbox = self.val_coco_evaluator.coco_eval[self.iou_types[1]].stats.tolist()
            # self.train_coco_evaluator.synchronize_between_processes()
            self.val_coco_evaluator = coco_eval.CocoEvaluator(self.coco_val_data_api, self.iou_types)

    def test_step(self, batch, batch_idx):
        self.backbone.eval()
        data, label, source_imgs, img_file_path_list = batch
        loc, conf, landms = self.backbone(data)
        detected_results = self.detector.forward(loc, conf, landms, self.priors)

        for index in range(loc.size(0)):
            a_loc = loc[index]
            a_conf = conf[index]
            a_landms = landms[index]
            a_img_data = source_imgs[index]
            a_gt = label[index]
            img_height = a_img_data.shape[0]
            img_width = a_img_data.shape[1]
            detected_result = self.detector.forward(a_loc, a_conf, a_landms, self.priors)
            # 计算mAP
            predict_dict_for_mAP = utils_fish_landmark_detection.detected_results_13_to_MeanAveragePrecision_preds_dict(
                detected_result)
            target_dict_for_mAP = utils_fish_landmark_detection.detected_results_gt_to_MeanAveragePrecision_target_dict(
                a_gt,
                img_height,
                img_width)
            self.detection_metric_for_valid.update(predict_dict_for_mAP, target_dict_for_mAP)
            if self.results_save_path_for_test is not None:
                a_img_file_path = img_file_path_list[index]
                a_img_name = a_img_file_path.stem + 'train_' + str(self.train_count) + 'batch_index_' + str(batch_idx)
                a_img_data = (a_img_data.detach().clone().cpu().int().np().transpose(1, 2, 0))
                a_img_data = np.ascontiguousarray(a_img_data)
                utils_fish_landmark_detection.display_IR_fish_detected_results(detected_result, a_img_data)
                utils_fish_landmark_detection.display_IR_fish_gt(a_gt, a_img_data)
                save_img_name = self.results_save_path_for_test + '/' + a_img_name + '.bmp'
                cv2.imwrite(save_img_name, a_img_data)

        if batch_idx == (self.trainer.num_val_batches[0] - 1):
            mAP = self.detection_metric_for_valid.compute()
            print('\n')
            print('test')
            print(f'mAP = {mAP["map"]}, mAP50 = {mAP["map_50"]}, mAP75 = {mAP["map_75"]}')

    def configure_optimizers(self):
        # non_frozen_parameters = [p for p in self.backbone.parameters() if p.requires_grad]
        # count = 0
        # for name, parameter in self.backbone.body.named_parameters():
        #     if count < 3:
        #         parameter.requires_grad = False
        #     else:
        #         parameter.requires_grad = True
        #     count = count + 1
        # for name, parameter in self.backbone.body.named_parameters():
        #     print(f'name is {name}')
        #     print(f'required grad is {parameter.requires_grad}')
        non_frozen_parameters = [p for p in self.backbone.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(
            non_frozen_parameters, lr=self.params.lr, momentum=self.params.momentum,
            weight_decay=self.params.weight_decay
        )
        # for a_param_group in optimizer.param_groups:
        #     a_param_group
        #     pass
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.75, patience=30, min_lr=0
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
        key_points_with_visibility_tensor[:, 0:2] = detected_result[:, 0:2]
        key_points_with_visibility_tensor[:, 3:5] = detected_result[:, 2:4]
        key_points_with_visibility_tensor[:, 6:8] = detected_result[:, 4:6]
        key_points_with_visibility_tensor[:, 9:11] = detected_result[:, 6:8]
        results_dict = {'labels': torch.ones(detected_result.shape[0]).int().to(self.pytorch_device),
                        'scores': detected_result[:, -1].to(self.pytorch_device),
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
                optimizer, mode="max", factor=0.75, patience=3, min_lr=0
            )
            trainer.optimizers = [optimizer]
            trainer.lr_schedulers = [lr_scheduler]


class Params:
    batch_size: int = 1
    num_workers = 1
    save_dir: str = None  # checkpoints will be saved to cwd (current working directory)
    # GPU: int = 1  # set to None for cpu training
    CPU = False
    lr: float = 0.001
    momentum = 0.1
    weight_decay = 5e-4
    gamma = 0.1
    precision: int = 32  # 未用
    num_of_classe: int = 2
    seed: int = 42
    max_epochs: int = 300
    patience: int = 50
    img_height: int = int(1080)
    img_width: int = int(1920)
    train_or_test = "Train"
    conf_thresh_for_test = 0.51
    nms_thresh_for_test = 0.05
    pos_sample_score_threshold = 30
    neg_pos_ratio = 2
    variance = [0.1, 0.2]
    landmark_size = 4
    pca_feature_size = 6


if __name__ == '__main__':
    # train with pyTorch lightning
    train_label_path_str = ('H:/code/python/IRFishDetection2.0.0/dataset2.1/poly_and_used_defined_label'
                            '/all_checked_label/user_defined_json_1.0_2.0/train1.1')
    # train_label_path_str = ('H:/code/python/IRFishDetection2.0.0/dataset2.1/someTest'
    #                         '/single_label_file_for_classification_test_2/landmarks')
    train_img_path_str = 'H:/code/python/IRFishDetection2.0.0/dataset2.1/img'
    test_label_path_str = ('H:/code/python/IRFishDetection2.0.0/dataset2.1/poly_and_used_defined_label'
                           '/all_checked_label/user_defined_json_1.0_2.0/test1.1')

    # test_label_path_str = ('H:/code/python/IRFishDetection2.0.0/dataset2.1/someTest'
    #                         '/single_label_file_for_classification_test_2/landmarks')

    # unknown_mask_path_str = ('H:/code/python/IRFishDetection2.0.0/dataset2.1/poly_and_used_defined_label'
    #                          '/all_checked_label/user_defined_json_1.0_2.0/unknown_sample_index')
    unknown_mask_path_str = ('H:/code/python/IRFishDetection2.0.0/dataset2.1/poly_and_used_defined_label'
                             '/all_checked_label/user_defined_json_1.0_2.0/unknown_sample_index_vgg_16_3_layer')
    # unknown_mask_path_str = ('H:/code/python/IRFishDetection2.0.0/dataset2.1/someTest'
    #                                        '/single_label_file_for_classification_test_2/mask')
    test_img_path_str = 'H:/code/python/IRFishDetection2.0.0/dataset2.1/img'
    ckp_save_dir = 'H:/code/python/IRFishDetection2.0.0/results/pcaPointLoss/ckp'

    # priors_file_name = 'H:/code/python/IRFishDetection2.0.0/dataset2.1/priors/retina_face_resnet50_priors.pt'
    # priors_file_name = 'H:/code/python/IRFishDetection2.0.0/dataset2.1/priors/retina_face_vgg16_pca_priors.pt'
    priors_file_name = 'H:/code/python/IRFishDetection2.0.0/dataset2.1/priors/retina_face_vgg16_3_layer_pca_priors.pt'
    pca_param_file_path = 'H:/code/python/IRFishDetection2.0.0/results/svd_feature/used_for_train/svd_features_2.5_1_1'

    valid_results_save_path = "H:/code/python/IRFishDetection2.0.0/results/pcaPointLoss/validation"
    # test_results_save_path = "H:/code/python/IRFishDetection2.0.0/results/pcaPointLoss/test"
    train_results_save_path = "H:/code/python/IRFishDetection2.0.0/results/pcaPointLoss/train"
    pos_landmarks_results_save_path = "H:/code/python/IRFishDetection2.0.0/results/pcaPointLoss/pos_landmarks"
    #ckp_file_path = 'H:/code/python/IRFishDetection2.0.0/results/pcaPointLoss/ckp/lightning_logs/version+vgg16+3layer+lstd+0.1+withnontransrotpca/checkpoints/epoch=42-step=17458.ckpt'
    # csv_results_file_path_for_train = "H:/code/python/IRFishDetection2.0.0/results/pcaPointLoss/csv"

    #valid_results_save_path = None
    # train_results_save_path = None
    test_results_save_path = None
    # pos_landmarks_results_save_path = None
    csv_results_file_path_for_train = None

    train_label_path = pathlib.Path(train_label_path_str)
    train_img_path = pathlib.Path(train_img_path_str)
    test_label_path = pathlib.Path(test_label_path_str)
    test_img_path = pathlib.Path(test_img_path_str)
    unknown_mask_path = pathlib.Path(unknown_mask_path_str)

    param = Params()

    train_and_valid_label_files = utils_fish_landmark_detection.get_filenames_of_path(train_label_path)
    train_size = int(np.ceil(len(train_and_valid_label_files) * 0.9))
    train_label_files = train_and_valid_label_files[0:train_size]
    # valid_lable_files = train_and_valid_label_files[train_size:]
    valid_lable_files = utils_fish_landmark_detection.get_filenames_of_path(test_label_path)
    test_label_files = utils_fish_landmark_detection.get_filenames_of_path(test_label_path)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(utils_fish_landmark_detection.ir_fish_img_means,
                             utils_fish_landmark_detection.ir_fish_img_stds)
    ])
    # transform = transforms.Compose([
    #     transforms.ToTensor()
    # ])

    data_transform = {
        "train": user_def_transform.Compose([user_def_transform.ToTensor(),
                                             user_def_transform.RandomHorizontalFlip(0.5)]),
        "val": user_def_transform.Compose([user_def_transform.ToTensor()])
    }

    train_dataset = IRFishDatasetCoco_16_unknown_mask.IRFishDatasetCoco_16_unknown_mask(train_img_path, unknown_mask_path,
                                                                                        train_label_files, param.landmark_size, data_transform["val"])
    valid_dataset = IRFishDatasetCoco_16_unknown_mask.IRFishDatasetCoco_16_unknown_mask(train_img_path, unknown_mask_path,
                                                                                        valid_lable_files, param.landmark_size, data_transform["val"])
    test_dataset = IRFishDatasetCoco_16_unknown_mask.IRFishDatasetCoco_16_unknown_mask(test_img_path, unknown_mask_path,
                                                                                       test_label_files, param.landmark_size, data_transform["val"])

    train_data_loader = data.DataLoader(train_dataset, param.batch_size, shuffle=True, num_workers=param.num_workers,
                                        collate_fn=IRFishDatasetCoco_16_unknown_mask.detection_collate)
    valid_data_loader = data.DataLoader(valid_dataset, param.batch_size, shuffle=True, num_workers=param.num_workers,
                                        collate_fn=IRFishDatasetCoco_16_unknown_mask.detection_collate)
    test_data_loader = data.DataLoader(test_dataset, param.batch_size, shuffle=True, num_workers=param.num_workers,
                                       collate_fn=IRFishDatasetCoco_16_unknown_mask.detection_collate)
    # test_data = next(iter(train_data_loader))

    coco_val_data_api = coco_utils.get_coco_api_from_dataset(valid_data_loader.dataset)
    coco_test_data_api = coco_utils.get_coco_api_from_dataset(test_data_loader.dataset)

    # 构建网络
    #cfg = cfg_vgg16_pca_2_layer
    cfg = cfg_vgg16_pca_3_layer
    device = torch.device("cpu" if param.CPU else "cuda")
    #FeatureExtractor = RetinaFaceLandmarkPCADetectionNet2Layer(cfg=cfg)
    FeatureExtractor = RetinaFaceLandmarkPCADetectionNet3Layer(cfg=cfg)
    FeatureExtractor.print_paramter_grad_state()

    priors = priorLandmarkGenerator.load_prior_fish_4_landmarks(priors_file_name)
    priors['prior_landmarks'] = priors['prior_landmarks'].to(device)
    priors['prior_landmarks'].requires_grad = False

    pca_param = torch.load(pca_param_file_path)
    loss = RetinaFaceLandmarkPCALossWithUnknownMask(param, pca_param)

    detectionRefinement = FishLandmarkRefinementWithLandmarkPCAnms(param, pca_param)

    # 训练任务
    ssd_key_pts_task = LitIRFishKeyPtsDetection(FeatureExtractor, loss, priors, detectionRefinement,
                                                param, pca_param,
                                                coco_val_data_api, coco_test_data_api,
                                                test_results_save_path, valid_results_save_path,
                                                train_results_save_path, pos_landmarks_results_save_path,
                                                csv_results_file_path_for_train)
    fineTuningCallback = FineTuningCallback(param)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    ssd_key_pts_trainer = Trainer(
        accelerator="gpu",
        precision=param.precision,  # try 16 with enable_pl_optimizer=False
        # callbacks=[checkpoint_callback, learningrate_callback, early_stopping_callback],
        callbacks=[lr_monitor, fineTuningCallback],
        default_root_dir=ckp_save_dir,  # where checkpoints are saved to
        # logger=neptune_logger,
        log_every_n_steps=1,
        num_sanity_val_steps=0,
        max_epochs=param.max_epochs,
    )

    # 训练
    # ssd_key_pts_trainer.fit(
    #     model=ssd_key_pts_task, train_dataloaders=train_data_loader, val_dataloaders=valid_data_loader
    # )
    if param.train_or_test == "Train":
        ssd_key_pts_trainer.fit(
            model=ssd_key_pts_task, train_dataloaders=train_data_loader, val_dataloaders=valid_data_loader #, ckpt_path = ckp_file_path
        )
    else:
        ssd_key_pts_trainer.test(ckpt_path="best", dataloaders=test_data_loader)
    # ssd_key_pts_trainer.test(ckpt_path="best", dataloaders=test_data_loader)
