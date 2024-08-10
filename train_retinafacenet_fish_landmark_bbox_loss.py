from __future__ import print_function
import os
import torch
import torch.utils.data as data
from torchvision import transforms
import time
import datetime
import math
import cv2
from config.config_retinaface_net import cfg_mnet, cfg_re50
from model.retinafaceDetection.FishLandmarkRefinementWithBBoxnms import FishLandmarkRefinementWithBBoxnms
from model.retinafaceDetection.RetinaFaceFishLandmarkDetectionNet3Layer import RetinaFaceLandmarkDetectionNet3Layer
from model.retinafaceDetection.RetinaFaceFishLandmarkDetectionNet2Layer import RetinaFaceLandmarkDetectionNet2Layer
from model.retinafaceDetection.priorBBoxGenerator import IRFishDetectionBBoxPriors
from model.retinafaceDetection.RetinaFaceFishBBoxLoss import RetinaFaceFishBBoxLoss
from utiles import utils_fish_landmark_detection
from dataset import IRFishDataset
from utiles import displayTool
import pathlib
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import Callback
from torchmetrics.detection import MeanAveragePrecision
from pytorch_lightning.callbacks import LearningRateMonitor
import torch.nn.functional as F

color_green = (0, 255, 0)
color_red = (255, 0, 0)
color_blue = (0, 0, 255)
color_white = (255, 255, 255)


class LitIRFishKeyPtsDetection(pl.LightningModule):
    def __init__(self, backbone, loss, priors, detector, params, results_save_path_for_test=None,
                 results_save_path_for_valid=None, results_save_path_for_train=None,
                 results_save_path_for_pos_sample=None):
        super().__init__()
        self.backbone = backbone
        self.loss = loss
        self.priors = priors
        self.detector = detector
        self.pointColorMap = {0: 'red', 1: 'orange', 2: 'yellow', 3: 'green'}
        self.train_count = 0
        self.results_save_path_for_test = results_save_path_for_test
        self.results_save_path_for_valid = results_save_path_for_valid
        self.results_save_path_for_train = results_save_path_for_train
        self.results_save_path_for_pos_sample = results_save_path_for_pos_sample

        self.params = params
        self.train_count = 0
        self.validation_count = 0
        self.detection_metric_for_valid = MeanAveragePrecision(iou_type="bbox")
        self.train_results_save_interval = 5
        self.validation_results_save_interval = 5
        self.max_map50_score_validation = 0


    def training_step(self, batch, batch_idx):
        # print(f'self.current_epoch is {self.current_epoch}')
        self.backbone.set_phase_train()
        data, label, source_imgs, img_file_path_list = batch
        out = self.backbone(data)
        loss_l, loss_c, loss_landm, pos_sample_index = self.loss(out, self.priors, label)
        loss_l = loss_l
        loss_c = loss_c
        loss_landm = loss_landm  # * 3
        loss = loss_l + loss_c  # + loss_landm
        self.log("train_loss_l", loss_l)
        self.log("train_loss_c", loss_c)
        self.log("train_loss_landm", loss_landm)
        self.log("train_loss", loss)
        print(f'loss score is {loss},loss_l is {loss_l},loss_c is {loss_c},loss_landm is {loss_landm}')
        self.train_count = self.train_count + 1

        # 显示gt和正样本,及正样本对应landmark的回归情况
        if self.results_save_path_for_pos_sample is not None:
            with torch.no_grad():
                for index_batch, a_img in enumerate(source_imgs):
                    pos_reg_encoded_bbox = out[0][index_batch][pos_sample_index].detach().cpu().clone().squeeze()
                    pos_prior_bbox = self.priors[pos_sample_index].detach().cpu().clone().squeeze()
                    pos_reg_decoded_bbox = utils_fish_landmark_detection.decode_bbox(
                        pos_reg_encoded_bbox, pos_prior_bbox, self.params.variance)
                    display_img = a_img.copy()
                    batch_label = label[index_batch].detach().clone()
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
                    # 显示正样本先验
                    displayTool.display_normalized_bbox(utils_fish_landmark_detection.point_form(pos_prior_bbox), display_img, color_red)
                    # 显示正样本栅格对应的预测值
                    displayTool.display_normalized_bbox(pos_reg_decoded_bbox, display_img, color_blue)
                    img_name = img_file_path_list[index_batch].stem
                    display_img = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
                    cv2.imwrite(self.results_save_path_for_pos_sample + '/' + img_name + '_epoch_' + str(
                        self.current_epoch) + '_batchIndex_' + str(batch_idx) + '.bmp', display_img)

        # 保存训练集的训练结果
        if self.current_epoch % self.train_results_save_interval == 0:
            if self.results_save_path_for_train is not None:
                data_for_show = data.detach().clone()
                self.backbone.set_phase_eval()
                with torch.no_grad():
                    loc, conf, landms = self.backbone(data_for_show)
                    for index in range(loc.size(0)):
                        a_loc = loc[index]
                        a_conf = conf[index]
                        a_landms = landms[index]
                        a_gt = label[index]
                        a_img_data = source_imgs[index]
                        detected_result = self.detector.forward(a_loc, a_conf, a_landms, self.priors)
                        a_img_file_path = img_file_path_list[index]
                        a_img_name = a_img_file_path.stem + '_train_' + str(self.train_count) + 'batch_index_' + str(
                            batch_idx)
                        # a_img_data = (a_img_data.detach().clone().cpu().int().numpy().transpose(1, 2, 0))
                        # a_img_data = np.ascontiguousarray(a_img_data)
                        # gt
                        displayTool.display_normalized_bbox(a_gt[:, 0:4], a_img_data, color_green)
                        displayTool.display_normalized_landmark_in_polygon_in_img(a_gt[:, 4:12], a_img_data,
                                                                                  color_green)
                        # detection results
                        displayTool.display_bbox_and_landmark_in_polygon_and_score(detected_result, a_img_data, color_red, color_white)
                        save_img_name = self.results_save_path_for_train + '/' + a_img_name + '.bmp'
                        display_img = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
                        cv2.imwrite(save_img_name, a_img_data)

        return loss

    # def validation_step(self, batch, batch_idx):
    #     self.validation_count = self.validation_count + 1
    #     if batch_idx == 0:
    #         self.detection_metric_for_valid.reset()
    #     self.backbone.set_phase_eval()
    #     print(self.backbone.phase)
    #     data, label, source_imgs, img_file_path_list = batch
    #     loc, conf, landms = self.backbone(data)
    #
    #     # 获得验证结果
    #     for index in range(loc.size(0)):
    #         a_loc = loc[index]
    #         a_conf = conf[index]
    #         a_landms = landms[index]
    #         a_img_data = source_imgs[index]
    #         a_gt = label[index]
    #         img_height = a_img_data.shape[0]
    #         img_width = a_img_data.shape[1]
    #         detected_result = self.detector.forward(a_loc, a_conf, a_landms, self.priors)
    #         # 计算mAP
    #         predict_dict_for_mAP = utils_fish_landmark_detection.detected_results_13_to_MeanAveragePrecision_preds_dict(
    #             detected_result)
    #         target_dict_for_mAP = utils_fish_landmark_detection.detected_results_gt_to_MeanAveragePrecision_target_dict(
    #             a_gt, img_height, img_width)
    #         self.detection_metric_for_valid.update(predict_dict_for_mAP, target_dict_for_mAP)
    #         if self.results_save_path_for_valid is not None:
    #             a_img_file_path = img_file_path_list[index]
    #             a_img_name = a_img_file_path.stem + 'train_' + str(self.validation_count) + 'batch_index_' + str(
    #                 batch_idx)
    #             # a_img_data = (a_img_data.detach().clone().cpu().int().numpy().transpose(1, 2, 0))
    #             # a_img_data = np.ascontiguousarray(a_img_data)
    #             displayTool.display_IR_fish_detected_results_bbox_and_pts(detected_result, a_img_data)
    #             displayTool.display_IR_fish_gt_bbox_and_pts(a_gt, a_img_data)
    #             save_img_name = self.results_save_path_for_valid + '/' + a_img_name + '.bmp'
    #             if self.current_epoch % self.validation_results_save_interval == 0:
    #                 cv2.imwrite(save_img_name, a_img_data)
    #
    #     if batch_idx == (self.trainer.num_val_batches[0] - 1):
    #         mAP = self.detection_metric_for_valid.compute()
    #         print('\n')
    #         print('validation')
    #         print(f'mAP = {mAP["map"]}, mAP50 = {mAP["map_50"]}, mAP75 = {mAP["map_75"]}')
    #         if mAP["map_50"] > self.max_map50_score_validation:
    #             self.max_map50_score_validation = mAP["map_50"]
    #             print(f'cur optimal score: mAP = {mAP["map"]}, mAP50 = {mAP["map_50"]}, mAP75 = {mAP["map_75"]}')
    #         print(f'cur optimal score: mAP50 = {self.max_map50_score_validation}')

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
                a_img_data = (a_img_data.detach().clone().cpu().int().numpy().transpose(1, 2, 0))
                a_img_data = np.ascontiguousarray(a_img_data)
                displayTool.display_IR_fish_detected_results_bbox_and_pts(detected_result, a_img_data)
                displayTool.display_IR_fish_gt_bbox_and_pts(a_gt, a_img_data)
                save_img_name = self.results_save_path_for_test + '/' + a_img_name + '.bmp'
                cv2.imwrite(save_img_name, a_img_data)

        if batch_idx == (self.trainer.num_val_batches[0] - 1):
            mAP = self.detection_metric_for_valid.compute()
            print('\n')
            print('test')
            print(f'mAP = {mAP["map"]}, mAP50 = {mAP["map_50"]}, mAP75 = {mAP["map_75"]}')

    def configure_optimizers(self):
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
    batch_size: int = 3
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
    max_epochs: int = 250
    patience: int = 50
    img_height: int = int(1080)
    img_width: int = int(1920)
    train_or_test = "Train"
    conf_thresh_for_test = 0.51
    nms_thresh_for_test = 0.4
    variance = [1, 1]
    overlap_threshold = 0.5
    neg_pos_ratio = 2
    landmark_size = 4


if __name__ == '__main__':
    # train with pyTorch lightning
    # train_label_path_str = ('H:/code/python/AboveWaterIRFishDetection1.0.0/data/experimentData1.0.0'
    #                         '/valid_keypoints_object_label/train_checked')
    train_label_path_str = 'H:/code/python/IRFishDetection2.0.0/dataset2.1/single_valid_keypoints_label_for_test'
    train_img_path_str = 'H:/code/python/AboveWaterIRFishDetection1.0.0/data/experimentData1.0.0/img_test_and_train'
    test_label_path_str = ('H:/code/python/AboveWaterIRFishDetection1.0.0/data/experimentData1.0.0'
                           '/valid_keypoints_object_label/test_checked')
    test_img_path_str = 'H:/code/python/AboveWaterIRFishDetection1.0.0/data/experimentData1.0.0/img_test_and_train'
    ckp_save_dir = 'H:/code/python/Pytorch_Retinaface-aboveWaterIRFishDetection/ckp_save'
    batch_train_process_results_display_dir = ('H:/code/python/AboveWaterIRFishDetection1.0.0/data/experimentData1.0.0'
                                               '/batch_train_process_results_display')
    priors_key_pts_save_path = 'H:/code/python/AboveWaterIRFishDetection1.0.0/data/experimentData1.0.0/prior_key_pts/prior_key_pts.pt'
    scale_parameters_save_path = 'H:/code/python/AboveWaterIRFishDetection1.0.0/data/experimentData1.0.0/prior_key_pts/prior_scale_parameters.pt'

    results_save_path_validation = "H:/code/python/IRFishDetection2.0.0/results/bboxLoss/valid"
    results_save_path_test = "H:/code/python/IRFishDetection2.0.0/results/bboxLoss/test"
    results_save_path_train = "H:/code/python/IRFishDetection2.0.0/results/bboxLoss/train"
    results_save_path_pos_sample = 'H:/code/python/IRFishDetection2.0.0/results/bboxLoss/pos_sample'

    results_save_path_validation = None
    # results_save_path_test = None
    # results_save_path_train = None
    # results_save_path_pos_sample = None

    train_label_path = pathlib.Path(train_label_path_str)
    train_img_path = pathlib.Path(train_img_path_str)
    test_label_path = pathlib.Path(test_label_path_str)
    test_img_path = pathlib.Path(test_img_path_str)

    param = Params()

    train_and_valid_label_files = utils_fish_landmark_detection.get_filenames_of_path(train_label_path)
    train_size = int(np.ceil(len(train_and_valid_label_files) * 0.9))
    train_label_files = train_and_valid_label_files[0:train_size]
    # valid_lable_files = train_and_valid_label_files[train_size:]
    valid_lable_files = utils_fish_landmark_detection.get_filenames_of_path(test_label_path)
    test_label_files = utils_fish_landmark_detection.get_filenames_of_path(test_label_path)

    # transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize(utils_ir_fish_detection.ir_fish_img_means,
    #                          utils_ir_fish_detection.ir_fish_img_stds)
    # ])
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_dataset = IRFishDataset.IRFishDataset(train_img_path, train_label_files, transform)
    valid_dataset = IRFishDataset.IRFishDataset(train_img_path, valid_lable_files, transform)
    test_dataset = IRFishDataset.IRFishDataset(test_img_path, test_label_files, transform)

    train_data_loader = data.DataLoader(train_dataset, param.batch_size, shuffle=True, num_workers=param.num_workers,
                                        collate_fn=IRFishDataset.detection_collate)
    valid_data_loader = data.DataLoader(valid_dataset, param.batch_size, shuffle=True, num_workers=param.num_workers,
                                        collate_fn=IRFishDataset.detection_collate)
    test_data_loader = data.DataLoader(test_dataset, param.batch_size, shuffle=True, num_workers=param.num_workers,
                                       collate_fn=IRFishDataset.detection_collate)

    # 构建网络
    cfg = cfg_re50
    device = torch.device("cpu" if param.CPU else "cuda")
    FeatureExtractor = RetinaFaceLandmarkDetectionNet2Layer(cfg=cfg)
    FeatureExtractor.print_paramter_grad_state()
    test_input_tensor = torch.zeros([1, 3, param.img_height, param.img_width])
    f_size = FeatureExtractor.feature_size_test_forward(test_input_tensor)
    loss = RetinaFaceFishBBoxLoss(param)
    priorbox = IRFishDetectionBBoxPriors(cfg, param, f_size[1])
    detectionRefinement = FishLandmarkRefinementWithBBoxnms(param)
    with torch.no_grad():
        priors = priorbox.forward()
        priors = priors.cuda()

    # 训练任务
    ssd_key_pts_task = LitIRFishKeyPtsDetection(FeatureExtractor, loss, priors, detectionRefinement, param,
                                                results_save_path_test,
                                                results_save_path_validation,
                                                results_save_path_train,
                                                results_save_path_pos_sample)
    fineTuningCallback = FineTuningCallback(param)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    ssd_key_pts_trainer = Trainer(
        accelerator="gpu",
        precision=param.precision,  # try 16 with enable_pl_optimizer=False
        # callbacks=[checkpoint_callback, learningrate_callback, early_stopping_callback],
        # callbacks=[lr_monitor, fineTuningCallback],
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
            model=ssd_key_pts_task, train_dataloaders=train_data_loader, val_dataloaders=valid_data_loader
        )
    else:
        ssd_key_pts_trainer.test(ckpt_path="best", dataloaders=test_data_loader)
    # ssd_key_pts_trainer.test(ckpt_path="best", dataloaders=test_data_loader)
