import torch
import torch.nn as nn
import torchvision.models._utils as _utils
import torch.nn.functional as F
import numpy as np
from model.netModule import MobileNetV1 as MobileNetV1
from model.netModule import FPN3 as FPN3
from model.netModule import SSH as SSH
import math





class ClassHead(nn.Module):
    def __init__(self, inchannels=512, num_anchors=3):
        super(ClassHead, self).__init__()
        self.num_anchors = num_anchors
        self.conv1x1 = nn.Conv2d(inchannels, self.num_anchors * 2, kernel_size=(1, 1), stride=1, padding=0)

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()

        return out.view(out.shape[0], -1, 2)


class BboxHead(nn.Module):
    def __init__(self, inchannels=512, num_anchors=3):
        super(BboxHead, self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels, num_anchors * 4, kernel_size=(1, 1), stride=1, padding=0)

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()

        return out.view(out.shape[0], -1, 4)


class LandmarkHead(nn.Module):
    def __init__(self, inchannels=512, num_anchors=3, f_size=5):
        super(LandmarkHead, self).__init__()
        self.f_size = f_size
        self.conv1x1 = nn.Conv2d(inchannels, num_anchors * self.f_size, kernel_size=(1, 1), stride=1,
                                 padding=0)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.conv1x1(x)
        # out = self.sigmoid(out)
        out = out.permute(0, 2, 3, 1).contiguous()

        return out.view(out.shape[0], -1, self.f_size)


class RetinaFaceLandmarkPCADetectionNet3Layer(nn.Module):
    def __init__(self, cfg=None, phase='train'):
        """
        :param cfg:  Network related settings.
        :param phase: train or test.
        """
        super(RetinaFaceLandmarkPCADetectionNet3Layer, self).__init__()
        self.phase = phase
        backbone = None
        if cfg['name'] == 'mobilenet0.25':
            backbone = MobileNetV1()
            if cfg['pretrain']:
                checkpoint = torch.load("./weights/mobilenetV1X0.25_pretrain.tar", map_location=torch.device('cpu'))
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in checkpoint['state_dict'].items():
                    name = k[7:]  # remove module.
                    new_state_dict[name] = v
                # load params
                backbone.load_state_dict(new_state_dict)
        elif cfg['name'] == 'Resnet50':
            import torchvision.models as models
            backbone = models.resnet50(pretrained=cfg['pretrain'])
        elif cfg['name'] == 'vgg16':
            import torchvision.models as models
            from torchvision.models import vgg16, VGG16_Weights
            backbone = vgg16(weights=VGG16_Weights.DEFAULT).features

        self.body = _utils.IntermediateLayerGetter(backbone, cfg['return_layers'])
        for a_model_layer_name in self.body:
            a_layer = self.body[a_model_layer_name]
            for a_param_name, a_param in a_layer.named_parameters():
                a_param.requires_grad = False

        in_channels_stage2 = cfg['in_channel']
        # in_channels_list = [
        #     in_channels_stage2 * 2,
        #     in_channels_stage2 * 4,
        #     in_channels_stage2 * 8,
        # ]
        in_channels_list = cfg['in_channels_list']
        out_channels = cfg['out_channel']
        self.fpn = FPN3(in_channels_list, out_channels)
        self.ssh1 = SSH(out_channels, out_channels)
        self.ssh2 = SSH(out_channels, out_channels)
        self.ssh3 = SSH(out_channels, out_channels)

        self.ClassHead = self._make_class_head(fpn_num=3, inchannels=cfg['out_channel'],
                                               anchor_num=cfg['prior_num_in_a_cell'])
        self.BboxHead = self._make_bbox_head(fpn_num=3, inchannels=cfg['out_channel'],
                                             anchor_num=cfg['prior_num_in_a_cell'])
        self.LandmarkHead = self._make_landmark_head(fpn_num=3, inchannels=cfg['out_channel'],
                                                     anchor_num=cfg['prior_num_in_a_cell'],
                                                     f_size=cfg['pca_feature_size'])

    def _make_class_head(self, fpn_num, inchannels, anchor_num):
        classhead = nn.ModuleList()
        for i in range(fpn_num):
            classhead.append(ClassHead(inchannels, anchor_num))
        return classhead

    def _make_bbox_head(self, fpn_num, inchannels, anchor_num):
        bboxhead = nn.ModuleList()
        for i in range(fpn_num):
            bboxhead.append(BboxHead(inchannels, anchor_num))
        return bboxhead

    def _make_landmark_head(self, fpn_num, inchannels, anchor_num, f_size):
        landmarkhead = nn.ModuleList()
        for i in range(fpn_num):
            landmarkhead.append(LandmarkHead(inchannels, anchor_num, f_size))
        return landmarkhead

    def forward(self, inputs):
        out = self.body(inputs)

        # FPN
        fpn = self.fpn(out)

        # SSH
        feature1 = self.ssh1(fpn[0])
        feature2 = self.ssh2(fpn[1])
        feature3 = self.ssh3(fpn[2])
        features = [feature1, feature2, feature3]

        bbox_regressions = torch.cat([self.BboxHead[i](feature) for i, feature in enumerate(features)], dim=1)
        classifications = torch.cat([self.ClassHead[i](feature) for i, feature in enumerate(features)], dim=1)
        ldm_regressions = torch.cat([self.LandmarkHead[i](feature) for i, feature in enumerate(features)], dim=1)

        if self.phase == 'train':
            output = (bbox_regressions, classifications, ldm_regressions)
        else:
            output = (bbox_regressions, F.softmax(classifications, dim=-1), ldm_regressions)
        return output

    def feature_size_test_forward(self, inputs):
        out = self.body(inputs)

        # FPN
        fpn = self.fpn(out)

        # SSH
        feature1 = self.ssh1(fpn[0])
        feature2 = self.ssh2(fpn[1])
        feature3 = self.ssh3(fpn[2])

        input_height = inputs.size(2)
        input_width = inputs.size(3)
        f1_height = feature1.size(2)
        f1_width = feature1.size(3)
        f2_height = feature2.size(2)
        f2_width = feature2.size(3)
        f3_height = feature3.size(2)
        f3_width = feature3.size(3)

        scale1 = np.round(input_height / f1_height)
        scale2 = np.round(input_height / f2_height)
        scale3 = np.round(input_height / f3_height)

        return ([input_height, input_width],
                [[f1_height, f1_width], [f2_height, f2_width], [f3_height, f3_width]],
                [scale1, scale2, scale3])

    def set_phase_train(self):
        self.train()
        self.phase = 'train'

    def set_phase_eval(self):
        # self.eval()
        self.phase = 'eval'

    def print_paramter_grad_state(self):
        for a_model_layer_name in self.body:
            a_layer = self.body[a_model_layer_name]
            for a_param_name, a_param in a_layer.named_parameters():
                print('_' * 20)
                print(f'body net param name is {a_param_name}')
                print(f'requires_grad is {a_param.requires_grad}')

        for a_param_name, a_param in self.fpn.named_parameters():
            print('_' * 20)
            print(f'FPN net param name is {a_param_name}')
            print(f'requires_grad is {a_param.requires_grad}')

        for a_param_name, a_param in self.ssh1.named_parameters():
            print('_' * 20)
            print(f'SSH1 net param name is {a_param_name}')
            print(f'requires_grad is {a_param.requires_grad}')

        for a_param_name, a_param in self.ssh2.named_parameters():
            print('_' * 20)
            print(f'SSH2 net param name is {a_param_name}')
            print(f'requires_grad is {a_param.requires_grad}')

        # for a_param_name, a_param in self.ssh3.named_parameters():
        #     print('_' * 20)
        #     print(f'SSH3 net param name is {a_param_name}')
        #     print(f'requires_grad is {a_param.requires_grad}')

        pass

# for a_model_layer in test_model_resnet_34_6_2:
#     print('_' * 20)
#     print(f'cur layer count is {cur_layer_count}')
#     print(f'resnet34 layer name is {type(a_model_layer)}')
#     for a_param_name, a_param in a_model_layer.named_parameters():
#         print(f'resnet34 param name is {a_param_name}')
#         print(f'requires_grad is {a_param.requires_grad}')
#     cur_layer_count = cur_layer_count + 1
