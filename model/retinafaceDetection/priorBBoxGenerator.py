import torch
from itertools import product as product
import numpy as np
from math import ceil


class IRFishDetectionBBoxPriors(object):
    def __init__(self, net_cfg, param, f_size):
        super(IRFishDetectionBBoxPriors, self).__init__()
        #self.min_sizes = cfg['min_sizes']
        self.prior_box_sizes = net_cfg.prior_box_sizes
        self.steps = net_cfg.steps
        self.clip = net_cfg.clip
        self.image_size = [param.img_height, param.img_width]
        self.feature_maps = f_size

    def forward(self):
        anchors = []
        for k, f in enumerate(self.feature_maps):
            #min_sizes = self.min_sizes[k]
            prior_box_sizes = self.prior_box_sizes[k]
            for i, j in product(range(f[0]), range(f[1])):
                for a_prior_box_size in prior_box_sizes:
                    s_kx = a_prior_box_size[0] / self.image_size[1]
                    s_ky = a_prior_box_size[1] / self.image_size[0]
                    dense_cx = [x * self.steps[k] / self.image_size[1] for x in [j + 0.5]]
                    dense_cy = [y * self.steps[k] / self.image_size[0] for y in [i + 0.5]]
                    for cy, cx in product(dense_cy, dense_cx):
                        anchors += [cx, cy, s_kx, s_ky]

                # for min_size in min_sizes:
                #     s_kx = min_size / self.image_size[1]
                #     s_ky = min_size / self.image_size[0]
                #     dense_cx = [x * self.steps[k] / self.image_size[1] for x in [j + 0.5]]
                #     dense_cy = [y * self.steps[k] / self.image_size[0] for y in [i + 0.5]]
                #     for cy, cx in product(dense_cy, dense_cx):
                #         anchors += [cx, cy, s_kx, s_ky]

        # back to torch land
        output = torch.Tensor(anchors).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output