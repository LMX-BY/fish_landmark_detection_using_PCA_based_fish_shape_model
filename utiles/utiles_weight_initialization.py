import torch
import torch.nn as nn


def init_weights_kaiming_uniform(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is None:
            return
        m.bias.data.fill_(0)


def init_weights_kaiming_normal(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is None:
            return
        m.bias.data.fill_(0)
