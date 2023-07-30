#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
File: voxnet.py
Created: 2020-01-21 21:32:40
Author : Yangmaonan
Email : 59786677@qq.com
Description: VoxNet 网络结构
'''

import torch
import torch.nn as nn
from collections import OrderedDict


class VoxNet(nn.Module):
    def __init__(self, n_classes):
        super(VoxNet, self).__init__()
        self.n_classes = n_classes
        self.feat = torch.nn.Sequential(OrderedDict([
            ('conv3d_1', torch.nn.Conv3d(in_channels=1,
                                         out_channels=32, kernel_size=5, stride=2)),
            ('relu1', torch.nn.ReLU()),
            ('drop1', torch.nn.Dropout(p=0.2)),
            ('conv3d_2', torch.nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3)),
            ('relu2', torch.nn.ReLU()),
            ('pool2', torch.nn.MaxPool3d(2)),
            ('drop2', torch.nn.Dropout(p=0.3))
        ]))

        self.mlp_f = torch.nn.Sequential(OrderedDict([
            ('fc1', torch.nn.Linear(1 * 6 * 6 * 6, 512)),
            ('relu1', torch.nn.ReLU()),
            ('drop3', torch.nn.Dropout(p=0.4)),
            ('fc2', torch.nn.Linear(512, 512))
        ]))

        self.mlp_c = torch.nn.Sequential(OrderedDict([
            ('fc1', torch.nn.Linear(32 * 6 * 6 * 6, 128)),
            ('relu1', torch.nn.ReLU()),
            ('drop3', torch.nn.Dropout(p=0.4))
            # ('fc2', torch.nn.Linear(128, self.n_classes))
        ]))
        self.fc3 = nn.Linear(128, n_classes)
        self.fc3over = nn.Linear(128, 100)

    def forward(self, x, global_ft=False):
        x = self.feat(x)
        b, n, _, _, _ = x.shape

        g_ft = x.view(b*n, -1)
        ft = self.mlp_f(g_ft)

        x = x.view(x.size(0), -1)
        x = self.mlp_c(x)
        out = self.fc3(x)
        out_over = self.fc3over(x)
        return [out, out_over], ft.view(b, n, -1)
        # if global_ft:
        #     return x, g_ft
        # else:
        #     return x


if __name__ == "__main__":
    voxnet = VoxNet(32, 10)
    data = torch.rand([256, 1, 32, 32, 32])
    voxnet(data)
