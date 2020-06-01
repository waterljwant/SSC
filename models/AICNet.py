#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
AICNet
jieli_cn@163.com
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
from .projection_layer import Project2Dto3D
from .DDR import BottleneckDDR2d, BottleneckDDR3d, DownsampleBlock3d


class BasicAIC3d(nn.Module):
    def __init__(self, channel, kernel=(3, 5, 7), dilation=(1, 1, 1), residual=True):
        super(BasicAIC3d, self).__init__()
        self.channel = channel
        self.residual = residual
        self.n = len(kernel)  # number of kernels
        self.conv_mx = nn.Conv3d(channel, 3 * self.n, (1, 1, 1), stride=1, padding=0, bias=False, dilation=1)
        self.softmax = nn.Softmax(dim=2)  # Applies the Softmax function in each axis

        # ---- Convs of each axis
        self.conv_1x1xk = nn.ModuleList()
        self.conv_1xkx1 = nn.ModuleList()
        self.conv_kx1x1 = nn.ModuleList()

        c = channel
        for _idx in range(self.n):
            k = kernel[_idx]
            d = dilation[_idx]
            p = k // 2 * d
            self.conv_1x1xk.append(nn.Conv3d(c, c, (1, 1, k), stride=1, padding=(0, 0, p), bias=True, dilation=(1, 1, d)))
            self.conv_1xkx1.append(nn.Conv3d(c, c, (1, k, 1), stride=1, padding=(0, p, 0), bias=True, dilation=(1, d, 1)))
            self.conv_kx1x1.append(nn.Conv3d(c, c, (k, 1, 1), stride=1, padding=(p, 0, 0), bias=True, dilation=(d, 1, 1)))

    def forward(self, x):
        mx = self.conv_mx(x)  # (BS, 3n, D, H, W)
        _bs, _tn, _d, _h, _w = mx.size()
        mx = mx.view(_bs, 3, -1, _d, _h, _w)  # (BS, 3, n, D, H, W)

        # print("After 'view', mx.size() is: {}".format(mx.size()))
        mx = self.softmax(mx)  # dim=2

        mx_c = torch.unsqueeze(mx, dim=3)  # (BS, 3, n, 1, D, H, W)
        mx_c = mx_c.expand(-1, -1, -1, self.channel, -1, -1, -1)  # (BS, 3, n, c, D, H, W)
        # mx1, mx2, mx3 = torch.split(mx_c, 1, dim=2)  # n x (BS, 3, 1, c, D, H, W)
        mx_list = torch.split(mx_c, 1, dim=2)  # n x (BS, 3, 1, c, D, H, W)

        mx_z_list = []
        mx_y_list = []
        mx_x_list = []
        for i in range(self.n):
            # mx_list[i] = torch.squeeze(mx_list[i], dim=2)  # (BS, 3, c, D, H, W)
            # mx_z, mx_y, mx_x = torch.split(mx_list[i], 1, dim=1)  # 3 x (BS, 1, c, D, H, W)
            mx_z, mx_y, mx_x = torch.split(torch.squeeze(mx_list[i], dim=2), 1, dim=1)  # 3 x (BS, 1, c, D, H, W)
            mx_z_list.append(torch.squeeze(mx_z, dim=1))  # (BS, c, D, H, W)
            mx_y_list.append(torch.squeeze(mx_y, dim=1))  # (BS, c, D, H, W)
            mx_x_list.append(torch.squeeze(mx_x, dim=1))  # (BS, c, D, H, W)

        # ------ x ------
        y_x = None
        for _idx in range(self.n):
            y1_x = self.conv_1x1xk[_idx](x)
            y1_x = F.relu(y1_x, inplace=True)
            y1_x = torch.mul(mx_x_list[_idx], y1_x)
            y_x = y1_x if y_x is None else y_x + y1_x

        # ------ y ------
        y_y = None
        for _idx in range(self.n):
            y1_y = self.conv_1xkx1[_idx](y_x)
            y1_y = F.relu(y1_y, inplace=True)
            y1_y = torch.mul(mx_y_list[_idx], y1_y)
            y_y = y1_y if y_y is None else y_y + y1_y

        # ------ z ------
        y_z = None
        for _idx in range(self.n):
            y1_z = self.conv_kx1x1[_idx](y_y)
            y1_z = F.relu(y1_z, inplace=True)
            y1_z = torch.mul(mx_z_list[_idx], y1_z)
            y_z = y1_z if y_z is None else y_z + y1_z

        y = F.relu(y_z + x, inplace=True) if self.residual else F.relu(y_z, inplace=True)
        return y


class BottleneckAIC3d(nn.Module):
    def __init__(self, c_in, c, c_out, kernel=(3, 5, 7), dilation=(1, 1, 1), residual=True, neighbours=0, pooling_kernel=0):
        super(BottleneckAIC3d, self).__init__()
        self.residual = residual
        self.conv_in = nn.Conv3d(c_in, c, kernel_size=1, bias=False)
        self.basic_aic = BasicAIC3d(c, kernel=kernel, dilation=dilation, residual=True)
        self.conv_out = nn.Conv3d(c, c_out, kernel_size=1, bias=False)

    def forward(self, x):
        y = self.conv_in(x)
        y = F.relu(y, inplace=True)

        y = self.basic_aic(y)

        y = self.conv_out(y)
        y = F.relu(y + x, inplace=True) if self.residual else F.relu(y, inplace=True)
        return y


class SSC_RGBD_AICNet(nn.Module):
    def __init__(self, num_classes=12):
        super(SSC_RGBD_AICNet, self).__init__()
        print('SSC_RGBD_AICNet.')

        w, h, d = 240, 144, 240
        k = ((3, 5, 7), (3, 5, 7), (3, 5, 7))
        ks = (3, 5, 7)

        # --- depth
        c_in, c, c_out, dilation, residual = 1, 4, 8, 1, True
        self.dep_feature2d = nn.Sequential(
            nn.Conv2d(c_in, c_out, 1, 1, 0),  # reduction
            BottleneckDDR2d(c_out, c, c_out, dilation=dilation, residual=residual),
            BottleneckDDR2d(c_out, c, c_out, dilation=dilation, residual=residual),
        )
        self.project_layer_dep = Project2Dto3D(w, h, d)  # w=240, h=144, d=240
        self.dep_feature3d = nn.Sequential(
            DownsampleBlock3d(8, 16),
            BottleneckDDR3d(c_in=16, c=8, c_out=16, dilation=1, residual=True),
            DownsampleBlock3d(16, 64),  # nn.MaxPool3d(kernel_size=2, stride=2)
            BottleneckDDR3d(c_in=64, c=16, c_out=64, dilation=1, residual=True),
        )

        # --- RGB
        c_in, c, c_out, dilation, residual = 3, 4, 8, 1, True
        self.rgb_feature2d = nn.Sequential(
            nn.Conv2d(c_in, c_out, 1, 1, 0),  # reduction
            BottleneckDDR2d(c_out, c, c_out, dilation=dilation, residual=residual),
            BottleneckDDR2d(c_out, c, c_out, dilation=dilation, residual=residual),
        )
        self.project_layer_rgb = Project2Dto3D(w, h, d)  # w=240, h=144, d=240
        self.rgb_feature3d = nn.Sequential(
            DownsampleBlock3d(8, 16),
            BottleneckDDR3d(c_in=16, c=8, c_out=16, dilation=1, residual=True),
            DownsampleBlock3d(16, 64),  # nn.MaxPool3d(kernel_size=2, stride=2)
            BottleneckDDR3d(c_in=64, c=16, c_out=64, dilation=1, residual=True),
        )

        ck = 64
        c = int(ck / 2)
        dilation = ((1, 1, 1), (1, 1, 1), (1, 1, 1))

        # ---- depth stream
        self.res3d_1d = BottleneckAIC3d(c_in=ck, c=c, c_out=ck, kernel=k[0], dilation=dilation[0], residual=True)
        self.res3d_2d = BottleneckAIC3d(c_in=ck, c=c, c_out=ck, kernel=k[1], dilation=dilation[1], residual=True)
        self.res3d_3d = BottleneckAIC3d(c_in=ck, c=c, c_out=ck, kernel=k[2], dilation=dilation[2], residual=True)
        # ---- rgb stream
        self.res3d_1r = BottleneckAIC3d(c_in=ck, c=c, c_out=ck, kernel=k[0], dilation=dilation[0], residual=True)
        self.res3d_2r = BottleneckAIC3d(c_in=ck, c=c, c_out=ck, kernel=k[1], dilation=dilation[1], residual=True)
        self.res3d_3r = BottleneckAIC3d(c_in=ck, c=c, c_out=ck, kernel=k[2], dilation=dilation[2], residual=True)

        d = (1, 1, 1)
        self.aspp_1 = BottleneckAIC3d(c_in=int(ck * 4), c=ck, c_out=int(ck * 4), kernel=ks, dilation=d, residual=True)
        self.aspp_2 = BottleneckAIC3d(c_in=int(ck * 4), c=ck, c_out=int(ck * 4), kernel=ks, dilation=d, residual=True)

        self.conv_out = nn.Sequential(
            nn.Conv3d(int(ck * 4), 128, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 128, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, num_classes, 1, 1, 0)
        )

        # ----  weights init
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.xavier_uniform_(m.weight.data)  # gain=1
                # nn.init.constant(m.bias.data, 0)
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, mean=0, std=0.1)

    def forward(self, x_depth=None, x_rgb=None, p=None):
        # input: x (BS, 3L, 240L, 144L, 240L)
        # print('SSC: x.shape', x.shape)
        f0_r = self.rgb_feature2d(x_rgb)
        f0_r = self.project_layer_rgb(f0_r, p)
        f0_r = self.rgb_feature3d(f0_r)

        f0_d = self.dep_feature2d(x_depth)
        f0_d = self.project_layer_dep(f0_d, p)
        f0_d = self.dep_feature3d(f0_d)

        # -------------------------------------------------------------------
        f0 = torch.add(f0_d, f0_r)

        f1_d = self.res3d_1d(f0_d)
        f1_r = self.res3d_1r(f0_r)

        f1 = torch.add(f1_d, f1_r)

        f2_d = self.res3d_2d(f1_d)
        f2_r = self.res3d_2r(f1_r)

        f2 = torch.add(f2_d, f2_r)

        f3_d = self.res3d_3d(f2_d)
        f3_r = self.res3d_3r(f2_r)

        f3 = torch.add(f3_d, f3_r)

        y = torch.cat((f0, f1, f2, f3), dim=1)  # channels concatenate
        # print('SSC: channels concatenate x', x.size())  # (BS, 256L, 60L, 36L, 60L)

        y = self.aspp_1(y)
        y = self.aspp_2(y)
        y = self.conv_out(y)  # (BS, 12L, 60L, 36L, 60L)
        return y
