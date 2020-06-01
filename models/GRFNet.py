#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
GRFNet
jieli_cn@163.com
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
from .projection_layer import Project2Dto3D
from .DDR import DDR_ASPP3d
from .DDR import BottleneckDDR2d, BottleneckDDR3d, DownsampleBlock3d


class Conv3dGRUCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size, bias=True):
        super(Conv3dGRUCell, self).__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.bias = bias

        self.in_conv = nn.Conv3d(in_channels=self.input_channels + self.hidden_channels,
                                 out_channels=2 * self.hidden_channels,
                                 kernel_size=self.kernel_size,
                                 stride=1,
                                 dilation=1,
                                 padding=self.padding,
                                 bias=self.bias)

        self.out_conv = nn.Conv3d(in_channels=self.input_channels + self.hidden_channels,
                                  out_channels=self.hidden_channels,
                                  kernel_size=self.kernel_size,
                                  stride=1,
                                  dilation=1,
                                  padding=self.padding,
                                  bias=self.bias)

    def forward(self, input_tensor, hidden_state):
        # print('input_tensor.size()', input_tensor.size(), 'hidden_state.size()', hidden_state.size())
        h_cur = hidden_state
        combined = torch.cat((input_tensor, h_cur), dim=1)  # concatenate along channel axis
        combined_conv = self.in_conv(combined)
        cc_r, cc_z = torch.split(combined_conv, self.hidden_channels, dim=1)
        # print('cc_r.size()', cc_r.size(), 'cc_z.size()', cc_z.size())
        r = torch.sigmoid(cc_r)  # reset gate
        z = torch.sigmoid(cc_z)  # update gate

        h_cur_bar = h_cur * r
        cc_h = self.out_conv(torch.cat((input_tensor, h_cur_bar), dim=1))
        h_bar = torch.tanh(cc_h)
        h_next = z * h_cur + (1 - z) * h_bar
        return h_next


class SSC_RGBD_GRFNet(nn.Module):
    def __init__(self, num_classes=12):
        super(SSC_RGBD_GRFNet, self).__init__()
        print('SSC_RGBD_GRFNet.')

        # --- depth
        c_in, c, c_out, dilation, residual = 1, 4, 8, 1, True
        self.dep_feature2d = nn.Sequential(
            nn.Conv2d(c_in, c_out, 1, 1, 0),  # reduction
            BottleneckDDR2d(c_out, c, c_out, dilation=dilation, residual=residual),
            BottleneckDDR2d(c_out, c, c_out, dilation=dilation, residual=residual),
        )
        self.project_layer_dep = Project2Dto3D(240, 144, 240)  # w=240, h=144, d=240
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
        self.project_layer_rgb = Project2Dto3D(240, 144, 240)  # w=240, h=144, d=240
        self.rgb_feature3d = nn.Sequential(
            DownsampleBlock3d(8, 16),
            BottleneckDDR3d(c_in=16, c=8, c_out=16, dilation=1, residual=True),
            DownsampleBlock3d(16, 64),  # nn.MaxPool3d(kernel_size=2, stride=2)
            BottleneckDDR3d(c_in=64, c=16, c_out=64, dilation=1, residual=True),
        )

        # -------------1/4
        ck = 64
        c = ck // 4

        # --- RGB
        self.res3d_1r = BottleneckDDR3d(c_in=ck, c=c, c_out=ck, dilation=2, residual=True)
        self.res3d_2r = BottleneckDDR3d(c_in=ck, c=c, c_out=ck, dilation=3, residual=True)
        self.res3d_3r = BottleneckDDR3d(c_in=ck, c=c, c_out=ck, dilation=5, residual=True)

        # --- Depth
        self.res3d_1d = BottleneckDDR3d(c_in=ck, c=c, c_out=ck, dilation=2, residual=True)
        self.res3d_2d = BottleneckDDR3d(c_in=ck, c=c, c_out=ck, dilation=3, residual=True)
        self.res3d_3d = BottleneckDDR3d(c_in=ck, c=c, c_out=ck, dilation=5, residual=True)

        # self.lstm = DDRConv3dLSTMCell(input_channels=128, hidden_channels=64, kernel_size=(3, 3, 3), bias=True)
        self.gru = Conv3dGRUCell(input_channels=64, hidden_channels=64, kernel_size=3, bias=True)

        self.aspp = DDR_ASPP3d(c_in=ck, c=16, c_out=64)

        self.conv_out = nn.Sequential(
            nn.Conv3d(320, 160, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv3d(160, num_classes, 1, 1, 0)
        )

        # ----  weights init
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                # nn.init.xavier_normal(m.weight.data, gain=math.sqrt(2. / n))
                # nn.init.xavier_uniform(m.weight.data, gain=math.sqrt(2. / n))
                nn.init.xavier_uniform_(m.weight.data)  # gain=1
                # nn.init.constant(m.bias.data, 0)
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, mean=0, std=0.1)

    def forward(self, x_depth=None, x_rgb=None, p=None):
        # input: x (BS, 3L, 240L, 144L, 240L)
        # print('SSC: x.shape', x.shape)

        if x_rgb is not None:
            x0_rgb = self.rgb_feature2d(x_rgb)
            x0_rgb = self.project_layer_rgb(x0_rgb, p)
            x0_rgb = self.rgb_feature3d(x0_rgb)
            # pass

        if x_depth is not None:
            x0_depth = self.dep_feature2d(x_depth)
            x0_depth = self.project_layer_dep(x0_depth, p)
            x0_depth = self.dep_feature3d(x0_depth)

        # -------------------------------------------------------------------
        # ---- 1/4

        x_4_d = self.res3d_1d(x0_depth)
        x_4_r = self.res3d_1r(x0_rgb)

        # f1 = torch.add(x_4_d, x_4_r)
        x_5_d = self.res3d_2d(x_4_d)
        x_5_r = self.res3d_2r(x_4_r)

        # f2 = torch.add(x_5_d, x_5_r)

        x_6_d = self.res3d_3d(x_5_d)
        x_6_r = self.res3d_3r(x_5_r)
        # f3 = torch.add(x_6_d, x_6_r)

        h0 = torch.add(x0_depth, x0_depth)

        # Fusion stage: 1
        h1_1 = self.gru(input_tensor=x0_depth, hidden_state=h0)
        h1 = self.gru(input_tensor=x0_rgb, hidden_state=h1_1)

        # Fusion stage: 2
        h2_1 = self.gru(input_tensor=x_4_d, hidden_state=h1)
        h2 = self.gru(input_tensor=x_4_r, hidden_state=h2_1)

        # Fusion stage: 3
        h3_1 = self.gru(input_tensor=x_5_d, hidden_state=h2)
        h3 = self.gru(input_tensor=x_5_r, hidden_state=h3_1)

        # Fusion stage: 4
        h4_1 = self.gru(input_tensor=x_6_d, hidden_state=h3)
        h4 = self.gru(input_tensor=x_6_r, hidden_state=h4_1)

        y = self.aspp(h4)
        y = self.conv_out(y)  # (BS, 12L, 60L, 36L, 60L)
        return y

