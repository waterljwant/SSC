#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
DDR
jieli_cn@163.com
"""

import torch
import torch.nn as nn
from torch.nn import functional as F


# ----------------------------------------------------------------------
class BasicDDR2d(nn.Module):
    def __init__(self, c, k=3, dilation=1, residual=True):
        super(BasicDDR2d, self).__init__()
        d = dilation
        p = k // 2 * d
        self.conv_1xk = nn.Conv2d(c, c, (1, k), stride=1, padding=(0, p), bias=True, dilation=(1, d))
        self.conv_kx1 = nn.Conv2d(c, c, (k, 1), stride=1, padding=(p, 0), bias=True, dilation=(d, 1))
        self.residual = residual

    def forward(self, x):
        y = self.conv_1xk(x)
        y = F.relu(y, inplace=True)
        y = self.conv_kx1(y)
        y = F.relu(y + x, inplace=True) if self.residual else F.relu(y, inplace=True)
        return y


# ----------------------------------------------------------------------
class BasicDDR3d(nn.Module):
    def __init__(self, c, k=3, dilation=1, stride=1, residual=True):
        super(BasicDDR3d, self).__init__()
        d = dilation
        p = k // 2 * d
        # p = (d * (k - 1) + 1) // 2
        s = stride
        # print("k:{}, d:{}, p:{}".format(k, d, p))
        self.conv_1x1xk = nn.Conv3d(c, c, (1, 1, k), stride=(1, 1, s), padding=(0, 0, p), bias=True, dilation=(1, 1, d))
        self.conv_1xkx1 = nn.Conv3d(c, c, (1, k, 1), stride=(1, s, 1), padding=(0, p, 0), bias=True, dilation=(1, d, 1))
        self.conv_kx1x1 = nn.Conv3d(c, c, (k, 1, 1), stride=(s, 1, 1), padding=(p, 0, 0), bias=True, dilation=(d, 1, 1))
        self.residual = residual

    def forward(self, x):
        y = self.conv_1x1xk(x)
        y = F.relu(y, inplace=True)
        y = self.conv_1xkx1(y)
        y = F.relu(y, inplace=True)
        y = self.conv_kx1x1(y)
        y = F.relu(y + x, inplace=True) if self.residual else F.relu(y, inplace=True)
        return y


class BottleneckDDR2d(nn.Module):
    def __init__(self, c_in, c, c_out, kernel=3, stride=1, dilation=1, residual=True):
        super(BottleneckDDR2d, self).__init__()
        s = stride
        k = kernel
        d = dilation
        p = k // 2 * d
        self.conv_in = nn.Conv2d(c_in, c, kernel_size=1, bias=False)
        self.conv_1xk = nn.Conv2d(c, c, (1, k), stride=s, padding=(0, p), bias=True, dilation=(1, d))
        self.conv_kx1 = nn.Conv2d(c, c, (k, 1), stride=s, padding=(p, 0), bias=True, dilation=(d, 1))
        self.conv_out = nn.Conv2d(c, c_out, kernel_size=1, bias=False)
        self.residual = residual

    def forward(self, x):
        y = self.conv_in(x)
        y = F.relu(y, inplace=True)
        y = self.conv_1xk(y)
        y = F.relu(y, inplace=True)
        y = self.conv_kx1(y)
        y = F.relu(y, inplace=True)
        y = self.conv_out(y)
        y = F.relu(y + x, inplace=True) if self.residual else F.relu(y, inplace=True)
        return y


class BottleneckDDR3d(nn.Module):
    def __init__(self, c_in, c, c_out, kernel=3, stride=1, dilation=1, residual=True):
        super(BottleneckDDR3d, self).__init__()
        s = stride
        k = kernel
        d = dilation
        p = k // 2 * d
        self.conv_in = nn.Conv3d(c_in, c, kernel_size=1, bias=False)
        self.conv1x1x3 = nn.Conv3d(c, c, (1, 1, k), stride=s, padding=(0, 0, p), bias=True, dilation=(1, 1, d))
        self.conv1x3x1 = nn.Conv3d(c, c, (1, k, 1), stride=s, padding=(0, p, 0), bias=True, dilation=(1, d, 1))
        self.conv3x1x1 = nn.Conv3d(c, c, (k, 1, 1), stride=s, padding=(p, 0, 0), bias=True, dilation=(d, 1, 1))
        self.conv_out = nn.Conv3d(c, c_out, kernel_size=1, bias=False)
        self.residual = residual

    def forward(self, x):
        y0 = self.conv_in(x)
        y0 = F.relu(y0, inplace=True)

        y1 = self.conv1x1x3(y0)
        y1 = F.relu(y1, inplace=True)

        y2 = self.conv1x3x1(y1) + y1
        y2 = F.relu(y2, inplace=True)

        y3 = self.conv3x1x1(y2) + y2 + y1
        y3 = F.relu(y3, inplace=True)

        y = self.conv_out(y3)

        y = F.relu(y + x, inplace=True) if self.residual else F.relu(y, inplace=True)
        return y


class DownsampleBlock3d(nn.Module):
    def __init__(self, c_in, c_out, k=3, s=2, p=1):
        super(DownsampleBlock3d, self).__init__()
        self.conv = nn.Conv3d(c_in, c_out-c_in, kernel_size=k, stride=s, padding=p, bias=False)
        self.pool = nn.MaxPool3d(2, stride=2)
        # self.bn = nn.BatchNorm2d(c_out, eps=1e-3)

    def forward(self, x):
        y = torch.cat([self.conv(x), self.pool(x)], 1)
        # y = self.bn(y)
        y = F.relu(y, inplace=True)
        return y


class DDR_ASPP3d(nn.Module):
    def __init__(self, c_in, c, c_out, residual=False):
        super(DDR_ASPP3d, self).__init__()
        print('DDR_ASPP3d: c_in:{}, c:{}, c_out:{}'.format(c_in, c, c_out))

        self.aspp0 = nn.Conv3d(c_in, c_out, kernel_size=1, stride=1, padding=0, dilation=1, bias=False)

        self.aspp1 = BottleneckDDR3d(c_in, c, c_out, dilation=6, residual=residual)

        self.aspp2 = BottleneckDDR3d(c_in, c, c_out, dilation=12, residual=residual)

        self.aspp3 = BottleneckDDR3d(c_in, c, c_out, dilation=18, residual=residual)

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool3d((1, 1, 1)),
                                             nn.Conv3d(c_in, c_out, 1, stride=1, bias=False))

    def forward(self, x):
        x0 = self.aspp0(x)
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x_ = self.global_avg_pool(x)

        # x_ = F.upsample(x_, size=x.size()[2:], mode='trilinear', align_corners=True)
        x_ = F.interpolate(x_, size=x.size()[2:], mode='trilinear', align_corners=True)

        x = torch.cat((x0, x1, x2, x3, x_), dim=1)
        # print(x0.shape, x1.shape, x2.shape, x3.shape, x_.shape, x.shape)
        return x

