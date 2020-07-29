#!/usr/bin/env python3
#-*- coding:utf-8 -*-
'''
backbone used in the arcface paper, including ResNet50-IR, ResNet101-IR, SEResNet50-IR, SEResNet101-IR
reference : https://github.com/wujiyang/Face_Pytorch/blob/master/backbone/arcfacenet.py
'''

import torch
from torch import nn
from collections import namedtuple
from IPython import embed


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class SEModule(nn.Module):
    def __init__(self, channels, reduction = 16):
        super(SEModule, self).__init__()
        self.se_layer = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0, bias=False),
            nn.PReLU(),
            nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0, bias=False),
            nn.Sigmoid())

    def forward(self, x):
        return x * self.se_layer(x)


class ResidualUnit_IR(nn.Module):
    def __init__(self, in_channel, out_channel, stride):
        super(ResidualUnit_IR, self).__init__()
        if in_channel == out_channel:
            self.downsample = nn.MaxPool2d(1, stride)
        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=(1, 1), stride=stride, bias=False),
                nn.BatchNorm2d(out_channel))

        self.res_layer = nn.Sequential(
            nn.BatchNorm2d(in_channel),
            nn.Conv2d(in_channel, out_channel, (3, 3), 1, 1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.PReLU(out_channel),
            nn.Conv2d(out_channel, out_channel, (3, 3), stride, 1, bias=False),
            nn.BatchNorm2d(out_channel))

    def forward(self, x):
        identity = self.downsample(x)
        res = self.res_layer(x)
        return identity + res

    
class ResidualUnit_IRSE(nn.Module):
    def __init__(self, in_channel, out_channel, stride = 2):
        super(ResidualUnit_IRSE, self).__init__()
        if in_channel == out_channel:
            self.downsample = nn.MaxPool2d(1, stride)
        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=(1, 1), stride=stride, bias=False),
                nn.BatchNorm2d(out_channel))

        self.res_layer = nn.Sequential(
            nn.BatchNorm2d(in_channel),
            nn.Conv2d(in_channel, out_channel, (3, 3), 1, 1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.PReLU(out_channel),
            nn.Conv2d(out_channel, out_channel, (3, 3), stride, 1, bias=False),
            nn.BatchNorm2d(out_channel),
            SEModule(out_channel, 16))

    def forward(self, x):
        identity = self.downsample(x)
        res = self.res_layer(x)
        return identity + res


class ArcRes(nn.Module):
    
    arcres_zoo = {
        'arcres50'  : [3, 4, 14, 3],
        'arcres100' : [3, 13, 30, 3],
        'arcres152' : [3, 8, 36, 3]}
    residual_zoo = {
        'ir'   : ResidualUnit_IR,
        'irse' : ResidualUnit_IRSE}
    
    def __init__(self, backbone = 'arcres100', embedding_dim = 512, drop_ratio = 0.4, mode = 'ir'):
        
        super(ArcRes, self).__init__()
        
        self.stage_layers  = self.arcres_zoo[backbone]
        self.embedding_dim = embedding_dim
        self.drop_ratio    = drop_ratio
        self.residual_unit = self.residual_zoo[mode]
        self.nums_filters  = [64, 64, 128, 256, 512]
        
        self.input_layer = nn.Sequential(
            nn.Conv2d(3, 64, (3, 3), 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.PReLU(64))
        
        self.body_layer = self._make_layers()
        
        self.output_layer = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.Dropout(drop_ratio),
            Flatten(),
            nn.Linear(512 * 7 * 7, embedding_dim),
            nn.BatchNorm1d(embedding_dim, affine=True))  # TODO::BUG
    
    
    @staticmethod
    def _build_stage(in_channel, out_channel, num_layers, stride = 2):
        Unit = namedtuple('Unit', ['in_channel', 'out_channel', 'stride']) 
        residual_units = [Unit(in_channel, out_channel, stride)]
        for i in range(1, num_layers):
            residual_units += [Unit(out_channel, out_channel, 1)]
        return residual_units
                
        
    def _make_layers(self):
        modules = []
        for stage_idx in range(len(self.stage_layers)):
            
            stage_units = self._build_stage(self.nums_filters[stage_idx], \
                                            self.nums_filters[stage_idx+1],
                                            self.stage_layers[stage_idx])
            for unit in stage_units:
                modules.append(self.residual_unit(unit.in_channel, unit.out_channel, unit.stride))
        return nn.Sequential(*modules)
 

    def forward(self, x):
        x = self.input_layer(x)
        x = self.body_layer(x)
        x = self.output_layer(x)
        return x    
    

if __name__ == '__main__':
    
    input = torch.Tensor(2, 3, 112, 112)
    backbone = ArcRes(backbone='arcres100', mode='ir')
    x = backbone(input)
    print(x.shape)
    embed()


        
    