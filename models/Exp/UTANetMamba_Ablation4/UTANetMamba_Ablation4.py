"""
Ablation Study 4: Baseline + Proposed Decoder + MoE (No Adjacent Parallel Guidance)
编码器 + 提出的解码器 + 专家模块（无邻接并行指导策略）
使用MoE但不使用邻接并行指导，门控注意力只传给上一层
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import List
import sys
import os

from .ta_mosc import MoE
from .modules_fast import FastFullScaleDecoder, GatedAttention

class UTANetMamba_Ablation4(nn.Module):
    """
    消融实验4: 编码器 + 提出的解码器 + MoE（无邻接并行指导）
    """
    def __init__(
        self,
        pretrained: bool = True,
        topk: int = 2,
        n_channels: int = 3,
        n_classes: int = 1,
        img_size: int = 224,
        deep_supervision: bool = True,
        cat_channels: int = 32
    ):
        super().__init__()
        self.n_classes = n_classes
        self.pretrained = pretrained
        self.img_size = img_size
        self.deep_supervision = deep_supervision
        
        self.resnet = models.resnet34(pretrained=True)
        self.filters_resnet = [64, 64, 128, 256, 512]
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(n_channels, self.filters_resnet[0], 3, 1, 1, bias=True),
            nn.BatchNorm2d(self.filters_resnet[0]),
            nn.ReLU(inplace=True)
        )
        self.maxpool = nn.MaxPool2d(2, 2)
        
        self.conv2 = self.resnet.layer1
        self.conv3 = self.resnet.layer2
        self.conv4 = self.resnet.layer3
        self.conv5 = self.resnet.layer4
        
        if pretrained:
            self.fuse = nn.Sequential(
                nn.Conv2d(512, 64, 1, 1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
            )
            self.moe = MoE(
                num_experts=4, 
                top=topk, 
                emb_size=64, 
                expert_type='mamba',
                H=img_size//2,
                W=img_size//2
            )
            
            self.docker1 = self._create_docker(64, self.filters_resnet[0])
            self.docker2 = self._create_docker(64, self.filters_resnet[1])
            self.docker3 = self._create_docker(64, self.filters_resnet[2])
            self.docker4 = self._create_docker(64, self.filters_resnet[3])
        
        self.decoder4 = FastFullScaleDecoder(
            in_channels_list=[256, 512],
            cat_channels=cat_channels,
            out_channels=256
        )
        self.att4 = GatedAttention(F_g=256, F_l=256, F_int=128)
        
        self.decoder3 = FastFullScaleDecoder(
            in_channels_list=[128, 256],
            cat_channels=cat_channels,
            out_channels=128
        )
        self.att3 = GatedAttention(F_g=128, F_l=128, F_int=64)
        
        self.decoder2 = FastFullScaleDecoder(
            in_channels_list=[64, 128],
            cat_channels=cat_channels,
            out_channels=64
        )
        self.att2 = GatedAttention(F_g=64, F_l=64, F_int=32)
        
        self.decoder1 = FastFullScaleDecoder(
            in_channels_list=[64, 64],
            cat_channels=cat_channels,
            out_channels=32
        )
        self.att1 = GatedAttention(F_g=32, F_l=32, F_int=16)
        
        if deep_supervision:
            self.ds4 = nn.Conv2d(256, n_classes, 1, bias=True)
            self.ds3 = nn.Conv2d(128, n_classes, 1, bias=True)
            self.ds2 = nn.Conv2d(64, n_classes, 1, bias=True)
            self.ds1 = nn.Conv2d(32, n_classes, 1, bias=True)
        
        self.final = nn.Sequential(
            nn.Conv2d(32, 16, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, n_classes, 1)
        )
        
    def _create_docker(self, in_ch: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, 1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x: torch.Tensor):
        input_size = x.shape[2:]
        
        e1 = self.conv1(x)
        e1_maxp = self.maxpool(e1)
        e2 = self.conv2(e1_maxp)
        e3 = self.conv3(e2)
        e4 = self.conv4(e3)
        e5 = self.conv5(e4)
        
        if self.pretrained:
            e1_resized = F.interpolate(e1, scale_factor=0.5, mode='bilinear', align_corners=True)
            e3_resized = F.interpolate(e3, scale_factor=2, mode='bilinear', align_corners=True)
            e4_resized = F.interpolate(e4, scale_factor=4, mode='bilinear', align_corners=True)
            
            fused = torch.cat([e1_resized, e2, e3_resized, e4_resized], dim=1)
            fused = self.fuse(fused)
            
            o1, o2, o3, o4, loss = self.moe(fused)
            
            o1 = self.docker1(o1)
            o2 = self.docker2(o2)
            o3 = self.docker3(o3)
            o4 = self.docker4(o4)
            
            o1 = F.interpolate(o1, scale_factor=2, mode='bilinear', align_corners=True)
            o3 = F.interpolate(o3, scale_factor=0.5, mode='bilinear', align_corners=True)
            o4 = F.interpolate(o4, scale_factor=0.25, mode='bilinear', align_corners=True)
            
            e1 = e1 + o1
            e2 = e2 + o2
            e3 = e3 + o3
            e4 = e4 + o4
        
        d4 = self.decoder4([e4, e5])
        d4_att = self.att4(d4, d4)
        
        d3 = self.decoder3([e3, d4_att])
        d3_att = self.att3(d3, d3)
        
        d2 = self.decoder2([e2, d3_att])
        d2_att = self.att2(d2, d2)
        
        d1 = self.decoder1([e1, d2_att])
        d1_att = self.att1(d1, d1)
        
        logits = self.final(d1_att)
        
        if self.deep_supervision:
            ds4_out = F.interpolate(self.ds4(d4_att), size=input_size, mode='bilinear', align_corners=True)
            ds3_out = F.interpolate(self.ds3(d3_att), size=input_size, mode='bilinear', align_corners=True)
            ds2_out = F.interpolate(self.ds2(d2_att), size=input_size, mode='bilinear', align_corners=True)
            ds1_out = F.interpolate(self.ds1(d1_att), size=input_size, mode='bilinear', align_corners=True)
            return [ds4_out, ds3_out, ds2_out, ds1_out, logits]
        else:
            return logits

def utanet_mamba_ablation4(input_channel=3, num_classes=1, pretrained=True):
    return UTANetMamba_Ablation4(
        n_channels=input_channel,
        n_classes=num_classes,
        pretrained=pretrained,
        deep_supervision=True
    )
