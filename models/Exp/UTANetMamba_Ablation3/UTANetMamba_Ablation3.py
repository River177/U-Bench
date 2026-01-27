"""
Ablation Study 3: Baseline + Proposed Decoder (No MoE, Gated Attention to Previous Layer Only)
编码器 + 提出的解码器（不包含MoE，门控注意力只传给上一层）
使用FastFullScaleDecoder和GatedAttention，但无专家模块，且门控注意力只传递给上一层
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import List

class FastFullScaleDecoder(nn.Module):
    """轻量化全尺度解码器"""
    def __init__(
        self, 
        in_channels_list: List[int],
        cat_channels: int = 32,
        out_channels: int = 128
    ):
        super().__init__()
        self.cat_channels = cat_channels
        self.num_inputs = len(in_channels_list)
        
        self.transforms = nn.ModuleList()
        for in_ch in in_channels_list:
            self.transforms.append(nn.Sequential(
                nn.Conv2d(in_ch, cat_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(cat_channels),
                nn.ReLU(inplace=True)
            ))
        
        self.fusion = nn.Sequential(
            nn.Conv2d(self.num_inputs * cat_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        target_h, target_w = features[0].size()[2], features[0].size()[3]
        
        processed_features = []
        for i, (feat, transform) in enumerate(zip(features, self.transforms)):
            x = transform(feat)
            h, w = x.size()[2], x.size()[3]
            
            if h > target_h and w > target_w:
                kernel_h = h // target_h
                kernel_w = w // target_w
                x = F.max_pool2d(x, kernel_size=(kernel_h, kernel_w), stride=(kernel_h, kernel_w))
            elif h < target_h and w < target_w:
                x = F.interpolate(x, size=(target_h, target_w), mode='bilinear', align_corners=True)
            
            processed_features.append(x)
        
        concat = torch.cat(processed_features, dim=1)
        out = self.fusion(concat)
        
        return out

class GatedAttention(nn.Module):
    """门控注意力模块"""
    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()
        
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, 1, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, 1, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, 1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(F_l, max(1, F_l // 16), 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(1, F_l // 16), F_l, 1, bias=True),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        
        if g1.size()[2:] != x1.size()[2:]:
            g1 = F.interpolate(g1, size=x1.size()[2:], mode='bilinear', align_corners=True)
        
        psi = self.relu(g1 + x1)
        spatial_att = self.psi(psi)
        channel_att = self.channel_att(x)
        
        return x * spatial_att * channel_att

class UTANetMamba_Ablation3(nn.Module):
    """
    消融实验3: 编码器 + 提出的解码器（无MoE，门控注意力只传给上一层）
    """
    def __init__(
        self,
        n_channels: int = 3,
        n_classes: int = 1,
        img_size: int = 224,
        deep_supervision: bool = True,
        cat_channels: int = 32
    ):
        super().__init__()
        self.n_classes = n_classes
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
        
    def forward(self, x: torch.Tensor):
        input_size = x.shape[2:]
        
        e1 = self.conv1(x)
        e1_maxp = self.maxpool(e1)
        e2 = self.conv2(e1_maxp)
        e3 = self.conv3(e2)
        e4 = self.conv4(e3)
        e5 = self.conv5(e4)
        
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

def utanet_mamba_ablation3(input_channel=3, num_classes=1, pretrained=True):
    return UTANetMamba_Ablation3(
        n_channels=input_channel,
        n_classes=num_classes,
        deep_supervision=True
    )
