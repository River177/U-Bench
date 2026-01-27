"""
Ablation Study 1: Encoder Only (Baseline)
只使用编码器，通过上采样得到最终结果
无解码器、无MoE、无门控注意力
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class UTANetMamba_Ablation1(nn.Module):
    """
    消融实验1: 仅编码器 + 上采样
    """
    def __init__(
        self,
        n_channels: int = 3,
        n_classes: int = 1,
        img_size: int = 224,
        deep_supervision: bool = True
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
        
        self.upsample = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        self.final = nn.Conv2d(32, n_classes, 1)
        
        if deep_supervision:
            self.ds5 = nn.Conv2d(512, n_classes, 1)
            self.ds4 = nn.Conv2d(256, n_classes, 1)
            self.ds3 = nn.Conv2d(128, n_classes, 1)
            self.ds2 = nn.Conv2d(64, n_classes, 1)
        
    def forward(self, x: torch.Tensor):
        input_size = x.shape[2:]
        
        e1 = self.conv1(x)
        e1_maxp = self.maxpool(e1)
        e2 = self.conv2(e1_maxp)
        e3 = self.conv3(e2)
        e4 = self.conv4(e3)
        e5 = self.conv5(e4)
        
        upsampled = F.interpolate(e5, size=input_size, mode='bilinear', align_corners=True)
        features = self.upsample(upsampled)
        logits = self.final(features)
        
        if self.deep_supervision:
            ds5_out = F.interpolate(self.ds5(e5), size=input_size, mode='bilinear', align_corners=True)
            ds4_out = F.interpolate(self.ds4(e4), size=input_size, mode='bilinear', align_corners=True)
            ds3_out = F.interpolate(self.ds3(e3), size=input_size, mode='bilinear', align_corners=True)
            ds2_out = F.interpolate(self.ds2(e2), size=input_size, mode='bilinear', align_corners=True)
            return [ds5_out, ds4_out, ds3_out, ds2_out, logits]
        else:
            return logits

def utanet_mamba_ablation1(input_channel=3, num_classes=1, pretrained=True):
    return UTANetMamba_Ablation1(
        n_channels=input_channel,
        n_classes=num_classes,
        deep_supervision=True
    )
