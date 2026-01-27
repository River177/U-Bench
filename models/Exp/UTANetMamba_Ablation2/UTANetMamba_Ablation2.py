"""
Ablation Study 2: Baseline + Simple Decoder
编码器 + 普通解码器（简单的跳跃连接，无门控注意力，无MoE）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class SimpleDecoder(nn.Module):
    """简单的解码器模块，使用跳跃连接"""
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class UTANetMamba_Ablation2(nn.Module):
    """
    消融实验2: 编码器 + 普通解码器
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
        
        self.decoder4 = SimpleDecoder(512, 256, 256)
        self.decoder3 = SimpleDecoder(256, 128, 128)
        self.decoder2 = SimpleDecoder(128, 64, 64)
        self.decoder1 = SimpleDecoder(64, 64, 32)
        
        if deep_supervision:
            self.ds4 = nn.Conv2d(256, n_classes, 1)
            self.ds3 = nn.Conv2d(128, n_classes, 1)
            self.ds2 = nn.Conv2d(64, n_classes, 1)
            self.ds1 = nn.Conv2d(32, n_classes, 1)
        
        self.final = nn.Conv2d(32, n_classes, 1)
        
    def forward(self, x: torch.Tensor):
        input_size = x.shape[2:]
        
        e1 = self.conv1(x)
        e1_maxp = self.maxpool(e1)
        e2 = self.conv2(e1_maxp)
        e3 = self.conv3(e2)
        e4 = self.conv4(e3)
        e5 = self.conv5(e4)
        
        d4 = self.decoder4(e5, e4)
        d3 = self.decoder3(d4, e3)
        d2 = self.decoder2(d3, e2)
        d1 = self.decoder1(d2, e1)
        
        logits = self.final(d1)
        
        if self.deep_supervision:
            ds4_out = F.interpolate(self.ds4(d4), size=input_size, mode='bilinear', align_corners=True)
            ds3_out = F.interpolate(self.ds3(d3), size=input_size, mode='bilinear', align_corners=True)
            ds2_out = F.interpolate(self.ds2(d2), size=input_size, mode='bilinear', align_corners=True)
            ds1_out = F.interpolate(self.ds1(d1), size=input_size, mode='bilinear', align_corners=True)
            return [ds4_out, ds3_out, ds2_out, ds1_out, logits]
        else:
            return logits

def utanet_mamba_ablation2(input_channel=3, num_classes=1, pretrained=True):
    return UTANetMamba_Ablation2(
        n_channels=input_channel,
        n_classes=num_classes,
        deep_supervision=True
    )
