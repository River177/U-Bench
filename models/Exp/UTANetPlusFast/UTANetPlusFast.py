"""
UTANet+ Fast: Optimized for Low VRAM Usage
通过减少融合通道数和裁剪长距离跳跃连接，显著降低显存占用。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Optional, Tuple, List
import sys
import os

# 导入本地模块
from .ta_mosc import MoE
from .modules_fast import (
    FastFullScaleDecoder,
    GatedAttention,
    DeepSupervisionHead
)

class UTANetPlusFast(nn.Module):
    """
    UTANet+ Fast 版本
    
    优化策略：
    1. FastFullScaleDecoder: 融合通道数从 64 降至 32
    2. Connection Pruning: 
       - Decoder1 (224x224): 仅融合 o1 (224), d2 (112), d3 (56)。丢弃极深层 d4 (28), e5 (14)。
       - Decoder2 (112x112): 融合 o2 (112), d3 (56), d4 (28), o1 (224)。
       - Decoder3 (56x56): 融合 o3 (56), d4 (28), e5 (14), o2 (112)。
       - Decoder4 (28x28): 融合 o4 (28), e5 (14), o3 (56)。
    
    Args:
        pretrained: 是否使用TA-MoSC
        topk: MoE top-k
        n_channels: 输入通道
        n_classes: 输出类别
        img_size: 图像尺寸
        deep_supervision: 是否深监督
        cat_channels: 解码器融合通道数，默认 32 (原版 64)
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
        
        # ========== 编码器：ResNet34 ==========
        self.resnet = models.resnet34(pretrained=True)
        self.filters_resnet = [64, 64, 128, 256, 512]
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(n_channels, self.filters_resnet[0], 3, 1, 1, bias=True),
            nn.BatchNorm2d(self.filters_resnet[0]),
            nn.ReLU(inplace=True)
        )
        self.maxpool = nn.MaxPool2d(2, 2)
        
        self.conv2 = self.resnet.layer1  # 64, 112x112
        self.conv3 = self.resnet.layer2  # 128, 56x56
        self.conv4 = self.resnet.layer3  # 256, 28x28
        self.conv5 = self.resnet.layer4  # 512, 14x14
        
        # ========== TA-MoSC模块 ==========
        if pretrained:
            self.fuse = nn.Sequential(
                nn.Conv2d(512, 64, 1, 1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
            )
            self.moe = MoE(num_experts=4, top=topk, emb_size=64)
            
            self.docker1 = self._create_docker(64, self.filters_resnet[0]) # 64
            self.docker2 = self._create_docker(64, self.filters_resnet[1]) # 64
            self.docker3 = self._create_docker(64, self.filters_resnet[2]) # 128
            self.docker4 = self._create_docker(64, self.filters_resnet[3]) # 256
        
        # ========== Fast Full-Scale Decoders ==========
        
        # Decoder 4 (target 28x28)
        # Inputs: o4(28), e5(14), o3(56) -> Pruned o1, o2
        # Channels: 256, 512, 128
        self.decoder4 = FastFullScaleDecoder(
            in_channels_list=[256, 512, 128],
            cat_channels=cat_channels,
            out_channels=256
        )
        self.att4 = GatedAttention(F_g=256, F_l=256, F_int=128)
        
        # Decoder 3 (target 56x56)
        # Inputs: o3(56), d4_att(256), e5(14), o2(112) -> Pruned o1
        # Channels: 128, 256, 512, 64
        self.decoder3 = FastFullScaleDecoder(
            in_channels_list=[128, 256, 512, 64],
            cat_channels=cat_channels,
            out_channels=128
        )
        self.att3 = GatedAttention(F_g=128, F_l=128, F_int=64)
        
        # Decoder 2 (target 112x112)
        # Inputs: o2(112), d3_att(128), d4_att(256), o1(64) -> Pruned e5
        # Channels: 64, 128, 256, 64
        self.decoder2 = FastFullScaleDecoder(
            in_channels_list=[64, 128, 256, 64],
            cat_channels=cat_channels,
            out_channels=64
        )
        self.att2 = GatedAttention(F_g=64, F_l=64, F_int=32)
        
        # Decoder 1 (target 224x224) - 最关键的显存瓶颈
        # Inputs: o1(64), d2_att(64), d3_att(128). 
        # Pruned d4_att(28), e5(14) to save VRAM on upsampling
        # Channels: 64, 64, 128
        self.decoder1 = FastFullScaleDecoder(
            in_channels_list=[64, 64, 128],
            cat_channels=cat_channels,
            out_channels=32
        )
        self.att1 = GatedAttention(F_g=32, F_l=32, F_int=16)
        
        # ========== Deep Supervision ==========
        if deep_supervision:
            self.ds4 = nn.Conv2d(256, n_classes, 1, bias=True)
            self.ds3 = nn.Conv2d(128, n_classes, 1, bias=True)
            self.ds2 = nn.Conv2d(64, n_classes, 1, bias=True)
            self.ds1 = nn.Conv2d(32, n_classes, 1, bias=True)
        
        # ========== Final Output ==========
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
        # Encoder
        e1 = self.conv1(x)           # 64, 224
        e1_maxp = self.maxpool(e1)   # 64, 112
        e2 = self.conv2(e1_maxp)     # 64, 112
        e3 = self.conv3(e2)          # 128, 56
        e4 = self.conv4(e3)          # 256, 28
        e5 = self.conv5(e4)          # 512, 14
        
        # TA-MoSC
        if self.pretrained:
            e1_resized = F.interpolate(e1, scale_factor=0.5, mode='bilinear', align_corners=True)
            e3_resized = F.interpolate(e3, scale_factor=2, mode='bilinear', align_corners=True)
            e4_resized = F.interpolate(e4, scale_factor=4, mode='bilinear', align_corners=True)
            
            fused = torch.cat([e1_resized, e2, e3_resized, e4_resized], dim=1)
            fused = self.fuse(fused)
            
            o1, o2, o3, o4, loss = self.moe(fused)
            
            o1 = self.docker1(o1) # 64, 112
            o2 = self.docker2(o2) # 64, 112
            o3 = self.docker3(o3) # 128, 112
            o4 = self.docker4(o4) # 256, 112
            
            # 调整到各层分辨率
            o1 = F.interpolate(o1, scale_factor=2, mode='bilinear', align_corners=True)    # 224
            # o2 保持 112
            o3 = F.interpolate(o3, scale_factor=0.5, mode='bilinear', align_corners=True)  # 56
            o4 = F.interpolate(o4, scale_factor=0.25, mode='bilinear', align_corners=True) # 28
            
        else:
            o1, o2, o3, o4 = e1, e2, e3, e4
            
        # Decoder
        
        # D4 (28x28): [o4, e5, o3]
        d4 = self.decoder4([o4, e5, o3])
        d4_att = self.att4(d4, d4)
        
        # D3 (56x56): [o3, d4_att, e5, o2]
        d3 = self.decoder3([o3, d4_att, e5, o2])
        d3_att = self.att3(d3, d3)
        
        # D2 (112x112): [o2, d3_att, d4_att, o1]
        d2 = self.decoder2([o2, d3_att, d4_att, o1])
        d2_att = self.att2(d2, d2)
        
        # D1 (224x224): [o1, d2_att, d3_att] - Pruned deep features
        d1 = self.decoder1([o1, d2_att, d3_att])
        d1_att = self.att1(d1, d1)
        
        logits = self.final(d1_att)
        
        # Return format for U-Bench framework
        if self.deep_supervision:
            # Generate deep supervision outputs - upsample to input size
            input_size = x.shape[2:]
            ds4_out = F.interpolate(self.ds4(d4_att), size=input_size, mode='bilinear', align_corners=True)
            ds3_out = F.interpolate(self.ds3(d3_att), size=input_size, mode='bilinear', align_corners=True)
            ds2_out = F.interpolate(self.ds2(d2_att), size=input_size, mode='bilinear', align_corners=True)
            ds1_out = F.interpolate(self.ds1(d1_att), size=input_size, mode='bilinear', align_corners=True)
            return [ds4_out, ds3_out, ds2_out, ds1_out, logits]
        else:
            return logits

def utanet_plus_fast(input_channel=3, num_classes=1, pretrained=True):
    """
    Factory function for UTANet+ Fast model compatible with U-Bench framework.
    
    Args:
        input_channel: Number of input channels (default: 3)
        num_classes: Number of output classes (default: 1)
        pretrained: Whether to use TA-MoSC module (default: True)
        
    Returns:
        UTANetPlusFast model instance
    """
    return UTANetPlusFast(
        n_channels=input_channel,
        n_classes=num_classes,
        pretrained=pretrained,
        deep_supervision=True  # Always enable for U-Bench compatibility
    )

if __name__ == "__main__":
    print("Testing UTANet+ Fast...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UTANetPlusFast(pretrained=True, deep_supervision=True).to(device)
    
    input_t = torch.randn(2, 3, 224, 224).to(device)
    print("Input:", input_t.shape)
    
    # Check forward pass
    out = model(input_t)
    print("Output:", out.shape)
    
    # Check params
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Params: {total_params:,}")
    
    # Check Memory (rough)
    if torch.cuda.is_available():
        print(f"Allocated Memory: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
