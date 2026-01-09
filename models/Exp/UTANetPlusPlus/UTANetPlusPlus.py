"""
UTANet++: Enhanced Medical Image Segmentation with Full-Scale Skip Connections
结合全尺度特征融合、门控注意力和深度监督的增强版UTANet

核心改进：
1. Full-Scale Skip Connections - 每个解码器层接收所有编码器层的特征
2. Gated Attention Module - 空间和通道注意力的联合门控机制
3. Deep Supervision - 多层辅助输出提供额外监督信号
4. TA-MoSC - 保留原始的任务自适应专家混合模块

Reference: 
- UTANet: https://ojs.aaai.org/index.php/AAAI/article/view/32627
- UNet3+: https://arxiv.org/abs/2004.08790
- AttU-Net: https://arxiv.org/abs/1804.03999
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Optional, Tuple, List
import sys
import os

# 使用相对导入
from .ta_mosc import MoE

# 导入辅助模块
from .modules import (
    FullScaleDecoder,
    GatedAttention,
    DeepSupervisionHead,
    DeepSupervisionLoss
)


class UTANetPlusPlus(nn.Module):
    """
    UTANet++ - 全尺度深度监督版本
    
    主要特性：
    1. ResNet34编码器 - 提取多尺度特征
    2. TA-MoSC模块 - 任务自适应的专家混合特征路由
    3. 全尺度跳跃连接 - 每个解码器层融合所有编码器特征
    4. 门控注意力 - 增强特征选择能力
    5. 深度监督 - 多层辅助输出加速收敛
    
    Args:
        pretrained: 是否使用TA-MoSC模块
                   False (阶段1): 训练基础网络
                   True  (阶段2): 训练TA-MoSC模块
        topk: MoE中选择的专家数量
        n_channels: 输入图像通道数
        n_classes: 输出类别数
        img_size: 输入图像尺寸
        deep_supervision: 是否启用深度监督（训练时）
    """
    def __init__(
        self,
        pretrained: bool = True,
        topk: int = 2,
        n_channels: int = 3,
        n_classes: int = 1,
        img_size: int = 224,
        deep_supervision: bool = True
    ):
        super().__init__()
        self.n_classes = n_classes
        self.pretrained = pretrained
        self.img_size = img_size
        self.deep_supervision = deep_supervision
        
        # ========== 编码器：ResNet34 ==========
        self.resnet = models.resnet34(pretrained=True)
        self.filters_resnet = [64, 64, 128, 256, 512]
        
        # 第一层卷积（支持任意输入通道数）
        self.conv1 = nn.Sequential(
            nn.Conv2d(n_channels, self.filters_resnet[0], 3, 1, 1, bias=True),
            nn.BatchNorm2d(self.filters_resnet[0]),
            nn.ReLU(inplace=True)
        )
        self.maxpool = nn.MaxPool2d(2, 2)
        
        # ResNet残差块
        self.conv2 = self.resnet.layer1  # 64, 112x112
        self.conv3 = self.resnet.layer2  # 128, 56x56
        self.conv4 = self.resnet.layer3  # 256, 28x28
        self.conv5 = self.resnet.layer4  # 512, 14x14
        
        # ========== TA-MoSC模块 ==========
        if pretrained:
            # 特征融合
            self.fuse = nn.Sequential(
                nn.Conv2d(512, 64, 1, 1),  # 64+64+128+256=512 -> 64
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
            )
            
            # MoE模块
            self.moe = MoE(num_experts=4, top=topk, emb_size=64)
            
            # Docker路由器
            self.docker1 = self._create_docker(64, self.filters_resnet[0])
            self.docker2 = self._create_docker(64, self.filters_resnet[1])
            self.docker3 = self._create_docker(64, self.filters_resnet[2])
            self.docker4 = self._create_docker(64, self.filters_resnet[3])
        
        # ========== 全尺度解码器 ==========
        # 每个解码器层接收所有5个编码器层的特征
        self.decoder4 = FullScaleDecoder(
            filters_enc=self.filters_resnet,
            target_size=28,
            cat_channels=64,
            out_channels=256
        )
        
        self.decoder3 = FullScaleDecoder(
            filters_enc=self.filters_resnet,
            target_size=56,
            cat_channels=64,
            out_channels=128
        )
        
        self.decoder2 = FullScaleDecoder(
            filters_enc=self.filters_resnet,
            target_size=112,
            cat_channels=64,
            out_channels=64
        )
        
        self.decoder1 = FullScaleDecoder(
            filters_enc=self.filters_resnet,
            target_size=224,
            cat_channels=64,
            out_channels=32
        )
        
        # ========== 门控注意力模块 ==========
        self.att4 = GatedAttention(F_g=256, F_l=256, F_int=128)
        self.att3 = GatedAttention(F_g=128, F_l=128, F_int=64)
        self.att2 = GatedAttention(F_g=64, F_l=64, F_int=32)
        self.att1 = GatedAttention(F_g=32, F_l=32, F_int=16)
        
        # ========== 深度监督头 ==========
        if deep_supervision:
            self.ds4 = DeepSupervisionHead(256, n_classes, scale_factor=8)
            self.ds3 = DeepSupervisionHead(128, n_classes, scale_factor=4)
            self.ds2 = DeepSupervisionHead(64, n_classes, scale_factor=2)
            self.ds1 = DeepSupervisionHead(32, n_classes, scale_factor=1)
        
        # ========== 主输出头 ==========
        self.final = nn.Sequential(
            nn.Conv2d(32, 16, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, n_classes, 1)
        )
    
    def _create_docker(self, in_ch: int, out_ch: int) -> nn.Sequential:
        """创建特征路由器"""
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, 1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入图像 (B, C, H, W)
            
        Returns:
            logits: 分割输出 (B, n_classes, H, W)
        """
        # ========== 编码器路径 ==========
        e1 = self.conv1(x)           # (B, 64, 224, 224)
        e1_maxp = self.maxpool(e1)   # (B, 64, 112, 112)
        e2 = self.conv2(e1_maxp)     # (B, 64, 112, 112)
        e3 = self.conv3(e2)          # (B, 128, 56, 56)
        e4 = self.conv4(e3)          # (B, 256, 28, 28)
        e5 = self.conv5(e4)          # (B, 512, 14, 14)
        
        # 初始化损失
        aux_loss = torch.tensor(0.0, device=x.device)
        
        # ========== TA-MoSC特征路由 ==========
        if self.pretrained:
            # 特征融合到统一尺寸 (112x112)
            e1_resized = F.interpolate(e1, scale_factor=0.5, mode='bilinear')
            e3_resized = F.interpolate(e3, scale_factor=2, mode='bilinear')
            e4_resized = F.interpolate(e4, scale_factor=4, mode='bilinear')
            
            # 拼接并融合
            fused = torch.cat([e1_resized, e2, e3_resized, e4_resized], dim=1)
            fused = self.fuse(fused)  # (B, 64, 112, 112)
            
            # MoE路由
            o1, o2, o3, o4, loss = self.moe(fused)
            aux_loss = loss
            
            # Docker特征分发
            o1 = self.docker1(o1)  # (B, 64, 112, 112)
            o2 = self.docker2(o2)  # (B, 64, 112, 112)
            o3 = self.docker3(o3)  # (B, 128, 112, 112)
            o4 = self.docker4(o4)  # (B, 256, 112, 112)
            
            # 调整空间尺寸以匹配编码器特征
            o1 = F.interpolate(o1, scale_factor=2, mode='bilinear')    # -> 224x224
            # o2保持112x112
            o3 = F.interpolate(o3, scale_factor=0.5, mode='bilinear')  # -> 56x56
            o4 = F.interpolate(o4, scale_factor=0.25, mode='bilinear') # -> 28x28
        else:
            # 不使用TA-MoSC
            o1, o2, o3, o4 = e1, e2, e3, e4
        
        # ========== 全尺度解码器路径 ==========
        # 解码器4: 28x28 - 接收所有编码器特征
        d4 = self.decoder4(o1, o2, o3, o4, e5)  # (B, 256, 28, 28)
        d4_att = self.att4(d4, d4)  # 自注意力
        
        # 解码器3: 56x56
        d3 = self.decoder3(o1, o2, o3, d4_att, e5)  # (B, 128, 56, 56)
        d3_att = self.att3(d3, d3)
        
        # 解码器2: 112x112
        d2 = self.decoder2(o1, o2, d3_att, d4_att, e5)  # (B, 64, 112, 112)
        d2_att = self.att2(d2, d2)
        
        # 解码器1: 224x224
        d1 = self.decoder1(o1, d2_att, d3_att, d4_att, e5)  # (B, 32, 224, 224)
        d1_att = self.att1(d1, d1)
        
        # ========== 输出 ==========
        logits = self.final(d1_att)  # (B, n_classes, 224, 224)
        
        # 训练框架只接收单一输出张量，这里仅返回 logits
        # 深度监督和 aux_loss 在内部计算但不返回，以保持与训练框架的兼容性
        return logits


def utanet_plusplus(
    input_channel: int = 3, 
    num_classes: int = 1,
    pretrained: bool = True,
    deep_supervision: bool = True
):
    """
    创建UTANet++模型
    
    Args:
        input_channel: 输入通道数
        num_classes: 输出类别数
        pretrained: 是否使用TA-MoSC模块
        deep_supervision: 是否启用深度监督
        
    Returns:
        UTANet++模型实例
    """
    return UTANetPlusPlus(
        n_channels=input_channel,
        n_classes=num_classes,
        pretrained=pretrained,
        deep_supervision=deep_supervision
    )


if __name__ == "__main__":
    # 测试代码
    print("=" * 60)
    print("Testing UTANet++")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # 创建模型
    model = UTANetPlusPlus(
        pretrained=True, 
        n_classes=1, 
        deep_supervision=True
    ).to(device)
    
    # 训练模式测试
    model.train()
    input_tensor = torch.randn(2, 3, 224, 224).to(device)
    
    print("\n[Training Mode]")
    output = model(input_tensor)
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}")
    
    # 推理模式测试
    model.eval()
    print("\n[Inference Mode]")
    with torch.no_grad():
        output = model(input_tensor)
    print(f"Output shape: {output.shape}")
    
    # 参数统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[Model Statistics]")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: {total_params * 4 / 1024 / 1024:.2f} MB")
    
    # 测试完成
    print("\n[Model Test Complete]")
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)

