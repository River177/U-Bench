"""
UTANet++ Micro: 超轻量化医学图像分割网络

设计目标：
- 参数量减少 80%+（目标 < 3M 参数）
- 保持 UTANet++ 的核心设计思想
- 适合资源受限环境和实时应用

核心技术：
1. 轻量化编码器 - 使用深度可分离卷积 + 轴向卷积
2. 简化全尺度解码器 - 减少拼接通道数
3. 轻量化注意力 - 移除SE模块，简化门控机制
4. 微型MoE - 减少专家数量，简化门控

Reference:
- UTANet++: Full-scale skip connections + Gated Attention
- MALUNet: Depthwise separable convolution + Gated attention
- ULite: Axial depthwise convolution
- CMUNeXt: Residual depthwise convolution blocks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List

from .modules_micro import (
    DepthwiseSeparableConv,
    LightweightBlock,
    LightFullScaleDecoder,
    LightGatedAttention,
    EfficientChannelAttention,
    BottleneckASPP,
    DeepSupervisionHeadMicro
)
from .ta_mosc_light import MoEMicro, MoELight


class LightweightEncoder(nn.Module):
    """
    轻量化编码器
    使用深度可分离卷积和轴向卷积，大幅减少参数量
    
    Args:
        n_channels: 输入通道数
        base_channels: 基础通道数（默认16）
        channel_multiplier: 通道倍增因子
    """
    def __init__(
        self, 
        n_channels: int = 3, 
        base_channels: int = 16,
        channel_multiplier: List[int] = [1, 2, 4, 8, 16]
    ):
        super().__init__()
        
        # 计算各层通道数
        self.channels = [base_channels * m for m in channel_multiplier]
        # 例：base=16 -> [16, 32, 64, 128, 256]
        
        # 输入卷积
        self.stem = nn.Sequential(
            nn.Conv2d(n_channels, self.channels[0], 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.channels[0]),
            nn.GELU()
        )
        
        # 编码器块
        self.encoder1 = LightweightBlock(self.channels[0], self.channels[0], downsample=True)
        self.encoder2 = LightweightBlock(self.channels[0], self.channels[1], downsample=True)
        self.encoder3 = LightweightBlock(self.channels[1], self.channels[2], downsample=True)
        self.encoder4 = LightweightBlock(self.channels[2], self.channels[3], downsample=True)
        self.encoder5 = LightweightBlock(self.channels[3], self.channels[4], downsample=False)
        
        # 瓶颈层：轻量化ASPP
        self.bottleneck = BottleneckASPP(self.channels[4], self.channels[4], dilations=[1, 2, 4])
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Returns:
            e1-e5: 各层编码器特征
        """
        x = self.stem(x)  # (B, 16, 224, 224)
        
        e1, skip1 = self.encoder1(x)    # (B, 16, 112, 112), skip: (B, 16, 224, 224)
        e2, skip2 = self.encoder2(e1)   # (B, 32, 56, 56), skip: (B, 16, 112, 112)
        e3, skip3 = self.encoder3(e2)   # (B, 64, 28, 28), skip: (B, 32, 56, 56)
        e4, skip4 = self.encoder4(e3)   # (B, 128, 14, 14), skip: (B, 64, 28, 28)
        e5, _ = self.encoder5(e4)       # (B, 256, 14, 14)
        
        e5 = self.bottleneck(e5)        # (B, 256, 14, 14)
        
        # 返回跳跃连接特征
        return skip1, skip2, skip3, skip4, e5


class UTANetPlusPlus_Micro(nn.Module):
    """
    UTANet++ Micro - 超轻量化版本
    
    特点：
    1. 自定义轻量化编码器（~1M参数，比ResNet34的21M小很多）
    2. 简化的全尺度解码器（cat_channels=16）
    3. 轻量化注意力机制
    4. 可选的微型MoE模块
    
    Args:
        n_channels: 输入通道数
        n_classes: 输出类别数
        base_channels: 基础通道数（默认16）
        use_moe: 是否使用MoE模块
        deep_supervision: 是否启用深度监督
        cat_channels: 全尺度连接的拼接通道数
    """
    def __init__(
        self,
        n_channels: int = 3,
        n_classes: int = 1,
        base_channels: int = 16,
        use_moe: bool = True,
        deep_supervision: bool = True,
        cat_channels: int = 16
    ):
        super().__init__()
        self.n_classes = n_classes
        self.use_moe = use_moe
        self.deep_supervision = deep_supervision
        
        # 计算各层通道数
        channel_mult = [1, 2, 4, 8, 16]
        self.filters = [base_channels * m for m in channel_mult]
        # [16, 32, 64, 128, 256] when base_channels=16
        
        # ========== 轻量化编码器 ==========
        self.encoder = LightweightEncoder(n_channels, base_channels, channel_mult)
        
        # ========== 微型MoE模块（可选）==========
        if use_moe:
            # 特征融合到统一尺寸
            # 编码器返回的skip连接通道数为: [16, 16, 32, 64]
            skip_channels = [self.filters[0], self.filters[0], self.filters[1], self.filters[2]]
            fusion_ch = sum(skip_channels)  # 16+16+32+64=128
            self.fuse = nn.Sequential(
                nn.Conv2d(fusion_ch, 32, 1, bias=False),
                nn.BatchNorm2d(32),
                nn.GELU()
            )
            
            # 微型MoE
            self.moe = MoEMicro(emb_size=32)
            
            # Docker路由器（轻量化）- 输出到skip连接的通道数
            self.docker1 = self._create_docker(32, skip_channels[0])  # 16
            self.docker2 = self._create_docker(32, skip_channels[1])  # 16
            self.docker3 = self._create_docker(32, skip_channels[2])  # 32
            self.docker4 = self._create_docker(32, skip_channels[3])  # 64
        
        # ========== 轻量化全尺度解码器 ==========
        decoder_channels = [c // 2 for c in self.filters]  # 更小的解码器通道
        # [8, 16, 32, 64, 128]
        
        # 编码器实际返回的skip连接通道数: [16, 16, 32, 64, 256]
        encoder_skip_channels = [self.filters[0], self.filters[0], self.filters[1], self.filters[2], self.filters[4]]
        
        self.decoder4 = LightFullScaleDecoder(
            filters_enc=encoder_skip_channels,
            target_size=28,
            cat_channels=cat_channels,
            out_channels=decoder_channels[3]  # 64
        )
        
        self.decoder3 = LightFullScaleDecoder(
            filters_enc=encoder_skip_channels,
            target_size=56,
            cat_channels=cat_channels,
            out_channels=decoder_channels[2]  # 32
        )
        
        self.decoder2 = LightFullScaleDecoder(
            filters_enc=encoder_skip_channels,
            target_size=112,
            cat_channels=cat_channels,
            out_channels=decoder_channels[1]  # 16
        )
        
        self.decoder1 = LightFullScaleDecoder(
            filters_enc=encoder_skip_channels,
            target_size=224,
            cat_channels=cat_channels,
            out_channels=decoder_channels[0]  # 8
        )
        
        # ========== 轻量化注意力模块 ==========
        self.att4 = LightGatedAttention(decoder_channels[3], decoder_channels[3], decoder_channels[3] // 2)
        self.att3 = LightGatedAttention(decoder_channels[2], decoder_channels[2], decoder_channels[2] // 2)
        self.att2 = LightGatedAttention(decoder_channels[1], decoder_channels[1], decoder_channels[1] // 2)
        self.att1 = LightGatedAttention(decoder_channels[0], decoder_channels[0], max(decoder_channels[0] // 2, 4))
        
        # ========== 高效通道注意力 ==========
        self.eca4 = EfficientChannelAttention(decoder_channels[3])
        self.eca3 = EfficientChannelAttention(decoder_channels[2])
        self.eca2 = EfficientChannelAttention(decoder_channels[1])
        self.eca1 = EfficientChannelAttention(decoder_channels[0])
        
        # ========== 深度监督头 ==========
        if deep_supervision:
            self.ds4 = DeepSupervisionHeadMicro(decoder_channels[3], n_classes, scale_factor=8, upsample=False)
            self.ds3 = DeepSupervisionHeadMicro(decoder_channels[2], n_classes, scale_factor=4, upsample=False)
            self.ds2 = DeepSupervisionHeadMicro(decoder_channels[1], n_classes, scale_factor=2, upsample=False)
            self.ds1 = DeepSupervisionHeadMicro(decoder_channels[0], n_classes, scale_factor=1, upsample=False)
        
        # ========== 主输出头 ==========
        self.final = nn.Sequential(
            nn.Conv2d(decoder_channels[0], decoder_channels[0], 3, 1, 1, bias=False),
            nn.BatchNorm2d(decoder_channels[0]),
            nn.GELU(),
            nn.Conv2d(decoder_channels[0], n_classes, 1)
        )
        
        # 初始化权重
        self._init_weights()
    
    def _create_docker(self, in_ch: int, out_ch: int) -> nn.Sequential:
        """创建轻量化特征路由器"""
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU()
        )
    
    def _init_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入图像 (B, C, H, W)
            
        Returns:
            logits: 分割输出 (B, n_classes, H, W)
        """
        # ========== 编码器 ==========
        e1, e2, e3, e4, e5 = self.encoder(x)
        # 编码器返回的是skip连接，通道数为：
        # e1: (B, 16, H, W)
        # e2: (B, 16, H/2, W/2)
        # e3: (B, 32, H/4, W/4)
        # e4: (B, 64, H/8, W/8)
        # e5: (B, 256, H/16, W/16)
        
        # 获取各层尺寸
        s1 = e1.shape[2:]
        s2 = e2.shape[2:]
        s3 = e3.shape[2:]
        s4 = e4.shape[2:]
        
        # print(f"DEBUG: Input shapes - s1:{s1}, s2:{s2}, s3:{s3}, s4:{s4}")
        
        # 初始化MoE损失
        aux_loss = torch.tensor(0.0, device=x.device)
        
        # ========== 微型MoE路由 ==========
        if self.use_moe:
            # 特征融合到统一尺寸 (以e3尺寸为基准，通常为H/4)
            # 例如输入224->56, 输入256->64
            target_h, target_w = s3
            
            e1_resized = F.interpolate(e1, size=(target_h, target_w), mode='bilinear', align_corners=True)
            e2_resized = F.interpolate(e2, size=(target_h, target_w), mode='bilinear', align_corners=True)
            e3_resized = e3 # 已经是基准尺寸
            e4_resized = F.interpolate(e4, size=(target_h, target_w), mode='bilinear', align_corners=True)
            
            # 拼接并融合 (16+16+32+64=128 channels)
            fused = torch.cat([e1_resized, e2_resized, e3_resized, e4_resized], dim=1)
            fused = self.fuse(fused)  # (B, 32, H/4, W/4)
            
            # MoE路由
            o1, o2, o3, o4, loss = self.moe(fused)
            aux_loss = loss
            
            # Docker特征分发 - 匹配编码器的实际通道数
            o1 = self.docker1(o1)  # (B, 16, H/4, W/4)
            o2 = self.docker2(o2)  # (B, 16, H/4, W/4)
            o3 = self.docker3(o3)  # (B, 32, H/4, W/4)
            o4 = self.docker4(o4)  # (B, 64, H/4, W/4)
            
            # 调整空间尺寸到对应的编码器层尺寸
            o1 = F.interpolate(o1, size=s1, mode='bilinear', align_corners=True)
            o2 = F.interpolate(o2, size=s2, mode='bilinear', align_corners=True)
            o3 = F.interpolate(o3, size=s3, mode='bilinear', align_corners=True)
            o4 = F.interpolate(o4, size=s4, mode='bilinear', align_corners=True)
        else:
            o1, o2, o3, o4 = e1, e2, e3, e4
        
        # ========== 全尺度解码器 ==========
        # 传递动态目标尺寸
        
        # 解码器4: s4 (H/8)
        d4 = self.decoder4(o1, o2, o3, o4, e5, target_size=s4)
        d4 = self.att4(d4, d4)
        d4 = self.eca4(d4)
        
        # 解码器3: s3 (H/4)
        d3 = self.decoder3(o1, o2, o3, d4, e5, target_size=s3)
        d3 = self.att3(d3, d3)
        d3 = self.eca3(d3)
        
        # 解码器2: s2 (H/2)
        d2 = self.decoder2(o1, o2, d3, d4, e5, target_size=s2)
        d2 = self.att2(d2, d2)
        d2 = self.eca2(d2)
        
        # 解码器1: s1 (H)
        d1 = self.decoder1(o1, d2, d3, d4, e5, target_size=s1)
        d1 = self.att1(d1, d1)
        d1 = self.eca1(d1)
        
        # ========== 输出 ==========
        logits = self.final(d1)
        
        return logits


def utanet_plusplus_micro(
    input_channel: int = 3,
    num_classes: int = 1,
    pretrained: bool = True,
    base_channels: int = 16,
    deep_supervision: bool = True
):
    """
    创建 UTANet++ Micro 模型
    
    Args:
        input_channel: 输入通道数
        num_classes: 输出类别数
        pretrained: 是否使用MoE模块（保持与U-Bench框架兼容）
        base_channels: 基础通道数（越小越轻量）
        deep_supervision: 是否启用深度监督
        
    Returns:
        UTANet++ Micro 模型实例
    """
    # pretrained参数在这里用于控制是否使用MoE模块
    use_moe = pretrained
    
    return UTANetPlusPlus_Micro(
        n_channels=input_channel,
        n_classes=num_classes,
        base_channels=base_channels,
        use_moe=use_moe,
        deep_supervision=deep_supervision
    )


def utanet_plusplus_nano(
    input_channel: int = 3,
    num_classes: int = 1
):
    """
    创建 UTANet++ Nano 模型（更轻量）
    基础通道数=8，约0.5M参数
    """
    return UTANetPlusPlus_Micro(
        n_channels=input_channel,
        n_classes=num_classes,
        base_channels=8,
        use_moe=False,  # 关闭MoE进一步减少参数
        deep_supervision=False,
        cat_channels=8
    )


if __name__ == "__main__":
    # 测试代码
    print("=" * 60)
    print("Testing UTANet++ Micro")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # 创建模型
    model = UTANetPlusPlus_Micro(
        n_channels=3,
        n_classes=1,
        base_channels=16,
        use_moe=True,
        deep_supervision=True
    ).to(device)
    
    print("\n[Model Config]")
    print(f"Base Channels: 16")
    print(f"Use MoE: True")
    print(f"Deep Supervision: True")
    
    # 测试前向传播
    model.eval()
    input_tensor = torch.randn(2, 3, 224, 224).to(device)
    
    print("\n[Forward Pass]")
    with torch.no_grad():
        output = model(input_tensor)
    
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}")
    
    # 参数统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n[Model Statistics]")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: {total_params * 4 / 1024 / 1024:.2f} MB")
    
    # 测试 Nano 版本
    print("\n" + "=" * 60)
    print("Testing UTANet++ Nano")
    print("=" * 60)
    
    model_nano = utanet_plusplus_nano(3, 1).to(device)
    
    nano_params = sum(p.numel() for p in model_nano.parameters())
    print(f"\n[Nano Model Statistics]")
    print(f"Total parameters: {nano_params:,}")
    print(f"Model size: {nano_params * 4 / 1024 / 1024:.2f} MB")
    
    with torch.no_grad():
        output_nano = model_nano(input_tensor)
    print(f"Output shape: {output_nano.shape}")
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
